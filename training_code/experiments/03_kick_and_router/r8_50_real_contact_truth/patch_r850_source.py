#!/usr/bin/env python3
"""Create an R8.50 probe from the exact accepted R8.42 source.

The large historical source remains immutable.  This structural patch has
anchor assertions and can be rerun deterministically on the cluster.
"""

import argparse
from pathlib import Path


FUNCTIONS = r'''

# ============================================================
# R8.50: actual filtered contact truth; no learning.
# ============================================================

def _r850_wilson_upper(successes, total, z=1.959963984540054):
    if total <= 0:
        return 1.0
    p = float(successes) / float(total)
    d = 1.0 + z * z / total
    c = p + z * z / (2.0 * total)
    r = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return min(1.0, (c + r) / d)


def _r850_place_valid_ball(env, sensor, mask, forward, lateral):
    if not mask.any():
        return
    ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    selected, _ = env._front_foot_positions()
    direction = env.strike_direction_world[ids]
    direction = direction / torch.clamp(
        torch.linalg.vector_norm(direction, dim=1, keepdim=True), min=1.0e-6
    )
    side = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)
    xy = selected[ids, :2] + forward[ids, None] * direction + lateral[ids, None] * side
    pose = torch.zeros(ids.numel(), 7, device=env.device)
    pose[:, :2] = xy
    pose[:, 2] = env.env_origins[ids, 2] + 0.115
    pose[:, 3] = 1.0
    env.ball.write_root_pose_to_sim(pose, env_ids=ids)
    env.ball.write_root_velocity_to_sim(torch.zeros(ids.numel(), 6, device=env.device), env_ids=ids)
    env.kick_ball_start_xy[ids] = xy
    env.ball_arm_step_buf[ids] = 0
    env.stable_consecutive_buf[ids] = 0
    env.previous_ball_direction_speed[ids] = 0.0
    env.peak_ball_direction_speed[ids] = 0.0
    sensor.reset(ids)


def _r850_controller(runner, env, models, bridge_age, swing_seen):
    lift_policy, lift_head, swing_policy, stability_head, recovery_policy, recovery_head, vector_head = models
    obs = TensorDict({"policy": env.low_obs_buf}, batch_size=[env.num_envs], device=env.device)
    obs_runner = obs.to(runner.device)
    lift_base = lift_policy.act_inference(obs_runner)
    lift_norm = _r820_actor_input(lift_policy, obs_runner)
    lift_action = lift_base + lift_head(lift_norm)
    bridge = env.bridge_seen_r852.detach().bool()
    bridge_age[:] = torch.where(bridge, bridge_age + 1, torch.zeros_like(bridge_age))
    alpha = _r833_swing_alpha(bridge_seen=bridge, bridge_age=bridge_age, mode="hold2")
    swing_mean = swing_policy.act_inference(obs_runner)
    swing_norm = _r820_actor_input(swing_policy, obs_runner)
    stable = swing_mean + 1.25 * stability_head(swing_norm)
    pre = (1.0 - alpha.unsqueeze(1)) * lift_action + alpha.unsqueeze(1) * stable
    recovery_norm = _r820_actor_input(recovery_policy, obs_runner)
    recovery = stable + 1.55 * recovery_head(recovery_norm) + 0.50 * vector_head(swing_norm)
    return torch.where(swing_seen.to(runner.device).unsqueeze(1), recovery, pre)


def _r850_load_models(runner, freeze_dir):
    freeze_dir = Path(freeze_dir)
    lift_path = freeze_dir / "artifacts/r821_lift_base.pt"
    swing_path = freeze_dir / "artifacts/r815_swing_base.pt"
    lift_policy = _r824_load_policy_copy(runner, lift_path)
    swing_policy = _r824_load_policy_copy(runner, swing_path)
    recovery_policy = _r824_load_policy_copy(runner, lift_path)
    bundle = torch.load(freeze_dir / "artifacts/r823_mode1_lift_head.pt", map_location=runner.device, weights_only=False)
    lift_head = R820LiftResidualHead(obs_dim=70, hidden=64, action_dim=12, scale=float(bundle["residual_scale"])).to(runner.device)
    lift_head.load_state_dict(bundle["state_dict"])
    lift_head.eval()
    stability_head, _ = _r835_load_warm_head(path=freeze_dir / "artifacts/r835_swing_stability_u32.pt", runner=runner)
    stability_head.eval()
    recovery_head = _r840_load_recovery_head(path=freeze_dir / "artifacts/r827_recovery_mode1_head.pt", runner=runner)
    vector_head = _r840_load_vector_head(
        checkpoint=freeze_dir / "artifacts/r839_recovery_vector_u32.pt",
        scale_source=freeze_dir / "artifacts/r835_swing_stability_u32.pt",
        runner=runner,
    )
    return (lift_policy, lift_head, swing_policy, stability_head, recovery_policy, recovery_head, vector_head)


def _r850_sensor_peaks(sensor, expected_filters):
    history = sensor.data.force_matrix_w_history
    if history is None or history.ndim != 5:
        raise RuntimeError(f"R850 expected force history [N,H,1,M,3], got {None if history is None else history.shape}")
    if history.shape[2] != 1 or history.shape[3] != expected_filters:
        raise RuntimeError(f"R850 contact shape mismatch: {tuple(history.shape)}, filters={expected_filters}")
    return torch.linalg.vector_norm(history[:, :, 0, :, :], dim=-1).amax(dim=1)


def _r850_contact_controls(env, sensor, body_names):
    """Far-ball null and deliberate FL overlap positive controls."""
    n, device = env.num_envs, env.device
    zeros = torch.zeros(n, 12, device=device)
    ids = torch.arange(n, device=device)
    far = torch.zeros(n, 7, device=device)
    far[:, :2] = env.env_origins[:, :2] + torch.tensor([3.0, 0.0], device=device)
    far[:, 2] = env.env_origins[:, 2] + 0.115
    far[:, 3] = 1.0
    env.ball.write_root_pose_to_sim(far, env_ids=ids)
    env.ball.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=ids)
    env.ball_placed[:] = True
    env.kick_active[:] = False
    sensor.reset(ids)
    for _ in range(3):
        env.kick_step(zeros)
    null_peaks = _r850_sensor_peaks(sensor, len(body_names))
    null_max = float(null_peaks.max().item())
    env.reset()
    selected, _ = env._front_foot_positions()
    overlap = torch.zeros(n, 7, device=device)
    overlap[:, :3] = selected
    overlap[:, 3] = 1.0
    env.ball.write_root_pose_to_sim(overlap, env_ids=ids)
    env.ball.write_root_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=ids)
    env.ball_placed[:] = True
    env.kick_active[:] = False
    sensor.reset(ids)
    positive = torch.zeros(n, device=device)
    fl_idx = body_names.index("FL_foot")
    for _ in range(3):
        env.kick_step(zeros)
        positive = torch.maximum(positive, _r850_sensor_peaks(sensor, len(body_names))[:, fl_idx])
    positive_rate = float((positive > 0.1).float().mean().item())
    result = {"far_ball_max_force_n": null_max, "overlap_fl_detection_rate": positive_rate,
              "far_ball_pass": null_max <= 0.1, "overlap_pass": positive_rate >= 0.80}
    if not result["far_ball_pass"] or not result["overlap_pass"]:
        raise RuntimeError(f"R850 contact controls failed: {result}")
    env.reset()
    sensor.reset(ids)
    print(f"[R850_CONTROLS] null={null_max:.6f}N overlap={100*positive_rate:.1f}%", flush=True)
    return result


def run_r850_real_contact_truth(*, runner, env, freeze_dir, output_dir, body_json, episodes_per_case, seed):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    body_meta = json.loads(Path(body_json).read_text(encoding="utf-8"))
    body_names = body_meta["body_names"]
    for required in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"):
        if required not in body_names:
            raise RuntimeError(f"R850 required body absent: {required}; names={body_names}")
    fl_idx = body_names.index("FL_foot")
    foot_indices = [body_names.index(x) for x in ("FL_foot", "FR_foot", "RL_foot", "RR_foot")]
    nonfoot_indices = [i for i in range(len(body_names)) if i not in foot_indices]
    sensor = env.base_env.scene["r850_ball_robot_contact"]
    models = _r850_load_models(runner, freeze_dir)
    env.r841_grace_steps = 6
    env.total_episode_steps = int(env.total_episode_steps) + 6
    env.max_episode_length = env.total_episode_steps
    env.max_low_episode_length = env.total_episode_steps
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    if isinstance(getattr(env, "generator", None), torch.Generator):
        env.generator.manual_seed(int(seed))
    env.set_curriculum_stage("recovery")
    env.reset()
    controls = _r850_contact_controls(env, sensor, body_names)
    n, device = env.num_envs, env.device
    forwards = torch.tensor([0.18, 0.20, 0.22, 0.24, 0.26], device=device)
    laterals = torch.tensor([-0.04, -0.02, 0.0, 0.02, 0.04], device=device)
    case_count = int(forwards.numel() * laterals.numel())
    case_id = torch.arange(n, device=device) % case_count
    case_f = forwards[case_id // laterals.numel()]
    case_y = laterals[case_id % laterals.numel()]
    placed = torch.zeros(n, dtype=torch.bool, device=device)
    settle = torch.zeros(n, dtype=torch.long, device=device)
    bridge_age = torch.zeros(n, dtype=torch.long, device=device)
    swing_seen = torch.zeros(n, dtype=torch.bool, device=device)
    precontact = torch.zeros(n, dtype=torch.bool, device=device)
    selected_seen = torch.zeros(n, dtype=torch.bool, device=device)
    wrong_before_selected = torch.zeros(n, dtype=torch.bool, device=device)
    body_before_selected = torch.zeros(n, dtype=torch.bool, device=device)
    real_strike = torch.zeros(n, dtype=torch.bool, device=device)
    threshold_seen = torch.zeros(n, 3, dtype=torch.bool, device=device)
    peak_force = torch.zeros(n, device=device)
    impulse = torch.zeros(n, device=device)
    first_phase = torch.full((n,), -1.0, device=device)
    completed = torch.zeros(case_count, dtype=torch.long, device=device)
    totals = [{"episodes": 0, "selected": 0, "clean": 0, "real_strike": 0, "precontact": 0,
               "wrong_first": 0, "body_first": 0, "t01": 0, "t1": 0, "t5": 0,
               "force_sum": 0.0, "impulse_sum": 0.0} for _ in range(case_count)]
    rows_path = output_dir / "r850_episode_rows.jsonl"
    rows_file = rows_path.open("w", encoding="utf-8")
    step_dt = float(getattr(env.base_env, "step_dt", 0.02))
    max_steps = max(20000, int(episodes_per_case) * case_count * 100)
    steps = 0
    with torch.no_grad():
        while int(completed.min().item()) < int(episodes_per_case):
            newly = env.ball_placed & (~env.kick_active) & (~placed)
            if newly.any():
                _r850_place_valid_ball(env, sensor, newly, case_f, case_y)
                placed[newly] = True
            inactive = placed & (~env.kick_active)
            if inactive.any():
                peaks_now = _r850_sensor_peaks(sensor, len(body_names))
                robot_touch = peaks_now.amax(dim=1) > 0.1
                ball_z = env.ball.data.root_pos_w[:, 2] - env.env_origins[:, 2]
                lin = torch.linalg.vector_norm(env.ball.data.root_lin_vel_w, dim=1)
                ang = torch.linalg.vector_norm(env.ball.data.root_ang_vel_w, dim=1)
                stable = inactive & ((ball_z - 0.115).abs() < 0.006) & (lin < 0.03) & (ang < 0.50) & (~robot_touch)
                settle[:] = torch.where(stable, settle + 1, torch.where(inactive, torch.zeros_like(settle), settle))
                hold = inactive & (settle < 12)
                env.ball_arm_step_buf[hold] = 0
                env.stable_consecutive_buf[hold] = 0
                precontact |= inactive & robot_touch
            action = _r850_controller(runner, env, models, bridge_age, swing_seen)
            active_before = env.kick_active.detach().bool().clone()
            phase_before = env._phase().detach().clone()
            selected_before, _ = env._front_foot_positions()
            ball_vel_before = env.ball.data.root_lin_vel_w.detach().clone()
            ball_pos_before = env.ball.data.root_pos_w.detach().clone()
            _, _, done, _, diag = env.kick_step(action.to(device))
            done_mask = done.to(device).bool()
            selected_after, _ = env._front_foot_positions()
            ball_vel_after = env.ball.data.root_lin_vel_w.detach()
            ball_pos_after = env.ball.data.root_pos_w.detach()
            direction = env.strike_direction_world.detach()
            direction = direction / torch.clamp(torch.linalg.vector_norm(direction, dim=1, keepdim=True), min=1.0e-6)
            pair_peak = _r850_sensor_peaks(sensor, len(body_names))
            selected_force = pair_peak[:, fl_idx]
            other_force = pair_peak[:, foot_indices[1:]].amax(dim=1)
            body_force = pair_peak[:, nonfoot_indices].amax(dim=1) if nonfoot_indices else torch.zeros_like(selected_force)
            physical = active_before & (~done_mask)
            selected_contact = physical & (selected_force > 1.0)
            wrong_contact = physical & (other_force > 1.0)
            body_contact = physical & (body_force > 1.0)
            not_yet = ~selected_seen
            wrong_before_selected |= wrong_contact & not_yet
            body_before_selected |= body_contact & not_yet
            selected_vel = (selected_after - selected_before) / max(step_dt, 1.0e-6)
            toward = (selected_vel[:, :2] * direction).sum(dim=1)
            delta_ball = ((ball_vel_after - ball_vel_before)[:, :2] * direction).sum(dim=1)
            first = selected_contact & (~selected_seen)
            first_phase[first] = phase_before[first]
            clean = first & (~precontact) & (~wrong_before_selected) & (~body_before_selected)
            real_strike |= clean & (toward > 0.0) & (delta_ball > 0.02)
            selected_seen |= selected_contact
            threshold_seen[:, 0] |= physical & (selected_force > 0.1)
            threshold_seen[:, 1] |= selected_contact
            threshold_seen[:, 2] |= physical & (selected_force > 5.0)
            peak_force = torch.maximum(peak_force, torch.where(physical, selected_force, torch.zeros_like(selected_force)))
            impulse += torch.where(physical, selected_force * step_dt, torch.zeros_like(selected_force))
            swing_seen |= env.r817_swing_now.detach().bool()
            if done_mask.any():
                ids = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
                for idx in ids.tolist():
                    cid = int(case_id[idx].item())
                    if completed[cid] >= int(episodes_per_case):
                        continue
                    clean_ep = bool(selected_seen[idx] and not precontact[idx] and not wrong_before_selected[idx] and not body_before_selected[idx])
                    row = {"case_id": cid, "forward_m": float(case_f[idx]), "lateral_m": float(case_y[idx]),
                           "selected_contact": bool(selected_seen[idx]), "clean_selected_contact": clean_ep,
                           "real_strike": bool(real_strike[idx]), "precontact": bool(precontact[idx]),
                           "wrong_before_selected": bool(wrong_before_selected[idx]), "body_before_selected": bool(body_before_selected[idx]),
                           "peak_selected_force_n": float(peak_force[idx]), "selected_impulse_ns": float(impulse[idx]),
                           "first_contact_phase": float(first_phase[idx]),
                           "threshold_0p1": bool(threshold_seen[idx, 0]), "threshold_1": bool(threshold_seen[idx, 1]),
                           "threshold_5": bool(threshold_seen[idx, 2])}
                    rows_file.write(json.dumps(row, sort_keys=True) + "\n")
                    t = totals[cid]; t["episodes"] += 1; t["selected"] += int(row["selected_contact"])
                    t["clean"] += int(clean_ep); t["real_strike"] += int(row["real_strike"])
                    t["precontact"] += int(row["precontact"]); t["wrong_first"] += int(row["wrong_before_selected"])
                    t["body_first"] += int(row["body_before_selected"]); t["t01"] += int(row["threshold_0p1"])
                    t["t1"] += int(row["threshold_1"]); t["t5"] += int(row["threshold_5"])
                    t["force_sum"] += row["peak_selected_force_n"]; t["impulse_sum"] += row["selected_impulse_ns"]
                    completed[cid] += 1
                placed[done_mask] = False; settle[done_mask] = 0; bridge_age[done_mask] = 0; swing_seen[done_mask] = False
                precontact[done_mask] = False; selected_seen[done_mask] = False; wrong_before_selected[done_mask] = False
                body_before_selected[done_mask] = False; real_strike[done_mask] = False; threshold_seen[done_mask] = False
                peak_force[done_mask] = 0.0; impulse[done_mask] = 0.0; first_phase[done_mask] = -1.0
            steps += 1
            if steps > max_steps:
                raise RuntimeError(f"R850 safety timeout completed={completed.tolist()}")
    rows_file.close()
    cases = []
    for cid, t in enumerate(totals):
        total = max(t["episodes"], 1)
        cases.append({"case_id": cid, "forward_m": float(forwards[cid // len(laterals)]),
                      "lateral_m": float(laterals[cid % len(laterals)]), **t,
                      "selected_rate": t["selected"] / total, "clean_rate": t["clean"] / total,
                      "real_strike_rate": t["real_strike"] / total,
                      "failure_wilson_upper": _r850_wilson_upper(total - t["real_strike"], total),
                      "mean_peak_force_n": t["force_sum"] / total, "mean_impulse_ns": t["impulse_sum"] / total})
    ranked = sorted(cases, key=lambda x: (x["real_strike_rate"], x["clean_rate"], x["mean_impulse_ns"], -x["precontact"]), reverse=True)
    summary = {"schema": "r850-real-contact-v1", "training": False, "source_contract": "R8.42 accepted ball-free",
               "ball_radius_m": 0.11, "ball_center_height_m": 0.115, "contact_thresholds_n": [0.1, 1.0, 5.0],
               "body_names": body_names, "controls": controls,
               "episodes_per_case": int(episodes_per_case), "steps": steps,
               "cases": cases, "winner": ranked[0],
               "next_gate": "R8.51 may train only if controls pass and winner has nonzero clean real_strike."}
    (output_dir / "r850_real_contact_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[R850_RESULT] winner={ranked[0]['case_id']} real={100*ranked[0]['real_strike_rate']:.2f}% clean={100*ranked[0]['clean_rate']:.2f}%", flush=True)
    print("R850_REAL_CONTACT_OK", flush=True)
'''


CFG = r'''
    if os.environ.get("R850_REAL_CONTACT_ONLY", "0") == "1":
        from isaaclab.sensors import ContactSensorCfg
        body_meta = json.loads(Path(os.environ["R850_BODY_NAMES_JSON"]).read_text(encoding="utf-8"))
        body_names = body_meta["body_names"]
        if not body_names:
            raise RuntimeError("R850 body discovery returned no names")
        env_cfg.scene.ball.spawn.activate_contact_sensors = True
        env_cfg.scene.r850_ball_robot_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/SoccerBall",
            update_period=0.0,
            history_length=8,
            track_pose=False,
            track_contact_points=False,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/{name}" for name in body_names],
        )

'''


BRANCH = r'''
    if os.environ.get("R850_REAL_CONTACT_ONLY", "0") == "1":
        run_r850_real_contact_truth(
            runner=runner,
            env=env,
            freeze_dir=Path(os.environ["R850_FREEZE_DIR"]).resolve(),
            output_dir=Path(os.environ["R850_OUTPUT"]).resolve(),
            body_json=Path(os.environ["R850_BODY_NAMES_JSON"]).resolve(),
            episodes_per_case=int(os.environ.get("R850_EPISODES_PER_CASE", "128")),
            seed=args_cli.seed,
        )
        return

'''


def replace_once(text: str, anchor: str, replacement: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(f"Expected exactly one anchor, found {count}: {anchor[:80]!r}")
    return text.replace(anchor, replacement, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    text = source.read_text(encoding="utf-8")
    text = replace_once(text, "\ndef main() -> None:\n", FUNCTIONS + "\n\ndef main() -> None:\n")
    text = replace_once(text, "    low_agent_cfg = instantiate_cfg(\n", CFG + "    low_agent_cfg = instantiate_cfg(\n")
    text = replace_once(text, "    if os.environ.get(\n        \"R841_GRACE_ONLY\",", BRANCH + "    if os.environ.get(\n        \"R841_GRACE_ONLY\",")
    compile(text, str(output), "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"R850_PATCH_OK source={source} output={output}")


if __name__ == "__main__":
    main()
