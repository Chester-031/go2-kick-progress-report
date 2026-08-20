#!/usr/bin/env python3
"""Add the R8.52 causal scan to the completed R8.51 generated source."""

import argparse
from pathlib import Path


FUNCTIONS = r'''

# ============================================================
# R8.52: no learning; residual gain x recovery-delay scan.
# ============================================================

def run_r852_gain_recovery_scan(*, runner, env, freeze_dir, kick_checkpoint,
                                recovery_checkpoint, r851_summary, body_json,
                                output_dir, episodes_per_case, seed):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    summary851 = json.loads(Path(r851_summary).read_text(encoding="utf-8"))
    body_names = json.loads(Path(body_json).read_text(encoding="utf-8"))["body_names"]
    top = torch.tensor(summary851["top_geometry"], device=env.device)
    components = _r848_load_frozen_components(runner=runner, freeze_dir=freeze_dir)
    for module in components.values():
        if isinstance(module, torch.nn.Module):
            module.eval()
            for p in module.parameters(): p.requires_grad_(False)
    _r851_load_recovery_vector(components, recovery_checkpoint, runner.device)
    bundle = torch.load(kick_checkpoint, map_location=runner.device, weights_only=False)
    model = R851Kick64ActorCritic(residual_scale=float(bundle["residual_scale"]), action_std=float(bundle["action_std"])).to(runner.device)
    model.load_state_dict(bundle["state_dict"]); model.eval()
    sensor = env.base_env.scene["r851_ball_robot_contact"]
    env.r841_grace_steps = 6
    env.total_episode_steps = env.settle_max_steps + env.ball_arm_steps + env.kick_episode_steps + 6
    env.max_episode_length = env.total_episode_steps; env.max_low_episode_length = env.total_episode_steps
    env.set_curriculum_stage("recovery"); torch.manual_seed(seed); np.random.seed(seed); env.reset()
    n, device = env.num_envs, env.device
    gains_table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], device=device)
    delays_table = torch.tensor([0, 2, 4, 6], dtype=torch.long, device=device)
    case_count = int(gains_table.numel() * delays_table.numel())
    case_id = torch.arange(n, device=device) % case_count
    gain = gains_table[case_id // delays_table.numel()]
    delay = delays_table[case_id % delays_table.numel()]
    forward = top[0, 0].repeat(n); lateral = top[0, 1].repeat(n); strength = torch.ones(n, device=device)
    placed = torch.zeros(n, dtype=torch.bool, device=device); settle = torch.zeros(n, dtype=torch.long, device=device)
    contaminated = torch.zeros(n, dtype=torch.bool, device=device); selected_seen = torch.zeros(n, dtype=torch.bool, device=device)
    wrong_seen = torch.zeros(n, dtype=torch.bool, device=device); body_seen = torch.zeros(n, dtype=torch.bool, device=device)
    real_seen = torch.zeros(n, dtype=torch.bool, device=device); contact_age = torch.zeros(n, dtype=torch.long, device=device)
    peak_speed = torch.zeros(n, device=device); max_disp = torch.zeros(n, device=device)
    gate_names = ("rec_candidate", "forward", "lift", "return", "contact", "support4", "base", "height", "upright", "body_ok", "touchdown", "all_now", "recovery")
    gates = {name: torch.zeros(n, dtype=torch.bool, device=device) for name in gate_names}
    completed = torch.zeros(case_count, dtype=torch.long, device=device)
    totals = [{"episodes": 0, "selected": 0, "clean": 0, "real": 0, "wrong": 0, "body": 0, "precontact": 0,
               "speed_sum": 0.0, "disp_sum": 0.0, **{name: 0 for name in gate_names}} for _ in range(case_count)]
    rows = (output_dir / "r852_episode_rows.jsonl").open("w", encoding="utf-8")
    step_dt = float(getattr(env.base_env, "step_dt", 0.02)); loops = 0
    with torch.no_grad():
        while int(completed.min()) < int(episodes_per_case):
            contaminated |= _r851_prepare_episode_state(env, sensor, body_names, placed, settle, forward, lateral, strength)
            obs_runner = TensorDict({"policy": env.low_obs_buf}, batch_size=[n], device=device).to(runner.device)
            obs64 = _r851_obs64(env.low_obs_buf).to(runner.device)
            active = env.kick_active.detach().bool().clone(); bridge = env.bridge_seen_r852.detach().bool().clone()
            recovery_control = real_seen & (contact_age >= delay)
            frozen, _, _ = _r851_frozen_action(runner, env, components, obs_runner, bridge, recovery_control)
            residual = model.mean(obs64) * gain.to(runner.device).unsqueeze(1)
            strike_active = active & bridge & (~recovery_control)
            action = frozen + strike_active.to(runner.device).unsqueeze(1) * residual
            selected_before, _ = env._front_foot_positions(); selected_before = selected_before.detach().clone()
            ball_vel_before = env.ball.data.root_lin_vel_w.detach().clone()
            direction = env.strike_direction_world.detach().clone(); direction = direction / torch.clamp(torch.linalg.vector_norm(direction, dim=1, keepdim=True), min=1e-6)
            _, _, done, _, _ = env.kick_step(action.to(device)); done = done.to(device).bool()
            cs = _r851_contact_state(env, sensor, body_names, selected_before, ball_vel_before, direction, active, done, step_dt)
            new_selected = cs["selected"] & (~selected_seen)
            wrong_seen |= cs["wrong"] & (~selected_seen); body_seen |= cs["body"] & (~selected_seen)
            clean_now = new_selected & (~contaminated) & (~wrong_seen) & (~body_seen)
            real_now = clean_now & (cs["toward"] > 0.0) & (cs["dv"] > 0.02)
            selected_seen |= cs["selected"]; real_seen |= real_now
            contact_age[:] = torch.where(real_seen, contact_age + 1, torch.zeros_like(contact_age))
            valid_gate = real_seen & (~done)
            attrs = {
                "rec_candidate": env.r829_rec_candidate, "forward": env.r829_forward_ok,
                "lift": env.r829_lift_ok, "return": env.r829_return_ok,
                "contact": env.r829_contact_ok, "support4": env.r829_support4_ok,
                "base": env.r829_base_ok, "height": env.r829_height_ok,
                "upright": env.r829_upright_ok, "body_ok": env.r829_body_ok,
                "touchdown": env.r829_touchdown_ok, "all_now": env.r829_all_now,
                "recovery": env.r829_recovery_seen,
            }
            for name, value in attrs.items(): gates[name] |= valid_gate & value.detach().bool()
            peak_speed = torch.maximum(peak_speed, torch.where(active & (~done), cs["speed"], torch.zeros_like(cs["speed"])))
            disp = ((env.ball.data.root_pos_w[:, :2] - env.kick_ball_start_xy) * direction).sum(1)
            max_disp = torch.maximum(max_disp, torch.where(active & (~done), disp, torch.zeros_like(disp)))
            if done.any():
                ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
                for idx in ids.tolist():
                    cid = int(case_id[idx]);
                    if int(completed[cid]) >= int(episodes_per_case): continue
                    clean_ep = bool(selected_seen[idx] and not contaminated[idx] and not wrong_seen[idx] and not body_seen[idx])
                    row = {"case_id": cid, "gain": float(gain[idx]), "delay_steps": int(delay[idx]),
                           "selected": bool(selected_seen[idx]), "clean": clean_ep, "real": bool(real_seen[idx]),
                           "wrong": bool(wrong_seen[idx]), "body": bool(body_seen[idx]), "precontact": bool(contaminated[idx]),
                           "peak_speed_mps": float(peak_speed[idx]), "directional_disp_m": float(max_disp[idx]),
                           **{name: bool(gates[name][idx]) for name in gate_names}}
                    rows.write(json.dumps(row, sort_keys=True) + "\n"); t = totals[cid]; t["episodes"] += 1
                    for name in ("selected", "clean", "real", "wrong", "body", "precontact"): t[name] += int(row[name])
                    for name in gate_names: t[name] += int(row[name])
                    t["speed_sum"] += row["peak_speed_mps"]; t["disp_sum"] += row["directional_disp_m"]; completed[cid] += 1
                for tensor in (placed, contaminated, selected_seen, wrong_seen, body_seen, real_seen): tensor[done] = False
                contact_age[done] = 0; settle[done] = 0; peak_speed[done] = 0.0; max_disp[done] = 0.0
                for value in gates.values(): value[done] = False
            loops += 1
            if loops > max(20000, episodes_per_case * case_count * 100): raise RuntimeError("R852 safety timeout")
    rows.close(); cases = []
    for cid, t in enumerate(totals):
        d = max(t["episodes"], 1); real_d = max(t["real"], 1)
        case = {"case_id": cid, "gain": float(gains_table[cid // len(delays_table)]), "delay_steps": int(delays_table[cid % len(delays_table)]),
                "episodes": t["episodes"], "selected_rate": t["selected"]/d, "clean_rate": t["clean"]/d,
                "real_rate": t["real"]/d, "wrong_rate": t["wrong"]/d, "body_rate": t["body"]/d,
                "precontact_rate": t["precontact"]/d, "mean_peak_speed_mps": t["speed_sum"]/d,
                "mean_directional_disp_m": t["disp_sum"]/d,
                **{f"{name}_given_real": t[name]/real_d for name in gate_names}}
        cases.append(case)
    by_delay = {}
    for delay_value in (0, 2, 4, 6):
        seq = sorted((x for x in cases if x["delay_steps"] == delay_value), key=lambda x:x["gain"])
        speeds = [x["mean_peak_speed_mps"] for x in seq]
        by_delay[str(delay_value)] = {"speeds": speeds, "strict_monotonic": all(b >= a + 0.005 for a,b in zip(speeds,speeds[1:]))}
    ranked_recovery = sorted(cases, key=lambda x:(x["recovery_given_real"], x["all_now_given_real"], x["touchdown_given_real"], x["mean_peak_speed_mps"]), reverse=True)
    summary = {"version":"r8_52_gain_recovery_scan", "training":False, "actual_contact":True,
               "parent":str(kick_checkpoint), "geometry":top[0].detach().cpu().tolist(),
               "episodes_per_case":int(episodes_per_case), "cases":cases, "strength_by_delay":by_delay,
               "best_recovery_case":ranked_recovery[0],
               "verdict":"USE_CAUSAL_GAIN_AND_DELAY_FOR_R853" if any(x["strict_monotonic"] for x in by_delay.values()) or ranked_recovery[0]["all_now_given_real"] > 0 else "NO_CAUSAL_CONTROL_FOUND_REDESIGN_ACTION"}
    (output_dir/"r852_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    b=ranked_recovery[0]
    print(f"[R852_RESULT] bestGain={b['gain']:.2f} delay={b['delay_steps']} real={100*b['real_rate']:.2f}% touchdown={100*b['touchdown_given_real']:.2f}% all={100*b['all_now_given_real']:.2f}% rec={100*b['recovery_given_real']:.2f}% verdict={summary['verdict']}",flush=True)
    print("R852_GAIN_RECOVERY_SCAN_OK",flush=True)
'''


BRANCH = r'''
    if os.environ.get("R852_GAIN_RECOVERY_ONLY", "0") == "1":
        run_r852_gain_recovery_scan(
            runner=runner, env=env,
            freeze_dir=Path(os.environ["R852_FREEZE_DIR"]).resolve(),
            kick_checkpoint=Path(os.environ["R852_KICK"]).resolve(),
            recovery_checkpoint=Path(os.environ["R852_RECOVERY"]).resolve(),
            r851_summary=Path(os.environ["R852_R851_SUMMARY"]).resolve(),
            body_json=Path(os.environ["R852_BODY_JSON"]).resolve(),
            output_dir=Path(os.environ["R852_OUTPUT"]).resolve(),
            episodes_per_case=int(os.environ.get("R852_EPISODES_PER_CASE", "128")),
            seed=args_cli.seed,
        )
        return

'''


def once(text, old, new):
    count=text.count(old)
    if count != 1: raise RuntimeError(f"expected one anchor, found {count}: {old[:100]!r}")
    return text.replace(old,new,1)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    source=Path(a.source).resolve(); output=Path(a.output).resolve(); text=source.read_text(encoding="utf-8")
    text=once(text,"\ndef main() -> None:\n",FUNCTIONS+"\n\ndef main() -> None:\n")
    text=once(text,'    if os.environ.get("R851_KICK64_ONLY", "0") == "1":\n        from isaaclab.sensors import ContactSensorCfg\n',
              '    if os.environ.get("R851_KICK64_ONLY", "0") == "1" or os.environ.get("R852_GAIN_RECOVERY_ONLY", "0") == "1":\n        from isaaclab.sensors import ContactSensorCfg\n')
    text=once(text,'    if os.environ.get("R851_KICK64_ONLY", "0") == "1":\n        run_r851_kick64(\n',BRANCH+'    if os.environ.get("R851_KICK64_ONLY", "0") == "1":\n        run_r851_kick64(\n')
    compile(text,str(output),"exec"); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(text,encoding="utf-8")
    print(f"R852_PATCH_OK source={source} output={output}")


if __name__=="__main__": main()

