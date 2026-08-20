#!/usr/bin/env python3
import argparse
from pathlib import Path


INSERT = r'''

# ============================================================
# R8.58: targeted low/mid static-ramp hold controller
# ============================================================

def _r858_candidates(device, fl):
    biases, labels = [], []
    for thigh in (0.12, 0.16, 0.20, 0.24, 0.28, 0.32):
        for calf in (-0.04, 0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24):
            for ramp in (0.0, 0.35, 0.70):
                b = torch.zeros(12, device=device)
                b[fl["thigh"]] = thigh
                b[fl["calf"]] = calf
                biases.append(b)
                labels.append({"mode": "static_ramp", "thigh": thigh, "calf": calf, "ramp": ramp})
    return torch.stack(biases), labels


def _r858_rollout(*, runner, env, components, kick, recovery, sensor, body_names,
                  geometry, strength, biases, labels, episodes_per_case, seed,
                  hold_steps=5, geometry_jitter=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    eg = getattr(env, "generator", None)
    if isinstance(eg, torch.Generator):
        eg.manual_seed(seed + 858)
    env.reset()
    n, device = env.num_envs, env.device
    cases = biases.shape[0]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 85800)
    schedule = _r857_schedule(cases, episodes_per_case, device, gen)
    total = int(schedule.numel())
    cursor = min(n, total)
    case_id = torch.zeros(n, dtype=torch.long, device=device)
    assigned = torch.zeros(n, dtype=torch.bool, device=device)
    case_id[:cursor] = schedule[:cursor]
    assigned[:cursor] = True
    ramps = torch.tensor([float(x.get("ramp", 0.0)) for x in labels], device=device)
    desired = torch.full((n,), float(strength), device=device)
    env_strength = torch.ones(n, device=device)
    forward = geometry[0].repeat(n)
    lateral = geometry[1].repeat(n)

    def reset_geometry(mask):
        ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return
        if geometry_jitter > 0.0:
            j = (2.0 * torch.rand((ids.numel(), 2), device=device, generator=gen) - 1.0) * geometry_jitter
            forward[ids] = geometry[0] + j[:, 0]
            lateral[ids] = geometry[1] + j[:, 1]
        else:
            forward[ids] = geometry[0]
            lateral[ids] = geometry[1]

    reset_geometry(torch.ones(n, dtype=torch.bool, device=device))
    placed = torch.zeros(n, dtype=torch.bool, device=device)
    settle = torch.zeros(n, dtype=torch.long, device=device)
    contaminated = torch.zeros(n, dtype=torch.bool, device=device)
    selected_seen = torch.zeros(n, dtype=torch.bool, device=device)
    wrong_seen = torch.zeros(n, dtype=torch.bool, device=device)
    body_seen = torch.zeros(n, dtype=torch.bool, device=device)
    real_seen = torch.zeros(n, dtype=torch.bool, device=device)
    touchdown_seen = torch.zeros(n, dtype=torch.bool, device=device)
    recovery_seen = torch.zeros(n, dtype=torch.bool, device=device)
    hold_remaining = torch.zeros(n, dtype=torch.long, device=device)
    hold_action = torch.zeros(n, 12, device=runner.device)
    completed = torch.zeros(cases, dtype=torch.long, device=device)
    realc = torch.zeros(cases, device=device)
    tdc = torch.zeros(cases, device=device)
    recc = torch.zeros(cases, device=device)
    wrongc = torch.zeros(cases, device=device)
    bodyc = torch.zeros(cases, device=device)
    step_dt = float(getattr(env.base_env, "step_dt", 0.02))
    scored = loops = 0
    with torch.no_grad():
        while scored < total:
            contaminated |= _r851_prepare_episode_state(env, sensor, body_names, placed, settle, forward, lateral, env_strength)
            obs = TensorDict({"policy": env.low_obs_buf}, batch_size=[n], device=device).to(runner.device)
            active = env.kick_active.bool().clone()
            bridge = env.bridge_seen_r852.bool().clone()
            phase = env._phase().clone()
            rc = real_seen & env.swing_seen_r89.bool() & (phase >= 0.78)
            holding = (hold_remaining > 0) & rc
            frozen, stable, snorm = _r851_frozen_action(runner, env, components, obs, bridge, torch.zeros_like(rc))
            kd = kick.mean(_r851_obs64(env.low_obs_buf).to(runner.device)) * _r853_gain(desired).to(runner.device).unsqueeze(1)
            rd = recovery.mean(_r854_condition(_r851_obs64(snorm), desired))
            pre = frozen + (active & bridge & (~rc)).to(runner.device).unsqueeze(1) * kd
            proposed = torch.where(rc.to(runner.device).unsqueeze(1), _r853_recovery_action(runner, components, obs, stable, rd), pre)
            age = (hold_steps - hold_remaining).clamp(min=0).float()
            mult = 1.0 + ramps[case_id] * age
            held = torch.clamp(hold_action + biases[case_id].to(runner.device) * mult.to(runner.device).unsqueeze(1), -1.0, 1.0)
            action = torch.where(holding.to(runner.device).unsqueeze(1), held, proposed)
            sb, _ = env._front_foot_positions()
            sb = sb.detach().clone()
            bv = env.ball.data.root_lin_vel_w.detach().clone()
            direction = env.strike_direction_world.detach().clone()
            direction = direction / torch.clamp(torch.linalg.vector_norm(direction, dim=1, keepdim=True), min=1e-6)
            _, _, done, _, _ = env.kick_step(action.to(device))
            done = done.to(device).bool()
            cs = _r851_contact_state(env, sensor, body_names, sb, bv, direction, active, done, step_dt)
            ns = cs["selected"] & (~selected_seen)
            wrong_seen |= cs["wrong"] & (~selected_seen)
            body_seen |= cs["body"] & (~selected_seen)
            rn = ns & (~contaminated) & (~wrong_seen) & (~body_seen) & (cs["toward"] > 0) & (cs["dv"] > 0.02)
            selected_seen |= cs["selected"]
            real_seen |= rn
            touchdown_seen |= real_seen & env.r829_touchdown_ok.bool()
            hold_remaining = torch.clamp(hold_remaining - 1, min=0)
            nl = real_seen & env.r829_all_now.bool() & (~holding) & (~done)
            hold_action[nl.to(runner.device)] = action[nl.to(runner.device)].detach()
            hold_remaining[nl] = hold_steps
            recovery_seen |= real_seen & env.r829_recovery_seen.bool()
            if done.any():
                ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
                score_ids = ids[assigned[ids]]
                if score_ids.numel():
                    cid = case_id[score_ids]
                    completed += torch.bincount(cid, minlength=cases)
                    realc += torch.bincount(cid, weights=real_seen[score_ids].float(), minlength=cases)
                    tdc += torch.bincount(cid, weights=(real_seen[score_ids] & touchdown_seen[score_ids]).float(), minlength=cases)
                    recc += torch.bincount(cid, weights=(real_seen[score_ids] & recovery_seen[score_ids]).float(), minlength=cases)
                    wrongc += torch.bincount(cid, weights=wrong_seen[score_ids].float(), minlength=cases)
                    bodyc += torch.bincount(cid, weights=body_seen[score_ids].float(), minlength=cases)
                    scored += int(score_ids.numel())
                assigned[ids] = False
                take = min(int(ids.numel()), total - cursor)
                if take > 0:
                    target = ids[:take]
                    case_id[target] = schedule[cursor:cursor + take]
                    assigned[target] = True
                    cursor += take
                for t in (placed, contaminated, selected_seen, wrong_seen, body_seen, real_seen, touchdown_seen, recovery_seen):
                    t[done] = False
                settle[done] = 0
                hold_remaining[done] = 0
                hold_action[done.to(runner.device)] = 0
                reset_geometry(done)
            loops += 1
            if loops > max(300000, total * (env.total_episode_steps + 100) // max(n, 1) * 5):
                raise RuntimeError("R858 balanced rollout timeout")
    if not bool(torch.all(completed == episodes_per_case)):
        raise RuntimeError(f"R858 imbalance completed={completed.tolist()} expected={episodes_per_case}")
    rows = []
    for i in range(cases):
        ep = int(completed[i]); real = max(float(realc[i]), 1.0); td = max(float(tdc[i]), 1.0)
        rows.append({"case": i, "episodes": ep, "real_rate": float(realc[i]) / ep,
                     "touchdown_given_real": float(tdc[i]) / real,
                     "recovery_given_real": float(recc[i]) / real,
                     "retention_given_touchdown": float(recc[i]) / td,
                     "wrong_rate": float(wrongc[i]) / ep, "body_rate": float(bodyc[i]) / ep})
    return rows


def _r858_bias(device, fl, controller):
    b = torch.zeros(1, 12, device=device)
    b[0, fl["thigh"]] = float(controller["thigh"])
    b[0, fl["calf"]] = float(controller["calf"])
    return b


def _r858_rank(row):
    c = row["controller"]
    return (row["recovery_given_real"], row["retention_given_touchdown"],
            row["touchdown_given_real"], -(abs(c["thigh"]) + abs(c["calf"]) + 0.1 * c.get("ramp", 0.0)))


def run_r858_targeted_hold(*, runner, env, freeze_dir, kick_checkpoint, r851_summary,
                           body_json, recovery_checkpoint, r857_winner, output_dir,
                           screen_episodes, semifinal_episodes, final_episodes, seed):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    geometry = torch.tensor(json.loads(Path(r851_summary).read_text(encoding="utf-8"))["top_geometry"][0], device=env.device)
    body_names = json.loads(Path(body_json).read_text(encoding="utf-8"))["body_names"]
    prior = json.loads(Path(r857_winner).read_text(encoding="utf-8"))
    components, kick, recovery = _r856_load_chain(runner, env, freeze_dir, kick_checkpoint, recovery_checkpoint)
    sensor = env.base_env.scene["r851_ball_robot_contact"]
    names, fl = _r856_fl_joint_ids(env)
    biases, labels = _r858_candidates(env.device, fl)
    scans, semifinals, finals, winners = {}, {}, {}, {}
    for si, strength in enumerate((0.75, 1.0)):
        rows = _r858_rollout(runner=runner, env=env, components=components, kick=kick, recovery=recovery,
            sensor=sensor, body_names=body_names, geometry=geometry, strength=strength,
            biases=biases, labels=labels, episodes_per_case=screen_episodes,
            seed=seed + 85800 + si, geometry_jitter=0.003)
        for row, label in zip(rows, labels): row["controller"] = label
        scans[str(strength)] = rows
        eligible = [x for x in rows if x["real_rate"] >= 0.90 and x["touchdown_given_real"] >= 0.75 and x["wrong_rate"] <= 0.02 and x["body_rate"] <= 0.02]
        top = sorted(eligible or rows, key=_r858_rank, reverse=True)[:8]
        semi = []
        for rank, candidate in enumerate(top):
            ctrl = candidate["controller"]
            rr = _r858_rollout(runner=runner, env=env, components=components, kick=kick, recovery=recovery,
                sensor=sensor, body_names=body_names, geometry=geometry, strength=strength,
                biases=_r858_bias(env.device, fl, ctrl), labels=[ctrl], episodes_per_case=semifinal_episodes,
                seed=seed + 858000 + si * 100 + rank, geometry_jitter=0.004)[0]
            rr["controller"] = ctrl; rr["screen_case"] = candidate["case"]; semi.append(rr)
        semifinals[str(strength)] = semi
        valid = [x for x in semi if x["real_rate"] >= 0.95 and x["touchdown_given_real"] >= 0.80 and x["wrong_rate"] <= 0.02 and x["body_rate"] <= 0.02]
        winner = max(valid or semi, key=_r858_rank); winners[str(strength)] = winner
        print(f"[R858_SEMIFINAL] s={strength} ctrl={winner['controller']} td={100*winner['touchdown_given_real']:.1f}% rec3={100*winner['recovery_given_real']:.1f}% retain={100*winner['retention_given_touchdown']:.1f}%", flush=True)
    high = dict(prior["by_strength"]["1.25"]); high["mode"] = "static_ramp"; high["ramp"] = 0.0
    winners["1.25"] = {"controller": high, "source": "frozen_r857"}
    for si, strength in enumerate((0.75, 1.0, 1.25)):
        ctrl = winners[str(strength)]["controller"]
        out = []
        for ci, jitter in enumerate((0.0, 0.004)):
            rr = _r858_rollout(runner=runner, env=env, components=components, kick=kick, recovery=recovery,
                sensor=sensor, body_names=body_names, geometry=geometry, strength=strength,
                biases=_r858_bias(env.device, fl, ctrl), labels=[ctrl], episodes_per_case=final_episodes,
                seed=seed + 8580000 + si * 10 + ci, geometry_jitter=jitter)[0]
            rr.update({"strength": strength, "controller": ctrl, "seed": seed + 8580000 + si * 10 + ci,
                       "geometry_jitter_m": jitter}); out.append(rr)
        finals[str(strength)] = out
        print(f"[R858_CONFIRM] s={strength} " + " ".join(f"j{x['geometry_jitter_m']:.3f}:td={100*x['touchdown_given_real']:.1f}%/rec3={100*x['recovery_given_real']:.1f}%" for x in out), flush=True)
    flat = [x for rows in finals.values() for x in rows]
    accepted = all(x["real_rate"] >= 0.95 and x["touchdown_given_real"] >= 0.80 and
                   x["recovery_given_real"] >= 0.60 and x["wrong_rate"] <= 0.02 and x["body_rate"] <= 0.02 for x in flat)
    winner_payload = {"version": "r8_58_targeted_hold", "joint_names": names, "fl_joint_ids": fl,
        "hold_steps": 5, "geometry": geometry.cpu().tolist(),
        "by_strength": {s: w["controller"] for s, w in winners.items()},
        "recovery_checkpoint": str(recovery_checkpoint)}
    winner_file = output_dir / "r858_hold_winner.json"
    winner_file.write_text(json.dumps(winner_payload, indent=2) + "\n")
    summary = {"version": "r8_58_targeted_static_ramp_hold", "training": False, "network_frozen": True,
        "reused_proven_chain": ["true_contact", "kick_strength_monotonic", "high_strength_hold"],
        "search_strengths": [0.75, 1.0], "high_strength_frozen": True,
        "candidate_count": len(labels), "screen_episodes_per_case": screen_episodes,
        "semifinal_topk": 8, "semifinal_episodes": semifinal_episodes,
        "final_episodes_per_condition": final_episodes, "final_conditions": ["exact_geometry", "uniform_4mm_geometry_jitter"],
        "scans": scans, "semifinals": semifinals, "winners": winners, "final_confirm": finals,
        "winner_file": str(winner_file), "accepted": accepted,
        "verdict": "READY_FOR_VIDEO_REVIEW" if accepted else "STOP_CONTROLLER_CLASS_INSUFFICIENT"}
    (output_dir / "r858_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[R858_RESULT] accepted={accepted} verdict={summary['verdict']}", flush=True)
    print("R858_TARGETED_HOLD_OK", flush=True)
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    text = Path(a.source).read_text(encoding="utf-8")
    marker = "\ndef main() -> None:\n"
    if marker not in text:
        raise SystemExit("main marker missing")
    text = text.replace(marker, INSERT + marker, 1)
    old = 'or os.environ.get("R857_BALANCED_SCAN_ONLY", "0") == "1":'
    new = 'or os.environ.get("R857_BALANCED_SCAN_ONLY", "0") == "1" or os.environ.get("R858_TARGETED_HOLD_ONLY", "0") == "1" or os.environ.get("R858_VIDEO_ONLY", "0") == "1":'
    if old not in text:
        raise SystemExit("sensor condition marker missing")
    text = text.replace(old, new, 1)
    branch = '''

    if os.environ.get("R858_TARGETED_HOLD_ONLY", "0") == "1":
        run_r858_targeted_hold(runner=runner, env=env, freeze_dir=Path(os.environ["R858_FREEZE_DIR"]).resolve(), kick_checkpoint=Path(os.environ["R858_KICK"]).resolve(), r851_summary=Path(os.environ["R858_R851_SUMMARY"]).resolve(), body_json=Path(os.environ["R858_BODY_JSON"]).resolve(), recovery_checkpoint=Path(os.environ["R858_RECOVERY"]).resolve(), r857_winner=Path(os.environ["R858_R857_WINNER"]).resolve(), output_dir=Path(os.environ["R858_OUTPUT"]).resolve(), screen_episodes=int(os.environ.get("R858_SCREEN_EPISODES", "24")), semifinal_episodes=int(os.environ.get("R858_SEMIFINAL_EPISODES", "256")), final_episodes=int(os.environ.get("R858_FINAL_EPISODES", "512")), seed=args_cli.seed)
        return

    if os.environ.get("R858_VIDEO_ONLY", "0") == "1":
        run_r856_video(runner=runner, env=env, freeze_dir=Path(os.environ["R856_FREEZE_DIR"]).resolve(), kick_checkpoint=Path(os.environ["R856_KICK"]).resolve(), r851_summary=Path(os.environ["R856_R851_SUMMARY"]).resolve(), body_json=Path(os.environ["R856_BODY_JSON"]).resolve(), recovery_checkpoint=Path(os.environ["R856_RECOVERY"]).resolve(), winner_json=Path(os.environ["R856_WINNER"]).resolve(), video_path=Path(args_cli.eval_video).resolve(), strength=float(os.environ["R856_VIDEO_STRENGTH"]), seed=args_cli.seed)
        return
'''
    target = '\n\n\n\n\n\n    if os.environ.get("R857_BALANCED_SCAN_ONLY", "0") == "1":'
    if target not in text:
        target = '\n    if os.environ.get("R857_BALANCED_SCAN_ONLY", "0") == "1":'
    if target not in text:
        raise SystemExit("branch insertion marker missing")
    text = text.replace(target, branch + target, 1)
    old_held = 'held=torch.clamp(proposed+bias,-1.0,1.0) if spec.get("mode")=="dynamic_target" else hold_action; action=torch.where(holding.to(runner.device).unsqueeze(1),held,proposed)'
    new_held = 'age=(int(winner["hold_steps"])-hold_remaining).clamp(min=0).float(); held=torch.clamp(hold_action+bias*(1.0+float(spec.get("ramp",0.0))*age.to(runner.device).unsqueeze(1)),-1.0,1.0) if spec.get("mode")=="static_ramp" else (torch.clamp(proposed+bias,-1.0,1.0) if spec.get("mode")=="dynamic_target" else hold_action); action=torch.where(holding.to(runner.device).unsqueeze(1),held,proposed)'
    if old_held not in text:
        raise SystemExit("video held marker missing")
    text = text.replace(old_held, new_held, 1)
    old_latch = 'hold_action[nl.to(runner.device)]=torch.clamp(action+bias,-1.0,1.0)[nl.to(runner.device)].detach(); hold_remaining[nl]=int(winner["hold_steps"])'
    new_latch = 'hold_action[nl.to(runner.device)]=(action if spec.get("mode")=="static_ramp" else torch.clamp(action+bias,-1.0,1.0))[nl.to(runner.device)].detach(); hold_remaining[nl]=int(winner["hold_steps"])'
    if old_latch not in text:
        raise SystemExit("video latch marker missing")
    text = text.replace(old_latch, new_latch, 1)
    Path(a.output).write_text(text, encoding="utf-8")
    print(f"R858_PATCH_OK source={a.source} output={a.output}")


if __name__ == "__main__":
    main()
