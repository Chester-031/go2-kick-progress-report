#!/usr/bin/env python3
"""Generate R8.51 from the existing project R8.49 source."""

import argparse
from pathlib import Path


FUNCTIONS = r'''

# ============================================================
# R8.51: integration-ready 64-D Kick residual trained from
# filtered physical contact.  Frozen Load/Lift/Swing/Recovery.
# ============================================================

class R851Kick64ActorCritic(torch.nn.Module):
    def __init__(self, residual_scale=0.10, action_std=0.05):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(64, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 64), torch.nn.ELU(),
            torch.nn.Linear(64, 12),
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(64, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 64), torch.nn.ELU(),
            torch.nn.Linear(64, 1),
        )
        self.register_buffer("action_std", torch.full((12,), float(action_std)))

    def mean(self, obs):
        return self.residual_scale * torch.tanh(self.actor(obs))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def sample(self, obs):
        mean = self.mean(obs)
        dist = torch.distributions.Normal(mean, self.action_std)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1), self.value(obs)

    def evaluate_actions(self, obs, action):
        dist = torch.distributions.Normal(self.mean(obs), self.action_std)
        return dist.log_prob(action).sum(-1), dist.entropy().sum(-1), self.value(obs)


def _r851_obs64(obs70):
    if obs70.shape[-1] != 70:
        raise RuntimeError(f"R851 expected historical 70-D input, got {tuple(obs70.shape)}")
    result = torch.cat((obs70[:, :63], obs70[:, 69:70]), dim=1)
    if result.shape[-1] != 64 or torch.any(result[:, 62] != 0.0):
        raise RuntimeError("R851 64-D contract or teacher-zero contract failed")
    return result


def _r851_load_strike64(path, device):
    bundle = torch.load(path, map_location=device, weights_only=False)
    old = bundle["state_dict"]
    model = R851Kick64ActorCritic(
        residual_scale=float(bundle["residual_scale"]),
        action_std=min(float(bundle["action_std"]), 0.05),
    ).to(device)
    new = model.state_dict()
    for key, value in old.items():
        if key in new and new[key].shape == value.shape:
            new[key] = value
    for prefix in ("actor", "critic"):
        key = f"{prefix}.0.weight"
        if key not in old or old[key].shape[1] != 70:
            raise RuntimeError(f"R851 cannot migrate {key}: {old.get(key, None)}")
        migrated = new[key].clone()
        migrated[:, :63] = old[key][:, :63]
        migrated[:, 63] = old[key][:, 69]
        new[key] = migrated
    model.load_state_dict(new)
    return model, {
        "parent_version": bundle.get("version", "unknown"),
        "migration": "70D[0:63,69] -> 64D",
    }


def _r851_load_recovery_vector(components, path, device):
    bundle = torch.load(path, map_location=device, weights_only=False)
    components["vector_head"].load_state_dict(bundle["state_dict"])
    components["vector_head"].eval()
    for p in components["vector_head"].parameters():
        p.requires_grad_(False)
    return bundle.get("version", "unknown")


def _r851_sensor_peaks(sensor, filters):
    history = sensor.data.force_matrix_w_history
    if history is None or history.ndim != 5:
        raise RuntimeError(f"R851 contact history shape invalid: {None if history is None else history.shape}")
    if history.shape[2] != 1 or history.shape[3] != filters:
        raise RuntimeError(f"R851 contact matrix mismatch: {tuple(history.shape)}, filters={filters}")
    return torch.linalg.vector_norm(history[:, :, 0, :, :], dim=-1).amax(dim=1)


def _r851_place(env, sensor, mask, forward, lateral, strength):
    if not mask.any():
        return
    ids = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    selected, _ = env._front_foot_positions()
    direction = env.strike_direction_world[ids]
    direction = direction / torch.clamp(torch.linalg.vector_norm(direction, dim=1, keepdim=True), min=1e-6)
    side = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)
    xy = selected[ids, :2] + forward[ids, None] * direction + lateral[ids, None] * side
    pose = torch.zeros(ids.numel(), 7, device=env.device)
    pose[:, :2] = xy
    pose[:, 2] = env.env_origins[ids, 2] + 0.115
    pose[:, 3] = 1.0
    env.ball.write_root_pose_to_sim(pose, env_ids=ids)
    env.ball.write_root_velocity_to_sim(torch.zeros(ids.numel(), 6, device=env.device), env_ids=ids)
    env.kick_ball_start_xy[ids] = xy
    env.strike_strength[ids] = strength[ids]
    env.ball_arm_step_buf[ids] = 0
    env.stable_consecutive_buf[ids] = 0
    env.previous_ball_direction_speed[ids] = 0.0
    env.peak_ball_direction_speed[ids] = 0.0
    sensor.reset(ids)


def _r851_strengths(n, device, update, generator):
    if update <= 4:
        return torch.ones(n, device=device)
    if update <= 8:
        anchors = torch.tensor([0.75, 1.0, 1.25], device=device)
        return anchors[torch.randint(0, 3, (n,), device=device, generator=generator)]
    return 0.70 + 0.60 * torch.rand(n, device=device, generator=generator)


def _r851_geometry(n, device, top_cases, update, generator):
    base = top_cases[torch.randint(0, top_cases.shape[0], (n,), device=device, generator=generator)]
    jitter = 0.0 if update <= 4 else (0.006 if update <= 8 else 0.012)
    if jitter:
        base = base + (2.0 * torch.rand(n, 2, device=device, generator=generator) - 1.0) * jitter
    return base[:, 0], base[:, 1]


def _r851_frozen_action(runner, env, components, obs_runner, bridge, recovery):
    return _r848_frozen_actions(
        runner=runner, obs_runner=obs_runner, bridge_before=bridge,
        recovery_before=recovery, components=components,
    )


def _r851_contact_state(env, sensor, body_names, selected_before, ball_vel_before, direction, active, done, step_dt):
    peaks = _r851_sensor_peaks(sensor, len(body_names))
    foot_names = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    foot_ids = [body_names.index(x) for x in foot_names]
    nonfoot = [i for i in range(len(body_names)) if i not in foot_ids]
    selected_force = peaks[:, foot_ids[0]]
    wrong_force = peaks[:, foot_ids[1:]].amax(1)
    body_force = peaks[:, nonfoot].amax(1) if nonfoot else torch.zeros_like(selected_force)
    physical = active & (~done)
    selected_after, _ = env._front_foot_positions()
    foot_velocity = (selected_after - selected_before) / max(step_dt, 1e-6)
    toward = (foot_velocity[:, :2] * direction).sum(1)
    ball_vel_after = env.ball.data.root_lin_vel_w.detach()
    dv = ((ball_vel_after - ball_vel_before)[:, :2] * direction).sum(1)
    speed = (ball_vel_after[:, :2] * direction).sum(1)
    return {
        "selected": physical & (selected_force > 1.0),
        "wrong": physical & (wrong_force > 1.0),
        "body": physical & (body_force > 1.0),
        "selected_force": selected_force,
        "toward": toward,
        "dv": dv,
        "speed": speed,
    }


def _r851_prepare_episode_state(env, sensor, body_names, placed, settle, forward, lateral, strength):
    newly = env.ball_placed & (~env.kick_active) & (~placed)
    if newly.any():
        _r851_place(env, sensor, newly, forward, lateral, strength)
        placed[newly] = True
    inactive = placed & (~env.kick_active)
    precontact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if inactive.any():
        peaks = _r851_sensor_peaks(sensor, len(body_names))
        touch = peaks.amax(1) > 0.1
        z = env.ball.data.root_pos_w[:, 2] - env.env_origins[:, 2]
        lin = torch.linalg.vector_norm(env.ball.data.root_lin_vel_w, dim=1)
        ang = torch.linalg.vector_norm(env.ball.data.root_ang_vel_w, dim=1)
        stable = inactive & ((z - 0.115).abs() < 0.006) & (lin < 0.03) & (ang < 0.50) & (~touch)
        settle[:] = torch.where(stable, settle + 1, torch.where(inactive, torch.zeros_like(settle), settle))
        hold = inactive & (settle < 12)
        env.ball_arm_step_buf[hold] = 0
        env.stable_consecutive_buf[hold] = 0
        precontact = inactive & touch
    return precontact


def _r851_eval(*, runner, env, components, model, sensor, body_names, top_cases, strength, episodes, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    g = torch.Generator(device=env.device); g.manual_seed(seed)
    env.reset()
    n, device = env.num_envs, env.device
    forward = top_cases[0, 0].repeat(n)
    lateral = top_cases[0, 1].repeat(n)
    strengths = torch.full((n,), float(strength), device=device)
    placed = torch.zeros(n, dtype=torch.bool, device=device)
    settle = torch.zeros(n, dtype=torch.long, device=device)
    contamination = torch.zeros(n, dtype=torch.bool, device=device)
    selected_seen = torch.zeros(n, dtype=torch.bool, device=device)
    wrong_seen = torch.zeros(n, dtype=torch.bool, device=device)
    body_seen = torch.zeros(n, dtype=torch.bool, device=device)
    real_seen = torch.zeros(n, dtype=torch.bool, device=device)
    recovery_seen = torch.zeros(n, dtype=torch.bool, device=device)
    recovery_control = torch.zeros(n, dtype=torch.bool, device=device)
    peak_speed = torch.zeros(n, device=device)
    max_disp = torch.zeros(n, device=device)
    completed = 0
    totals = {"selected": 0, "clean": 0, "real": 0, "recovery": 0, "wrong": 0, "body": 0, "precontact": 0}
    speed_sum = disp_sum = 0.0
    step_dt = float(getattr(env.base_env, "step_dt", 0.02))
    loops = 0
    with torch.no_grad():
        while completed < episodes:
            contamination |= _r851_prepare_episode_state(env, sensor, body_names, placed, settle, forward, lateral, strengths)
            obs_runner = TensorDict({"policy": env.low_obs_buf}, batch_size=[n], device=device).to(runner.device)
            active = env.kick_active.detach().bool().clone()
            bridge = env.bridge_seen_r852.detach().bool().clone()
            frozen, _, _ = _r851_frozen_action(runner, env, components, obs_runner, bridge, recovery_control)
            obs64 = _r851_obs64(env.low_obs_buf).to(runner.device)
            residual = model.mean(obs64)
            strike_active = active & bridge & (~recovery_control)
            action = frozen + strike_active.to(runner.device).unsqueeze(1) * residual
            selected_before, _ = env._front_foot_positions(); selected_before = selected_before.detach().clone()
            ball_vel_before = env.ball.data.root_lin_vel_w.detach().clone()
            ball_pos_before = env.ball.data.root_pos_w.detach().clone()
            direction = env.strike_direction_world.detach().clone()
            direction = direction / torch.clamp(torch.linalg.vector_norm(direction, dim=1, keepdim=True), min=1e-6)
            _, _, done, _, _ = env.kick_step(action.to(device)); done = done.to(device).bool()
            cs = _r851_contact_state(env, sensor, body_names, selected_before, ball_vel_before, direction, active, done, step_dt)
            new_selected = cs["selected"] & (~selected_seen)
            wrong_seen |= cs["wrong"] & (~selected_seen)
            body_seen |= cs["body"] & (~selected_seen)
            clean_now = new_selected & (~contamination) & (~wrong_seen) & (~body_seen)
            real_now = clean_now & (cs["toward"] > 0.0) & (cs["dv"] > 0.02)
            selected_seen |= cs["selected"]
            real_seen |= real_now
            recovery_control |= real_now
            recovery_seen |= real_seen & env.r829_recovery_seen.detach().bool()
            peak_speed = torch.maximum(peak_speed, torch.where(active & (~done), cs["speed"], torch.zeros_like(cs["speed"])))
            disp = ((env.ball.data.root_pos_w[:, :2] - env.kick_ball_start_xy) * direction).sum(1)
            max_disp = torch.maximum(max_disp, torch.where(active & (~done), disp, torch.zeros_like(disp)))
            if done.any():
                ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
                take = ids[:max(0, episodes - completed)]
                totals["selected"] += int(selected_seen[take].sum())
                clean_ep = selected_seen[take] & (~contamination[take]) & (~wrong_seen[take]) & (~body_seen[take])
                totals["clean"] += int(clean_ep.sum()); totals["real"] += int(real_seen[take].sum())
                totals["recovery"] += int((real_seen[take] & recovery_seen[take]).sum())
                totals["wrong"] += int(wrong_seen[take].sum()); totals["body"] += int(body_seen[take].sum())
                totals["precontact"] += int(contamination[take].sum())
                speed_sum += float(peak_speed[take].sum()); disp_sum += float(max_disp[take].sum())
                completed += int(take.numel())
                for tensor in (placed, contamination, selected_seen, wrong_seen, body_seen, real_seen, recovery_seen, recovery_control): tensor[done] = False
                settle[done] = 0; peak_speed[done] = 0.0; max_disp[done] = 0.0
            loops += 1
            if loops > max(10000, episodes * (env.total_episode_steps + 100)):
                raise RuntimeError("R851 eval safety timeout")
    denom = max(completed, 1); real_denom = max(totals["real"], 1)
    return {"strength": float(strength), "episodes": completed,
            "selected_rate": totals["selected"] / denom, "clean_rate": totals["clean"] / denom,
            "real_strike_rate": totals["real"] / denom, "recovery_given_real": totals["recovery"] / real_denom,
            "wrong_rate": totals["wrong"] / denom, "body_rate": totals["body"] / denom,
            "precontact_rate": totals["precontact"] / denom,
            "mean_peak_direction_speed_mps": speed_sum / denom, "mean_directional_disp_m": disp_sum / denom}


def run_r851_kick64(*, runner, env, freeze_dir, strike_checkpoint, recovery_checkpoint,
                    r850_summary, body_json, output_dir, updates, rollout_steps, eval_episodes, seed):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    r850 = json.loads(Path(r850_summary).read_text(encoding="utf-8"))
    body_names = json.loads(Path(body_json).read_text(encoding="utf-8"))["body_names"]
    controls = r850.get("controls", {})
    if not controls.get("far_ball_pass") or not controls.get("overlap_pass"):
        raise RuntimeError(f"R851 refuses uncalibrated contact source: {controls}")
    ranked = sorted(r850["cases"], key=lambda x: (x["real_strike_rate"], x["clean_rate"], x["mean_impulse_ns"]), reverse=True)
    top_cases = torch.tensor([[x["forward_m"], x["lateral_m"]] for x in ranked[:3]], device=env.device)
    components = _r848_load_frozen_components(runner=runner, freeze_dir=freeze_dir)
    for module in components.values():
        if isinstance(module, torch.nn.Module):
            module.eval()
            for p in module.parameters(): p.requires_grad_(False)
    recovery_parent = _r851_load_recovery_vector(components, recovery_checkpoint, runner.device)
    model, migration = _r851_load_strike64(strike_checkpoint, runner.device)
    sensor = env.base_env.scene["r851_ball_robot_contact"]
    env.r841_grace_steps = 6
    env.total_episode_steps = env.settle_max_steps + env.ball_arm_steps + env.kick_episode_steps + 6
    env.max_episode_length = env.total_episode_steps; env.max_low_episode_length = env.total_episode_steps
    env.set_curriculum_stage("recovery")
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-4)
    gamma, gae_lambda, clip_eps = 0.99, 0.95, 0.20
    n, device = env.num_envs, env.device
    generator = torch.Generator(device=device); generator.manual_seed(seed)
    history = []
    for update in range(1, updates + 1):
        env.reset()
        forward, lateral = _r851_geometry(n, device, top_cases, update, generator)
        strengths = _r851_strengths(n, device, update, generator)
        placed = torch.zeros(n, dtype=torch.bool, device=device); settle = torch.zeros(n, dtype=torch.long, device=device)
        contaminated = torch.zeros(n, dtype=torch.bool, device=device); selected_seen = torch.zeros(n, dtype=torch.bool, device=device)
        wrong_seen = torch.zeros(n, dtype=torch.bool, device=device); body_seen = torch.zeros(n, dtype=torch.bool, device=device)
        real_seen = torch.zeros(n, dtype=torch.bool, device=device); recovery_control = torch.zeros(n, dtype=torch.bool, device=device)
        recovery_rewarded = torch.zeros(n, dtype=torch.bool, device=device)
        stores = [[] for _ in range(7)]
        counts = {"active": 0, "contact": 0, "real": 0, "wrong": 0, "body": 0, "recovery": 0}
        step_dt = float(getattr(env.base_env, "step_dt", 0.02))
        model.train()
        for _ in range(rollout_steps):
            contaminated |= _r851_prepare_episode_state(env, sensor, body_names, placed, settle, forward, lateral, strengths)
            obs_runner = TensorDict({"policy": env.low_obs_buf}, batch_size=[n], device=device).to(runner.device)
            obs64 = _r851_obs64(env.low_obs_buf).to(runner.device)
            active = env.kick_active.detach().bool().clone(); bridge = env.bridge_seen_r852.detach().bool().clone()
            frozen, _, _ = _r851_frozen_action(runner, env, components, obs_runner, bridge, recovery_control)
            residual, logp, value = model.sample(obs64)
            strike_active = active & bridge & (~recovery_control)
            action = frozen + strike_active.to(runner.device).unsqueeze(1) * residual
            selected_before, _ = env._front_foot_positions(); selected_before = selected_before.detach().clone()
            ball_vel_before = env.ball.data.root_lin_vel_w.detach().clone()
            direction = env.strike_direction_world.detach().clone(); direction = direction / torch.clamp(torch.linalg.vector_norm(direction, dim=1, keepdim=True), min=1e-6)
            _, _, done, _, _ = env.kick_step(action.to(device)); done = done.to(device).bool()
            cs = _r851_contact_state(env, sensor, body_names, selected_before, ball_vel_before, direction, active, done, step_dt)
            new_selected = cs["selected"] & (~selected_seen)
            new_wrong = cs["wrong"] & (~wrong_seen) & (~selected_seen)
            new_body = cs["body"] & (~body_seen) & (~selected_seen)
            wrong_seen |= new_wrong; body_seen |= new_body
            clean_contact = new_selected & (~contaminated) & (~wrong_seen) & (~body_seen)
            real_now = clean_contact & (cs["toward"] > 0.0) & (cs["dv"] > 0.02)
            selected_seen |= cs["selected"]; real_seen |= real_now; recovery_control |= real_now
            target_speed = 0.16 + (strengths - 0.70) * (0.30 / 0.60)
            speed_score = torch.exp(-torch.abs(cs["speed"] - target_speed) / 0.10)
            new_recovery = real_seen & env.r829_recovery_seen.detach().bool() & (~recovery_rewarded)
            reward = (1.5 * clean_contact.float() + 4.0 * real_now.float() +
                      1.5 * real_seen.float() * speed_score + 3.0 * new_recovery.float() -
                      6.0 * new_wrong.float() - 6.0 * new_body.float())
            reward = reward - (0.015 * strike_active.to(runner.device).float() * residual.pow(2).sum(1)).to(device)
            recovery_rewarded |= new_recovery
            counts["active"] += int(strike_active.sum()); counts["contact"] += int(clean_contact.sum())
            counts["real"] += int(real_now.sum()); counts["wrong"] += int(new_wrong.sum()); counts["body"] += int(new_body.sum()); counts["recovery"] += int(new_recovery.sum())
            for store, item in zip(stores, (obs64.detach(), residual.detach(), logp.detach(), value.detach(), reward.to(runner.device).detach(), done.to(runner.device).float(), strike_active.to(runner.device).float())): store.append(item)
            if done.any():
                for tensor in (placed, contaminated, selected_seen, wrong_seen, body_seen, real_seen, recovery_control, recovery_rewarded): tensor[done] = False
                settle[done] = 0
                fresh_f, fresh_y = _r851_geometry(n, device, top_cases, update, generator); fresh_s = _r851_strengths(n, device, update, generator)
                forward[done] = fresh_f[done]; lateral[done] = fresh_y[done]; strengths[done] = fresh_s[done]
        obs_t, act_t, old_logp_t, value_t, reward_t, done_t, active_t = [torch.stack(x) for x in stores]
        with torch.no_grad(): bootstrap = model.value(_r851_obs64(env.low_obs_buf).to(runner.device))
        advantages = torch.zeros_like(reward_t); last = torch.zeros(n, device=runner.device)
        for t in reversed(range(rollout_steps)):
            next_value = bootstrap if t == rollout_steps - 1 else value_t[t + 1]
            nonterminal = 1.0 - done_t[t]
            delta = reward_t[t] + gamma * next_value * nonterminal - value_t[t]
            last = delta + gamma * gae_lambda * nonterminal * last; advantages[t] = last
        returns = advantages + value_t
        flat = [x.reshape(-1, x.shape[-1]) if x.ndim == 3 else x.reshape(-1) for x in (obs_t, act_t, old_logp_t, returns, active_t)]
        flat_obs, flat_act, flat_logp, flat_returns, flat_active_float = flat; flat_active = flat_active_float > 0.5
        flat_adv = advantages.reshape(-1); active_adv = flat_adv[flat_active]
        norm_adv = (flat_adv - active_adv.mean()) / (active_adv.std(unbiased=False) + 1e-8) if active_adv.numel() else flat_adv
        losses = []
        total = flat_obs.shape[0]
        for _epoch in range(4):
            perm = torch.randperm(total, device=runner.device)
            for start in range(0, total, 8192):
                idx = perm[start:start + 8192]; new_logp, entropy, new_value = model.evaluate_actions(flat_obs[idx], flat_act[idx]); mask = flat_active[idx]
                if mask.any():
                    ratio = torch.exp(new_logp[mask] - flat_logp[idx][mask]); adv = norm_adv[idx][mask]
                    actor_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv).mean()
                else: actor_loss = new_value.sum() * 0.0
                value_loss = (new_value - flat_returns[idx]).pow(2).mean()
                loss = actor_loss + 0.5 * value_loss - 0.001 * entropy.mean()
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                losses.append((float(actor_loss.detach()), float(value_loss.detach())))
        record = {"update": update, "curriculum": "fixed" if update <= 4 else ("three_strength" if update <= 8 else "continuous"), **counts,
                  "actor_loss": float(np.mean([x[0] for x in losses])), "value_loss": float(np.mean([x[1] for x in losses]))}
        history.append(record)
        print(f"[R851_UPDATE] u={update:02d} stage={record['curriculum']} active={counts['active']} contact={counts['contact']} real={counts['real']} recovery={counts['recovery']} wrong={counts['wrong']} body={counts['body']} actorLoss={record['actor_loss']:.5f} valueLoss={record['value_loss']:.5f}", flush=True)
        torch.save({"version": "r8_51_kick64_true_contact", "obs_dim": 64, "action_dim": 12,
                    "residual_scale": model.residual_scale, "action_std": float(model.action_std[0]),
                    "state_dict": model.state_dict(), "migration": migration, "history": history,
                    "top_geometry": top_cases.detach().cpu().tolist()}, output_dir / "kick64_latest.pt")
    model.eval()
    confirms = [_r851_eval(runner=runner, env=env, components=components, model=model, sensor=sensor, body_names=body_names,
                           top_cases=top_cases, strength=s, episodes=eval_episodes, seed=seed + 851000 + i)
                for i, s in enumerate((0.75, 1.0, 1.25))]
    speeds = [x["mean_peak_direction_speed_mps"] for x in confirms]
    monotonic = speeds[1] >= speeds[0] + 0.015 and speeds[2] >= speeds[1] + 0.015
    mid = confirms[1]
    accepted = (mid["real_strike_rate"] >= 0.80 and mid["clean_rate"] >= 0.80 and
                mid["recovery_given_real"] >= 0.30 and mid["wrong_rate"] <= 0.02 and mid["body_rate"] <= 0.02 and monotonic)
    final_ckpt = output_dir / "kick64_final.pt"
    torch.save({"version": "r8_51_kick64_true_contact", "obs_dim": 64, "action_dim": 12,
                "residual_scale": model.residual_scale, "action_std": float(model.action_std[0]),
                "state_dict": model.state_dict(), "migration": migration, "history": history,
                "top_geometry": top_cases.detach().cpu().tolist(), "strength_confirm": confirms}, final_ckpt)
    summary = {"version": "r8_51_kick64_true_contact", "training": True, "actual_contact": True,
               "obs_contract": "64D=70D[0:63]+70D[69]", "frozen_load_lift_swing_recovery": True,
               "strike_parent": str(strike_checkpoint), "recovery_parent": str(recovery_checkpoint),
               "recovery_parent_version": recovery_parent, "r850_parent": str(r850_summary),
               "top_geometry": top_cases.detach().cpu().tolist(), "updates": updates, "rollout_steps": rollout_steps,
               "history": history, "strength_confirm": confirms, "strength_monotonic": monotonic,
               "checkpoint": str(final_ckpt), "accepted": accepted,
               "verdict": "READY_FOR_4_SKILL_ROUTER" if accepted else "KEEP_R851_NOT_READY_FOR_ROUTER"}
    (output_dir / "r851_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[R851_CONFIRM] low={speeds[0]:.3f} mid={speeds[1]:.3f} high={speeds[2]:.3f} monotonic={monotonic} midReal={100*mid['real_strike_rate']:.2f}% midRec={100*mid['recovery_given_real']:.2f}%", flush=True)
    print(f"[R851_RESULT] accepted={accepted} verdict={summary['verdict']} checkpoint={final_ckpt}", flush=True)
    print("R851_KICK64_OK", flush=True)
'''


CFG = r'''
    if os.environ.get("R851_KICK64_ONLY", "0") == "1":
        from isaaclab.sensors import ContactSensorCfg
        body_meta = json.loads(Path(os.environ["R851_BODY_JSON"]).read_text(encoding="utf-8"))
        body_names = body_meta["body_names"]
        env_cfg.scene.ball.spawn.activate_contact_sensors = True
        env_cfg.scene.r851_ball_robot_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/SoccerBall", update_period=0.0, history_length=8,
            track_pose=False, track_contact_points=False,
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/{name}" for name in body_names],
        )

'''


BRANCH = r'''
    if os.environ.get("R851_KICK64_ONLY", "0") == "1":
        run_r851_kick64(
            runner=runner, env=env,
            freeze_dir=Path(os.environ["R851_FREEZE_DIR"]).resolve(),
            strike_checkpoint=Path(os.environ["R851_STRIKE"]).resolve(),
            recovery_checkpoint=Path(os.environ["R851_RECOVERY"]).resolve(),
            r850_summary=Path(os.environ["R851_R850_SUMMARY"]).resolve(),
            body_json=Path(os.environ["R851_BODY_JSON"]).resolve(),
            output_dir=Path(os.environ["R851_OUTPUT"]).resolve(),
            updates=int(os.environ.get("R851_UPDATES", "12")),
            rollout_steps=int(os.environ.get("R851_ROLLOUT_STEPS", "192")),
            eval_episodes=int(os.environ.get("R851_EVAL_EPISODES", "512")),
            seed=args_cli.seed,
        )
        return

'''


def replace_once(text: str, anchor: str, replacement: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(f"expected one anchor, found {count}: {anchor[:100]!r}")
    return text.replace(anchor, replacement, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.source).resolve(); output = Path(args.output).resolve()
    text = source.read_text(encoding="utf-8")
    text = replace_once(text, "\ndef main() -> None:\n", FUNCTIONS + "\n\ndef main() -> None:\n")
    text = replace_once(text, "    low_agent_cfg = instantiate_cfg(\n", CFG + "    low_agent_cfg = instantiate_cfg(\n")
    text = replace_once(text, "    if os.environ.get(\n        \"R849_RECOVERY_FINETUNE_ONLY\",", BRANCH + "    if os.environ.get(\n        \"R849_RECOVERY_FINETUNE_ONLY\",")
    compile(text, str(output), "exec")
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(text, encoding="utf-8")
    print(f"R851_PATCH_OK source={source} output={output}")


if __name__ == "__main__":
    main()

