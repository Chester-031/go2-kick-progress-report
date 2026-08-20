#!/usr/bin/env python3
"""Add strength-conditioned dense-reward Recovery64 PPO to R8.53 source."""

import argparse
from pathlib import Path


FUNCTIONS = r'''

# ============================================================
# R8.54: strength-conditioned, touchdown-dense Recovery64 PPO.
# ============================================================

def _r854_condition(obs64, desired_strength):
    """Expose high-level desired strength without changing legacy env timing."""
    conditioned=obs64.clone()
    conditioned[:,63]=desired_strength.to(conditioned.device)
    return conditioned


def _r854_curriculum(update, n, device, generator):
    anchors=torch.tensor([0.75,1.0,1.25],device=device)
    if update<=8:
        return torch.full((n,),1.25,device=device),"high_bootstrap"
    if update<=16:
        choices=anchors[1:]
        return choices[torch.randint(0,2,(n,),device=device,generator=generator)],"mid_high"
    return anchors[torch.randint(0,3,(n,),device=device,generator=generator)],"all_strengths"


def _r854_eval(*,runner,env,components,kick,recovery,sensor,body_names,geometry,desired_strength,episodes,seed):
    torch.manual_seed(seed); np.random.seed(seed); env.reset(); n,device=env.num_envs,env.device
    desired=torch.full((n,),float(desired_strength),device=device); env_strength=torch.ones(n,device=device)
    forward=geometry[0].repeat(n); lateral=geometry[1].repeat(n)
    placed=torch.zeros(n,dtype=torch.bool,device=device); settle=torch.zeros(n,dtype=torch.long,device=device)
    contaminated=torch.zeros(n,dtype=torch.bool,device=device); selected_seen=torch.zeros(n,dtype=torch.bool,device=device)
    wrong_seen=torch.zeros(n,dtype=torch.bool,device=device); body_seen=torch.zeros(n,dtype=torch.bool,device=device)
    real_seen=torch.zeros(n,dtype=torch.bool,device=device); recovery_seen=torch.zeros(n,dtype=torch.bool,device=device)
    gates={name:torch.zeros(n,dtype=torch.bool,device=device) for name in ("candidate","return","contact","support4","base","height","upright","touchdown","all_now")}
    peak_speed=torch.zeros(n,device=device); max_disp=torch.zeros(n,device=device); completed=0
    totals={"real":0,"clean":0,"wrong":0,"body":0,"recovery":0,**{k:0 for k in gates}}; speed_sum=disp_sum=0.0
    step_dt=float(getattr(env.base_env,"step_dt",0.02)); loops=0
    with torch.no_grad():
        while completed<episodes:
            contaminated|=_r851_prepare_episode_state(env,sensor,body_names,placed,settle,forward,lateral,env_strength)
            obs_runner=TensorDict({"policy":env.low_obs_buf},batch_size=[n],device=device).to(runner.device)
            active=env.kick_active.detach().bool().clone(); bridge=env.bridge_seen_r852.detach().bool().clone(); phase=env._phase().detach().clone()
            recovery_control=real_seen&env.swing_seen_r89.detach().bool()&(phase>=0.78)
            frozen,stable,swing_norm=_r851_frozen_action(runner,env,components,obs_runner,bridge,torch.zeros_like(recovery_control))
            kick_obs=_r851_obs64(env.low_obs_buf).to(runner.device); kick_delta=kick.mean(kick_obs)*_r853_gain(desired).to(runner.device).unsqueeze(1)
            rec_obs=_r854_condition(_r851_obs64(swing_norm),desired); rec_delta=recovery.mean(rec_obs)
            pre=frozen+(active&bridge&(~recovery_control)).to(runner.device).unsqueeze(1)*kick_delta
            rec_action=_r853_recovery_action(runner,components,obs_runner,stable,rec_delta)
            action=torch.where(recovery_control.to(runner.device).unsqueeze(1),rec_action,pre)
            selected_before,_=env._front_foot_positions(); selected_before=selected_before.detach().clone(); ball_vel_before=env.ball.data.root_lin_vel_w.detach().clone()
            direction=env.strike_direction_world.detach().clone(); direction=direction/torch.clamp(torch.linalg.vector_norm(direction,dim=1,keepdim=True),min=1e-6)
            _,_,done,_,_=env.kick_step(action.to(device)); done=done.to(device).bool()
            cs=_r851_contact_state(env,sensor,body_names,selected_before,ball_vel_before,direction,active,done,step_dt)
            new_selected=cs["selected"]&(~selected_seen); wrong_seen|=cs["wrong"]&(~selected_seen); body_seen|=cs["body"]&(~selected_seen)
            clean_now=new_selected&(~contaminated)&(~wrong_seen)&(~body_seen); real_now=clean_now&(cs["toward"]>0)&(cs["dv"]>0.02)
            selected_seen|=cs["selected"]; real_seen|=real_now; recovery_seen|=real_seen&env.r829_recovery_seen.detach().bool(); valid=real_seen&(~done)
            attrs={"candidate":env.r829_rec_candidate,"return":env.r829_return_ok,"contact":env.r829_contact_ok,"support4":env.r829_support4_ok,"base":env.r829_base_ok,"height":env.r829_height_ok,"upright":env.r829_upright_ok,"touchdown":env.r829_touchdown_ok,"all_now":env.r829_all_now}
            for k,v in attrs.items(): gates[k]|=valid&v.detach().bool()
            peak_speed=torch.maximum(peak_speed,torch.where(active&(~done),cs["speed"],torch.zeros_like(cs["speed"])))
            disp=((env.ball.data.root_pos_w[:,:2]-env.kick_ball_start_xy)*direction).sum(1); max_disp=torch.maximum(max_disp,torch.where(active&(~done),disp,torch.zeros_like(disp)))
            if done.any():
                ids=torch.nonzero(done,as_tuple=False).squeeze(-1); take=ids[:max(0,episodes-completed)]
                clean_ep=selected_seen[take]&(~contaminated[take])&(~wrong_seen[take])&(~body_seen[take]); totals["clean"]+=int(clean_ep.sum()); totals["real"]+=int(real_seen[take].sum()); totals["wrong"]+=int(wrong_seen[take].sum()); totals["body"]+=int(body_seen[take].sum()); totals["recovery"]+=int((real_seen[take]&recovery_seen[take]).sum())
                for k in gates: totals[k]+=int(gates[k][take].sum())
                speed_sum+=float(peak_speed[take].sum()); disp_sum+=float(max_disp[take].sum()); completed+=int(take.numel())
                for tensor in (placed,contaminated,selected_seen,wrong_seen,body_seen,real_seen,recovery_seen): tensor[done]=False
                for tensor in gates.values(): tensor[done]=False
                settle[done]=0; peak_speed[done]=0; max_disp[done]=0
            loops+=1
            if loops>max(10000,episodes*(env.total_episode_steps+100)): raise RuntimeError("R854 eval timeout")
    d=max(completed,1); rd=max(totals["real"],1)
    return {"strength":float(desired_strength),"episodes":completed,"clean_rate":totals["clean"]/d,"real_rate":totals["real"]/d,"wrong_rate":totals["wrong"]/d,"body_rate":totals["body"]/d,"recovery_given_real":totals["recovery"]/rd,"mean_peak_speed_mps":speed_sum/d,"mean_disp_m":disp_sum/d,**{f"{k}_given_real":totals[k]/rd for k in gates}}


def run_r854_strength_recovery(*,runner,env,freeze_dir,kick_checkpoint,r851_summary,body_json,r853_checkpoint,output_dir,updates,rollout_steps,eval_episodes,seed):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); s851=json.loads(Path(r851_summary).read_text(encoding="utf-8")); body_names=json.loads(Path(body_json).read_text(encoding="utf-8"))["body_names"]
    geometry=torch.tensor(s851["top_geometry"][0],device=env.device); components=_r848_load_frozen_components(runner=runner,freeze_dir=freeze_dir)
    for module in components.values():
        if isinstance(module,torch.nn.Module):
            module.eval()
            for p in module.parameters(): p.requires_grad_(False)
    kb=torch.load(kick_checkpoint,map_location=runner.device,weights_only=False); kick=R851Kick64ActorCritic(residual_scale=float(kb["residual_scale"]),action_std=float(kb["action_std"])).to(runner.device); kick.load_state_dict(kb["state_dict"]); kick.eval()
    for p in kick.parameters(): p.requires_grad_(False)
    rb=torch.load(r853_checkpoint,map_location=runner.device,weights_only=False); recovery=R853Recovery64ActorCritic(scale=float(rb["scale"]),action_std=0.06).to(runner.device); recovery.load_state_dict(rb["state_dict"],strict=False); recovery.action_std.fill_(0.06)
    initial={k:p.detach().clone() for k,p in recovery.actor.named_parameters()}; optimizer=torch.optim.Adam(recovery.parameters(),lr=2.0e-4); sensor=env.base_env.scene["r851_ball_robot_contact"]
    env.r841_grace_steps=6; env.total_episode_steps=env.settle_max_steps+env.ball_arm_steps+env.kick_episode_steps+6; env.max_episode_length=env.total_episode_steps; env.max_low_episode_length=env.total_episode_steps; env.set_curriculum_stage("recovery")
    n,device=env.num_envs,env.device; generator=torch.Generator(device=device); generator.manual_seed(seed); history=[]; anchors=torch.tensor([0.75,1.0,1.25],device=device)
    for update in range(1,updates+1):
        env.reset(); desired,stage=_r854_curriculum(update,n,device,generator); env_strength=torch.ones(n,device=device); forward=geometry[0].repeat(n); lateral=geometry[1].repeat(n)
        placed=torch.zeros(n,dtype=torch.bool,device=device); settle=torch.zeros(n,dtype=torch.long,device=device); contaminated=torch.zeros(n,dtype=torch.bool,device=device); selected_seen=torch.zeros(n,dtype=torch.bool,device=device); wrong_seen=torch.zeros(n,dtype=torch.bool,device=device); body_seen=torch.zeros(n,dtype=torch.bool,device=device); real_seen=torch.zeros(n,dtype=torch.bool,device=device)
        prev_delta=torch.zeros(n,12,device=runner.device); stores=[[] for _ in range(7)]; counts={str(float(s.item())):{"real":0,"all":0,"recovery":0} for s in anchors}; step_dt=float(getattr(env.base_env,"step_dt",0.02)); recovery.train()
        for _ in range(rollout_steps):
            contaminated|=_r851_prepare_episode_state(env,sensor,body_names,placed,settle,forward,lateral,env_strength)
            obs_runner=TensorDict({"policy":env.low_obs_buf},batch_size=[n],device=device).to(runner.device); active=env.kick_active.detach().bool().clone(); bridge=env.bridge_seen_r852.detach().bool().clone(); phase=env._phase().detach().clone(); recovery_control=real_seen&env.swing_seen_r89.detach().bool()&(phase>=0.78)
            frozen,stable,swing_norm=_r851_frozen_action(runner,env,components,obs_runner,bridge,torch.zeros_like(recovery_control)); kick_delta=kick.mean(_r851_obs64(env.low_obs_buf).to(runner.device))*_r853_gain(desired).to(runner.device).unsqueeze(1)
            rec_obs=_r854_condition(_r851_obs64(swing_norm),desired); sample,logp,value=recovery.sample(rec_obs); pre=frozen+(active&bridge&(~recovery_control)).to(runner.device).unsqueeze(1)*kick_delta; rec_action=_r853_recovery_action(runner,components,obs_runner,stable,sample); action=torch.where(recovery_control.to(runner.device).unsqueeze(1),rec_action,pre)
            selected_before,_=env._front_foot_positions(); selected_before=selected_before.detach().clone(); ball_vel_before=env.ball.data.root_lin_vel_w.detach().clone(); direction=env.strike_direction_world.detach().clone(); direction=direction/torch.clamp(torch.linalg.vector_norm(direction,dim=1,keepdim=True),min=1e-6)
            _,_,done,_,_=env.kick_step(action.to(device)); done=done.to(device).bool(); cs=_r851_contact_state(env,sensor,body_names,selected_before,ball_vel_before,direction,active,done,step_dt)
            new_selected=cs["selected"]&(~selected_seen); wrong_seen|=cs["wrong"]&(~selected_seen); body_seen|=cs["body"]&(~selected_seen); real_now=new_selected&(~contaminated)&(~wrong_seen)&(~body_seen)&(cs["toward"]>0)&(cs["dv"]>0.02); selected_seen|=cs["selected"]; real_seen|=real_now
            fwd=env.r817_foot_forward; lift=env.r817_foot_lift; pose=torch.exp(-(fwd/0.045).square()-(lift/0.015).square()); contact=torch.clamp((env.r817_selected_fz-2.0)/14.0,0.0,1.0); support=(env.r829_support_count>=4).float(); touchdown=env.r829_touchdown_ok.float(); all_now=env.r829_all_now.float(); hold=torch.clamp(env.r829_recovery_hold.float()/3.0,0.0,1.0); recovered=env.r829_recovery_seen.float()
            rate=(sample-prev_delta).pow(2).mean(1); dense=recovery_control.float()*(1.2*pose+1.2*contact+0.4*support+1.5*touchdown+3.0*all_now+2.0*hold+6.0*recovered)-5.0*(cs["wrong"]|cs["body"]).float(); reward=dense.to(runner.device)-0.010*sample.pow(2).sum(1)-0.020*rate
            for s in anchors:
                key=str(float(s.item())); mask=desired.eq(s); counts[key]["real"]+=int((real_now&mask).sum()); counts[key]["all"]+=int((env.r829_all_now.bool()&mask).sum()); counts[key]["recovery"]+=int((env.r829_recovery_seen.bool()&mask).sum())
            for store,item in zip(stores,(rec_obs.detach(),sample.detach(),logp.detach(),value.detach(),reward.detach(),done.to(runner.device).float(),recovery_control.to(runner.device).float())): store.append(item)
            prev_delta=torch.where(recovery_control.to(runner.device).unsqueeze(1),sample.detach(),torch.zeros_like(prev_delta))
            if done.any():
                for tensor in (placed,contaminated,selected_seen,wrong_seen,body_seen,real_seen): tensor[done]=False
                settle[done]=0; prev_delta[done.to(runner.device)]=0; replacement,_=_r854_curriculum(update,int(done.sum()),device,generator); desired[done]=replacement
        obs_t,act_t,old_logp_t,value_t,reward_t,done_t,active_t=[torch.stack(x) for x in stores]; final_obs=_r854_condition(_r851_obs64(_r820_actor_input(components["swing_policy"],TensorDict({"policy":env.low_obs_buf},batch_size=[n],device=device).to(runner.device))),desired); bootstrap=recovery.value(final_obs).detach(); adv=torch.zeros_like(reward_t); last=torch.zeros(n,device=runner.device)
        for t in reversed(range(rollout_steps)):
            nv=bootstrap if t==rollout_steps-1 else value_t[t+1]; non=1-done_t[t]; delta=reward_t[t]+0.99*nv*non-value_t[t]; last=delta+0.99*0.95*non*last; adv[t]=last
        ret=adv+value_t; fo=obs_t.reshape(-1,64); fa=act_t.reshape(-1,12); fl=old_logp_t.reshape(-1); fr=ret.reshape(-1); fm=active_t.reshape(-1)>0.5; fav=adv.reshape(-1); aa=fav[fm]; nav=(fav-aa.mean())/(aa.std(unbiased=False)+1e-8) if aa.numel() else fav; losses=[]
        for _epoch in range(5):
            perm=torch.randperm(fo.shape[0],device=runner.device)
            for start in range(0,fo.shape[0],8192):
                idx=perm[start:start+8192]; nlog,ent,nval=recovery.evaluate_actions(fo[idx],fa[idx]); mask=fm[idx]
                if mask.any(): ratio=torch.exp(nlog[mask]-fl[idx][mask]); a=nav[idx][mask]; actor=-torch.min(ratio*a,torch.clamp(ratio,0.8,1.2)*a).mean()
                else: actor=nval.sum()*0
                value_loss=(nval-fr[idx]).pow(2).mean(); anchor=sum((p-initial[k]).pow(2).mean() for k,p in recovery.actor.named_parameters()); loss=actor+0.5*value_loss+0.01*anchor-0.001*ent.mean(); optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(recovery.parameters(),1.0); optimizer.step(); losses.append((float(actor.detach()),float(value_loss.detach()),float(anchor.detach())))
        rec={"update":update,"stage":stage,"by_strength":counts,"actor_loss":float(np.mean([x[0] for x in losses])),"value_loss":float(np.mean([x[1] for x in losses])),"anchor":float(np.mean([x[2] for x in losses]))}; history.append(rec); compact=" ".join(f"s{k}:{v['real']}/{v['all']}/{v['recovery']}" for k,v in counts.items()); print(f"[R854_UPDATE] u={update:02d} stage={stage} {compact} actorLoss={rec['actor_loss']:.5f} valueLoss={rec['value_loss']:.5f} anchor={rec['anchor']:.7f}",flush=True)
        torch.save({"version":"r8_54_strength_conditioned_recovery","obs_dim":64,"scale":recovery.scale,"action_std":float(recovery.action_std[0]),"state_dict":recovery.state_dict(),"strength_slot":63,"parent":str(r853_checkpoint),"history":history},output_dir/"recovery64_latest.pt")
    recovery.eval(); confirms=[_r854_eval(runner=runner,env=env,components=components,kick=kick,recovery=recovery,sensor=sensor,body_names=body_names,geometry=geometry,desired_strength=s,episodes=eval_episodes,seed=seed+854000+i) for i,s in enumerate((0.75,1.0,1.25))]
    speeds=[x["mean_peak_speed_mps"] for x in confirms]; monotonic=speeds[1]>=speeds[0]+0.01 and speeds[2]>=speeds[1]+0.01
    accepted=monotonic and all(x["real_rate"]>=0.95 and x["touchdown_given_real"]>=0.80 and x["recovery_given_real"]>=0.60 and x["wrong_rate"]<=0.02 and x["body_rate"]<=0.02 for x in confirms)
    final=output_dir/"recovery64_final.pt"; torch.save({"version":"r8_54_strength_conditioned_recovery","obs_dim":64,"scale":recovery.scale,"action_std":float(recovery.action_std[0]),"state_dict":recovery.state_dict(),"strength_slot":63,"parent":str(r853_checkpoint),"history":history,"strength_confirm":confirms},final)
    summary={"version":"r8_54_strength_conditioned_recovery","training":True,"actual_contact":True,"kick_frozen":True,"r853_warm_start":str(r853_checkpoint),"strength_observation":"64D slot63 overwritten with desired strength; environment strength remains 1.0","curriculum":"updates1-8 high; 9-16 mid+high; 17+ all","dense_reward":"pose+contact+support4+touchdown+all_now+hold3","geometry":geometry.detach().cpu().tolist(),"strength_gain":{"0.75":0.5,"1.0":1.0,"1.25":2.0},"updates":updates,"rollout_steps":rollout_steps,"history":history,"strength_confirm":confirms,"strength_monotonic":monotonic,"checkpoint":str(final),"accepted":accepted,"verdict":"READY_FOR_4_SKILL_ROUTER" if accepted else "KEEP_R854_RECOVERY_NOT_READY"}
    (output_dir/"r854_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8"); print(f"[R854_CONFIRM] low={speeds[0]:.3f}/{100*confirms[0]['touchdown_given_real']:.1f}/{100*confirms[0]['recovery_given_real']:.1f} mid={speeds[1]:.3f}/{100*confirms[1]['touchdown_given_real']:.1f}/{100*confirms[1]['recovery_given_real']:.1f} high={speeds[2]:.3f}/{100*confirms[2]['touchdown_given_real']:.1f}/{100*confirms[2]['recovery_given_real']:.1f} monotonic={monotonic}",flush=True); print(f"[R854_RESULT] accepted={accepted} verdict={summary['verdict']} checkpoint={final}",flush=True); print("R854_STRENGTH_RECOVERY_OK",flush=True)
'''


BRANCH = r'''
    if os.environ.get("R854_STRENGTH_RECOVERY_ONLY", "0") == "1":
        run_r854_strength_recovery(runner=runner,env=env,freeze_dir=Path(os.environ["R854_FREEZE_DIR"]).resolve(),kick_checkpoint=Path(os.environ["R854_KICK"]).resolve(),r851_summary=Path(os.environ["R854_R851_SUMMARY"]).resolve(),body_json=Path(os.environ["R854_BODY_JSON"]).resolve(),r853_checkpoint=Path(os.environ["R854_R853_CHECKPOINT"]).resolve(),output_dir=Path(os.environ["R854_OUTPUT"]).resolve(),updates=int(os.environ.get("R854_UPDATES","28")),rollout_steps=int(os.environ.get("R854_ROLLOUT_STEPS","256")),eval_episodes=int(os.environ.get("R854_EVAL_EPISODES","1024")),seed=args_cli.seed)
        return

'''


def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"anchor count {text.count(old)}: {old[:100]!r}")
    return text.replace(old, new, 1)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--source",required=True); parser.add_argument("--output",required=True); args=parser.parse_args()
    source=Path(args.source).resolve(); output=Path(args.output).resolve(); text=source.read_text(encoding="utf-8")
    text=once(text,"\ndef main() -> None:\n",FUNCTIONS+"\n\ndef main() -> None:\n")
    text=once(text,'    if os.environ.get("R851_KICK64_ONLY", "0") == "1" or os.environ.get("R853_PHASE_RECOVERY64_ONLY", "0") == "1":\n        from isaaclab.sensors import ContactSensorCfg\n','    if os.environ.get("R851_KICK64_ONLY", "0") == "1" or os.environ.get("R853_PHASE_RECOVERY64_ONLY", "0") == "1" or os.environ.get("R854_STRENGTH_RECOVERY_ONLY", "0") == "1":\n        from isaaclab.sensors import ContactSensorCfg\n')
    text=once(text,'    if os.environ.get("R853_PHASE_RECOVERY64_ONLY", "0") == "1":\n        run_r853_phase_recovery64(',BRANCH+'    if os.environ.get("R853_PHASE_RECOVERY64_ONLY", "0") == "1":\n        run_r853_phase_recovery64(')
    compile(text,str(output),"exec"); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(text,encoding="utf-8"); print(f"R854_PATCH_OK source={source} output={output}")


if __name__ == "__main__":
    main()
