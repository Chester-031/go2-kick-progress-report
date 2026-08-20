#!/usr/bin/env python3
"""Add phase-aware Recovery64 PPO to a completed R8.51 generated source."""

import argparse
from pathlib import Path


FUNCTIONS = r'''

# ============================================================
# R8.53: phase-aware true-contact Recovery64 PPO.
# ============================================================

class R853Recovery64ActorCritic(torch.nn.Module):
    def __init__(self, scale, action_std=0.04):
        super().__init__(); self.scale=float(scale)
        self.actor=torch.nn.Sequential(torch.nn.Linear(64,64),torch.nn.ELU(),torch.nn.Linear(64,64),torch.nn.ELU(),torch.nn.Linear(64,12))
        self.critic=torch.nn.Sequential(torch.nn.Linear(64,128),torch.nn.ELU(),torch.nn.Linear(128,64),torch.nn.ELU(),torch.nn.Linear(64,1))
        self.register_buffer("action_std",torch.full((12,),float(action_std)))
    def mean(self,obs): return self.scale*torch.tanh(self.actor(obs))
    def value(self,obs): return self.critic(obs).squeeze(-1)
    def sample(self,obs):
        mean=self.mean(obs); dist=torch.distributions.Normal(mean,self.action_std); action=dist.sample()
        return action,dist.log_prob(action).sum(-1),self.value(obs)
    def evaluate_actions(self,obs,action):
        dist=torch.distributions.Normal(self.mean(obs),self.action_std)
        return dist.log_prob(action).sum(-1),dist.entropy().sum(-1),self.value(obs)


def _r853_load_recovery64(vector_path, scale_path, device):
    vector=torch.load(vector_path,map_location=device,weights_only=False)
    scale_bundle=torch.load(scale_path,map_location=device,weights_only=False)
    model=R853Recovery64ActorCritic(scale=0.50*float(scale_bundle["scale"]),action_std=0.04).to(device)
    old=vector["state_dict"]; new=model.state_dict()
    mapping={"actor.0.weight":"net.0.weight","actor.0.bias":"net.0.bias","actor.2.weight":"net.2.weight","actor.2.bias":"net.2.bias","actor.4.weight":"net.4.weight","actor.4.bias":"net.4.bias"}
    for dst,src in mapping.items():
        if dst=="actor.0.weight":
            migrated=new[dst].clone(); migrated[:,:63]=old[src][:,:63]; migrated[:,63]=old[src][:,69]; new[dst]=migrated
        else: new[dst]=old[src]
    model.load_state_dict(new)
    return model,{"parent":str(vector_path),"migration":"R839 normalized 70D[0:63,69] -> Recovery64","initial_scale":model.scale}


def _r853_gain(desired):
    low=0.5+(desired-0.75)*2.0
    high=1.0+(desired-1.0)*4.0
    return torch.where(desired<=1.0,low,high).clamp(0.5,2.0)


def _r853_recovery_action(runner,components,obs_runner,stable,recovery_delta):
    norm=_r820_actor_input(components["recovery_policy"],obs_runner)
    return stable+1.55*components["recovery_head"](norm)+recovery_delta


def _r853_eval(*,runner,env,components,kick,recovery,sensor,body_names,geometry,desired_strength,episodes,seed):
    torch.manual_seed(seed); np.random.seed(seed); env.reset(); n,device=env.num_envs,env.device
    desired=torch.full((n,),float(desired_strength),device=device); env_strength=torch.ones(n,device=device)
    forward=geometry[0].repeat(n); lateral=geometry[1].repeat(n)
    placed=torch.zeros(n,dtype=torch.bool,device=device); settle=torch.zeros(n,dtype=torch.long,device=device)
    contaminated=torch.zeros(n,dtype=torch.bool,device=device); selected_seen=torch.zeros(n,dtype=torch.bool,device=device)
    wrong_seen=torch.zeros(n,dtype=torch.bool,device=device); body_seen=torch.zeros(n,dtype=torch.bool,device=device)
    real_seen=torch.zeros(n,dtype=torch.bool,device=device); recovery_seen=torch.zeros(n,dtype=torch.bool,device=device)
    gates={name:torch.zeros(n,dtype=torch.bool,device=device) for name in ("candidate","return","contact","support4","base","height","upright","touchdown","all_now")}
    peak_speed=torch.zeros(n,device=device); max_disp=torch.zeros(n,device=device)
    completed=0; totals={"real":0,"clean":0,"wrong":0,"body":0,"recovery":0,**{k:0 for k in gates}}; speed_sum=disp_sum=0.0
    step_dt=float(getattr(env.base_env,"step_dt",0.02)); loops=0
    with torch.no_grad():
        while completed<episodes:
            contaminated|=_r851_prepare_episode_state(env,sensor,body_names,placed,settle,forward,lateral,env_strength)
            obs_runner=TensorDict({"policy":env.low_obs_buf},batch_size=[n],device=device).to(runner.device)
            active=env.kick_active.detach().bool().clone(); bridge=env.bridge_seen_r852.detach().bool().clone(); phase=env._phase().detach().clone()
            recovery_control=real_seen & env.swing_seen_r89.detach().bool() & (phase>=0.78)
            frozen,stable,swing_norm=_r851_frozen_action(runner,env,components,obs_runner,bridge,torch.zeros_like(recovery_control))
            kick_obs=_r851_obs64(env.low_obs_buf).to(runner.device); kick_delta=kick.mean(kick_obs)*_r853_gain(desired).to(runner.device).unsqueeze(1)
            recovery_obs=_r851_obs64(swing_norm); recovery_delta=recovery.mean(recovery_obs)
            strike_active=active & bridge & (~recovery_control); pre=frozen+strike_active.to(runner.device).unsqueeze(1)*kick_delta
            rec_action=_r853_recovery_action(runner,components,obs_runner,stable,recovery_delta)
            action=torch.where(recovery_control.to(runner.device).unsqueeze(1),rec_action,pre)
            selected_before,_=env._front_foot_positions(); selected_before=selected_before.detach().clone(); ball_vel_before=env.ball.data.root_lin_vel_w.detach().clone()
            direction=env.strike_direction_world.detach().clone(); direction=direction/torch.clamp(torch.linalg.vector_norm(direction,dim=1,keepdim=True),min=1e-6)
            _,_,done,_,_=env.kick_step(action.to(device)); done=done.to(device).bool()
            cs=_r851_contact_state(env,sensor,body_names,selected_before,ball_vel_before,direction,active,done,step_dt)
            new_selected=cs["selected"]&(~selected_seen); wrong_seen|=cs["wrong"]&(~selected_seen); body_seen|=cs["body"]&(~selected_seen)
            clean_now=new_selected&(~contaminated)&(~wrong_seen)&(~body_seen); real_now=clean_now&(cs["toward"]>0)&(cs["dv"]>0.02)
            selected_seen|=cs["selected"]; real_seen|=real_now; recovery_seen|=real_seen&env.r829_recovery_seen.detach().bool()
            valid=real_seen&(~done)
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
            if loops>max(10000,episodes*(env.total_episode_steps+100)): raise RuntimeError("R853 eval timeout")
    d=max(completed,1); rd=max(totals["real"],1)
    return {"strength":float(desired_strength),"episodes":completed,"clean_rate":totals["clean"]/d,"real_rate":totals["real"]/d,"wrong_rate":totals["wrong"]/d,"body_rate":totals["body"]/d,"recovery_given_real":totals["recovery"]/rd,"mean_peak_speed_mps":speed_sum/d,"mean_disp_m":disp_sum/d,**{f"{k}_given_real":totals[k]/rd for k in gates}}


def run_r853_phase_recovery64(*,runner,env,freeze_dir,kick_checkpoint,r851_summary,body_json,output_dir,updates,rollout_steps,eval_episodes,seed):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); s851=json.loads(Path(r851_summary).read_text(encoding="utf-8")); body_names=json.loads(Path(body_json).read_text(encoding="utf-8"))["body_names"]
    geometry=torch.tensor(s851["top_geometry"][0],device=env.device); components=_r848_load_frozen_components(runner=runner,freeze_dir=freeze_dir)
    for module in components.values():
        if isinstance(module,torch.nn.Module): module.eval(); [p.requires_grad_(False) for p in module.parameters()]
    kb=torch.load(kick_checkpoint,map_location=runner.device,weights_only=False); kick=R851Kick64ActorCritic(residual_scale=float(kb["residual_scale"]),action_std=float(kb["action_std"])).to(runner.device); kick.load_state_dict(kb["state_dict"]); kick.eval(); [p.requires_grad_(False) for p in kick.parameters()]
    recovery,migration=_r853_load_recovery64(Path(freeze_dir)/"artifacts/r839_recovery_vector_u32.pt",Path(freeze_dir)/"artifacts/r835_swing_stability_u32.pt",runner.device)
    initial={k:p.detach().clone() for k,p in recovery.actor.named_parameters()}; optimizer=torch.optim.Adam(recovery.parameters(),lr=1.0e-4); sensor=env.base_env.scene["r851_ball_robot_contact"]
    env.r841_grace_steps=6; env.total_episode_steps=env.settle_max_steps+env.ball_arm_steps+env.kick_episode_steps+6; env.max_episode_length=env.total_episode_steps; env.max_low_episode_length=env.total_episode_steps; env.set_curriculum_stage("recovery")
    n,device=env.num_envs,env.device; generator=torch.Generator(device=device); generator.manual_seed(seed); history=[]
    for update in range(1,updates+1):
        env.reset(); anchors=torch.tensor([0.75,1.0,1.25],device=device); desired=anchors[torch.randint(0,3,(n,),device=device,generator=generator)]; env_strength=torch.ones(n,device=device); forward=geometry[0].repeat(n); lateral=geometry[1].repeat(n)
        placed=torch.zeros(n,dtype=torch.bool,device=device); settle=torch.zeros(n,dtype=torch.long,device=device); contaminated=torch.zeros(n,dtype=torch.bool,device=device); selected_seen=torch.zeros(n,dtype=torch.bool,device=device); wrong_seen=torch.zeros(n,dtype=torch.bool,device=device); body_seen=torch.zeros(n,dtype=torch.bool,device=device); real_seen=torch.zeros(n,dtype=torch.bool,device=device); rewarded=torch.zeros(n,dtype=torch.bool,device=device); all_rewarded=torch.zeros(n,dtype=torch.bool,device=device)
        stores=[[] for _ in range(7)]; counts={"active":0,"candidate":0,"all":0,"recovery":0,"real":0}; step_dt=float(getattr(env.base_env,"step_dt",0.02)); recovery.train()
        for _ in range(rollout_steps):
            contaminated|=_r851_prepare_episode_state(env,sensor,body_names,placed,settle,forward,lateral,env_strength)
            obs_runner=TensorDict({"policy":env.low_obs_buf},batch_size=[n],device=device).to(runner.device); active=env.kick_active.detach().bool().clone(); bridge=env.bridge_seen_r852.detach().bool().clone(); phase=env._phase().detach().clone(); recovery_control=real_seen&env.swing_seen_r89.detach().bool()&(phase>=0.78)
            frozen,stable,swing_norm=_r851_frozen_action(runner,env,components,obs_runner,bridge,torch.zeros_like(recovery_control)); kick_delta=kick.mean(_r851_obs64(env.low_obs_buf).to(runner.device))*_r853_gain(desired).to(runner.device).unsqueeze(1)
            rec_obs=_r851_obs64(swing_norm); sample,logp,value=recovery.sample(rec_obs); pre=frozen+(active&bridge&(~recovery_control)).to(runner.device).unsqueeze(1)*kick_delta; rec_action=_r853_recovery_action(runner,components,obs_runner,stable,sample); action=torch.where(recovery_control.to(runner.device).unsqueeze(1),rec_action,pre)
            selected_before,_=env._front_foot_positions(); selected_before=selected_before.detach().clone(); ball_vel_before=env.ball.data.root_lin_vel_w.detach().clone(); direction=env.strike_direction_world.detach().clone(); direction=direction/torch.clamp(torch.linalg.vector_norm(direction,dim=1,keepdim=True),min=1e-6)
            _,_,done,_,_=env.kick_step(action.to(device)); done=done.to(device).bool(); cs=_r851_contact_state(env,sensor,body_names,selected_before,ball_vel_before,direction,active,done,step_dt)
            new_selected=cs["selected"]&(~selected_seen); wrong_seen|=cs["wrong"]&(~selected_seen); body_seen|=cs["body"]&(~selected_seen); real_now=new_selected&(~contaminated)&(~wrong_seen)&(~body_seen)&(cs["toward"]>0)&(cs["dv"]>0.02); selected_seen|=cs["selected"]; real_seen|=real_now
            candidate=real_seen&env.r829_rec_candidate.detach().bool(); all_now=real_seen&env.r829_all_now.detach().bool(); new_all=all_now&(~all_rewarded); new_recovery=real_seen&env.r829_recovery_seen.detach().bool()&(~rewarded)
            return_score=torch.exp(-env.r817_foot_forward.abs()/0.05-env.r817_foot_lift.abs()/0.02); body_score=torch.exp(-env.r829_base_displacement/0.065-env.r829_height_error/0.035-env.r829_upright_error/0.025)
            reward=recovery_control.float()*(0.55*return_score+0.35*body_score)+0.5*candidate.float()+3.0*new_all.float()+7.0*new_recovery.float()-5.0*(cs["wrong"]|cs["body"]).float(); reward=reward-(0.01*recovery_control.to(runner.device).float()*sample.pow(2).sum(1)).to(device)
            all_rewarded|=new_all; rewarded|=new_recovery; counts["active"]+=int(recovery_control.sum()); counts["candidate"]+=int(candidate.sum()); counts["all"]+=int(new_all.sum()); counts["recovery"]+=int(new_recovery.sum()); counts["real"]+=int(real_now.sum())
            for store,item in zip(stores,(rec_obs.detach(),sample.detach(),logp.detach(),value.detach(),reward.to(runner.device).detach(),done.to(runner.device).float(),recovery_control.to(runner.device).float())): store.append(item)
            if done.any():
                for tensor in (placed,contaminated,selected_seen,wrong_seen,body_seen,real_seen,rewarded,all_rewarded): tensor[done]=False
                settle[done]=0; desired[done]=anchors[torch.randint(0,3,(int(done.sum()),),device=device,generator=generator)]
        obs_t,act_t,old_logp_t,value_t,reward_t,done_t,active_t=[torch.stack(x) for x in stores]; bootstrap=recovery.value(_r851_obs64(_r820_actor_input(components["swing_policy"],TensorDict({"policy":env.low_obs_buf},batch_size=[n],device=device).to(runner.device)))).detach()
        adv=torch.zeros_like(reward_t); last=torch.zeros(n,device=runner.device)
        for t in reversed(range(rollout_steps)):
            nv=bootstrap if t==rollout_steps-1 else value_t[t+1]; non=1-done_t[t]; delta=reward_t[t]+0.99*nv*non-value_t[t]; last=delta+0.99*0.95*non*last; adv[t]=last
        ret=adv+value_t; fo=obs_t.reshape(-1,64); fa=act_t.reshape(-1,12); fl=old_logp_t.reshape(-1); fr=ret.reshape(-1); fm=active_t.reshape(-1)>0.5; fav=adv.reshape(-1); aa=fav[fm]; nav=(fav-aa.mean())/(aa.std(unbiased=False)+1e-8) if aa.numel() else fav; losses=[]
        for _epoch in range(4):
            perm=torch.randperm(fo.shape[0],device=runner.device)
            for start in range(0,fo.shape[0],8192):
                idx=perm[start:start+8192]; nlog,ent,nval=recovery.evaluate_actions(fo[idx],fa[idx]); mask=fm[idx]
                if mask.any(): ratio=torch.exp(nlog[mask]-fl[idx][mask]); a=nav[idx][mask]; actor=-torch.min(ratio*a,torch.clamp(ratio,0.8,1.2)*a).mean()
                else: actor=nval.sum()*0
                value_loss=(nval-fr[idx]).pow(2).mean(); anchor=sum((p-initial[k]).pow(2).mean() for k,p in recovery.actor.named_parameters()); loss=actor+0.5*value_loss+0.02*anchor-0.001*ent.mean(); optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(recovery.parameters(),1.0); optimizer.step(); losses.append((float(actor.detach()),float(value_loss.detach()),float(anchor.detach())))
        rec={"update":update,**counts,"actor_loss":float(np.mean([x[0] for x in losses])),"value_loss":float(np.mean([x[1] for x in losses])),"anchor":float(np.mean([x[2] for x in losses]))}; history.append(rec); print(f"[R853_UPDATE] u={update:02d} active={counts['active']} real={counts['real']} candidate={counts['candidate']} all={counts['all']} recovery={counts['recovery']} actorLoss={rec['actor_loss']:.5f} valueLoss={rec['value_loss']:.5f} anchor={rec['anchor']:.7f}",flush=True)
        torch.save({"version":"r8_53_phase_recovery64","obs_dim":64,"scale":recovery.scale,"action_std":float(recovery.action_std[0]),"state_dict":recovery.state_dict(),"migration":migration,"history":history},output_dir/"recovery64_latest.pt")
    recovery.eval(); confirms=[_r853_eval(runner=runner,env=env,components=components,kick=kick,recovery=recovery,sensor=sensor,body_names=body_names,geometry=geometry,desired_strength=s,episodes=eval_episodes,seed=seed+853000+i) for i,s in enumerate((0.75,1.0,1.25))]
    speeds=[x["mean_peak_speed_mps"] for x in confirms]; monotonic=speeds[1]>=speeds[0]+0.01 and speeds[2]>=speeds[1]+0.01; mid=confirms[1]; accepted=mid["real_rate"]>=0.95 and mid["candidate_given_real"]>=0.95 and mid["touchdown_given_real"]>=0.80 and mid["recovery_given_real"]>=0.60 and mid["wrong_rate"]<=0.02 and mid["body_rate"]<=0.02 and monotonic
    final=output_dir/"recovery64_final.pt"; torch.save({"version":"r8_53_phase_recovery64","obs_dim":64,"scale":recovery.scale,"action_std":float(recovery.action_std[0]),"state_dict":recovery.state_dict(),"migration":migration,"history":history,"strength_confirm":confirms},final)
    summary={"version":"r8_53_phase_recovery64","training":True,"actual_contact":True,"kick_frozen":True,"accepted_r839_parent":True,"r849_used":False,"geometry":geometry.detach().cpu().tolist(),"strength_gain":{"0.75":0.5,"1.0":1.0,"1.25":2.0},"recovery_gate":"real_strike & swing_seen & phase>=0.78","updates":updates,"rollout_steps":rollout_steps,"history":history,"strength_confirm":confirms,"strength_monotonic":monotonic,"checkpoint":str(final),"accepted":accepted,"verdict":"READY_FOR_4_SKILL_ROUTER" if accepted else "KEEP_R853_RECOVERY_NOT_READY"}
    (output_dir/"r853_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8"); print(f"[R853_CONFIRM] low={speeds[0]:.3f} mid={speeds[1]:.3f} high={speeds[2]:.3f} monotonic={monotonic} candidate={100*mid['candidate_given_real']:.2f}% touchdown={100*mid['touchdown_given_real']:.2f}% recovery={100*mid['recovery_given_real']:.2f}%",flush=True); print(f"[R853_RESULT] accepted={accepted} verdict={summary['verdict']} checkpoint={final}",flush=True); print("R853_PHASE_RECOVERY64_OK",flush=True)
'''


BRANCH=r'''
    if os.environ.get("R853_PHASE_RECOVERY64_ONLY", "0") == "1":
        run_r853_phase_recovery64(runner=runner,env=env,freeze_dir=Path(os.environ["R853_FREEZE_DIR"]).resolve(),kick_checkpoint=Path(os.environ["R853_KICK"]).resolve(),r851_summary=Path(os.environ["R853_R851_SUMMARY"]).resolve(),body_json=Path(os.environ["R853_BODY_JSON"]).resolve(),output_dir=Path(os.environ["R853_OUTPUT"]).resolve(),updates=int(os.environ.get("R853_UPDATES","20")),rollout_steps=int(os.environ.get("R853_ROLLOUT_STEPS","256")),eval_episodes=int(os.environ.get("R853_EVAL_EPISODES","1024")),seed=args_cli.seed)
        return

'''


def once(text,old,new):
    if text.count(old)!=1: raise RuntimeError(f"anchor count {text.count(old)}: {old[:80]!r}")
    return text.replace(old,new,1)
def main():
    p=argparse.ArgumentParser();p.add_argument("--source",required=True);p.add_argument("--output",required=True);a=p.parse_args();src=Path(a.source).resolve();out=Path(a.output).resolve();text=src.read_text(encoding="utf-8")
    text=once(text,"\ndef main() -> None:\n",FUNCTIONS+"\n\ndef main() -> None:\n")
    text=once(text,'    if os.environ.get("R851_KICK64_ONLY", "0") == "1":\n        from isaaclab.sensors import ContactSensorCfg\n','    if os.environ.get("R851_KICK64_ONLY", "0") == "1" or os.environ.get("R853_PHASE_RECOVERY64_ONLY", "0") == "1":\n        from isaaclab.sensors import ContactSensorCfg\n')
    text=once(text,'    if os.environ.get("R851_KICK64_ONLY", "0") == "1":\n        run_r851_kick64(\n',BRANCH+'    if os.environ.get("R851_KICK64_ONLY", "0") == "1":\n        run_r851_kick64(\n')
    compile(text,str(out),"exec");out.parent.mkdir(parents=True,exist_ok=True);out.write_text(text,encoding="utf-8");print(f"R853_PATCH_OK source={src} output={out}")
if __name__=="__main__":main()
