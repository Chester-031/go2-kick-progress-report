#!/usr/bin/env python3
"""Add FL joint-target hold-bias scan and exact-policy video branch to R8.55."""

import argparse
from pathlib import Path


FUNCTIONS = r'''

# ============================================================
# R8.56: deterministic FL touchdown-hold bias scan + video.
# ============================================================

def _r856_fl_joint_ids(env):
    robot_names=list(getattr(env.robot,"joint_names",getattr(env.robot.data,"joint_names",[])))
    manager=env.base_env.action_manager; terms=getattr(manager,"_terms",{})
    term=terms.get("joint_pos") if isinstance(terms,dict) else None
    if term is None and isinstance(terms,(list,tuple)) and len(terms)==1: term=terms[0]
    joint_ids=getattr(term,"_joint_ids",slice(None)) if term is not None else slice(None)
    if isinstance(joint_ids,slice): action_joint_ids=list(range(len(robot_names)))[joint_ids]
    elif torch.is_tensor(joint_ids): action_joint_ids=[int(x) for x in joint_ids.detach().cpu().tolist()]
    else: action_joint_ids=[int(x) for x in joint_ids]
    names=[robot_names[i] for i in action_joint_ids]
    if len(names)!=12: raise RuntimeError(f"R856 action joint names invalid: robot={robot_names} action_ids={action_joint_ids} action={names}")
    def one(kind):
        ids=[i for i,n in enumerate(names) if "FL" in n.upper() and kind in n.lower()]
        if len(ids)!=1: raise RuntimeError(f"R856 cannot resolve FL {kind}: names={names} ids={ids}")
        return ids[0]
    ids={"hip":one("hip"),"thigh":one("thigh"),"calf":one("calf")}
    print(f"[R856_JOINTS] robot_names={robot_names} action_joint_ids={action_joint_ids} action_names={names} FL_action={ids}",flush=True)
    return names,ids


def _r856_bias_grid(device,fl):
    vals=(-0.12,-0.08,-0.04,0.0,0.04,0.08,0.12); rows=[]
    for thigh in vals:
        for calf in vals:
            b=torch.zeros(12,device=device); b[fl["thigh"]]=thigh; b[fl["calf"]]=calf; rows.append(b)
    return torch.stack(rows),[{"thigh":t,"calf":c} for t in vals for c in vals]


def _r856_rollout_cases(*,runner,env,components,kick,recovery,sensor,body_names,geometry,strength,biases,episodes_per_case,seed,hold_steps=5):
    torch.manual_seed(seed); np.random.seed(seed); env.reset(); n,device=env.num_envs,env.device; cases=biases.shape[0]; case_id=torch.arange(n,device=device)%cases
    desired=torch.full((n,),float(strength),device=device); env_strength=torch.ones(n,device=device); forward=geometry[0].repeat(n); lateral=geometry[1].repeat(n)
    placed=torch.zeros(n,dtype=torch.bool,device=device); settle=torch.zeros(n,dtype=torch.long,device=device); contaminated=torch.zeros(n,dtype=torch.bool,device=device); selected_seen=torch.zeros(n,dtype=torch.bool,device=device); wrong_seen=torch.zeros(n,dtype=torch.bool,device=device); body_seen=torch.zeros(n,dtype=torch.bool,device=device); real_seen=torch.zeros(n,dtype=torch.bool,device=device); touchdown_seen=torch.zeros(n,dtype=torch.bool,device=device); recovery_seen=torch.zeros(n,dtype=torch.bool,device=device); latch_seen=torch.zeros(n,dtype=torch.bool,device=device); hold_remaining=torch.zeros(n,dtype=torch.long,device=device); hold_action=torch.zeros(n,12,device=runner.device)
    completed=torch.zeros(cases,dtype=torch.long,device=device); real_count=torch.zeros(cases,device=device); td_count=torch.zeros(cases,device=device); rec_count=torch.zeros(cases,device=device); wrong_count=torch.zeros(cases,device=device); body_count=torch.zeros(cases,device=device); latch_count=torch.zeros(cases,device=device); step_dt=float(getattr(env.base_env,"step_dt",0.02)); loops=0
    with torch.no_grad():
        while not bool(torch.all(completed>=episodes_per_case)):
            contaminated|=_r851_prepare_episode_state(env,sensor,body_names,placed,settle,forward,lateral,env_strength); obs_runner=TensorDict({"policy":env.low_obs_buf},batch_size=[n],device=device).to(runner.device); active=env.kick_active.detach().bool().clone(); bridge=env.bridge_seen_r852.detach().bool().clone(); phase=env._phase().detach().clone(); recovery_control=real_seen&env.swing_seen_r89.detach().bool()&(phase>=0.78); holding=(hold_remaining>0)&recovery_control
            frozen,stable,swing_norm=_r851_frozen_action(runner,env,components,obs_runner,bridge,torch.zeros_like(recovery_control)); kick_delta=kick.mean(_r851_obs64(env.low_obs_buf).to(runner.device))*_r853_gain(desired).to(runner.device).unsqueeze(1); rec_obs=_r854_condition(_r851_obs64(swing_norm),desired); rec_delta=recovery.mean(rec_obs); pre=frozen+(active&bridge&(~recovery_control)).to(runner.device).unsqueeze(1)*kick_delta; proposed=torch.where(recovery_control.to(runner.device).unsqueeze(1),_r853_recovery_action(runner,components,obs_runner,stable,rec_delta),pre); action=torch.where(holding.to(runner.device).unsqueeze(1),hold_action,proposed)
            selected_before,_=env._front_foot_positions(); selected_before=selected_before.detach().clone(); ball_vel_before=env.ball.data.root_lin_vel_w.detach().clone(); direction=env.strike_direction_world.detach().clone(); direction=direction/torch.clamp(torch.linalg.vector_norm(direction,dim=1,keepdim=True),min=1e-6); _,_,done,_,_=env.kick_step(action.to(device)); done=done.to(device).bool(); cs=_r851_contact_state(env,sensor,body_names,selected_before,ball_vel_before,direction,active,done,step_dt)
            new_selected=cs["selected"]&(~selected_seen); wrong_seen|=cs["wrong"]&(~selected_seen); body_seen|=cs["body"]&(~selected_seen); real_now=new_selected&(~contaminated)&(~wrong_seen)&(~body_seen)&(cs["toward"]>0)&(cs["dv"]>0.02); selected_seen|=cs["selected"]; real_seen|=real_now; touchdown_seen|=real_seen&env.r829_touchdown_ok.detach().bool(); hold_remaining=torch.clamp(hold_remaining-1,min=0); new_latch=real_seen&env.r829_all_now.detach().bool()&(~holding)&(~done); biased=torch.clamp(action+biases[case_id].to(action.device),-1.0,1.0); hold_action[new_latch.to(runner.device)]=biased[new_latch.to(runner.device)].detach(); hold_remaining[new_latch]=hold_steps; latch_seen|=new_latch; recovery_seen|=real_seen&env.r829_recovery_seen.detach().bool()
            if done.any():
                ids=torch.nonzero(done,as_tuple=False).squeeze(-1); cid=case_id[ids]; accept=completed[cid]<episodes_per_case; ids=ids[accept]; cid=case_id[ids]
                if ids.numel():
                    ones=torch.ones(ids.numel(),device=device); completed+=torch.bincount(cid,minlength=cases); real_count+=torch.bincount(cid,weights=real_seen[ids].float(),minlength=cases); td_count+=torch.bincount(cid,weights=(real_seen[ids]&touchdown_seen[ids]).float(),minlength=cases); rec_count+=torch.bincount(cid,weights=(real_seen[ids]&recovery_seen[ids]).float(),minlength=cases); wrong_count+=torch.bincount(cid,weights=wrong_seen[ids].float(),minlength=cases); body_count+=torch.bincount(cid,weights=body_seen[ids].float(),minlength=cases); latch_count+=torch.bincount(cid,weights=latch_seen[ids].float(),minlength=cases)
                for t in (placed,contaminated,selected_seen,wrong_seen,body_seen,real_seen,touchdown_seen,recovery_seen,latch_seen): t[done]=False
                settle[done]=0; hold_remaining[done]=0; hold_action[done.to(runner.device)]=0
            loops+=1
            if loops>max(200000,episodes_per_case*(env.total_episode_steps+100)*4): raise RuntimeError("R856 scan timeout")
    rows=[]
    for i in range(cases):
        ep=max(int(completed[i]),1); real=max(float(real_count[i]),1.0); rows.append({"case":i,"episodes":ep,"real_rate":float(real_count[i])/ep,"touchdown_given_real":float(td_count[i])/real,"recovery_given_real":float(rec_count[i])/real,"latch_given_real":float(latch_count[i])/real,"wrong_rate":float(wrong_count[i])/ep,"body_rate":float(body_count[i])/ep})
    return rows


def _r856_load_chain(runner,env,freeze_dir,kick_checkpoint,recovery_checkpoint):
    components=_r848_load_frozen_components(runner=runner,freeze_dir=freeze_dir)
    for module in components.values():
        if isinstance(module,torch.nn.Module):
            module.eval()
            for p in module.parameters(): p.requires_grad_(False)
    kb=torch.load(kick_checkpoint,map_location=runner.device,weights_only=False); kick=R851Kick64ActorCritic(residual_scale=float(kb["residual_scale"]),action_std=float(kb["action_std"])).to(runner.device); kick.load_state_dict(kb["state_dict"]); kick.eval()
    rb=torch.load(recovery_checkpoint,map_location=runner.device,weights_only=False); recovery=R853Recovery64ActorCritic(scale=float(rb["scale"]),action_std=float(rb.get("action_std",0.05))).to(runner.device); recovery.load_state_dict(rb["state_dict"],strict=False); recovery.eval()
    env.r841_grace_steps=6; env.total_episode_steps=env.settle_max_steps+env.ball_arm_steps+env.kick_episode_steps+6; env.max_episode_length=env.total_episode_steps; env.max_low_episode_length=env.total_episode_steps; env.set_curriculum_stage("recovery")
    return components,kick,recovery


def run_r856_hold_scan(*,runner,env,freeze_dir,kick_checkpoint,r851_summary,body_json,recovery_checkpoint,output_dir,episodes_per_case,confirm_episodes,seed):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); geometry=torch.tensor(json.loads(Path(r851_summary).read_text(encoding="utf-8"))["top_geometry"][0],device=env.device); body_names=json.loads(Path(body_json).read_text(encoding="utf-8"))["body_names"]; components,kick,recovery=_r856_load_chain(runner,env,freeze_dir,kick_checkpoint,recovery_checkpoint); sensor=env.base_env.scene["r851_ball_robot_contact"]; joint_names,fl=_r856_fl_joint_ids(env); biases,labels=_r856_bias_grid(env.device,fl); scans={}; winners={}
    for si,strength in enumerate((0.75,1.0,1.25)):
        rows=_r856_rollout_cases(runner=runner,env=env,components=components,kick=kick,recovery=recovery,sensor=sensor,body_names=body_names,geometry=geometry,strength=strength,biases=biases,episodes_per_case=episodes_per_case,seed=seed+85600+si)
        for row,label in zip(rows,labels): row["bias"]=label
        eligible=[r for r in rows if r["real_rate"]>=0.90 and r["touchdown_given_real"]>=0.75 and r["wrong_rate"]<=0.02 and r["body_rate"]<=0.02]; pool=eligible or rows; winner=max(pool,key=lambda r:(r["recovery_given_real"],r["touchdown_given_real"],-(abs(r["bias"]["thigh"])+abs(r["bias"]["calf"])))); scans[str(strength)]=rows; winners[str(strength)]=winner; print(f"[R856_SCAN] strength={strength} winner={winner['case']} thigh={winner['bias']['thigh']:+.3f} calf={winner['bias']['calf']:+.3f} real={100*winner['real_rate']:.1f}% td={100*winner['touchdown_given_real']:.1f}% rec3={100*winner['recovery_given_real']:.1f}%",flush=True)
    confirms=[]
    for si,strength in enumerate((0.75,1.0,1.25)):
        w=winners[str(strength)]; b=torch.zeros(1,12,device=env.device); b[0,fl["thigh"]]=w["bias"]["thigh"]; b[0,fl["calf"]]=w["bias"]["calf"]; row=_r856_rollout_cases(runner=runner,env=env,components=components,kick=kick,recovery=recovery,sensor=sensor,body_names=body_names,geometry=geometry,strength=strength,biases=b,episodes_per_case=confirm_episodes,seed=seed+856000+si)[0]; row["strength"]=strength; row["bias"]=w["bias"]; confirms.append(row)
    accepted=all(x["real_rate"]>=0.95 and x["touchdown_given_real"]>=0.80 and x["recovery_given_real"]>=0.60 and x["wrong_rate"]<=0.02 and x["body_rate"]<=0.02 for x in confirms); winner_payload={"version":"r8_56_hold_bias","joint_names":joint_names,"fl_joint_ids":fl,"hold_steps":5,"geometry":geometry.detach().cpu().tolist(),"by_strength":{str(x["strength"]):x["bias"] for x in confirms},"recovery_checkpoint":str(recovery_checkpoint)}; (output_dir/"r856_hold_winner.json").write_text(json.dumps(winner_payload,indent=2)+"\n",encoding="utf-8"); summary={"version":"r8_56_hold_bias_scan","training":False,"actual_contact":True,"network_frozen":True,"grid_values":[-0.12,-0.08,-0.04,0.0,0.04,0.08,0.12],"episodes_per_case":episodes_per_case,"confirm_episodes":confirm_episodes,"scans":scans,"winners":winners,"confirm":confirms,"winner_file":str(output_dir/"r856_hold_winner.json"),"accepted":accepted,"verdict":"READY_FOR_VIDEO_REVIEW" if accepted else "KEEP_R856_HOLD_SCAN_NOT_READY"}; (output_dir/"r856_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8"); print("[R856_CONFIRM] "+" ".join(f"s{x['strength']} td={100*x['touchdown_given_real']:.1f}% rec3={100*x['recovery_given_real']:.1f}% bias={x['bias']}" for x in confirms),flush=True); print(f"[R856_RESULT] accepted={accepted} verdict={summary['verdict']}",flush=True); print("R856_HOLD_SCAN_OK",flush=True)


def run_r856_video(*,runner,env,freeze_dir,kick_checkpoint,r851_summary,body_json,recovery_checkpoint,winner_json,video_path,strength,seed):
    import imageio.v2 as imageio
    torch.manual_seed(seed); np.random.seed(seed); video_path=Path(video_path); video_path.parent.mkdir(parents=True,exist_ok=True); geometry=torch.tensor(json.loads(Path(r851_summary).read_text(encoding="utf-8"))["top_geometry"][0],device=env.device); body_names=json.loads(Path(body_json).read_text(encoding="utf-8"))["body_names"]; winner=json.loads(Path(winner_json).read_text(encoding="utf-8")); components,kick,recovery=_r856_load_chain(runner,env,freeze_dir,kick_checkpoint,recovery_checkpoint); sensor=env.base_env.scene["r851_ball_robot_contact"]; n=env.num_envs
    if n!=1: raise RuntimeError("R856 video requires num_envs=1")
    bias=torch.zeros(1,12,device=runner.device); spec=winner["by_strength"][str(float(strength))]; fl=winner["fl_joint_ids"]; bias[0,int(fl["thigh"])]=float(spec["thigh"]); bias[0,int(fl["calf"])]=float(spec["calf"]); desired=torch.tensor([strength],device=env.device); env_strength=torch.ones(1,device=env.device); forward=geometry[0].repeat(1); lateral=geometry[1].repeat(1); placed=torch.zeros(1,dtype=torch.bool,device=env.device); settle=torch.zeros(1,dtype=torch.long,device=env.device); real_seen=torch.zeros(1,dtype=torch.bool,device=env.device); selected_seen=torch.zeros_like(real_seen); contaminated=torch.zeros_like(real_seen); wrong_seen=torch.zeros_like(real_seen); body_seen=torch.zeros_like(real_seen); hold_remaining=torch.zeros(1,dtype=torch.long,device=env.device); hold_action=torch.zeros(1,12,device=runner.device); recovery_seen=torch.zeros_like(real_seen); touchdown_seen=torch.zeros_like(real_seen); step_dt=float(getattr(env.base_env,"step_dt",0.02)); render_env=env.low_env; seen=set()
    while not callable(getattr(render_env,"render",None)):
        if id(render_env) in seen: raise RuntimeError("R856 render unwrap cycle")
        seen.add(id(render_env)); nxt=getattr(render_env,"env",None)
        if nxt is None or nxt is render_env: nxt=getattr(render_env,"unwrapped",None)
        if nxt is None or nxt is render_env: raise RuntimeError("R856 no render-capable env")
        render_env=nxt
    writer=imageio.get_writer(video_path,fps=15); frames=0; done=torch.zeros(1,dtype=torch.bool,device=env.device)
    try:
        while not bool(done[0]):
            contaminated|=_r851_prepare_episode_state(env,sensor,body_names,placed,settle,forward,lateral,env_strength); obs_runner=TensorDict({"policy":env.low_obs_buf},batch_size=[1],device=env.device).to(runner.device); active=env.kick_active.detach().bool().clone(); bridge=env.bridge_seen_r852.detach().bool().clone(); phase=env._phase().detach().clone(); rec_control=real_seen&env.swing_seen_r89.detach().bool()&(phase>=0.78); holding=(hold_remaining>0)&rec_control; frozen,stable,swing_norm=_r851_frozen_action(runner,env,components,obs_runner,bridge,torch.zeros_like(rec_control)); kd=kick.mean(_r851_obs64(env.low_obs_buf).to(runner.device))*_r853_gain(desired).to(runner.device).unsqueeze(1); rd=recovery.mean(_r854_condition(_r851_obs64(swing_norm),desired)); pre=frozen+(active&bridge&(~rec_control)).to(runner.device).unsqueeze(1)*kd; proposed=torch.where(rec_control.to(runner.device).unsqueeze(1),_r853_recovery_action(runner,components,obs_runner,stable,rd),pre); action=torch.where(holding.to(runner.device).unsqueeze(1),hold_action,proposed)
            sb,_=env._front_foot_positions(); sb=sb.detach().clone(); bv=env.ball.data.root_lin_vel_w.detach().clone(); direction=env.strike_direction_world.detach().clone(); direction=direction/torch.clamp(torch.linalg.vector_norm(direction,dim=1,keepdim=True),min=1e-6); _,_,done,_,_=env.kick_step(action.to(env.device)); done=done.to(env.device).bool(); cs=_r851_contact_state(env,sensor,body_names,sb,bv,direction,active,done,step_dt); ns=cs["selected"]&(~selected_seen); wrong_seen|=cs["wrong"]&(~selected_seen); body_seen|=cs["body"]&(~selected_seen); rn=ns&(~contaminated)&(~wrong_seen)&(~body_seen)&(cs["toward"]>0)&(cs["dv"]>0.02); selected_seen|=cs["selected"]; real_seen|=rn; touchdown_seen|=real_seen&env.r829_touchdown_ok.bool(); hold_remaining=torch.clamp(hold_remaining-1,min=0); nl=real_seen&env.r829_all_now.bool()&(~holding)&(~done); hold_action[nl.to(runner.device)]=torch.clamp(action+bias,-1.0,1.0)[nl.to(runner.device)].detach(); hold_remaining[nl]=int(winner["hold_steps"]); recovery_seen|=real_seen&env.r829_recovery_seen.bool(); frame=render_env.render()
            if frame is None: raise RuntimeError("R856 render returned no frame")
            writer.append_data(frame); frames+=1
            if frames>500: raise RuntimeError("R856 video timeout")
    finally: writer.close()
    payload={"version":"r8_56_video","strength":float(strength),"seed":seed,"geometry":geometry.detach().cpu().tolist(),"bias":spec,"frames":frames,"real_contact":bool(real_seen[0]),"touchdown":bool(touchdown_seen[0]),"recovery3":bool(recovery_seen[0]),"wrong":bool(wrong_seen[0]),"body":bool(body_seen[0]),"video":str(video_path)}; video_path.with_suffix(".json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    if not payload["real_contact"] or payload["wrong"] or payload["body"]: raise RuntimeError(f"R856 video guard failed: {payload}")
    video_path.with_suffix(video_path.suffix+".ready").touch(); print(f"R856_EVAL_VIDEO={video_path}",flush=True); print(f"[R856_VIDEO] strength={strength} frames={frames} real={payload['real_contact']} touchdown={payload['touchdown']} recovery3={payload['recovery3']}",flush=True); print("R856_EVAL_VIDEO_READY",flush=True)
'''


BRANCH = r'''
    if os.environ.get("R856_HOLD_SCAN_ONLY", "0") == "1":
        run_r856_hold_scan(runner=runner,env=env,freeze_dir=Path(os.environ["R856_FREEZE_DIR"]).resolve(),kick_checkpoint=Path(os.environ["R856_KICK"]).resolve(),r851_summary=Path(os.environ["R856_R851_SUMMARY"]).resolve(),body_json=Path(os.environ["R856_BODY_JSON"]).resolve(),recovery_checkpoint=Path(os.environ["R856_RECOVERY"]).resolve(),output_dir=Path(os.environ["R856_OUTPUT"]).resolve(),episodes_per_case=int(os.environ.get("R856_EPISODES_PER_CASE","64")),confirm_episodes=int(os.environ.get("R856_CONFIRM_EPISODES","1024")),seed=args_cli.seed)
        return
    if os.environ.get("R856_VIDEO_ONLY", "0") == "1":
        run_r856_video(runner=runner,env=env,freeze_dir=Path(os.environ["R856_FREEZE_DIR"]).resolve(),kick_checkpoint=Path(os.environ["R856_KICK"]).resolve(),r851_summary=Path(os.environ["R856_R851_SUMMARY"]).resolve(),body_json=Path(os.environ["R856_BODY_JSON"]).resolve(),recovery_checkpoint=Path(os.environ["R856_RECOVERY"]).resolve(),winner_json=Path(os.environ["R856_WINNER"]).resolve(),video_path=Path(args_cli.eval_video).resolve(),strength=float(os.environ["R856_VIDEO_STRENGTH"]),seed=args_cli.seed)
        return

'''


def once(text,old,new):
    if text.count(old)!=1: raise RuntimeError(f"anchor count {text.count(old)}: {old[:100]!r}")
    return text.replace(old,new,1)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",required=True); p.add_argument("--output",required=True); a=p.parse_args(); source=Path(a.source).resolve(); output=Path(a.output).resolve(); text=source.read_text(encoding="utf-8")
    text=once(text,"\ndef main() -> None:\n",FUNCTIONS+"\n\ndef main() -> None:\n")
    text=once(text,' or os.environ.get("R855_TOUCHDOWN_HOLD_ONLY", "0") == "1":\n        from isaaclab.sensors import ContactSensorCfg\n',' or os.environ.get("R855_TOUCHDOWN_HOLD_ONLY", "0") == "1" or os.environ.get("R856_HOLD_SCAN_ONLY", "0") == "1" or os.environ.get("R856_VIDEO_ONLY", "0") == "1":\n        from isaaclab.sensors import ContactSensorCfg\n')
    text=once(text,'    if os.environ.get("R855_TOUCHDOWN_HOLD_ONLY", "0") == "1":\n        run_r855_touchdown_hold(',BRANCH+'    if os.environ.get("R855_TOUCHDOWN_HOLD_ONLY", "0") == "1":\n        run_r855_touchdown_hold(')
    compile(text,str(output),"exec"); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(text,encoding="utf-8"); print(f"R856_PATCH_OK source={source} output={output}")


if __name__=="__main__": main()
