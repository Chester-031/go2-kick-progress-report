#!/usr/bin/env python3
"""Correct R8.56 env/case confounding with balanced randomized crossover."""
import argparse
from pathlib import Path

FUNCTIONS=r'''

# ============================================================
# R8.57: balanced randomized crossover hold-controller scan.
# ============================================================

def _r857_candidates(device,fl):
    vals=(-0.12,-0.08,-0.04,0.0,0.04,0.08,0.12); biases=[]; labels=[]
    for mode in ("static_latch","dynamic_target"):
        for thigh in vals:
            for calf in vals:
                b=torch.zeros(12,device=device); b[fl["thigh"]]=thigh; b[fl["calf"]]=calf; biases.append(b); labels.append({"mode":mode,"thigh":thigh,"calf":calf})
    return torch.stack(biases),labels


def _r857_schedule(cases,repeats,device,generator):
    return torch.cat([torch.randperm(cases,device=device,generator=generator) for _ in range(repeats)])


def _r857_balanced_rollout(*,runner,env,components,kick,recovery,sensor,body_names,geometry,strength,biases,labels,episodes_per_case,seed,hold_steps=5):
    torch.manual_seed(seed); np.random.seed(seed); env.reset(); n,device=env.num_envs,env.device; cases=biases.shape[0]; generator=torch.Generator(device=device); generator.manual_seed(seed+5700); schedule=_r857_schedule(cases,episodes_per_case,device,generator); total=int(schedule.numel()); cursor=min(n,total); case_id=torch.zeros(n,dtype=torch.long,device=device); assigned=torch.zeros(n,dtype=torch.bool,device=device); case_id[:cursor]=schedule[:cursor]; assigned[:cursor]=True
    mode_dynamic_cases=torch.tensor([x["mode"]=="dynamic_target" for x in labels],dtype=torch.bool,device=device); desired=torch.full((n,),float(strength),device=device); env_strength=torch.ones(n,device=device); forward=geometry[0].repeat(n); lateral=geometry[1].repeat(n); placed=torch.zeros(n,dtype=torch.bool,device=device); settle=torch.zeros(n,dtype=torch.long,device=device); contaminated=torch.zeros(n,dtype=torch.bool,device=device); selected_seen=torch.zeros(n,dtype=torch.bool,device=device); wrong_seen=torch.zeros(n,dtype=torch.bool,device=device); body_seen=torch.zeros(n,dtype=torch.bool,device=device); real_seen=torch.zeros(n,dtype=torch.bool,device=device); touchdown_seen=torch.zeros(n,dtype=torch.bool,device=device); recovery_seen=torch.zeros(n,dtype=torch.bool,device=device); hold_remaining=torch.zeros(n,dtype=torch.long,device=device); hold_action=torch.zeros(n,12,device=runner.device)
    completed=torch.zeros(cases,dtype=torch.long,device=device); realc=torch.zeros(cases,device=device); tdc=torch.zeros(cases,device=device); recc=torch.zeros(cases,device=device); wrongc=torch.zeros(cases,device=device); bodyc=torch.zeros(cases,device=device); step_dt=float(getattr(env.base_env,"step_dt",0.02)); scored=0; loops=0
    with torch.no_grad():
        while scored<total:
            contaminated|=_r851_prepare_episode_state(env,sensor,body_names,placed,settle,forward,lateral,env_strength); obs=TensorDict({"policy":env.low_obs_buf},batch_size=[n],device=device).to(runner.device); active=env.kick_active.bool().clone(); bridge=env.bridge_seen_r852.bool().clone(); phase=env._phase().clone(); rc=real_seen&env.swing_seen_r89.bool()&(phase>=0.78); holding=(hold_remaining>0)&rc; frozen,stable,snorm=_r851_frozen_action(runner,env,components,obs,bridge,torch.zeros_like(rc)); kd=kick.mean(_r851_obs64(env.low_obs_buf).to(runner.device))*_r853_gain(desired).to(runner.device).unsqueeze(1); rd=recovery.mean(_r854_condition(_r851_obs64(snorm),desired)); pre=frozen+(active&bridge&(~rc)).to(runner.device).unsqueeze(1)*kd; proposed=torch.where(rc.to(runner.device).unsqueeze(1),_r853_recovery_action(runner,components,obs,stable,rd),pre); dynamic=mode_dynamic_cases[case_id]; biased_dynamic=torch.clamp(proposed+biases[case_id].to(proposed.device),-1.0,1.0); held=torch.where(dynamic.to(runner.device).unsqueeze(1),biased_dynamic,hold_action); action=torch.where(holding.to(runner.device).unsqueeze(1),held,proposed)
            sb,_=env._front_foot_positions(); sb=sb.detach().clone(); bv=env.ball.data.root_lin_vel_w.detach().clone(); direction=env.strike_direction_world.detach().clone(); direction=direction/torch.clamp(torch.linalg.vector_norm(direction,dim=1,keepdim=True),min=1e-6); _,_,done,_,_=env.kick_step(action.to(device)); done=done.to(device).bool(); cs=_r851_contact_state(env,sensor,body_names,sb,bv,direction,active,done,step_dt); ns=cs["selected"]&(~selected_seen); wrong_seen|=cs["wrong"]&(~selected_seen); body_seen|=cs["body"]&(~selected_seen); rn=ns&(~contaminated)&(~wrong_seen)&(~body_seen)&(cs["toward"]>0)&(cs["dv"]>0.02); selected_seen|=cs["selected"]; real_seen|=rn; touchdown_seen|=real_seen&env.r829_touchdown_ok.bool(); hold_remaining=torch.clamp(hold_remaining-1,min=0); nl=real_seen&env.r829_all_now.bool()&(~holding)&(~done); hold_action[nl.to(runner.device)]=torch.clamp(action+biases[case_id].to(action.device),-1.0,1.0)[nl.to(runner.device)].detach(); hold_remaining[nl]=hold_steps; recovery_seen|=real_seen&env.r829_recovery_seen.bool()
            if done.any():
                ids=torch.nonzero(done,as_tuple=False).squeeze(-1); score_ids=ids[assigned[ids]]
                if score_ids.numel():
                    cid=case_id[score_ids]; completed+=torch.bincount(cid,minlength=cases); realc+=torch.bincount(cid,weights=real_seen[score_ids].float(),minlength=cases); tdc+=torch.bincount(cid,weights=(real_seen[score_ids]&touchdown_seen[score_ids]).float(),minlength=cases); recc+=torch.bincount(cid,weights=(real_seen[score_ids]&recovery_seen[score_ids]).float(),minlength=cases); wrongc+=torch.bincount(cid,weights=wrong_seen[score_ids].float(),minlength=cases); bodyc+=torch.bincount(cid,weights=body_seen[score_ids].float(),minlength=cases); scored+=int(score_ids.numel())
                assigned[ids]=False
                available=total-cursor; take=min(int(ids.numel()),available)
                if take>0: target=ids[:take]; case_id[target]=schedule[cursor:cursor+take]; assigned[target]=True; cursor+=take
                for t in (placed,contaminated,selected_seen,wrong_seen,body_seen,real_seen,touchdown_seen,recovery_seen): t[done]=False
                settle[done]=0; hold_remaining[done]=0; hold_action[done.to(runner.device)]=0
            loops+=1
            if loops>max(300000,total*(env.total_episode_steps+100)//max(n,1)*5): raise RuntimeError("R857 balanced scan timeout")
    if not bool(torch.all(completed==episodes_per_case)): raise RuntimeError(f"R857 imbalance completed={completed.tolist()} expected={episodes_per_case}")
    rows=[]
    for i in range(cases):
        ep=int(completed[i]); real=max(float(realc[i]),1.0); td=max(float(tdc[i]),1.0); rows.append({"case":i,"episodes":ep,"real_rate":float(realc[i])/ep,"touchdown_given_real":float(tdc[i])/real,"recovery_given_real":float(recc[i])/real,"retention_given_touchdown":float(recc[i])/td,"wrong_rate":float(wrongc[i])/ep,"body_rate":float(bodyc[i])/ep})
    return rows


def run_r857_balanced_scan(*,runner,env,freeze_dir,kick_checkpoint,r851_summary,body_json,recovery_checkpoint,output_dir,episodes_per_case,confirm_episodes,seed):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); geometry=torch.tensor(json.loads(Path(r851_summary).read_text(encoding="utf-8"))["top_geometry"][0],device=env.device); body_names=json.loads(Path(body_json).read_text(encoding="utf-8"))["body_names"]; components,kick,recovery=_r856_load_chain(runner,env,freeze_dir,kick_checkpoint,recovery_checkpoint); sensor=env.base_env.scene["r851_ball_robot_contact"]; names,fl=_r856_fl_joint_ids(env); biases,labels=_r857_candidates(env.device,fl); scans={}; winners={}; confirms={}
    for si,strength in enumerate((0.75,1.0,1.25)):
        rows=_r857_balanced_rollout(runner=runner,env=env,components=components,kick=kick,recovery=recovery,sensor=sensor,body_names=body_names,geometry=geometry,strength=strength,biases=biases,labels=labels,episodes_per_case=episodes_per_case,seed=seed+85700+si)
        for r,l in zip(rows,labels): r["controller"]=l
        eligible=[r for r in rows if r["real_rate"]>=0.95 and r["touchdown_given_real"]>=0.80 and r["wrong_rate"]<=0.02 and r["body_rate"]<=0.02]; pool=eligible or rows; winner=max(pool,key=lambda r:(r["recovery_given_real"],r["retention_given_touchdown"],-(abs(r["controller"]["thigh"])+abs(r["controller"]["calf"])))); scans[str(strength)]=rows; winners[str(strength)]=winner; print(f"[R857_SCREEN] s={strength} case={winner['case']} ctrl={winner['controller']} td={100*winner['touchdown_given_real']:.1f}% rec3={100*winner['recovery_given_real']:.1f}% retain={100*winner['retention_given_touchdown']:.1f}%",flush=True)
        b=torch.zeros(1,12,device=env.device); b[0,fl["thigh"]]=winner["controller"]["thigh"]; b[0,fl["calf"]]=winner["controller"]["calf"]; lab=[winner["controller"]]; seed_rows=[]
        for ci,cseed in enumerate((seed+857000+si*10,seed+857000+si*10+1)):
            row=_r857_balanced_rollout(runner=runner,env=env,components=components,kick=kick,recovery=recovery,sensor=sensor,body_names=body_names,geometry=geometry,strength=strength,biases=b,labels=lab,episodes_per_case=confirm_episodes,seed=cseed)[0]; row["seed"]=cseed; row["strength"]=strength; row["controller"]=winner["controller"]; seed_rows.append(row)
        confirms[str(strength)]=seed_rows; print(f"[R857_CONFIRM] s={strength} "+" ".join(f"seed{x['seed']} td={100*x['touchdown_given_real']:.1f}% rec3={100*x['recovery_given_real']:.1f}% retain={100*x['retention_given_touchdown']:.1f}%" for x in seed_rows),flush=True)
    flat=[x for rows in confirms.values() for x in rows]; accepted=all(x["real_rate"]>=0.95 and x["touchdown_given_real"]>=0.80 and x["recovery_given_real"]>=0.60 and x["wrong_rate"]<=0.02 and x["body_rate"]<=0.02 for x in flat); winner_payload={"version":"r8_57_balanced_hold","joint_names":names,"fl_joint_ids":fl,"hold_steps":5,"geometry":geometry.cpu().tolist(),"by_strength":{s:w["controller"] for s,w in winners.items()},"recovery_checkpoint":str(recovery_checkpoint)}; (output_dir/"r857_hold_winner.json").write_text(json.dumps(winner_payload,indent=2)+"\n"); summary={"version":"r8_57_balanced_randomized_hold_scan","training":False,"network_frozen":True,"diagnosis":"R856 fixed env-to-case confounding removed","assignment":"per-repeat shuffled complete case permutation; exact equal episode counts","candidate_modes":["static_latch","dynamic_target"],"candidate_count":len(labels),"episodes_per_case":episodes_per_case,"confirm_episodes_per_seed":confirm_episodes,"confirm_seeds":2,"scans":scans,"winners":winners,"confirm":confirms,"winner_file":str(output_dir/"r857_hold_winner.json"),"accepted":accepted,"verdict":"READY_FOR_VIDEO_REVIEW" if accepted else "KEEP_R857_HOLD_CONTROLLER_NOT_READY"}; (output_dir/"r857_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(f"[R857_RESULT] accepted={accepted} verdict={summary['verdict']}",flush=True); print("R857_BALANCED_SCAN_OK",flush=True)
'''

BRANCH=r'''
    if os.environ.get("R857_BALANCED_SCAN_ONLY", "0") == "1":
        run_r857_balanced_scan(runner=runner,env=env,freeze_dir=Path(os.environ["R857_FREEZE_DIR"]).resolve(),kick_checkpoint=Path(os.environ["R857_KICK"]).resolve(),r851_summary=Path(os.environ["R857_R851_SUMMARY"]).resolve(),body_json=Path(os.environ["R857_BODY_JSON"]).resolve(),recovery_checkpoint=Path(os.environ["R857_RECOVERY"]).resolve(),output_dir=Path(os.environ["R857_OUTPUT"]).resolve(),episodes_per_case=int(os.environ.get("R857_EPISODES_PER_CASE","64")),confirm_episodes=int(os.environ.get("R857_CONFIRM_EPISODES","512")),seed=args_cli.seed)
        return

'''

def once(text,old,new):
    if text.count(old)!=1: raise RuntimeError(f"anchor count {text.count(old)}: {old[:100]!r}")
    return text.replace(old,new,1)
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();src=Path(a.source).resolve();out=Path(a.output).resolve();text=src.read_text()
    text=once(text,'\ndef main() -> None:\n',FUNCTIONS+'\n\ndef main() -> None:\n'); text=once(text,' or os.environ.get("R856_VIDEO_ONLY", "0") == "1":\n        from isaaclab.sensors import ContactSensorCfg\n',' or os.environ.get("R856_VIDEO_ONLY", "0") == "1" or os.environ.get("R857_BALANCED_SCAN_ONLY", "0") == "1":\n        from isaaclab.sensors import ContactSensorCfg\n'); text=once(text,'    if os.environ.get("R856_HOLD_SCAN_ONLY", "0") == "1":\n        run_r856_hold_scan(',BRANCH+'    if os.environ.get("R856_HOLD_SCAN_ONLY", "0") == "1":\n        run_r856_hold_scan(')
    text=once(text,'pre=frozen+(active&bridge&(~rec_control)).to(runner.device).unsqueeze(1)*kd; proposed=torch.where(rec_control.to(runner.device).unsqueeze(1),_r853_recovery_action(runner,components,obs_runner,stable,rd),pre); action=torch.where(holding.to(runner.device).unsqueeze(1),hold_action,proposed)','pre=frozen+(active&bridge&(~rec_control)).to(runner.device).unsqueeze(1)*kd; proposed=torch.where(rec_control.to(runner.device).unsqueeze(1),_r853_recovery_action(runner,components,obs_runner,stable,rd),pre); held=torch.clamp(proposed+bias,-1.0,1.0) if spec.get("mode")=="dynamic_target" else hold_action; action=torch.where(holding.to(runner.device).unsqueeze(1),held,proposed)')
    compile(text,str(out),'exec');out.parent.mkdir(parents=True,exist_ok=True);out.write_text(text);print(f'R857_PATCH_OK source={src} output={out}')
if __name__=='__main__':main()
