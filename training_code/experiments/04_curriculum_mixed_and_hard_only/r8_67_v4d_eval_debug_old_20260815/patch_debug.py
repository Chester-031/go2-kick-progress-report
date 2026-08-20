#!/usr/bin/env python3
from pathlib import Path
import argparse, json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    text=Path(a.source).read_text(encoding='utf-8')
    start=text.index('def run_v4d_evaluation(')
    end=text.index('\ndef main()', start)
    body=text[start:end]
    markers=[
        ('    high_obs = high_adapter.get_observations().to(\n        high_runner.device\n    )\n', '    print("[R867_MARKER] after_env_reset", flush=True)\n'),
        ('            for high_step in range(steps):\n', '            for high_step in range(steps):\n                print(f"[R867_MARKER] before_high_inference step={high_step}", flush=True)\n'),
        ('                high_action = policy_module(\n                    high_runner\n                ).act_inference(high_obs)\n', '                high_action = policy_module(\n                    high_runner\n                ).act_inference(high_obs)\n                print(f"[R867_MARKER] after_high_inference step={high_step}", flush=True)\n'),
        ('                    (\n                        low_obs,\n                        _,\n                        _,\n                        _,\n                        _,\n                    ) = env.low_step(\n', '                    print(f"[R867_MARKER] before_low_step step={high_step} low={low_index}", flush=True)\n                    (\n                        low_obs,\n                        _,\n                        _,\n                        _,\n                        _,\n                    ) = env.low_step(\n'),
        ('                    low_obs = low_obs.to(\n                        low_runner.device\n                    )\n', '                    low_obs = low_obs.to(\n                        low_runner.device\n                    )\n                    print(f"[R867_MARKER] after_low_step step={high_step} low={low_index}", flush=True)\n'),
        ('                high_diag = env.end_high_action()\n', '                high_diag = env.end_high_action()\n                print(f"[R867_MARKER] after_end_high_action step={high_step}", flush=True)\n'),
    ]
    for needle,repl in markers:
        pos=body.find(needle)
        if pos < 0: raise RuntimeError('missing marker needle')
        body=body[:pos]+repl+body[pos+len(needle):]
    text=text[:start]+body+text[end:]
    out=Path(a.out); out.write_text(text,encoding='utf-8')
    Path(str(out)+'.patch_report.json').write_text(json.dumps({'version':'r8_67_v4d_eval_debug','training':False,'steps':1,'video':False,'markers':6},indent=2)+'\n')
    print('R867_DEBUG_PATCH_OK')
if __name__=='__main__': main()
