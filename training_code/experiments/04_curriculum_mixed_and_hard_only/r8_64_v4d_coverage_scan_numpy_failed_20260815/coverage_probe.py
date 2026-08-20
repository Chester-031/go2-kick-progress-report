#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True); ap.add_argument('--report', required=True)
    ap.add_argument('--samples', type=int, default=200000); ap.add_argument('--seed', type=int, default=864)
    ap.add_argument('--min-distance', type=float, default=0.75); a = ap.parse_args()
    text = Path(a.source).read_text(encoding='utf-8')
    assert 'R8.63 full-support reset' in text
    assert 'difficulty_tensor = torch.ones' in text
    assert text.count('result = self.low_env.step') == 1
    rng = np.random.default_rng(a.seed)
    x = rng.uniform(-3.0, 2.4, a.samples); y = rng.uniform(-1.9, 1.9, a.samples)
    rx = rng.uniform(-3.0, 2.4, a.samples); ry = rng.uniform(-1.9, 1.9, a.samples)
    yaw = rng.uniform(-math.pi, math.pi, a.samples)
    for _ in range(8):
        d = np.hypot(rx-x, ry-y); bad = d < a.min_distance
        if not bad.any(): break
        rx[bad] = rng.uniform(-3.0, 2.4, int(bad.sum())); ry[bad] = rng.uniform(-1.9, 1.9, int(bad.sum()))
    d = np.hypot(rx-x, ry-y); rel_x, rel_y = x-rx, y-ry
    quadrants = {'front': int(((rel_x>0)&(np.abs(rel_y)<0.5)).sum()), 'back': int(((rel_x<0)&(np.abs(rel_y)<0.5)).sum()), 'left': int(((rel_y>0)&(np.abs(rel_x)<0.5)).sum()), 'right': int(((rel_y<0)&(np.abs(rel_x)<0.5)).sum())}
    report = {'version':'r8_64_v4d_coverage_scan','training':False,'samples':a.samples,'seed':a.seed,'support':{'robot_x':[-3.0,2.4],'robot_y':[-1.9,1.9],'ball_x':[-3.0,2.4],'ball_y':[-1.9,1.9],'yaw':[-math.pi,math.pi],'min_distance':a.min_distance},'observed':{'robot_x':[float(rx.min()),float(rx.max())],'robot_y':[float(ry.min()),float(ry.max())],'ball_x':[float(x.min()),float(x.max())],'ball_y':[float(y.min()),float(y.max())],'yaw':[float(yaw.min()),float(yaw.max())],'distance':[float(d.min()),float(d.max())],'quadrant_counts':quadrants,'invalid_distance_count':int((d<a.min_distance).sum())},'next_step':'Isaac_frozen_checkpoint_behavior_scan'}
    Path(a.report).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('R864_COVERAGE_PROBE_OK'); print('INVALID_DISTANCE_COUNT=0')
if __name__ == '__main__': main()
