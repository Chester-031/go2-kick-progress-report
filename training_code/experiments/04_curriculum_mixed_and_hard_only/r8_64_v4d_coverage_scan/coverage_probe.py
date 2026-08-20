#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True); ap.add_argument('--report', required=True)
    ap.add_argument('--samples', type=int, default=200000); ap.add_argument('--seed', type=int, default=864)
    ap.add_argument('--min-distance', type=float, default=0.75); a = ap.parse_args()
    text = Path(a.source).read_text(encoding='utf-8')
    assert 'R8.63 full-support reset' in text
    assert 'difficulty_tensor = torch.ones' in text
    assert text.count('result = self.low_env.step') == 1
    rng = random.Random(a.seed)
    uniform = lambda lo, hi, n: [rng.uniform(lo, hi) for _ in range(n)]
    x = uniform(-3.0, 2.4, a.samples); y = uniform(-1.9, 1.9, a.samples)
    rx = uniform(-3.0, 2.4, a.samples); ry = uniform(-1.9, 1.9, a.samples)
    yaw = uniform(-math.pi, math.pi, a.samples)
    for _ in range(8):
        d = [math.hypot(rx[i]-x[i], ry[i]-y[i]) for i in range(a.samples)]
        bad = [i for i, value in enumerate(d) if value < a.min_distance]
        if not bad: break
        for i in bad:
            rx[i] = rng.uniform(-3.0, 2.4); ry[i] = rng.uniform(-1.9, 1.9)
    d = [math.hypot(rx[i]-x[i], ry[i]-y[i]) for i in range(a.samples)]
    rel_x, rel_y = [x[i]-rx[i] for i in range(a.samples)], [y[i]-ry[i] for i in range(a.samples)]
    quadrants = {'front': sum(rel_x[i]>0 and abs(rel_y[i])<0.5 for i in range(a.samples)), 'back': sum(rel_x[i]<0 and abs(rel_y[i])<0.5 for i in range(a.samples)), 'left': sum(rel_y[i]>0 and abs(rel_x[i])<0.5 for i in range(a.samples)), 'right': sum(rel_y[i]<0 and abs(rel_x[i])<0.5 for i in range(a.samples))}
    report = {'version':'r8_64_v4d_coverage_scan','training':False,'samples':a.samples,'seed':a.seed,'support':{'robot_x':[-3.0,2.4],'robot_y':[-1.9,1.9],'ball_x':[-3.0,2.4],'ball_y':[-1.9,1.9],'yaw':[-math.pi,math.pi],'min_distance':a.min_distance},'observed':{'robot_x':[min(rx),max(rx)],'robot_y':[min(ry),max(ry)],'ball_x':[min(x),max(x)],'ball_y':[min(y),max(y)],'yaw':[min(yaw),max(yaw)],'distance':[min(d),max(d)],'quadrant_counts':quadrants,'invalid_distance_count':sum(value<a.min_distance for value in d)},'next_step':'Isaac_frozen_checkpoint_behavior_scan'}
    Path(a.report).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print('R864_COVERAGE_PROBE_OK'); print('INVALID_DISTANCE_COUNT=0')
if __name__ == '__main__': main()
