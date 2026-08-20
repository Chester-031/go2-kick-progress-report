#!/usr/bin/env python3
from pathlib import Path
import argparse, json

MARK = 'print("[R868_MARKER] {name}", flush=True)'

def insert_after_line(lines, idx, text):
    indent = lines[idx][:len(lines[idx]) - len(lines[idx].lstrip())]
    lines.insert(idx + 1, indent + text + "\n")

def insert_before_line(lines, idx, text):
    indent = lines[idx][:len(lines[idx]) - len(lines[idx].lstrip())]
    lines.insert(idx, indent + text + "\n")

def call_end(lines, start):
    depth = 0
    seen = False
    for i in range(start, len(lines)):
        line = lines[i]
        for ch in line:
            if ch == '(':
                depth += 1; seen = True
            elif ch == ')':
                depth -= 1
        if seen and depth <= 0:
            return i
    return start

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    text = Path(a.source).read_text(encoding='utf-8')
    lines = text.splitlines(keepends=True)
    build_idx = next((i for i, line in enumerate(lines)
                      if line.startswith('def build_parser')), None)
    if build_idx is None:
        raise RuntimeError('build_parser definition not found')
    lines[build_idx:build_idx] = [
        'import faulthandler as _r868_faulthandler\n',
        '_r868_faulthandler.enable()\n',
        '_r868_faulthandler.dump_traceback_later(45, repeat=True)\n',
        'print("[R868_MARKER] module_loaded", flush=True)\n',
        '\n',
    ]
    inserted = []
    for i, line in enumerate(list(lines)):
        if line.startswith('def main('):
            insert_after_line(lines, i, MARK.format(name='main_enter'))
            inserted.append('main_enter')
            break
    # Recompute line positions after the main marker.
    i = 0
    while i < len(lines):
        line = lines[i]
        if 'env = HighLevelSoccerVecEnv(' in line:
            insert_before_line(lines, i, MARK.format(name='before_env_construct'))
            i += 1
            end = call_end(lines, i)
            insert_after_line(lines, end, MARK.format(name='after_env_construct'))
            inserted += ['before_env_construct', 'after_env_construct']
            break
        i += 1
    # Instrument the evaluation function at all high-value boundaries.
    start = next((i for i,l in enumerate(lines) if l.startswith('def run_v4d_evaluation(')), None)
    if start is not None:
        end = next((i for i in range(start+1, len(lines)) if lines[i].startswith('def main()')), len(lines))
        i = start
        while i < end:
            line = lines[i]
            if line.strip() == 'env.set_teacher_scale(0.0)':
                insert_after_line(lines, i, MARK.format(name='after_teacher_scale'))
                end += 1; i += 1
            elif line.strip() == 'env.reset()':
                insert_before_line(lines, i, MARK.format(name='before_eval_reset'))
                insert_after_line(lines, i+1, MARK.format(name='after_eval_reset'))
                end += 2; i += 2
            elif 'high_adapter = V4DHighLevelEnvAdapter(env)' in line:
                insert_before_line(lines, i, MARK.format(name='before_high_adapter'))
                insert_after_line(lines, i+1, MARK.format(name='after_high_adapter'))
                end += 2; i += 2
            elif 'high_obs = high_adapter.get_observations().to(' in line:
                insert_before_line(lines, i, MARK.format(name='before_high_obs'))
                end += 1; i += 1
            elif 'for high_step in range(steps):' in line:
                insert_after_line(lines, i, MARK.format(name='before_high_loop'))
                end += 1; i += 1
            i += 1
    out = Path(a.out)
    out.write_text(''.join(lines), encoding='utf-8')
    Path(str(out)+'.patch_report.json').write_text(json.dumps({
        'version':'r8_68_v4d_hard_diag', 'training':False,
        'video':False, 'faulthandler_seconds':45,
        'markers':inserted + ['after_teacher_scale','before_eval_reset',
                              'after_eval_reset','before_high_adapter',
                              'after_high_adapter','before_high_obs',
                              'before_high_loop']}, indent=2)+'\n')
    print('R868_HARD_DIAG_PATCH_OK')

if __name__ == '__main__':
    main()
