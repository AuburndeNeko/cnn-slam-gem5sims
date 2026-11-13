#!/usr/bin/env python3
"""
Run SCALE-Sim one layer at a time, each in a fresh process, and merge the
per-layer reports into single COMPUTE / BANDWIDTH / DETAILED_ACCESS CSVs.

Why: memory is freed after every layer, and each layer's output folder is
deleted as soon as its reports are merged. The run is resumable: layers that
are already in the merged reports are skipped, so after a crash or Ctrl-C you
simply rerun the same command.

Needs run_scalesim.py in the same folder as this script.
Run from inside the SCALE-Sim folder (its default layout path is relative):

  python3 ../accel/run_layers.py ../accel/os64_2048.cfg \
      ../accel/vits_block_518x1708.csv ../accel/results

Merged reports end up in <results>/<run_name>/, with LayerID = row number in
the topology CSV (0 = first layer), exactly like a normal SCALE-Sim run.

A fifth argument conv selects convolution topologies.
"""
import os
import re
import shutil
import subprocess
import sys
import time

REPORTS = ['COMPUTE_REPORT.csv', 'BANDWIDTH_REPORT.csv', 'DETAILED_ACCESS_REPORT.csv']


def main():
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    cfg_path, topo_path, out_dir = sys.argv[1:4]
    mode = sys.argv[4] if len(sys.argv) == 5 else 'gemm'
    if mode not in ('gemm', 'conv'):
        sys.exit(f'mode must be gemm or conv, got {mode}')
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_scalesim.py')
    if not os.path.exists(runner):
        sys.exit(f'run_scalesim.py not found next to this script ({runner})')

    cfg_text = open(cfg_path).read()
    m = re.search(r'^\s*run_name\s*[=:]\s*(\S+)', cfg_text, re.M)
    if not m:
        sys.exit(f'no run_name line in {cfg_path}')
    run_name = m.group(1)

    lines = [l.rstrip('\n') for l in open(topo_path) if l.strip()]
    header, layers = lines[0], lines[1:]

    final_dir = os.path.join(out_dir, run_name)
    tmp_dir = os.path.join(out_dir, run_name + '_tmp')
    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # Layers already merged (for resuming).
    done = set()
    first_report = os.path.join(final_dir, REPORTS[0])
    if os.path.exists(first_report):
        for row in open(first_report).readlines()[1:]:
            if row.strip():
                done.add(int(row.split(',')[0]))

    for idx, layer in enumerate(layers):
        name = layer.split(',')[0].strip()
        if idx in done:
            print(f'[layers] {idx} {name}: already done, skipping', flush=True)
            continue

        # One-layer config (own run_name) and one-layer topology.
        sub = f'{run_name}_L{idx}'
        cfg_i = os.path.join(tmp_dir, f'L{idx}.cfg')
        topo_i = os.path.join(tmp_dir, f'L{idx}.csv')
        with open(cfg_i, 'w') as f:
            f.write(cfg_text[:m.start(1)] + sub + cfg_text[m.end(1):])
        with open(topo_i, 'w') as f:
            f.write(header + '\n' + layer + '\n')

        print(f'[layers] {idx} {name}: running ...', flush=True)
        t0 = time.time()
        result = subprocess.run([sys.executable, runner, cfg_i, topo_i, tmp_dir, mode])
        if result.returncode != 0:
            sys.exit(f'[layers] {idx} {name} failed (exit {result.returncode}). '
                     f'Fix the cause and rerun the same command to resume.')

        # Merge this layer's reports, renumbering LayerID, then delete its folder.
        src = os.path.join(tmp_dir, sub)
        for rep in REPORTS:
            rows = [l.rstrip('\n') for l in open(os.path.join(src, rep)) if l.strip()]
            dst = os.path.join(final_dir, rep)
            new_file = not os.path.exists(dst)
            with open(dst, 'a') as f:
                if new_file:
                    f.write(rows[0] + '\n')
                for row in rows[1:]:
                    f.write(str(idx) + ',' + row.split(',', 1)[1] + '\n')
        shutil.rmtree(src)
        print(f'[layers] {idx} {name}: done in {time.time() - t0:.0f} s', flush=True)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f'[layers] all layers done; merged reports in {final_dir}', flush=True)


if __name__ == '__main__':
    main()
