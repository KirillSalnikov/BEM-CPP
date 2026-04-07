#!/usr/bin/env python3
"""
Cluster BEM-CUDA: distribute orientation averaging across GPU nodes.

Usage:
  python3 cluster_bem.py [bem_cuda args...] --nodes host1 host2 ...

Each node runs bem_cuda with --orient-range for its chunk.
Results are merged by summing partial Mueller matrices.

Requirements:
  - bem_cuda binary at ~/BEM-CUDA-merged/bin/bem_cuda on each node
  - SSH key auth to all nodes
  - Same source code version on all nodes
"""

import subprocess, json, sys, os, time, tempfile, argparse, math

def parse_args():
    # Split into our args and bem_cuda args
    argv = sys.argv[1:]

    # Find --nodes flag
    nodes = []
    bem_args = []
    i = 0
    while i < len(argv):
        if argv[i] == '--nodes':
            i += 1
            while i < len(argv) and not argv[i].startswith('--'):
                nodes.append(argv[i])
                i += 1
        else:
            bem_args.append(argv[i])
            i += 1

    if not nodes:
        print("Error: --nodes required (e.g. --nodes 172.16.1.86 172.16.1.104)")
        sys.exit(1)

    return bem_args, nodes


def get_orient_count(bem_args):
    """Compute total orientations from CLI args."""
    n_alpha, n_beta, n_gamma = 8, 8, 1
    beta_sym, gamma_sym = 1, 1
    gamma_mirror = False

    i = 0
    while i < len(bem_args):
        if bem_args[i] == '--orient' and i+3 < len(bem_args):
            n_alpha = int(bem_args[i+1])
            n_beta = int(bem_args[i+2])
            n_gamma = int(bem_args[i+3])
            i += 4
        elif bem_args[i] == '--orient-sym' and i+2 < len(bem_args):
            beta_sym = int(bem_args[i+1])
            gamma_sym = int(bem_args[i+2])
            i += 3
        elif bem_args[i] == '--gamma-mirror':
            gamma_mirror = True
            i += 1
        else:
            i += 1

    # Replicate generate_orientations logic
    if beta_sym == 2:
        nb = n_beta  # GL(2*n_beta) keeps n_beta positive nodes
    else:
        nb = n_beta

    return n_alpha * nb * n_gamma


def split_ranges(n_total, n_nodes):
    """Split n_total orientations across n_nodes."""
    chunk = n_total // n_nodes
    remainder = n_total % n_nodes
    ranges = []
    start = 0
    for i in range(n_nodes):
        end = start + chunk + (1 if i < remainder else 0)
        ranges.append((start, end))
        start = end
    return ranges


def run_on_node(node, bem_args, i0, i1, outfile):
    """Launch bem_cuda on a remote node via SSH."""
    args_str = ' '.join(bem_args)
    cmd = (f"ssh -o ConnectTimeout=15 -o ServerAliveInterval=30 {node} "
           f"'cd ~/BEM-CUDA-merged && bin/bem_cuda {args_str} "
           f"--orient-range {i0} {i1} --out {outfile}'")
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def merge_results(result_files, nodes, output_file):
    """Merge partial Mueller results by summing."""
    merged = None

    for node, rfile in zip(nodes, result_files):
        # Download result
        local = f"/tmp/bem_partial_{node.replace('.','_')}.json"
        subprocess.run(f"scp {node}:~/BEM-CUDA-merged/{rfile} {local}",
                      shell=True, capture_output=True)

        with open(local) as f:
            data = json.load(f)

        if merged is None:
            merged = data
        else:
            # Sum Mueller matrices
            for i in range(4):
                for j in range(4):
                    for t in range(len(merged['mueller'][i][j])):
                        merged['mueller'][i][j][t] += data['mueller'][i][j][t]

            # Sum timings
            for key in ['solve_s', 'farfield_s']:
                merged['timing'][key] = max(merged['timing'][key], data['timing'][key])
            merged['timing']['total_s'] = max(merged['timing']['total_s'], data['timing']['total_s'])

    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)

    return merged


def main():
    bem_args, nodes = parse_args()

    # Remove --out from bem_args if present, we'll set our own
    output_file = "result.json"
    filtered_args = []
    i = 0
    while i < len(bem_args):
        if bem_args[i] == '--out' and i+1 < len(bem_args):
            output_file = bem_args[i+1]
            i += 2
        else:
            filtered_args.append(bem_args[i])
            i += 1
    bem_args = filtered_args

    # Remove --orient-range if user accidentally passed it
    filtered_args = []
    i = 0
    while i < len(bem_args):
        if bem_args[i] == '--orient-range':
            i += 3  # skip --orient-range I0 I1
        else:
            filtered_args.append(bem_args[i])
            i += 1
    bem_args = filtered_args

    n_total = get_orient_count(bem_args)
    n_nodes = len(nodes)
    ranges = split_ranges(n_total, n_nodes)

    print(f"=== Cluster BEM-CUDA ===")
    print(f"  Total orientations: {n_total}")
    print(f"  Nodes: {n_nodes}")
    for i, (node, (i0, i1)) in enumerate(zip(nodes, ranges)):
        print(f"    [{i}] {node}: orient [{i0}, {i1}) = {i1-i0} orients")
    print()

    # Launch on all nodes
    procs = []
    result_files = []
    t0 = time.time()
    for i, (node, (i0, i1)) in enumerate(zip(nodes, ranges)):
        rfile = f"cluster_part_{i}.json"
        result_files.append(rfile)
        print(f"  Launching on {node} [{i0},{i1})...")
        p = run_on_node(node, bem_args, i0, i1, rfile)
        procs.append((node, p))

    # Wait for all
    print(f"\n  Waiting for {n_nodes} nodes...")
    for node, p in procs:
        out, _ = p.communicate()
        rc = p.returncode
        lines = out.decode().strip().split('\n') if out else []
        last_lines = lines[-5:] if len(lines) > 5 else lines
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"\n  [{node}] {status}")
        for line in last_lines:
            print(f"    {line}")

    elapsed = time.time() - t0
    print(f"\n  All nodes done in {elapsed:.1f}s")

    # Merge
    print(f"\n  Merging results...")
    merged = merge_results(result_files, nodes, output_file)

    m11 = merged['mueller'][0][0]
    print(f"  M11[0]={m11[0]:.6f}, M11[90]={m11[90]:.6f}, M11[180]={m11[180]:.6f}")
    print(f"  Output: {output_file}")

    # Cleanup remote files
    for node, rfile in zip(nodes, result_files):
        subprocess.run(f"ssh {node} 'rm -f ~/BEM-CUDA-merged/{rfile}'",
                      shell=True, capture_output=True)

    print(f"\n=== Done ({elapsed:.1f}s wall time) ===")


if __name__ == '__main__':
    main()
