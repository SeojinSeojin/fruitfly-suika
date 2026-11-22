"""Import a reproducible directed 512-node FlyWire subgraph."""
import argparse
import hashlib
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

NODES = 512
INPUTS = 128
STEPS = 6


def distances(edges):
    outgoing = [[] for _ in range(NODES)]
    for a, b, _, _ in edges:
        outgoing[a].append(b)
    depth = [-1] * NODES
    queue = deque(range(INPUTS))
    for a in queue:
        depth[a] = 0
    while queue:
        a = queue.popleft()
        for b in outgoing[a]:
            if depth[b] == -1:
                depth[b] = depth[a] + 1
                queue.append(b)
    return depth


def rewire(edges):
    rng = np.random.default_rng(6319)
    original = {(a, b) for a, b, _, _ in edges}
    for _ in range(20):
        swapped = [list(edge) for edge in edges]
        pairs = set(original)
        for i, j in rng.integers(len(edges), size=(len(edges) * 20, 2)):
            a, b, _, _ = swapped[i]
            c, d, _, _ = swapped[j]
            if a == c or b == d or a == d or c == b or (a, d) in pairs or (c, b) in pairs:
                continue
            pairs.difference_update(((a, b), (c, d)))
            pairs.update(((a, d), (c, b)))
            swapped[i][1], swapped[j][1] = d, b
        depth = distances(swapped)
        if min(depth) >= 0 and max(depth) <= STEPS:
            for axis in (0, 1):
                assert np.array_equal(np.bincount([e[axis] for e in edges], minlength=NODES),
                                      np.bincount([e[axis] for e in swapped], minlength=NODES))
            return swapped, 1 - len(pairs & original) / len(edges), depth
    raise ValueError('Could not find a degree-preserving reachable control')


def import_graph(connections, coordinates):
    import pandas as pd

    table = pd.read_csv(connections, usecols=['pre_root_id', 'post_root_id', 'neuropil', 'syn_count'])
    table = table[table.pre_root_id != table.post_root_id]
    pairs = table.groupby(['pre_root_id', 'post_root_id'], sort=True).syn_count.sum()
    outgoing = {}
    for (source, target), count in pairs.items():
        outgoing.setdefault(source, []).append((target, count))
    anchor = min(outgoing, key=lambda node: (-sum(w for _, w in outgoing[node]), node))
    ids, visited = [anchor], {anchor}
    for source in ids:
        for target, _ in sorted(outgoing.get(source, []), key=lambda edge: (-edge[1], edge[0])):
            if target not in visited:
                ids.append(target)
                visited.add(target)
            if len(ids) == NODES:
                break
        if len(ids) == NODES:
            break
    if len(ids) != NODES:
        raise ValueError('Selection did not reach 512 neurons')
    index = {node: i for i, node in enumerate(ids)}
    subset = table[table.pre_root_id.isin(ids) & table.post_root_id.isin(ids)]
    edges = []
    for (a, b), rows in subset.groupby(['pre_root_id', 'post_root_id'], sort=True):
        regions = rows.dropna(subset=['neuropil']).groupby('neuropil').syn_count.sum()
        edges.append([index[a], index[b], int(rows.syn_count.sum()),
                      {k: int(v) for k, v in regions.items()}])
    positions = {}
    if coordinates:
        coords = pd.read_csv(coordinates)
        for row in coords[coords.root_id.isin(ids)].itertuples():
            positions.setdefault(row.root_id, list(map(int, row.position.strip('[]').split())))
    return [str(node) for node in ids], edges, [positions.get(node) for node in ids]


def fixture():
    rng = np.random.default_rng(901)
    pairs = {(a, (a + offset) % NODES) for a in range(NODES) for offset in (1, 31, 79)}
    edges = [[a, b, int(rng.integers(5, 45)), {}] for a, b in sorted(pairs)]
    return [f'fixture-{i}' for i in range(NODES)], edges, [None] * NODES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--connections', type=Path)
    parser.add_argument('--coordinates', type=Path)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.synthetic == bool(args.connections):
        parser.error('Provide exactly one of --connections and --synthetic')
    started = time.perf_counter()
    ids, edges, positions = fixture() if args.synthetic else import_graph(args.connections, args.coordinates)
    depth = distances(edges)
    assert min(depth) >= 0 and max(depth) <= STEPS
    swapped, changed, control_depth = rewire(edges)
    sources = {}
    for path in (args.connections, args.coordinates):
        if path:
            with path.open('rb') as source:
                sources[path.name] = hashlib.file_digest(source, 'sha256').hexdigest()
    graph = dict(release='synthetic fixture' if args.synthetic else 'FlyWire FAFB v783',
                 source_sha256=sources, ids=ids, edges=edges, rewired=swapped,
                 positions=positions, input_nodes=list(range(INPUTS)), readout_nodes=list(range(NODES)),
                 recurrent_steps=STEPS, depth=depth, control_depth=control_depth, fraction_changed=changed,
                 annotation_coverage=sum(sum(e[3].values()) for e in edges) / sum(e[2] for e in edges))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(graph, separators=(',', ':')) + '\n')
    print(f'{len(edges)} edges, {changed:.2%} changed, {max(depth)}/{max(control_depth)} hops, '
          f'{sum(p is not None for p in positions)} coordinates, {time.perf_counter()-started:.1f}s')


if __name__ == '__main__':
    main()
