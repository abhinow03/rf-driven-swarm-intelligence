"""
Graph construction: drone positions at one timestep -> a PyG graph.

Migrated verbatim from the notebooks. The only change is that
``edge_threshold`` is now a required argument instead of reading a module-level
global ``CFG`` (removes hidden global state).
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data


def build_graph(positions, edge_threshold: float) -> Data:
    """Build a PyG ``Data`` object from (6, 3) normalised positions.

    Returns a graph with:
        data.x          node features  (6, 3)
        data.edge_index COO edge list  (2, E)
        data.edge_attr  edge features  (E, 4) = (dx, dy, dz, dist)

    Edges connect drone pairs closer than ``edge_threshold`` (undirected, so
    both i->j and j->i are added). If no pair is close enough, self-loops are
    added so GATv2 does not crash on isolated nodes.
    """
    if isinstance(positions, np.ndarray):
        positions = torch.tensor(positions, dtype=torch.float32)

    n = positions.shape[0]
    src_list, dst_list, attr_list = [], [], []

    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[j] - positions[i]
            dist = torch.norm(diff).item()
            if dist < edge_threshold:
                src_list.append(i)
                dst_list.append(j)
                attr_list.append(torch.cat([diff, torch.tensor([dist])]))
                src_list.append(j)
                dst_list.append(i)
                attr_list.append(torch.cat([-diff, torch.tensor([dist])]))

    if len(src_list) == 0:
        src_list = list(range(n))
        dst_list = list(range(n))
        attr_list = [torch.zeros(4) for _ in range(n)]

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.stack(attr_list, dim=0)
    return Data(x=positions, edge_index=edge_index, edge_attr=edge_attr)


def sequence_to_graphs(seq, edge_threshold: float):
    """Convert one (T, 6, 3) sequence into a list of T PyG graphs."""
    return [build_graph(seq[t], edge_threshold) for t in range(seq.shape[0])]
