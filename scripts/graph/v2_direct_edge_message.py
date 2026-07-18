#!/usr/bin/env python3
"""Direct-edge graph message layer for V2 pair experiments."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from scripts.graph import v2_direct_pair_graph_config as cfg

GRAPH_L1 = 1e-4


class EdgeMapper:
    """Namespace placeholder for type checkers; instantiate with make_edge_mapper."""


def make_edge_mapper(torch, upstream_dim: int = 9, hidden_size: int = 24, target_dim: int = 5):
    """Create the horizon-wise upstream-change to downstream-correction mapper."""

    class _EdgeMapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(upstream_dim, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, target_dim),
            )

        def forward(self, aligned_upstream):
            return self.net(aligned_upstream)

    return _EdgeMapper()


def masked_l1(torch, prediction, target, mask):
    """Masked mean absolute error."""
    valid = mask.bool() & torch.isfinite(prediction) & torch.isfinite(target)
    if not bool(valid.any()):
        return torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    return torch.abs(prediction[valid] - target[valid]).mean()


class DirectEdgeMessageModel:
    """Torch-backed namespace placeholder; call constructor after importing torch."""

    def __new__(cls, *args, **kwargs):
        import torch

        return make_direct_edge_message_model(torch, *args, **kwargs)


def make_direct_edge_message_model(
    torch,
    sequence_input_dim: int,
    current_input_dim: int,
    hidden_size: int = 64,
    current_hidden_size: int = 16,
    output_steps: int = cfg.OUTPUT_STEPS,
    target_dim: int = len(cfg.TARGET_FEATURE_COLUMNS),
):
    """Create a self D backbone plus explicit graph correction model."""

    class _DirectEdgeMessageModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output_steps = int(output_steps)
            self.target_dim = int(target_dim)
            self.sequence_encoder = torch.nn.GRU(sequence_input_dim, hidden_size, batch_first=True)
            self.current_encoder = torch.nn.Sequential(
                torch.nn.Linear(current_input_dim, current_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(current_hidden_size, hidden_size),
                torch.nn.ReLU(),
            )
            self.self_head = torch.nn.Linear(hidden_size * 2, output_steps * target_dim)
            self.edge_mapper = make_edge_mapper(torch, upstream_dim=9, hidden_size=24, target_dim=target_dim)

        def forward(self, sequence_x, current_level, aligned_upstream, graph_enabled=True, return_parts=False):
            _, hidden = self.sequence_encoder(sequence_x)
            sequence_state = hidden[-1]
            current_state = self.current_encoder(current_level)
            state = torch.cat([sequence_state, current_state], dim=-1)
            self_delta = self.self_head(state).view(-1, self.output_steps, self.target_dim)
            if graph_enabled:
                graph_delta = self.edge_mapper(aligned_upstream)
            else:
                graph_delta = torch.zeros_like(self_delta)
            final_delta = self_delta + graph_delta
            if return_parts:
                return final_delta, {
                    "self_delta": self_delta,
                    "graph_delta": graph_delta,
                    "final_delta": final_delta,
                }
            return final_delta

    return _DirectEdgeMessageModel()


def graph_loss(torch, final_delta, target_delta, mask, graph_delta, regularization: float = GRAPH_L1):
    """Main masked L1 plus small fixed graph-correction penalty."""
    return masked_l1(torch, final_delta, target_delta, mask) + float(regularization) * torch.abs(graph_delta).mean()


def block_shuffle(source: np.ndarray, block_steps: int = 6, seed: int = cfg.PILOT_SEED) -> np.ndarray:
    """Shuffle complete source windows while preserving values and shape."""
    arr = np.asarray(source)
    rng = np.random.default_rng(seed)
    if arr.shape[0] <= 1:
        return arr.copy()
    block_ids = np.arange(int(np.ceil(arr.shape[0] / block_steps)))
    rng.shuffle(block_ids)
    indices = np.concatenate(
        [
            np.arange(block * block_steps, min((block + 1) * block_steps, arr.shape[0]))
            for block in block_ids
        ]
    )
    return arr[indices[: arr.shape[0]]].copy()


def upstream_component(target: str, strict_edges) -> set[str]:
    """Return the undirected strict component containing target."""
    graph: dict[str, set[str]] = defaultdict(set)
    for source, dest in strict_edges:
        source = str(source)
        dest = str(dest)
        graph[source].add(dest)
        graph[dest].add(source)
    seen = {str(target)}
    queue = deque([str(target)])
    while queue:
        node = queue.popleft()
        for nxt in graph[node]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def choose_wrong_source(target: str, strict_edges, candidates: list[str] | tuple[str, ...]) -> str:
    """Choose the first candidate outside the target's strict connected component."""
    component = upstream_component(target, strict_edges)
    for candidate in candidates:
        if str(candidate) not in component:
            return str(candidate)
    raise ValueError(f"No wrong-source candidate outside component for {target}")
