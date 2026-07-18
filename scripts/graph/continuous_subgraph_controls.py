#!/usr/bin/env python3
"""Negative graph controls for the continuous-subgraph experiment."""

from __future__ import annotations

import numpy as np


def reverse_edges(edges: np.ndarray) -> np.ndarray:
    edge_array = np.asarray(edges, dtype=int)
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("edges must have shape [edge, 2]")
    return edge_array[:, ::-1].copy()


def degree_matched_random_edges(
    edges: np.ndarray,
    node_count: int,
    seed: int,
    max_attempts: int = 2000,
) -> np.ndarray:
    """Randomize edges with directed double-edge swaps and exact node degrees."""
    edge_array = np.asarray(edges, dtype=int)
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("edges must have shape [edge, 2]")
    if len(edge_array) < 2:
        raise ValueError("At least two edges are required for a degree-matched swap")
    if edge_array.min() < 0 or edge_array.max() >= node_count:
        raise ValueError("edge endpoint is outside node_count")

    rng = np.random.default_rng(seed)
    randomized = edge_array.copy()
    edge_set = {tuple(edge) for edge in randomized.tolist()}
    successful = 0
    target_swaps = max(1, len(edge_array) * 2)
    for _ in range(max_attempts):
        first, second = rng.choice(len(randomized), size=2, replace=False)
        a, b = randomized[first]
        c, d = randomized[second]
        candidate_one = (int(a), int(d))
        candidate_two = (int(c), int(b))
        old_one = tuple(randomized[first])
        old_two = tuple(randomized[second])
        if candidate_one[0] == candidate_one[1] or candidate_two[0] == candidate_two[1]:
            continue
        remaining = edge_set - {old_one, old_two}
        if candidate_one in remaining or candidate_two in remaining or candidate_one == candidate_two:
            continue
        if {candidate_one, candidate_two} == {old_one, old_two}:
            continue
        edge_set = remaining | {candidate_one, candidate_two}
        randomized[first] = candidate_one
        randomized[second] = candidate_two
        successful += 1
        if successful >= target_swaps:
            break
    if successful == 0 or np.array_equal(randomized, edge_array):
        raise ValueError("Could not construct a non-identical degree-matched graph")
    return randomized


def validate_wrong_relation_edges(
    strict_edges: np.ndarray,
    wrong_edges: np.ndarray,
) -> np.ndarray:
    strict = {tuple(edge) for edge in np.asarray(strict_edges, dtype=int).tolist()}
    wrong = np.asarray(wrong_edges, dtype=int)
    overlap = strict & {tuple(edge) for edge in wrong.tolist()}
    if overlap:
        raise ValueError(f"Wrong-relation control contains a strict edge: {sorted(overlap)}")
    if np.any(wrong[:, 0] == wrong[:, 1]):
        raise ValueError("Wrong-relation control cannot contain self loops")
    return wrong.copy()


def block_shuffle_by_split(
    values: np.ndarray,
    split_labels: np.ndarray,
    block_steps: int,
    seed: int,
) -> np.ndarray:
    """Destroy temporal alignment without moving observations across splits."""
    source = np.asarray(values)
    labels = np.asarray(split_labels)
    if len(source) != len(labels):
        raise ValueError("values and split_labels must have the same first dimension")
    if block_steps <= 0:
        raise ValueError("block_steps must be positive")
    output = source.copy()
    rng = np.random.default_rng(seed)
    for split in dict.fromkeys(labels.tolist()):
        target_idx = np.flatnonzero(labels == split)
        blocks = [target_idx[start : start + block_steps] for start in range(0, len(target_idx), block_steps)]
        order = rng.permutation(len(blocks))
        source_idx = np.concatenate([blocks[position] for position in order])
        output[target_idx] = source[source_idx]
    return output
