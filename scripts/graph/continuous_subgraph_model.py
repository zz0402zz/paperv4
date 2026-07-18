#!/usr/bin/env python3
"""Shared current-anchored change GRU with a zero-initialized graph correction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ForecastOutput:
    level: torch.Tensor
    delta: torch.Tensor
    self_delta: torch.Tensor
    graph_delta: torch.Tensor


class ContinuousSubgraphForecaster(nn.Module):
    """Predict direct multi-horizon changes and optionally add upstream messages."""

    def __init__(
        self,
        num_nodes: int,
        input_features: int = 9,
        target_features: int = 5,
        output_steps: int = 9,
        hidden_size: int = 32,
        station_embedding_size: int = 8,
        current_state_size: int = 16,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.input_features = int(input_features)
        self.target_features = int(target_features)
        self.output_steps = int(output_steps)
        self.hidden_size = int(hidden_size)

        self.history_encoder = nn.GRU(input_features, hidden_size, batch_first=True)
        self.station_embedding = nn.Embedding(num_nodes, station_embedding_size)
        self.current_encoder = nn.Sequential(
            nn.Linear(target_features, current_state_size),
            nn.ReLU(),
        )
        self.self_head = nn.Linear(
            hidden_size + station_embedding_size + current_state_size,
            output_steps * target_features,
        )

        self.graph_encoder = nn.GRU(input_features, hidden_size, batch_first=True)
        self.edge_projection = nn.Linear(hidden_size, output_steps * target_features, bias=False)
        self.graph_gate_logits = nn.Parameter(
            torch.zeros(num_nodes, output_steps, target_features)
        )

    def _encode_history(self, history_diffs: torch.Tensor, encoder: nn.GRU) -> torch.Tensor:
        if history_diffs.ndim != 4:
            raise ValueError("history_diffs must have shape [batch, time, node, feature]")
        batch, steps, nodes, features = history_diffs.shape
        if nodes != self.num_nodes or features != self.input_features:
            raise ValueError(
                f"Expected nodes/features {self.num_nodes}/{self.input_features}, "
                f"received {nodes}/{features}"
            )
        flattened = history_diffs.permute(0, 2, 1, 3).reshape(
            batch * nodes, steps, features
        )
        _, hidden = encoder(flattened)
        return hidden[-1].reshape(batch, nodes, self.hidden_size)

    def self_delta(
        self,
        history_diffs: torch.Tensor,
        current_targets: torch.Tensor,
    ) -> torch.Tensor:
        history_state = self._encode_history(history_diffs, self.history_encoder)
        batch, nodes, _ = history_state.shape
        if current_targets.shape != (batch, nodes, self.target_features):
            raise ValueError(
                "current_targets must have shape "
                f"[{batch}, {nodes}, {self.target_features}]"
            )
        current_state = self.current_encoder(current_targets)
        station_ids = torch.arange(nodes, device=history_diffs.device)
        station_state = self.station_embedding(station_ids)[None, :, :].expand(
            batch, -1, -1
        )
        fused = torch.cat((history_state, current_state, station_state), dim=-1)
        delta = self.self_head(fused).reshape(
            batch, nodes, self.output_steps, self.target_features
        )
        return delta.permute(0, 2, 1, 3)

    def graph_correction(
        self,
        aligned_upstream_diffs: torch.Tensor,
        edge_index: torch.Tensor,
        edge_flow_strength: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map observed aligned upstream changes only to declared edge targets."""
        state = self._encode_history(aligned_upstream_diffs, self.graph_encoder)
        batch = state.shape[0]
        if edge_index.ndim != 2 or edge_index.shape[1] != 2:
            raise ValueError("edge_index must have shape [edge, 2]")
        source = edge_index[:, 0].long()
        target = edge_index[:, 1].long()
        edge_count = edge_index.shape[0]
        messages = self.edge_projection(state[:, source, :]).reshape(
            batch, edge_count, self.output_steps, self.target_features
        )

        if edge_flow_strength is None:
            flow = torch.ones(
                batch,
                edge_count,
                device=messages.device,
                dtype=messages.dtype,
            )
        else:
            flow = edge_flow_strength.to(device=messages.device, dtype=messages.dtype)
            if flow.ndim == 1:
                flow = flow[None, :].expand(batch, -1)
            if flow.shape != (batch, edge_count):
                raise ValueError(
                    f"edge_flow_strength must have shape [{batch}, {edge_count}]"
                )
        messages = messages * flow[:, :, None, None]
        aggregated = torch.zeros(
            batch,
            self.num_nodes,
            self.output_steps,
            self.target_features,
            device=messages.device,
            dtype=messages.dtype,
        )
        aggregated.index_add_(1, target, messages)
        gate = torch.tanh(self.graph_gate_logits)[None, :, :, :]
        gated = aggregated * gate
        return gated.permute(0, 2, 1, 3)

    def forward(
        self,
        history_diffs: torch.Tensor,
        current_targets: torch.Tensor,
        edge_index: torch.Tensor,
        edge_flow_strength: torch.Tensor | None = None,
        aligned_upstream_diffs: torch.Tensor | None = None,
        graph_enabled: bool = True,
    ) -> ForecastOutput:
        self_change = self.self_delta(history_diffs, current_targets)
        if graph_enabled:
            graph_change = self.graph_correction(
                history_diffs if aligned_upstream_diffs is None else aligned_upstream_diffs,
                edge_index,
                edge_flow_strength,
            )
        else:
            graph_change = torch.zeros_like(self_change)
        delta = self_change + graph_change
        level = current_targets[:, None, :, :] + delta
        return ForecastOutput(
            level=level,
            delta=delta,
            self_delta=self_change,
            graph_delta=graph_change,
        )
