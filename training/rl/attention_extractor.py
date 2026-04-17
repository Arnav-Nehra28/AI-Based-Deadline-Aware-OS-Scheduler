from __future__ import annotations

from typing import Any

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SchedulerAttentionExtractor(BaseFeaturesExtractor):
    """
    Cross-attention feature extractor for scheduler observations.

    - Task encoder: MLP(task_dim -> candidate_hidden_dim)
    - Candidate encoder: shared MLP(candidate_dim -> candidate_hidden_dim)
    - Cross-attention: task query attends over candidate embeddings
    - Fleet encoder: MLP(fleet_dim -> fleet_hidden_dim)
    - Output: MLP(attended_task + fleet_emb -> features_dim)
    """

    def __init__(
        self,
        observation_space: Any,
        features_dim: int = 256,
        task_hidden_dim: int = 128,
        candidate_hidden_dim: int = 128,
        fleet_hidden_dim: int = 64,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__(observation_space, features_dim)

        task_dim = int(observation_space.spaces["task_features"].shape[0])
        candidate_shape = observation_space.spaces["candidate_features"].shape
        fleet_dim = int(observation_space.spaces["fleet_summary"].shape[0])
        if len(candidate_shape) != 2:
            raise ValueError(
                "candidate_features must be rank-2 [num_candidates, candidate_dim] "
                f"but got shape={candidate_shape}."
            )

        candidate_dim = int(candidate_shape[1])
        if candidate_hidden_dim % max(1, attention_heads) != 0:
            raise ValueError(
                "candidate_hidden_dim must be divisible by attention_heads. "
                f"Received {candidate_hidden_dim} and {attention_heads}."
            )

        self.task_encoder = nn.Sequential(
            nn.Linear(task_dim, task_hidden_dim),
            nn.GELU(),
            nn.Linear(task_hidden_dim, candidate_hidden_dim),
            nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, candidate_hidden_dim),
            nn.GELU(),
            nn.Linear(candidate_hidden_dim, candidate_hidden_dim),
            nn.GELU(),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=candidate_hidden_dim,
            num_heads=attention_heads,
            dropout=float(attention_dropout),
            batch_first=True,
        )
        self.fleet_encoder = nn.Sequential(
            nn.Linear(fleet_dim, fleet_hidden_dim),
            nn.GELU(),
            nn.Linear(fleet_hidden_dim, fleet_hidden_dim),
            nn.GELU(),
        )
        self.output_head = nn.Sequential(
            nn.Linear(candidate_hidden_dim + fleet_hidden_dim, features_dim),
            nn.GELU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        task_tensor = observations["task_features"].float()
        candidate_tensor = observations["candidate_features"].float()
        fleet_tensor = observations["fleet_summary"].float()

        task_embedding = self.task_encoder(task_tensor).unsqueeze(1)  # [B, 1, H]
        candidate_embedding = self.candidate_encoder(candidate_tensor)  # [B, K, H]

        # Candidate row 0 feature is set to 1.0 only for real candidate slots.
        candidate_presence = candidate_tensor[..., 0] > 0.5  # [B, K]
        key_padding_mask = ~candidate_presence
        all_masked = torch.all(key_padding_mask, dim=1)
        if torch.any(all_masked):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        attended, _ = self.cross_attention(
            query=task_embedding,
            key=candidate_embedding,
            value=candidate_embedding,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attended_task = attended.squeeze(1)
        fleet_embedding = self.fleet_encoder(fleet_tensor)
        joined = torch.cat([attended_task, fleet_embedding], dim=1)
        return self.output_head(joined)
