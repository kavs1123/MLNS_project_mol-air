from typing import Optional

import torch
import torch.nn as nn

from drl.policy_dist import CategoricalDist, CategoricalSACDist


class CategoricalPolicy(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_discrete_actions: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype=None,
        temperature: float = 1.0
    ) -> None:
        super().__init__()

        self._layer = nn.Linear(
            in_features,
            num_discrete_actions,
            bias,
            device,
            dtype
        )

        self._temperature = temperature

    def forward(self, x: torch.Tensor) -> CategoricalDist:
        logits = self._layer(x) / self._temperature
        return CategoricalDist(logits=logits)


class SACPolicy(nn.Module):
    """Policy network outputting CategoricalDist for discrete actions."""

    def __init__(self, in_features: int, num_actions: int, bias: bool = True,
                 device: Optional[torch.device] = None,
                 dtype=None):
        super().__init__()
        self.logits_head = nn.Linear(in_features, num_actions, bias=bias, 
                                    device=device, dtype=dtype)
        self.temperature = nn.Parameter(torch.ones(1, device=device, dtype=dtype))

    def forward(self, embed: torch.Tensor):
        logits = self.logits_head(embed)
        return CategoricalSACDist(logits=logits, temperature=self.temperature)
        
