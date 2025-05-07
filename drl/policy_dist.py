from typing import Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class CategoricalDist:
    """
    Categorical policy distribution for the discrete action type.
    
    `*batch_shape` depends on the input of the algorithm you are using.
    
    * simple batch: `*batch_shape` = `(batch_size,)`
    * sequence batch: `*batch_shape` = `(num_seq, seq_len)`
    
    Args:
        probs (Tensor): categorical probabilities `(*batch_shape, num_actions)` which is typically the output of neural network
    """
    def __init__(
        self, 
        probs: Optional[torch.Tensor] = None, 
        logits: Optional[torch.Tensor] = None
    ) -> None:
        # Ensure logits or probs create a valid probability distribution
        if logits is not None:
            # Handle NaN or extreme values in logits
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), 
                               torch.zeros_like(logits), 
                               logits)
            
            # Create categorical distribution from normalized logits
            self._dist = Categorical(logits=logits)
        else:
            # Ensure probs are valid probabilities that sum to 1
            if probs is not None:
                # Normalize probs to ensure they sum to 1
                probs = F.softmax(probs, dim=-1)
                self._dist = Categorical(probs=probs)
            else:
                raise ValueError("Either logits or probs must be provided")

    def sample(self) -> torch.Tensor:
        """Sample an action `(*batch_shape, 1)` from the policy distribution."""
        return self._dist.sample().unsqueeze_(dim=-1)
    
    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """
        Returns the log of the porability mass/density function accroding to the `action`.

        Args:
            action (Tensor): `(*batch_shape, 1)`

        Returns:
            log_prob (Tensor): `(*batch_shape, 1)`
        """
        action = action.squeeze(dim=-1)
        return self._dist.log_prob(action).unsqueeze_(dim=-1)
    
    def entropy(self) -> torch.Tensor:
        """
        Returns the entropy of this distribution `(*batch_shape, num_branches)`. 
        """
        return self._dist.entropy().unsqueeze_(dim=-1)
    
    def apply_temperature(self, temperature):
        """Apply temperature to the logits."""
        if temperature == 1.0:
            return self
        logits = self._dist.logits / temperature
        return CategoricalDist(logits=logits)