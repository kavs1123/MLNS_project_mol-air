from typing import Optional

import torch
from torch.distributions import Categorical
from torch.distributions import Independent,Normal

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
        self._dist = Categorical(probs=probs, logits=logits)

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
    
#! Changes for SAC
class CategoricalSACDist:
    def __init__(
        self, 
        probs: Optional[torch.Tensor] = None, 
        logits: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> None:
        self._dist = Categorical(probs=probs, logits=logits)
        self.temperature = temperature
        self.logits = logits if logits is not None else torch.log(probs + 1e-8)
        
    def sample(self) -> torch.Tensor:
        """Sample an action (*batch_shape, 1) from the policy distribution."""
        return self._dist.sample().unsqueeze_(dim=-1)
    
    def rsample(self, temperature=None) -> torch.Tensor:
        """Reparameterized sample using Gumbel-Softmax trick."""
        if temperature is None:
            temperature = self.temperature
            
        # Sample from Gumbel distribution
        u = torch.rand_like(self.logits)
        gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
        
        # Gumbel-softmax sample
        y = self.logits + gumbel_noise
        y = torch.softmax(y / temperature, dim=-1)
        
        # Get argmax action for discrete output (straight-through estimator)
        _, indices = y.max(dim=-1)
        return indices.unsqueeze(-1)
    
    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        """Returns log probability with any SAC-specific adjustments."""
        action = action.squeeze(dim=-1)
        log_probs = self._dist.log_prob(action).unsqueeze_(dim=-1)
        return log_probs
    
    def entropy(self) -> torch.Tensor:
        """Returns the entropy of this distribution."""
        return self._dist.entropy().unsqueeze_(dim=-1)
