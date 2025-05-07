from typing import Optional, Tuple

import torch
import torch.nn as nn

from drl.agent.agent import Agent, agent_config
from drl.agent.net import PretrainedRecurrentNetwork
from drl.exp import Experience
from drl.policy_dist import CategoricalDist

@agent_config(name="Pretrained Agent")
class PretrainedRecurrentAgent(Agent):
    def __init__(
        self,
        network: PretrainedRecurrentNetwork,
        num_envs: int = 1,
        device: Optional[str] = None,
        temperature: float = 1.0,
        **kwargs
    ):
        super().__init__(num_envs, network, device)
        
        self._network = network
        self._temperature = temperature
        
        hidden_state_shape = (network.hidden_state_shape()[0], self._num_envs, network.hidden_state_shape()[1])
        self._hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._next_hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._prev_terminated = torch.zeros(self._num_envs, 1, device=self.device)
        
    @torch.no_grad()
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        self._hidden_state = self._next_hidden_state * (1.0 - self._prev_terminated)
        policy_dist_seq, next_hidden_state = self._network.forward(
            obs.unsqueeze(dim=1),
            self._hidden_state
        )
        
        # Apply temperature if needed
        if self._temperature != 1.0:
            policy_dist_seq = policy_dist_seq.apply_temperature(self._temperature)
            
        self._next_hidden_state = next_hidden_state
        return policy_dist_seq.sample().squeeze(dim=1)
    
    def update(self, exp: Experience) -> Optional[dict]:
        self._prev_terminated = exp.terminated
        return None

    def inference_agent(self, num_envs: int = 1, device: Optional[str] = None) -> Agent:
        return PretrainedRecurrentAgent(
            self._network, 
            num_envs, 
            device or str(self.device),
            temperature=self._temperature
        )
    
    @property
    def state_dict(self) -> dict:
        """Returns the state dict of the agent."""
        return {
            "net": self._network.model().state_dict()
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Loads the state dict into the agent."""
        self._network.model().load_state_dict(state_dict["net"])
