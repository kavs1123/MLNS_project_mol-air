from typing import Dict, Optional

import torch

import drl.rl_loss as L
import drl.util.func as util_f
from drl.agent.agent import Agent, agent_config
from drl.agent.config import RecurrentA3CConfig
from drl.agent.net import RecurrentA3CNetwork
from drl.agent.trajectory import RecurrentA3CExperience, RecurrentA3CTrajectory
from drl.exp import Experience
from drl.net import Trainer
from drl.util import IncrementalMean


@agent_config(name="Recurrent A3C")
class RecurrentA3C(Agent):
    def __init__(
        self, 
        config: RecurrentA3CConfig,
        network: RecurrentA3CNetwork,
        trainer: Trainer,
        num_envs: int,
        device: Optional[str] = None
    ) -> None:
        super().__init__(num_envs, network, device)
        
        self._config = config
        self._network = network
        self._trainer = trainer
        self._trajectory = RecurrentA3CTrajectory(self._config.n_steps)
        self._state_value: torch.Tensor = None # type: ignore    
        
        hidden_state_shape = (network.hidden_state_shape()[0], self._num_envs, network.hidden_state_shape()[1])
        self._hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._next_hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._prev_terminated = torch.zeros(self._num_envs, 1, device=self.device)
        
        # log data
        self._actor_average_loss = IncrementalMean()
        self._critic_average_loss = IncrementalMean()
    
    @property
    def config_dict(self) -> dict:
        return self._config.__dict__
        
    @torch.no_grad()
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        # update hidden state H_t
        self._hidden_state = self._next_hidden_state * (1.0 - self._prev_terminated)
        
        # feed forward
        policy_dist_seq, state_value_seq, next_hidden_state = self._network.forward(
            obs.unsqueeze(dim=1),
            self._hidden_state
        )
        
        # action sampling
        action_seq = policy_dist_seq.sample()
        action = action_seq.squeeze(dim=1)
        self._state_value = state_value_seq.squeeze_(dim=1)

        
        self._next_hidden_state = next_hidden_state
        
        return action
    
    def update(self, exp: Experience) -> Optional[dict]:
        self._prev_terminated = exp.terminated
        
        self._trajectory.add(RecurrentA3CExperience(
            **exp.__dict__,
            state_value=self._state_value,
            hidden_state=self._hidden_state,
            next_hidden_state=self._next_hidden_state,
        ))
        
        if self._trajectory.reached_n_steps:
            self._train()
            
    def inference_agent(self, num_envs: int = 1, device: Optional[str] = None) -> Agent:
        return RecurrentA3CInference(self._network, num_envs, device or str(self.device))
    
    def _train(self):
        exp_batch = self._trajectory.sample()
        advantage,target_state_value = self._compute_adv_target(exp_batch)
        
        # feed forward
        policy_dist_seq, state_value_seq, _ = self._network.forward(
            exp_batch.obs.unsqueeze(dim=1),
            exp_batch.hidden_state
        )
        
        # compute actor loss
        action_log_prob = policy_dist_seq.log_prob(exp_batch.action)
        actor_loss = L.a3c_loss(action_log_prob, advantage)
        
        # compute critic loss
        critic_loss = L.bellman_value_loss(state_value_seq.squeeze(dim=1), target_state_value)
        
        # compute entropy
        entropy = policy_dist_seq.entropy().mean()
        
        # train step
        loss = actor_loss + self._config.critic_loss_coef * critic_loss - self._config.entropy_coef * entropy
        self._trainer.step(loss, self.training_steps)
        self._tick_training_steps()
        
        # update log data
        self._actor_average_loss.update(actor_loss.item())
        self._critic_average_loss.update(critic_loss.item())
    
    def _compute_adv_target(self, exp_batch: RecurrentA3CExperience):
        """
        Compute advantage (batch_size, 1) and target state value (batch_size, 1).
        """
        
        # (num_envs, *obs_shape)
        final_next_obs = exp_batch.next_obs[-self._num_envs:]
        final_next_hidden_state = self._next_hidden_state
        
        with torch.no_grad():
            # compute final next state value
            _, final_next_state_value_seq, _ = self._network.forward(
                final_next_obs.unsqueeze(dim=1), # (num_envs, 1, *obs_shape) because sequence length is 1
                final_next_hidden_state
            )
        
        # (num_envs, 1, 1) -> (num_envs, 1)
        final_next_state_value = final_next_state_value_seq.squeeze_(dim=1)
        # (num_envs x (n_steps + 1), 1)
        entire_state_value = torch.cat((exp_batch.state_value, final_next_state_value), dim=0)
        
        # (num_envs x T, 1) -> (num_envs, T)
        b2e = lambda x: util_f.batch_to_perenv(x, self._num_envs)
        entire_state_value = b2e(entire_state_value).squeeze_(dim=-1)
        reward = b2e(exp_batch.reward).squeeze_(dim=-1)
        terminated = b2e(exp_batch.terminated).squeeze_(dim=-1)
        
        # compute advantage (num_envs, n_steps) using GAE
        advantage = L.gae(
            entire_state_value,
            reward,
            terminated,
            self._config.gamma,
            self._config.lam
        )
        
        # compute target state value (num_envs, n_steps)
        target_state_value = advantage + entire_state_value[:, :-1]
        
        # (num_envs, n_steps) -> (num_envs x n_steps, 1)
        e2b = lambda x: util_f.perenv_to_batch(x).unsqueeze_(dim=-1)
        advantage = e2b(advantage)
        target_state_value = e2b(target_state_value)
        
        return advantage, target_state_value
    
    @property
    def log_data(self) -> Dict[str, tuple]:
        ld = super().log_data
        if self._actor_average_loss.count > 0:
            ld["Training/Actor Loss"] = (self._actor_average_loss.mean, self.training_steps)
            ld["Training/Critic Loss"] = (self._critic_average_loss.mean, self.training_steps)
            self._actor_average_loss.reset()
            self._critic_average_loss.reset()
        return ld


@agent_config(name="Recurrent A3C Inference")
class RecurrentA3CInference(Agent):
    def __init__(self, network: RecurrentA3CNetwork, num_envs: int, device: Optional[str] = None) -> None:
        super().__init__(num_envs, network, device)
        
        self._network = network
        
        hidden_state_shape = (network.hidden_state_shape()[0], self._num_envs, network.hidden_state_shape()[1])
        self._hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._next_hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._prev_terminated = torch.zeros(self._num_envs, 1, device=self.device)
        
    @torch.no_grad()
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        self._hidden_state = self._next_hidden_state * (1.0 - self._prev_terminated)
        policy_dist_seq, _, next_hidden_state = self._network.forward(
            obs.unsqueeze(dim=1),
            self._hidden_state
        )
        self._next_hidden_state = next_hidden_state
        return policy_dist_seq.sample().squeeze(dim=1)
    
    def update(self, exp: Experience) -> Optional[dict]:
        self._prev_terminated = exp.terminated

    def inference_agent(self, num_envs: int = 1, device: Optional[str] = None) -> Agent:
        return RecurrentA3CInference(self._network, num_envs, device or str(self.device))