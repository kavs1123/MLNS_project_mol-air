from typing import Dict, Optional

import torch
import torch.nn.functional as F
import numpy as np

import drl.rl_loss as L
import drl.util.func as util_f
from drl.agent.agent import Agent, agent_config
from drl.agent.config import RecurrentPPORNDConfig
from drl.agent.net import RecurrentPPORNDNetwork
from drl.agent.trajectory import (RecurrentPPORNDExperience,
                                  RecurrentPPORNDTrajectory)
from drl.exp import Experience
from drl.net import Network, Trainer
from drl.util import (IncrementalMean, IncrementalMeanVarianceFromBatch,
                      TruncatedSequenceGenerator)

@agent_config("Recurrent PPO RND")
class RecurrentPPORND(Agent):
    def __init__(
        self, 
        config: RecurrentPPORNDConfig,
        network: RecurrentPPORNDNetwork,
        trainer: Trainer,
        num_envs: int,
        device: Optional[str] = None
    ) -> None:
        super().__init__(num_envs, network, device)
        
        self._config = config
        self._network = network
        self._trainer = trainer
        self._trajectory = RecurrentPPORNDTrajectory(self._config.n_steps)
        
        self._action_log_prob: torch.Tensor = None # type: ignore
        self._epi_state_value: torch.Tensor = None # type: ignore
        self._nonepi_state_value: torch.Tensor = None # type: ignore
        self._prev_discounted_rnd_int_return = 0.0
        self._current_init_norm_steps = 0
        # compute normalization parameters of non-episodic reward of each env along time steps
        self._rnd_int_return_mean_var = IncrementalMeanVarianceFromBatch(dim=1)
        # compute normalization parameters of each feature of next observation along batches
        self._obs_feature_mean_var = IncrementalMeanVarianceFromBatch(dim=0)
        # compute normalization parameters of each feature of next hidden state along batches
        self._hidden_state_feature_mean_var = IncrementalMeanVarianceFromBatch(dim=0)
        hidden_state_shape = (network.hidden_state_shape()[0], self._num_envs, network.hidden_state_shape()[1])
        self._hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._next_hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._prev_terminated = torch.zeros(self._num_envs, 1, device=self.device)
                
        # log data
        self._actor_avg_loss = IncrementalMean()
        self._epi_critic_avg_loss = IncrementalMean()
        self._nonepi_critic_avg_loss = IncrementalMean()
        self._rnd_avg_loss = IncrementalMean()
        self._avg_rnd_int_reward = IncrementalMean()
        self._episodes = np.zeros((self.num_envs, 1), dtype=np.int32)
        self._rnd_int_rewards = []
        self._avg_rnd_int_rewards = [IncrementalMeanVarianceFromBatch() for _ in range(self.num_envs)]
        
        self._time_steps = 0
    
    @property
    def config_dict(self) -> dict:
        return self._config.__dict__
    
    @torch.no_grad()
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        # update hidden state H_t
        self._hidden_state = self._next_hidden_state * (1.0 - self._prev_terminated)
        
        # feed forward
        # when interacting with environment, sequence_length must be 1
        # that is (seq_batch_size, seq_len) = (num_envs, 1)
        policy_dist_seq, epi_state_value_seq, nonepi_state_value_seq, next_hidden_state = self._network.forward_actor_critic(
            obs.unsqueeze(dim=1),
            self._hidden_state
        )
        
        # action sampling
        action_seq = policy_dist_seq.sample()
        
        # (num_envs, 1, *shape) -> (num_envs, *shape)
        action = action_seq.squeeze(dim=1)
        self._action_log_prob = policy_dist_seq.log_prob(action_seq).squeeze_(dim=1)
        self._epi_state_value = epi_state_value_seq.squeeze_(dim=1)
        self._nonepi_state_value = nonepi_state_value_seq.squeeze_(dim=1)
        
        self._next_hidden_state = next_hidden_state
        
        return action
    
    def update(self, exp: Experience) -> Optional[dict]:
        self._time_steps += 1
        self._prev_terminated = exp.terminated
        
        # (D x num_layers, num_envs, H) -> (num_envs, D x num_layers, H)
        next_hidden_state_along_envs = self._next_hidden_state.swapdims(0, 1)
        
        # initialize normalization parameters
        if (self._config.init_norm_steps is not None) and (self._current_init_norm_steps < self._config.init_norm_steps):
            self._current_init_norm_steps += 1
            self._obs_feature_mean_var.update(exp.next_obs)
            self._hidden_state_feature_mean_var.update(next_hidden_state_along_envs)
            return
        
        # compute RND intrinsic reward
        normalized_next_obs = self._normalize_obs(exp.next_obs.to(device=self.device))
        normalized_next_hidden_state = self._normalize_hidden_state(next_hidden_state_along_envs)
        rnd_int_reward = self._compute_rnd_int_reward(normalized_next_obs, normalized_next_hidden_state)
        self._rnd_int_rewards.append(rnd_int_reward.detach().cpu().numpy())
        
        # add an experience
        self._trajectory.add(RecurrentPPORNDExperience(
            **exp.__dict__,
            rnd_int_reward=rnd_int_reward,
            action_log_prob=self._action_log_prob,
            epi_state_value=self._epi_state_value,
            nonepi_state_value=self._nonepi_state_value,
            hidden_state=self._hidden_state,
            next_hidden_state=self._next_hidden_state
        ))
        
        if self._trajectory.reached_n_steps:
            metric_info_dicts = self._train()
            info_dict = {"metric": metric_info_dicts}
            return info_dict
    
    def inference_agent(self, num_envs: int = 1, device: Optional[str] = None) -> Agent:
        return RecurrentPPORNDInference(self._network, num_envs, device or str(self.device))
    
    def _train(self):
        exp_batch = self._trajectory.sample()
        # compute advantage, extrinsic and intrinsic target state value
        advantage, epi_target_state_value, nonepi_target_state_value, metric_info_dicts = self._compute_adv_target(exp_batch)
        
        # batch (batch_size, *shape) to truncated sequence (seq_batch_size, seq_len, *shape)
        seq_generator = TruncatedSequenceGenerator(
            self._config.seq_len,
            self._num_envs,
            self._config.n_steps,
            self._config.padding_value
        )
        
        def add_to_seq_gen(batch, start_idx = 0, seq_len = 0):
            seq_generator.add(util_f.batch_to_perenv(batch, self._num_envs), start_idx, seq_len)
    
        add_to_seq_gen(exp_batch.hidden_state.swapdims(0, 1), seq_len=1)
        add_to_seq_gen(exp_batch.next_hidden_state.swapdims(0, 1))
        add_to_seq_gen(exp_batch.obs)
        add_to_seq_gen(exp_batch.next_obs)
        add_to_seq_gen(exp_batch.action)
        add_to_seq_gen(exp_batch.action_log_prob)
        add_to_seq_gen(advantage)
        add_to_seq_gen(epi_target_state_value)
        add_to_seq_gen(nonepi_target_state_value)
        
        sequences = seq_generator.generate(util_f.batch_to_perenv(exp_batch.terminated, self._num_envs).squeeze_(dim=-1))
        (mask, seq_init_hidden_state, next_hidden_state_seq, obs_seq, next_obs_seq, action_seq, old_action_log_prob_seq, 
         advantage_seq, epi_target_state_value_seq, nonepi_target_state_value_seq) = sequences

        entire_seq_batch_size = len(mask)
        # (entire_seq_batch_size, 1, D x num_layers, H) -> (D x num_layers, entire_seq_batch_size, H)
        seq_init_hidden_state = seq_init_hidden_state.squeeze_(dim=1).swapdims_(0, 1)
        
        # update the normalization parameters of the observation and the hidden state
        # when masked by mask, (entire_seq_batch_size, seq_len,) -> (masked_batch_size,)
        masked_next_obs = next_obs_seq[mask]
        masked_next_hidden_state = next_hidden_state_seq[mask]
        self._obs_feature_mean_var.update(masked_next_obs)
        self._hidden_state_feature_mean_var.update(masked_next_hidden_state)
        
        # normalize the observation and the hidden state
        normalized_next_obs_seq = next_obs_seq
        normalized_hidden_state_seq = next_hidden_state_seq
        normalized_next_obs_seq[mask] = self._normalize_obs(masked_next_obs)
        normalized_hidden_state_seq[mask] = self._normalize_hidden_state(masked_next_hidden_state)
        
        for _ in range(self._config.epoch):
            # if seq_mini_batch_size is None, use the entire sequence batch to executes iteration only once
            # otherwise, use the randomly shuffled mini batch to executes iteration multiple times
            if self._config.seq_mini_batch_size is None:
                shuffled_seq = torch.arange(entire_seq_batch_size)
                seq_mini_batch_size = entire_seq_batch_size
            else:
                shuffled_seq = torch.randperm(entire_seq_batch_size)
                seq_mini_batch_size = self._config.seq_mini_batch_size
                
            for i in range(entire_seq_batch_size // seq_mini_batch_size):
                # when sliced by sample_seq, (entire_seq_batch_size,) -> (seq_mini_batch_size,)
                sample_seq = shuffled_seq[seq_mini_batch_size * i : seq_mini_batch_size * (i + 1)]
                # when masked by sample_mask, (seq_mini_batch_size, seq_len) -> (masked_batch_size,)
                sample_mask = mask[sample_seq]
                
                # feed forward
                sample_policy_dist_seq, sample_epi_state_value_seq, sample_nonepi_state_value_seq, _ = self._network.forward_actor_critic(
                    obs_seq[sample_seq],
                    seq_init_hidden_state[:, sample_seq]
                )
                predicted_feature, target_feature = self._network.forward_rnd(
                    normalized_next_obs_seq[sample_seq][sample_mask],
                    normalized_hidden_state_seq[sample_seq][sample_mask].flatten(1, 2)
                )
                
                # compute actor loss
                sample_new_action_log_prob_seq = sample_policy_dist_seq.log_prob(action_seq[sample_seq])
                actor_loss = L.ppo_clipped_loss(
                    advantage_seq[sample_seq][sample_mask],
                    old_action_log_prob_seq[sample_seq][sample_mask],
                    sample_new_action_log_prob_seq[sample_mask],
                    self._config.epsilon_clip
                )
                
                # compute critic loss
                epi_critic_loss = L.bellman_value_loss(
                    sample_epi_state_value_seq[sample_mask],
                    epi_target_state_value_seq[sample_seq][sample_mask],
                )
                nonepi_critic_loss = L.bellman_value_loss(
                    sample_nonepi_state_value_seq[sample_mask],
                    nonepi_target_state_value_seq[sample_seq][sample_mask],
                )
                critic_loss = epi_critic_loss + nonepi_critic_loss
                
                # compute entropy
                entropy = sample_policy_dist_seq.entropy()[sample_mask].mean()
                
                # compute RND loss
                rnd_loss = L.rnd_loss(
                    predicted_feature,
                    target_feature,
                    self._config.rnd_pred_exp_proportion
                )
                
                # train step
                loss = actor_loss + self._config.critic_loss_coef * critic_loss - self._config.entropy_coef * entropy + rnd_loss
                self._trainer.step(loss, self.training_steps)
                self._tick_training_steps()
                
                # update log data
                self._actor_avg_loss.update(actor_loss.item())
                self._epi_critic_avg_loss.update(epi_critic_loss.item())
                self._nonepi_critic_avg_loss.update(nonepi_critic_loss.item())
                self._rnd_avg_loss.update(rnd_loss.item())
                
        return metric_info_dicts
    
    def _compute_adv_target(self, exp_batch: RecurrentPPORNDExperience):
        """
        Compute advantage `(batch_size, 1)`, episodic and non-episodic target state value `(batch_size, 1)`.
        """
        # (num_envs, *obs_shape)
        final_next_obs = exp_batch.next_obs[-self._num_envs:]
        final_next_hidden_state = self._next_hidden_state
        
        with torch.no_grad():
            # compute final next state value
            _, final_epi_next_state_value_seq, final_nonepi_next_state_value_seq, _ = self._network.forward_actor_critic(
                final_next_obs.unsqueeze(dim=1), # (num_envs, 1, *obs_shape) because sequence length is 1
                final_next_hidden_state
            )
        
        # (num_envs, 1, 1) -> (num_envs, 1)
        final_epi_next_state_value = final_epi_next_state_value_seq.squeeze_(dim=1)
        final_nonepi_next_state_value = final_nonepi_next_state_value_seq.squeeze_(dim=1)
        # (num_envs x (n_steps + 1), 1)
        entire_epi_state_value = torch.cat((exp_batch.epi_state_value, final_epi_next_state_value), dim=0)
        entire_nonepi_state_value = torch.cat((exp_batch.nonepi_state_value, final_nonepi_next_state_value), dim=0)
        
        # (num_envs x T, 1) -> (num_envs, T)
        b2e = lambda x: util_f.batch_to_perenv(x, self._num_envs)
        entire_epi_state_value = b2e(entire_epi_state_value).squeeze_(dim=-1)
        entire_nonepi_state_value = b2e(entire_nonepi_state_value).squeeze_(dim=-1)
        reward = b2e(exp_batch.reward).squeeze_(dim=-1)
        rnd_int_reward = b2e(exp_batch.rnd_int_reward).squeeze_(dim=-1)
        terminated = b2e(exp_batch.terminated).squeeze_(dim=-1)
        
        # compute discounted RND intrinsic returns
        discounted_rnd_int_return = torch.empty_like(rnd_int_reward)
        for t in range(self._config.n_steps):
            self._prev_discounted_rnd_int_return = rnd_int_reward[:, t] + self._config.gamma_n * self._prev_discounted_rnd_int_return
            discounted_rnd_int_return[:, t] = self._prev_discounted_rnd_int_return
        
        # update RND intrinsic reward normalization parameters
        self._rnd_int_return_mean_var.update(discounted_rnd_int_return)
        
        # normalize RND intrinsic rewards
        rnd_int_reward /= torch.sqrt(self._rnd_int_return_mean_var.variance).unsqueeze(dim=-1) + 1e-8
        self._avg_rnd_int_reward.update(rnd_int_reward.mean().item())
        
        metric_info_dicts = []
        for env_id in range(self.num_envs):
            end_idxes = torch.where(terminated[env_id])[0].cpu() + 1
            start = 0
            for end in end_idxes:
                mean, _ = self._avg_rnd_int_rewards[env_id].update(rnd_int_reward[env_id, start:end].cpu())
                metric_info_dicts.append({
                    "episode_metric": {
                        "keys": {
                            "episode": self._episodes[env_id].item(),
                            "env_id": env_id
                        },
                        "values": {
                            "avg_rnd_int_reward": mean.item()
                        }
                    }
                })
                self._episodes[env_id] += 1
                self._avg_rnd_int_rewards[env_id].reset()
                start = end
            if start < len(rnd_int_reward[env_id]):
                self._avg_rnd_int_rewards[env_id].update(rnd_int_reward[env_id, start:].cpu())
        
        # compute advantage (num_envs, n_steps) using GAE
        epi_advantage = L.gae(
            entire_epi_state_value,
            reward,
            terminated,
            self._config.gamma,
            self._config.lam
        )
        nonepi_advantage = L.gae(
            entire_nonepi_state_value,
            rnd_int_reward, # RND intrinsic reward is non-episodic stream
            torch.zeros_like(terminated), # non-episodic
            self._config.gamma_n,
            self._config.lam
        )
        advantage = epi_advantage + self._config.nonepi_adv_coef * nonepi_advantage
        
        # compute target state value (num_envs, n_steps)
        epi_target_state_value = epi_advantage + entire_epi_state_value[:, :-1]
        nonepi_target_state_value = nonepi_advantage + entire_nonepi_state_value[:, :-1]
        
        # (num_envs, n_steps) -> (num_envs x n_steps, 1)
        e2b = lambda x: util_f.perenv_to_batch(x).unsqueeze_(dim=-1)
        advantage = e2b(advantage)
        epi_target_state_value = e2b(epi_target_state_value)
        nonepi_target_state_value = e2b(nonepi_target_state_value)
        
        return advantage, epi_target_state_value, nonepi_target_state_value, metric_info_dicts
    
    def _compute_rnd_int_reward(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Compute intrinsic reward.

        Args:
            obs (Tensor): `(batch_size, *obs_shape)`
            hidden_state (Tensor): `(batch_size, D x num_layers, H)`

        Returns:
            int_reward (Tensor): intrinsic reward `(batch_size, 1)`
        """
        with torch.no_grad():
            predicted_feature, target_feature = self._network.forward_rnd(obs, hidden_state.flatten(1, 2))
            int_reward = 0.5 * ((target_feature - predicted_feature)**2).sum(dim=1, keepdim=True)
            return int_reward
    
    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self._config.init_norm_steps is None:
            return obs
        obs_feature_mean = self._obs_feature_mean_var.mean
        obs_feature_std = torch.sqrt(self._obs_feature_mean_var.variance) + 1e-8
        normalized_obs = (obs - obs_feature_mean) / obs_feature_std
        return normalized_obs.clamp(self._config.obs_norm_clip_range[0], self._config.obs_norm_clip_range[1])

    def _normalize_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self._config.init_norm_steps is None:
            return hidden_state
        hidden_state_feature_mean = self._hidden_state_feature_mean_var.mean
        hidden_state_feature_std = torch.sqrt(self._hidden_state_feature_mean_var.variance) + 1e-8
        normalized_hidden_state = (hidden_state - hidden_state_feature_mean) / hidden_state_feature_std
        return normalized_hidden_state.clamp(self._config.hidden_state_norm_clip_range[0], self._config.hidden_state_norm_clip_range[1])
    
    @property
    def log_data(self) -> Dict[str, tuple]:
        ld = super().log_data
        if self._actor_avg_loss.count > 0:
            ld["Training/Actor Loss"] = (self._actor_avg_loss.mean, self.training_steps)
            ld["Training/Extrinsic Critic Loss"] = (self._epi_critic_avg_loss.mean, self.training_steps)
            ld["Training/Intrinsic Critic Loss"] = (self._nonepi_critic_avg_loss.mean, self.training_steps)
            ld["Training/RND Loss"] = (self._rnd_avg_loss.mean, self.training_steps)
            self._actor_avg_loss.reset()
            self._epi_critic_avg_loss.reset()
            self._nonepi_critic_avg_loss.reset()
            self._rnd_avg_loss.reset()
        return ld

class RecurrentPPORNDInference(Agent):
    def __init__(self, network: RecurrentPPORNDNetwork, num_envs: int, device: Optional[str] = None) -> None:
        super().__init__(num_envs, network, device)
        
        self._network = network
        
        hidden_state_shape = (network.hidden_state_shape()[0], self._num_envs, network.hidden_state_shape()[1])
        self._hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._next_hidden_state = torch.zeros(hidden_state_shape, device=self.device)
        self._prev_terminated = torch.zeros(self._num_envs, 1, device=self.device)
        
    @torch.no_grad()
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        self._hidden_state = self._next_hidden_state * (1.0 - self._prev_terminated)
        policy_dist_seq, _, _, next_hidden_state = self._network.forward_actor_critic(
            obs.unsqueeze(dim=1),
            self._hidden_state
        )
        self._next_hidden_state = next_hidden_state
        return policy_dist_seq.sample().squeeze(dim=1)
    
    def update(self, exp: Experience) -> Optional[dict]:
        self._prev_terminated = exp.terminated

    def inference_agent(self, num_envs: int = 1, device: Optional[str] = None) -> Agent:
        return RecurrentPPORNDInference(self._network, num_envs, device or str(self.device))
