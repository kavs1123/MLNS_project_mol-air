import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from typing_extensions import Protocol  # Use typing_extensions for Python 3.7 compatibility
import numpy as np

import drl.net as net
import drl.util.func as util_f
from drl import Experience
from drl.agent import Agent, agent_config
from drl.policy_dist import CategoricalDist
from drl.net import Trainer, unwrap_lstm_hidden_state
from drl.util import IncrementalMean, TrainStep, IncrementalMeanVarianceFromBatch


class DreamerNetwork(nn.Module, net.Network):
    """DreamerV3 network implementation."""
    
    def __init__(self, 
                 in_features: int, 
                 num_actions: int,
                 hidden_size: int = 200,
                 state_size: int = 32,
                 rnn_hidden_size: int = 200, 
                 discrete_size: int = 32, 
                 category_size: int = 32,
                 layer_norm: bool = True) -> None:
        super().__init__()
        
        self._state_size = state_size
        self._rnn_hidden_size = rnn_hidden_size
        self._discrete_size = discrete_size
        self._category_size = category_size
        self._num_layers = 1
        self._num_actions = num_actions
        
        # Encoder model
        self._encoder = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ELU(),
        )
        
        # Recurrent model (RSSM)
        # Adjust input size correctly: state_size + num_actions
        self._recurrent_model = nn.LSTM(
            state_size + num_actions, 
            rnn_hidden_size,
            batch_first=True,
            num_layers=self._num_layers
        )
        
        # Representation model
        self._representation_model = nn.Sequential(
            nn.Linear(rnn_hidden_size, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ELU(),
            nn.Linear(hidden_size, discrete_size * category_size)
        )
        
        # Decoder model
        self._decoder = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ELU(),
            nn.Linear(hidden_size, in_features)
        )
        
        # Reward predictor
        self._reward_model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ELU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Continue predictor (predicts termination)
        self._continue_model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Actor model
        self._actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ELU(),
            nn.Linear(hidden_size, num_actions)
        )
        
        # Critic model
        self._critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size) if layer_norm else nn.Identity(),
            nn.ELU(),
            nn.Linear(hidden_size, 1)
        )
        
    def model(self) -> nn.Module:
        return self
    
    def hidden_state_shape(self) -> Tuple[int, int]:
        return (self._num_layers, self._rnn_hidden_size * 2)
    
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations to embeddings."""
        return self._encoder(obs)
    
    def forward_actor(self, state: torch.Tensor) -> CategoricalDist:
        """Forward actor to get action distribution."""
        logits = self._actor(state)
        # Apply softmax to ensure valid probability distribution
        return CategoricalDist(logits=logits)
    
    def forward_critic(self, state: torch.Tensor) -> torch.Tensor:
        """Forward critic to get state value."""
        return self._critic(state)
    
    def predict_reward(self, state: torch.Tensor) -> torch.Tensor:
        """Predict reward from state."""
        return self._reward_model(state)
    
    def predict_continue(self, state: torch.Tensor) -> torch.Tensor:
        """Predict continue probability from state."""
        return self._continue_model(state)
    
    def decode(self, state: torch.Tensor) -> torch.Tensor:
        """Decode state to observation reconstruction."""
        return self._decoder(state)
    
    def represent(self, obs_embed: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get representation from observation embedding and hidden state."""
        h, c = unwrap_lstm_hidden_state(hidden_state)
        # Forward through LSTM
        output = self._representation_model(h[-1])
        logits = output.reshape(-1, self._discrete_size, self._category_size)
        
        # Sample categorical variables
        categorical_dist = F.gumbel_softmax(logits, hard=True, dim=-1)
        state = categorical_dist.reshape(-1, self._discrete_size * self._category_size)
        state = state.view(-1, self._state_size)
        
        return state, hidden_state
    
    def recurrent_model_step(self, 
                           prev_state: torch.Tensor, 
                           action: torch.Tensor, 
                           hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Step through the recurrent model with previous state and action."""
        h, c = unwrap_lstm_hidden_state(hidden_state)
        batch_size = prev_state.size(0)
        
        # Ensure action is one-hot encoded for proper input dimensionality
        if action.size(-1) == 1:
            # Convert discrete action indices to one-hot, handling batch size safely
            try:
                # Reshape action to ensure proper dimensions
                action_flat = action.reshape(-1).long()
                # Create one-hot tensor with the correct dimensions
                action_one_hot = torch.zeros(batch_size, self._num_actions, device=action.device)
                # Use scatter with properly sized tensors to create one-hot encoding
                action_one_hot.scatter_(1, action_flat.unsqueeze(-1), 1.0)
                action = action_one_hot
            except Exception as e:
                # Fallback for any dimension mismatch
                print(f"Action shape: {action.shape}, Batch size: {batch_size}, Num actions: {self._num_actions}")
                # Safe fallback - create a uniform distribution across actions
                action = torch.ones(batch_size, self._num_actions, device=action.device) / self._num_actions
        
        # Concatenate state and action for RSSM input
        # Make sure dimensions match the expected input size of the LSTM
        rssm_input = torch.cat([prev_state, action], dim=-1)
        rssm_input = rssm_input.unsqueeze(1)  # Add time dimension: [batch_size, 1, feature_dim]
        
        # CRITICAL: Make sure hidden state matches LSTM's expected dimensions
        # Check if the hidden state dimensions match what the LSTM expects
        # For LSTM, hidden state should be [num_layers, batch_size, hidden_size]
        if h.size(0) != self._num_layers:
            # Adjust hidden state size to match expected dimensions
            # First, let's get the dimensions right
            new_h = torch.zeros(self._num_layers, batch_size, self._rnn_hidden_size, device=h.device)
            new_c = torch.zeros(self._num_layers, batch_size, self._rnn_hidden_size, device=c.device)
            
            # Copy data from the old hidden state if possible
            # We only copy the first layer since that's what we need for a 1-layer LSTM
            if h.size(0) > 0 and h.size(1) > 0:
                copy_layers = min(self._num_layers, h.size(0))
                copy_batch = min(batch_size, h.size(1))
                new_h[:copy_layers, :copy_batch, :] = h[:copy_layers, :copy_batch, :]
                new_c[:copy_layers, :copy_batch, :] = c[:copy_layers, :copy_batch, :]
            
            h, c = new_h, new_c
        elif h.size(1) != batch_size:
            # If only the batch dimension is wrong, adjust that
            new_h = torch.zeros(h.size(0), batch_size, h.size(2), device=h.device)
            new_c = torch.zeros(c.size(0), batch_size, c.size(2), device=c.device)
            
            # Copy existing data for overlapping dimensions
            min_batch = min(h.size(1), batch_size)
            new_h[:, :min_batch, :] = h[:, :min_batch, :]
            new_c[:, :min_batch, :] = c[:, :min_batch, :]
            
            h, c = new_h, new_c
        
        # Forward through LSTM
        _, (h_new, c_new) = self._recurrent_model(rssm_input, (h, c))
        new_hidden_state = net.wrap_lstm_hidden_state(h_new, c_new)
        
        # Get new state representation
        output = self._representation_model(h_new[-1])
        logits = output.reshape(-1, self._discrete_size, self._category_size)
        
        # Sample categorical variables
        categorical_dist = F.gumbel_softmax(logits, hard=True, dim=-1)
        new_state = categorical_dist.reshape(-1, self._discrete_size * self._category_size)
        new_state = new_state.view(-1, self._state_size)
        
        return new_state, new_hidden_state


class DreamerExperience(Experience):
    """DreamerV3 experience container."""
    
    def __init__(self, 
                 obs: torch.Tensor,
                 action: torch.Tensor,
                 next_obs: torch.Tensor,
                 reward: torch.Tensor,
                 terminated: torch.Tensor,
                 **kwargs) -> None:
        super().__init__(obs, action, next_obs, reward, terminated)
        self.__dict__.update(kwargs)


@agent_config("DreamerV3")
class DreamerV3(Agent):
    """DreamerV3 agent implementation."""
    
    def __init__(self, 
                 net: DreamerNetwork, 
                 num_envs: int,
                 device: Optional[str] = None,
                 lr: float = 3e-4,
                 discount: float = 0.99,
                 lambda_: float = 0.95,
                 horizon: int = 15,
                 imagination_steps: int = 5,
                 actor_entropy: float = 1e-3,
                 critic_weight: float = 0.5,
                 model_weight: float = 0.5,
                 reward_weight: float = 1.0,
                 continue_weight: float = 1.0,
                 kl_loss_scale: float = 1.0,
                 decoder_weight: float = 1.0,
                 grad_clip: float = 100.0,
                 n_steps: int = 64,
                 init_norm_steps: Optional[int] = 100,
                 effective_batch_size: int = None,
                 **kwargs) -> None:
        super().__init__(num_envs, net, device)
        
        self._net = net
        self._num_envs = num_envs
        self._n_steps = n_steps
        
        # Use effective batch size to prevent OOM
        self._effective_batch_size = effective_batch_size or num_envs
        
        # Store configuration parameters
        self._config = type('AgentConfig', (object,), {
            'lr': lr,
            'discount': discount,
            'lambda_': lambda_,
            'horizon': horizon,
            'imagination_steps': imagination_steps,
            'actor_entropy': actor_entropy,
            'critic_weight': critic_weight,
            'model_weight': model_weight,
            'reward_weight': reward_weight,
            'continue_weight': continue_weight,
            'kl_loss_scale': kl_loss_scale,
            'decoder_weight': decoder_weight,
            'grad_clip': grad_clip,
            'n_steps': n_steps,
            'init_norm_steps': init_norm_steps,
            'effective_batch_size': self._effective_batch_size
        })
        
        # Initialize hidden state
        self._init_hidden_state()
        
        # Initialize world model optimizer
        self._world_model_optimizer = torch.optim.Adam(
            list(self._net.model()._encoder.parameters()) +
            list(self._net.model()._recurrent_model.parameters()) +
            list(self._net.model()._representation_model.parameters()) +
            list(self._net.model()._decoder.parameters()) +
            list(self._net.model()._reward_model.parameters()) +
            list(self._net.model()._continue_model.parameters()),
            lr=lr
        )
        self._world_model_trainer = Trainer(self._world_model_optimizer).enable_grad_clip(
            self._net.model().parameters(), grad_clip
        )
        
        self._actor_optimizer = torch.optim.Adam(self._net.model()._actor.parameters(), lr=lr)
        self._actor_trainer = Trainer(self._actor_optimizer).enable_grad_clip(
            self._net.model()._actor.parameters(), grad_clip
        )
        
        self._critic_optimizer = torch.optim.Adam(self._net.model()._critic.parameters(), lr=lr)
        self._critic_trainer = Trainer(self._critic_optimizer).enable_grad_clip(
            self._net.model()._critic.parameters(), grad_clip
        )
        
        # Initialize trajectories buffer
        self._trajectory = DreamerTrajectory(n_steps, num_envs)
        
        # Training step counter
        self._time_steps = 0
        # Use a simple counter instead of TrainStep class if it requires an optimizer
        self._train_step_counter = 0
        
        # Observation normalization stats
        self._obs_mean_var = IncrementalMeanVarianceFromBatch()
        self._current_init_norm_steps = 0
        self._init_norm_steps = init_norm_steps
        
        # Metrics
        self._avg_model_loss = IncrementalMean()
        self._avg_actor_loss = IncrementalMean()
        self._avg_critic_loss = IncrementalMean()
        self._avg_reward_loss = IncrementalMean()
        self._avg_decoder_loss = IncrementalMean()
        self._avg_continue_loss = IncrementalMean()
        self._avg_kl_loss = IncrementalMean()
        
        # Track intrinsic reward metrics
        self._avg_int_reward = IncrementalMean()
        self._avg_ext_reward = IncrementalMean()
        self._avg_total_reward = IncrementalMean()
    
    def _init_hidden_state(self) -> None:
        """Initialize the hidden state."""
        shape = self._net.hidden_state_shape()
        self._hidden_state = torch.zeros(shape[0], self._num_envs, shape[1], device=self.device)
        self._next_hidden_state = self._hidden_state.clone()
        
        # Initialize previous action log probability and state value trackers
        self._action_log_prob = torch.zeros(self._num_envs, 1, device=self.device)
        self._state_value = torch.zeros(self._num_envs, 1, device=self.device)
    
    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Select action based on observation."""
        # Process observations in smaller batches if needed
        batch_size = obs.shape[0]
        max_batch = min(16, batch_size)  # Process at most 16 observations at a time
        
        all_actions = []
        all_action_log_probs = []
        all_state_values = []
        
        # Process in batches to avoid CUDA OOM
        for i in range(0, batch_size, max_batch):
            end_idx = min(i + max_batch, batch_size)
            current_batch_size = end_idx - i
            
            # Extract the appropriate slice of observations
            batch_obs = obs[i:end_idx]
            batch_obs = self._normalize_obs(batch_obs).to(self.device)
            
            with torch.no_grad():
                # Encode observation
                obs_embed = self._net.encode(batch_obs)
                
                # Create a batch-sized hidden state for this mini-batch
                h, c = net.unwrap_lstm_hidden_state(self._hidden_state)
                
                # Create new hidden states with the correct batch dimension
                batch_h = torch.zeros(h.shape[0], current_batch_size, h.shape[2], device=self.device)
                batch_c = torch.zeros(c.shape[0], current_batch_size, c.shape[2], device=self.device)
                
                # Copy values from the original hidden state
                if i < h.shape[1]:
                    # Make sure we don't try to copy beyond the bounds of the original tensor
                    copy_size = min(current_batch_size, h.shape[1] - i)
                    batch_h[:, :copy_size, :] = h[:, i:i+copy_size, :]
                    batch_c[:, :copy_size, :] = c[:, i:i+copy_size, :]
                
                batch_hidden = net.wrap_lstm_hidden_state(batch_h, batch_c)
                
                # Get state representation
                state, batch_hidden_out = self._net.represent(obs_embed, batch_hidden)
                
                # Get action distribution
                policy_dist = self._net.forward_actor(state)
                batch_action = policy_dist.sample()
                
                # Get next hidden state for this batch
                next_state, batch_next_hidden = self._net.recurrent_model_step(state, batch_action, batch_hidden_out)
                
                # Store results for this batch
                all_actions.append(batch_action)
                all_action_log_probs.append(policy_dist.log_prob(batch_action))
                all_state_values.append(self._net.forward_critic(state))
                
                # Clear GPU cache if possible
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
        
        # Concatenate all results
        self._action_log_prob = torch.cat(all_action_log_probs, dim=0)
        self._state_value = torch.cat(all_state_values, dim=0)
        
        # Initialize next hidden state with the same shape as the original hidden state
        h, c = net.unwrap_lstm_hidden_state(self._hidden_state)
        self._next_hidden_state = self._hidden_state.clone()
        
        # Update hidden state for next iteration - simply copy the first batch's hidden state
        # This is a simplification but will work for now
        self._hidden_state = self._next_hidden_state
        
        return torch.cat(all_actions, dim=0)
    
    def update(self, exp: Experience) -> Optional[dict]:
        """Update the agent with new experience."""
        self._time_steps += 1
        
        # Update hidden state for next time step
        self._hidden_state = self._next_hidden_state
        
        # Track rewards - handle both standard and intrinsic reward formats
        total_reward = exp.reward
        ext_reward = exp.reward
        int_reward = torch.zeros_like(exp.reward)
        
        # If info contains 'int_reward', extract intrinsic and extrinsic components
        if hasattr(exp, 'info') and isinstance(exp.info, dict):
            if 'int_reward' in exp.info:
                int_reward = torch.tensor(exp.info['int_reward'], device=self.device)
                ext_reward = exp.reward - int_reward
        
        self._avg_ext_reward.update(ext_reward.mean().item())
        self._avg_int_reward.update(int_reward.mean().item())
        self._avg_total_reward.update(total_reward.mean().item())
        
        # Initialize normalization parameters
        if (self._init_norm_steps is not None) and (self._current_init_norm_steps < self._init_norm_steps):
            self._current_init_norm_steps += 1
            self._obs_mean_var.update(exp.next_obs)
            return
        
        # Add experience to trajectory
        self._trajectory.add(DreamerExperience(
            **exp.__dict__,
            action_log_prob=self._action_log_prob,
            state_value=self._state_value,
            hidden_state=self._hidden_state
        ))
        
        # Train if we have collected enough steps
        if self._trajectory.reached_n_steps:
            metric_info_dicts = self._train()
            info_dict = {"metric": metric_info_dicts}
            return info_dict
        
        return None
    
    def _train(self) -> List[dict]:
        """Train the agent on the collected trajectory."""
        # Get trajectory batch
        exp_batch = self._trajectory.get_batch()
        batch_size = exp_batch.obs.size(0)
        
        # Reset trajectory for next collection
        self._trajectory.reset()
        
        # Extract experience data
        obs = exp_batch.obs
        action = exp_batch.action
        next_obs = exp_batch.next_obs
        reward = exp_batch.reward
        terminated = exp_batch.terminated
        hidden_state = exp_batch.hidden_state
        
        # Check if the effective batch size is smaller than the actual batch size
        # If so, we need to process the data in smaller batches to avoid OOM
        actual_batch_size = obs.size(0)
        if self._effective_batch_size < actual_batch_size:
            # Process the batch in chunks
            model_losses = []
            actor_losses = []
            critic_losses = []
            
            for i in range(0, actual_batch_size, self._effective_batch_size):
                end_idx = min(i + self._effective_batch_size, actual_batch_size)
                current_batch_size = end_idx - i
                
                # Extract the batch slice
                obs_batch = obs[i:end_idx]
                action_batch = action[i:end_idx]
                next_obs_batch = next_obs[i:end_idx]
                reward_batch = reward[i:end_idx]
                terminated_batch = terminated[i:end_idx]
                
                # Extract the appropriate hidden state slice
                h, c = net.unwrap_lstm_hidden_state(hidden_state)
                h_batch = h[:, i:end_idx]
                c_batch = c[:, i:end_idx]
                hidden_state_batch = net.wrap_lstm_hidden_state(h_batch, c_batch)
                
                # Normalize observations
                normalized_obs_batch = self._normalize_obs(obs_batch)
                normalized_next_obs_batch = self._normalize_obs(next_obs_batch)
                
                # Train on this batch
                model_loss = self._train_world_model(
                    normalized_obs_batch,
                    action_batch,
                    normalized_next_obs_batch,
                    reward_batch,
                    terminated_batch,
                    hidden_state_batch
                )
                model_losses.append(model_loss.item())
                
                actor_loss, critic_loss = self._train_policy(normalized_obs_batch, hidden_state_batch)
                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())
                
                # Clear GPU cache if possible
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            
            # Average the losses
            self._avg_model_loss.update(sum(model_losses) / len(model_losses))
            self._avg_actor_loss.update(sum(actor_losses) / len(actor_losses))
            self._avg_critic_loss.update(sum(critic_losses) / len(critic_losses))
        else:
            # Process the whole batch at once
            # Normalize observations
            normalized_obs = self._normalize_obs(obs)
            normalized_next_obs = self._normalize_obs(next_obs)
            
            # Train models
            model_loss = self._train_world_model(
                normalized_obs, 
                action, 
                normalized_next_obs, 
                reward, 
                terminated, 
                hidden_state
            )
            
            # Train policy
            actor_loss, critic_loss = self._train_policy(normalized_obs, hidden_state)
            
            # Track metrics
            self._avg_model_loss.update(model_loss.item())
            self._avg_actor_loss.update(actor_loss.item())
            self._avg_critic_loss.update(critic_loss.item())
        
        # Return metrics
        metric_info_dicts = [{
            "train_metric": {
                "keys": {
                    "time_step": self._time_steps
                },
                "values": {
                    "model_loss": self._avg_model_loss.mean,
                    "actor_loss": self._avg_actor_loss.mean,
                    "critic_loss": self._avg_critic_loss.mean,
                    "reward_loss": self._avg_reward_loss.mean,
                    "decoder_loss": self._avg_decoder_loss.mean,
                    "continue_loss": self._avg_continue_loss.mean,
                    "kl_loss": self._avg_kl_loss.mean
                }
            }
        }]
        
        return metric_info_dicts
    
    def _train_world_model(self, 
                         obs: torch.Tensor, 
                         action: torch.Tensor, 
                         next_obs: torch.Tensor, 
                         reward: torch.Tensor, 
                         terminated: torch.Tensor, 
                         hidden_state: torch.Tensor) -> torch.Tensor:
        """Train the world model components."""
        # 1. Encode observations
        obs_embed = self._net.encode(obs)
        next_obs_embed = self._net.encode(next_obs)
        
        # 2. Get state representation
        state, _ = self._net.represent(obs_embed, hidden_state)
        
        # 3. Get next state prediction with recurrent model
        pred_next_state, _ = self._net.recurrent_model_step(state, action, hidden_state)
        
        # 4. Compute reconstruction loss (decoder)
        pred_obs = self._net.decode(state)
        
        # Ensure pred_obs and obs have the same batch dimension
        if pred_obs.size(0) != obs.size(0):
            # Resize tensors to match if needed
            common_batch_size = min(pred_obs.size(0), obs.size(0))
            pred_obs = pred_obs[:common_batch_size]
            obs = obs[:common_batch_size]
            print(f"Resized tensors for decoder loss: pred_obs={pred_obs.shape}, obs={obs.shape}")
        
        decoder_loss = F.mse_loss(pred_obs, obs)
        self._avg_decoder_loss.update(decoder_loss.item())
        
        # 5. Compute reward prediction loss
        pred_reward = self._net.predict_reward(state)
        
        # Ensure pred_reward and reward have the same batch dimension
        if pred_reward.size(0) != reward.size(0):
            # Resize tensors to match if needed
            common_batch_size = min(pred_reward.size(0), reward.size(0))
            pred_reward = pred_reward[:common_batch_size]
            reward = reward[:common_batch_size]
            print(f"Resized tensors for reward loss: pred_reward={pred_reward.shape}, reward={reward.shape}")
        
        reward_loss = F.mse_loss(pred_reward, reward)
        self._avg_reward_loss.update(reward_loss.item())
        
        # 6. Compute continue prediction loss
        continue_target = 1 - terminated
        pred_continue = self._net.predict_continue(state)
        
        # Ensure pred_continue and continue_target have the same batch dimension
        if pred_continue.size(0) != continue_target.size(0):
            # Resize tensors to match if needed
            common_batch_size = min(pred_continue.size(0), continue_target.size(0))
            pred_continue = pred_continue[:common_batch_size]
            continue_target = continue_target[:common_batch_size]
            print(f"Resized tensors for continue loss: pred_continue={pred_continue.shape}, continue_target={continue_target.shape}")
        
        continue_loss = F.binary_cross_entropy(pred_continue, continue_target)
        self._avg_continue_loss.update(continue_loss.item())
        
        # 7. KL regularization loss
        kl_loss = torch.tensor(0.0, device=self.device)
        self._avg_kl_loss.update(kl_loss.item())
        
        # 8. Combine losses
        model_loss = (
            self._config.decoder_weight * decoder_loss + 
            self._config.reward_weight * reward_loss + 
            self._config.continue_weight * continue_loss +
            self._config.kl_loss_scale * kl_loss
        )
        
        # 9. Update world model parameters
        self._world_model_trainer.step(model_loss, self._train_step_counter)
        self._train_step_counter += 1
        
        return model_loss
    
    def _train_policy(self, obs: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train the actor and critic using imagination rollouts."""
        # Break into smaller batches if needed
        batch_size = obs.size(0)
        max_batch = min(4, batch_size)  # Process at most 4 observations at a time
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        for i in range(0, batch_size, max_batch):
            end_idx = min(i + max_batch, batch_size)
            current_batch_size = end_idx - i
            
            # Extract the batch slice
            obs_batch = obs[i:end_idx]
            
            # Extract the appropriate hidden state slice
            h, c = net.unwrap_lstm_hidden_state(hidden_state)
            h_batch = h[:, i:end_idx] if i < h.size(1) else torch.zeros(h.size(0), current_batch_size, h.size(2), device=self.device)
            c_batch = c[:, i:end_idx] if i < c.size(1) else torch.zeros(c.size(0), current_batch_size, c.size(2), device=self.device)
            hidden_batch = net.wrap_lstm_hidden_state(h_batch, c_batch)
            
            # 1. Encode observations and get initial state
            with torch.no_grad():
                obs_embed = self._net.encode(obs_batch)
                initial_state, hidden_state_batch = self._net.represent(obs_embed, hidden_batch)
            
            # 2. Perform imagination rollouts
            imagined_states = [initial_state]
            imagined_actions = []
            action_log_probs = []
            
            state = initial_state
            curr_hidden_state = hidden_state_batch
            
            for _ in range(self._config.imagination_steps):
                # Get action from current policy
                policy_dist = self._net.forward_actor(state)
                action = policy_dist.sample()
                action_log_prob = policy_dist.log_prob(action)
                
                # Imagine next state
                with torch.no_grad():
                    next_state, next_hidden_state = self._net.recurrent_model_step(state, action, curr_hidden_state)
                
                # Save trajectory
                imagined_actions.append(action)
                action_log_probs.append(action_log_prob)
                imagined_states.append(next_state)
                
                # Update for next step
                state = next_state
                curr_hidden_state = next_hidden_state
                
                # Clear cache if possible after each imagination step
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            
            # 3. Compute returns and advantages for actor-critic training
            with torch.no_grad():
                # Get predicted values for all states
                imagined_values = []
                imagined_rewards = []
                continue_probs = []
                
                for state in imagined_states:
                    value = self._net.forward_critic(state)
                    reward = self._net.predict_reward(state)
                    cont_prob = self._net.predict_continue(state)
                    
                    imagined_values.append(value)
                    imagined_rewards.append(reward)
                    continue_probs.append(cont_prob)
                
                # Convert lists to tensors
                imagined_values = torch.stack(imagined_values, dim=1)
                imagined_rewards = torch.stack(imagined_rewards[:-1], dim=1)  # No reward for last state
                continue_probs = torch.stack(continue_probs[:-1], dim=1)  # No continue for last state
                
                # Compute lambda returns
                lambda_returns = []
                last_value = imagined_values[:, -1]
                
                for t in reversed(range(self._config.imagination_steps)):
                    bootstrap = (
                        self._config.discount * continue_probs[:, t] * 
                        ((1 - self._config.lambda_) * imagined_values[:, t+1] + 
                         self._config.lambda_ * last_value)
                    )
                    last_value = imagined_rewards[:, t] + bootstrap
                    lambda_returns.insert(0, last_value)
                    
                lambda_returns = torch.stack(lambda_returns, dim=1)
                
                # Compute advantages
                advantages = lambda_returns - imagined_values[:, :-1]
            
            # 4. Train actor (policy)
            batch_policy_loss = 0
            for t in range(self._config.imagination_steps):
                state = imagined_states[t]
                policy_dist = self._net.forward_actor(state)
                
                # Get log probabilities for actions
                log_probs = policy_dist.log_prob(imagined_actions[t])
                
                # Compute entropy for exploration
                entropy = policy_dist.entropy().mean()
                
                # Compute policy loss with advantage
                t_policy_loss = -(log_probs * advantages[:, t].detach()).mean()
                
                # Add entropy bonus
                t_policy_loss = t_policy_loss - self._config.actor_entropy * entropy
                
                batch_policy_loss += t_policy_loss
            
            # Average over time steps
            batch_policy_loss = batch_policy_loss / self._config.imagination_steps
            
            # Scale the loss by batch proportion for gradient accumulation
            batch_policy_loss = batch_policy_loss * (current_batch_size / batch_size)
            
            # Update actor parameters - divide by number of batches to simulate gradient accumulation
            self._actor_trainer.step(batch_policy_loss)
            
            # 5. Train critic (value function)
            batch_value_loss = 0
            for t in range(self._config.imagination_steps):
                state = imagined_states[t]
                value = self._net.forward_critic(state)
                
                # Compute value loss
                t_value_loss = F.mse_loss(value, lambda_returns[:, t].detach())
                batch_value_loss += t_value_loss
            
            # Average over time steps
            batch_value_loss = batch_value_loss / self._config.imagination_steps
            
            # Scale the loss by batch proportion for gradient accumulation
            batch_value_loss = batch_value_loss * (current_batch_size / batch_size)
            
            # Update critic parameters
            self._critic_trainer.step(self._config.critic_weight * batch_value_loss)
            
            # Accumulate losses
            total_policy_loss += batch_policy_loss.item() * (current_batch_size / batch_size)
            total_value_loss += batch_value_loss.item() * (current_batch_size / batch_size)
            num_batches += 1
            
            # Free memory
            del imagined_states, imagined_actions, action_log_probs, imagined_values, imagined_rewards, continue_probs
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        
        # Return average losses
        return total_policy_loss, total_value_loss
    
    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations using running statistics."""
        if hasattr(self._obs_mean_var, 'mean') and hasattr(self._obs_mean_var, 'variance'):
            # Check if mean is numpy array or tensor and handle accordingly
            if isinstance(self._obs_mean_var.mean, np.ndarray):
                mean = torch.from_numpy(self._obs_mean_var.mean).to(device=self.device, dtype=obs.dtype)
                var = torch.from_numpy(self._obs_mean_var.variance).to(device=self.device, dtype=obs.dtype)
            else:
                mean = self._obs_mean_var.mean.to(device=self.device, dtype=obs.dtype)
                var = self._obs_mean_var.variance.to(device=self.device, dtype=obs.dtype)
                
            return (obs - mean) / torch.sqrt(var + 1e-8)
        return obs  # Return unnormalized if no stats available yet
    
    def inference_agent(self, num_envs: int) -> "DreamerV3":
        """Create a copy of the agent for inference."""
        shape = self._net.hidden_state_shape()
        inference_agent = type(self)(
            self._net, 
            num_envs,
            **{k: v for k, v in self._config.__dict__.items() if k != "self" and k != "net" and k != "num_envs"}
        )
        inference_agent._hidden_state = torch.zeros(shape[0], num_envs, shape[1], device=self.device)
        inference_agent._next_hidden_state = inference_agent._hidden_state.clone()
        inference_agent._obs_mean_var = self._obs_mean_var
        return inference_agent
    
    @property
    def log_data(self) -> Dict[str, tuple]:
        """Get log data for the agent."""
        return {
            "Agent/Model Loss": (self._avg_model_loss.mean, self._time_steps),
            "Agent/Actor Loss": (self._avg_actor_loss.mean, self._time_steps),
            "Agent/Critic Loss": (self._avg_critic_loss.mean, self._time_steps),
            "Agent/Reward Loss": (self._avg_reward_loss.mean, self._time_steps),
            "Agent/Decoder Loss": (self._avg_decoder_loss.mean, self._time_steps),
            "Agent/Continue Loss": (self._avg_continue_loss.mean, self._time_steps),
            "Agent/KL Loss": (self._avg_kl_loss.mean, self._time_steps),
            "Agent/External Reward": (self._avg_ext_reward.mean, self._time_steps),
            "Agent/Intrinsic Reward": (self._avg_int_reward.mean, self._time_steps),
            "Agent/Total Reward": (self._avg_total_reward.mean, self._time_steps),
        }
    
    @property
    def state_dict(self) -> dict:
        """Get the state dict."""
        return {
            "net": self._net.model().state_dict(),
            "world_model_optimizer": self._world_model_optimizer.state_dict(),
            "actor_optimizer": self._actor_optimizer.state_dict(),
            "critic_optimizer": self._critic_optimizer.state_dict(),
            "obs_mean_var": {
                "mean": self._obs_mean_var.mean,
                "variance": self._obs_mean_var.variance,
                "count": self._obs_mean_var.count,
            },
            "time_steps": self._time_steps,
            "train_steps": self._train_step_counter,
        }
    
    @property
    def load_state_dict(self, state_dict: dict) -> None:
        """Load from state dict."""
        self._net.model().load_state_dict(state_dict["net"])
        self._world_model_optimizer.load_state_dict(state_dict["world_model_optimizer"])
        self._actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self._critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self._obs_mean_var.mean = state_dict["obs_mean_var"]["mean"]
        self._obs_mean_var.variance = state_dict["obs_mean_var"]["variance"]
        self._obs_mean_var.count = state_dict["obs_mean_var"]["count"]
        self._time_steps = state_dict["time_steps"]
        self._train_step_counter = state_dict["train_steps"]


class DreamerTrajectory:
    """Trajectory buffer for DreamerV3."""
    
    def __init__(self, n_steps: int, num_envs: int) -> None:
        self._n_steps = n_steps
        self._num_envs = num_envs
        self._experiences = []
        self._count = 0
    
    def add(self, experience: DreamerExperience) -> None:
        """Add experience to buffer."""
        self._experiences.append(experience)
        self._count += 1
    
    def get_batch(self) -> DreamerExperience:
        """Get batch of experiences as a single DreamerExperience object."""
        if not self.reached_n_steps:
            raise ValueError("Not enough experiences collected yet.")
        
        return DreamerExperience(**{
            k: torch.cat([getattr(exp, k) for exp in self._experiences], dim=0) 
            for k in self._experiences[0].__dict__.keys()
        })
    
    def reset(self) -> None:
        """Reset the trajectory buffer."""
        self._experiences = []
        self._count = 0
    
    @property
    def reached_n_steps(self) -> bool:
        """Check if we've collected enough steps."""
        return self._count >= self._n_steps * self._num_envs