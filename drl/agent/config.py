from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class RecurrentPPOConfig:
    """
    Recurrent PPO configurations.
    """
    n_steps: int
    epoch: int
    seq_len: int
    seq_mini_batch_size: int
    padding_value: float = 0.0
    gamma: float = 0.99
    lam: float = 0.95
    epsilon_clip: float = 0.2
    critic_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    
@dataclass(frozen=True)
class RecurrentPPORNDConfig:
    """
    Recurrent PPO with RND configurations.
    """
    n_steps: int
    epoch: int
    seq_len: int
    seq_mini_batch_size: Optional[int] = None
    padding_value: float = 0.0
    gamma: float = 0.99
    gamma_n: float = 0.99
    nonepi_adv_coef: float = 1.0
    lam: float = 0.95
    epsilon_clip: float = 0.2
    critic_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    rnd_pred_exp_proportion: float = 0.25
    init_norm_steps: Optional[int] = 50
    obs_norm_clip_range: Tuple[float, float] = (-5.0, 5.0)
    hidden_state_norm_clip_range: Tuple[float, float] = (-5.0, 5.0)

@dataclass(frozen=True)
class RecurrentA3CConfig:
    """
    Recurrent A3C configurations.
    """
    n_steps: int
    seq_len: int
    seq_mini_batch_size: int
    padding_value: float = 0.0
    gamma: float = 1.0
    lam: float = 0.99
    entropy_coef: float = 0.001
    critic_loss_coef: float = 0.5
    learning_rate: float = 0.07
    grad_norm_clip: float = 160.0
    cg_iters: int = 20
    tr_delta = 0.99
    line_search_steps : int = 20

@dataclass(frozen=True)
class RecurrentA3CRNDConfig:
    """
    Recurrent A3C with RND configurations.
    """
    n_steps: int
    epoch: int
    seq_len: int
    seq_mini_batch_size: Optional[int] = None
    padding_value: float = 0.0
    gamma: float = 0.99
    gamma_n: float = 0.99
    lam: float = 0.95
    nonepi_adv_coef: float = 1.0
    critic_loss_coef: float = 0.5
    entropy_coef: float = 0.001
    learning_rate: float = 0.0007
    grad_norm_clip: float = 40.0
    rnd_pred_exp_proportion: float = 0.25
    init_norm_steps: Optional[int] = 50
    obs_norm_clip_range: Tuple[float, float] = (-5.0, 5.0)
    hidden_state_norm_clip_range: Tuple[float, float] = (-5.0, 5.0)
    cg_iters: int = 1000
    tr_delta = 100.0
    line_search_steps : int = 1000
    fvp_damping: float = 1e-4

@dataclass(frozen=True)
class RecurrentSACConfig:
    """
    Recurrent PPO configurations.
    """
    n_steps: int
    epoch: int
    seq_len: int
    seq_mini_batch_size: int
    padding_value: float = 0.0
    gamma: float = 0.99
    epsilon_clip: float = 0.2
    critic_loss_coef: float = 0.75
    target_entropy: float = 0.1
    init_alpha: float = 0.1
    auto_entropy: bool = True