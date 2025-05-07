from dataclasses import dataclass
from typing import Optional, Tuple, Type, Dict, Any, List

import torch
import torch.optim as optim

import drl
import drl.agent as agent
import train.net as net
import util
from envs import Env
from envs.chem_env import make_async_chem_env
from train.train import Train
from train.pretrain import Pretrain, SelfiesDataset
from util import instance_from_dict
from train.inference import Inference
from train.net import SelfiesPretrainedNet
import os

class ConfigParsingError(Exception):
    pass

def yaml_to_config_dict(file_path: str) -> Tuple[str, dict]:
    try:
        config_dict = util.load_yaml(file_path)
    except FileNotFoundError:
        raise ConfigParsingError(f"Config file not found: {file_path}")
    
    try:
        config_id = tuple(config_dict.keys())[0]
    except:
        raise ConfigParsingError("YAML config file must start with the training ID.")
    config = config_dict[config_id]
    return config_id, config

@dataclass(frozen=True)
class CommonConfig:
    num_envs: int = 1
    seed: Optional[int] = None
    device: Optional[str] = None
    lr: float = 1e-3
    grad_clip_max_norm: float = 5.0
    pretrained_path: Optional[str] = None
    num_inference_envs: int = 0


class AgentFactory:
    """Factory class for creating agent instances based on configuration."""
    
    @staticmethod
    def create_agent(agent_config: dict, env_config: dict, num_envs: int, device: Optional[str] = None) -> agent.Agent:
        """Create an agent based on the provided configuration."""
        agent_type = agent_config.get("type", "Pretrained").lower()
        
        if agent_type == "ppo":
            return AgentFactory.__create_ppo_agent(agent_config, env_config, num_envs, device)
        elif agent_type == "rnd":
            return AgentFactory.__create_rnd_agent(agent_config, env_config, num_envs, device)
        elif agent_type == "pretrained":
            return AgentFactory.__create_pretrained_agent(agent_config, env_config, num_envs, device)
        elif agent_type == "dreamerv3":
            return AgentFactory.__create_dreamerv3_agent(agent_config, env_config, num_envs, device)
        else:
            raise ValueError(f"Agent type {agent_type} is not supported.")
    
    @staticmethod
    def __create_pretrained_agent(agent_config: dict, env_config: dict, num_envs: int, device: Optional[str] = None) -> agent.PretrainedRecurrentAgent:
        device = device or (agent_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        with torch.device(device):
            net_obj = SelfiesPretrainedNet(env_config["num_actions"])
            return agent.PretrainedRecurrentAgent(
                net_obj,
                num_envs=num_envs,
                device=device,
                temperature=float(agent_config.get("temperature", 1.0))
            )
    
    @staticmethod
    def __create_ppo_agent(agent_config: dict, env_config: dict, num_envs: int, device: Optional[str] = None) -> agent.RecurrentPPO:
        device = device or (agent_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        with torch.device(device):
            config = instance_from_dict(agent.RecurrentPPOConfig, agent_config)
            network = net.SelfiesRecurrentPPONet(
                env_config["obs_shape"][0],
                env_config["num_actions"]
            )
            trainer = drl.Trainer(optim.Adam(
                network.parameters(),
                lr=agent_config.get("lr", 1e-3)
            )).enable_grad_clip(network.parameters(), max_norm=agent_config.get("grad_clip_max_norm", 5.0))
            
            return agent.RecurrentPPO(
                config=config,
                network=network,
                trainer=trainer,
                num_envs=num_envs,
                device=device
            )
    
    @staticmethod
    def __create_rnd_agent(agent_config: dict, env_config: dict, num_envs: int, device: Optional[str] = None) -> agent.RecurrentPPORND:
        device = device or (agent_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        with torch.device(device):
            config = instance_from_dict(agent.RecurrentPPORNDConfig, agent_config)
            network = net.SelfiesRecurrentPPORNDNet(
                env_config["obs_shape"][0],
                env_config["num_actions"],
                temperature=float(agent_config.get("temperature", 1.0))
            )
            trainer = drl.Trainer(optim.Adam(
                network.parameters(),
                lr=agent_config.get("lr", 1e-3)
            )).enable_grad_clip(network.parameters(), max_norm=agent_config.get("grad_clip_max_norm", 5.0))
            
            return agent.RecurrentPPORND(
                config=config,
                network=network,
                trainer=trainer,
                num_envs=num_envs,
                device=device
            )
            
    @staticmethod
    def __create_dreamerv3_agent(agent_config: dict, env_config: dict, num_envs: int, device: Optional[str] = None) -> agent.DreamerV3:
        # Check for GPU device
        is_cuda = device == 'cuda' or (device is None and torch.cuda.is_available())
        
        # If we have a cuda device, handle memory limitations
        if is_cuda:
            # Check available GPU memory and adjust batch size accordingly
            free_memory = 0
            if torch.cuda.is_available():
                try:
                    # Get free memory in bytes, convert to GB
                    free_memory = torch.cuda.mem_get_info()[0] / (1024**3)
                    print(f"Available GPU memory: {free_memory:.2f} GB")
                except:
                    # Older PyTorch versions don't have mem_get_info
                    free_memory = 2.0  # Assume 2GB as a conservative default
            
            # Set effective batch size based on available memory
            effective_batch_size = min(num_envs, max(1, int(free_memory / 0.5)))  # Estimate 0.5GB per env
            if effective_batch_size < num_envs:
                print(f"WARNING: Reducing effective batch size for DreamerV3 to avoid CUDA OOM. Original: {num_envs}, New: {effective_batch_size}")
                agent_config['effective_batch_size'] = effective_batch_size
        
        device = device or (agent_config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        network = net.SelfiesDreamerV3Net(
            env_config["obs_shape"][0],
            env_config["num_actions"],
            temperature=float(agent_config.get("temperature", 1.0))
        )
        network = network.to(device)
        
        # Create a new dictionary with proper parameter types
        kwargs = {}
        
        # Add essential parameters
        kwargs['net'] = network
        kwargs['num_envs'] = num_envs
        kwargs['device'] = device
        
        # Extract configuration parameters, ensuring proper type conversion
        for k, v in agent_config.items():
            if k not in ["type", "device", "temperature"]:
                if isinstance(v, str):
                    if k in ['lr', 'discount', 'lambda_', 'actor_entropy', 'critic_weight', 
                           'model_weight', 'reward_weight', 'continue_weight', 'kl_loss_scale', 
                           'decoder_weight', 'grad_clip']:
                        kwargs[k] = float(v)
                    elif k in ['horizon', 'imagination_steps', 'n_steps']:
                        kwargs[k] = int(v)
                    elif k == 'init_norm_steps':
                        kwargs[k] = int(v) if v.lower() != 'null' and v.lower() != 'none' else None
                    else:
                        kwargs[k] = v
                else:
                    kwargs[k] = v
        
        # Debug print to verify parameter types
        print(f"DreamerV3 Parameters: {kwargs}")
        
        return agent.DreamerV3(**kwargs)


class MolRLTrainFactory:
    """
    Factory class creates a Train instance from a dictionary config.
    """
    @staticmethod
    def from_yaml(file_path: str) -> "MolRLTrainFactory":
        """
        Create a `MolRLTrainFactory` from a YAML file.
        """
        config_id, config = yaml_to_config_dict(file_path)
        return MolRLTrainFactory(config_id, config)
    
    def __init__(self, id: str, config: dict):
        self._id = id
        self._config = config
        self._agent_config = self._config.get("Agent", dict())
        self._env_config = self._config.get("Env", dict())
        self._train_config = self._config.get("Train", dict())
        self._count_int_reward_config = self._config.get("CountIntReward", dict())
        self._common_config = instance_from_dict(CommonConfig, self._train_config)
        self._pretrained = None
        
    def create_train(self) -> Train:
        self._train_setup()
        
        try:
            env = self._create_env()
            inference_env = self._create_inference_env()
        except TypeError:
            raise ConfigParsingError("Invalid Env config. Missing arguments or wrong type.")
        try:
            agent = self._create_agent(env)
        except TypeError:
            raise ConfigParsingError("Invalid Agent config. Missing arguments or wrong type.")
        try:
            smiles_or_selfies_refset = util.load_smiles_or_selfies(self._train_config["refset_path"]) if "refset_path" in self._train_config else None
            train = instance_from_dict(Train, {
                "env": env,
                "agent": agent,
                "id": self._id,
                "inference_env": inference_env,
                "smiles_or_selfies_refset": smiles_or_selfies_refset,
                **self._train_config,
            })
        except TypeError:
            raise ConfigParsingError("Invalid Train config. Missing arguments or wrong type.")
        return train
    
    def _train_setup(self):
        util.logger.enable(self._id, enable_log_file=False)
        util.try_create_dir(util.logger.dir())
        config_to_save = {self._id: self._config}
        util.save_yaml(f"{util.logger.dir()}/config.yaml", config_to_save)
        
        if self._common_config.seed is not None:
            util.seed(self._common_config.seed)
            
        if self._common_config.pretrained_path is not None:
            self._pretrained = torch.load(self._common_config.pretrained_path)
        else:
            if os.path.exists(f"{util.logger.dir()}/pretrained_models/best.pt"):
                self._pretrained = torch.load(f"{util.logger.dir()}/pretrained_models/best.pt")
    
        if "vocab_path" in self._env_config:
            vocab, max_str_len = util.load_vocab(self._env_config["vocab_path"])
            self._env_config["vocabulary"] = vocab
            self._env_config["max_str_len"] = max_str_len
        else:
            if os.path.exists(f"{util.logger.dir()}/vocab.json"):
                vocab, max_str_len = util.load_vocab(f"{util.logger.dir()}/vocab.json")
                self._env_config["vocabulary"] = vocab
                self._env_config["max_str_len"] = max_str_len
        util.logger.disable()
        
    def _create_env(self) -> Env:        
        env = make_async_chem_env(
            num_envs=self._common_config.num_envs,
            seed=self._common_config.seed,
            **{**self._env_config, **self._count_int_reward_config}
        )
        return env
    
    def _create_inference_env(self) -> Optional[Env]:
        if self._common_config.num_inference_envs == 0:
            return None
        
        env = make_async_chem_env(
            num_envs=self._common_config.num_inference_envs,
            seed=self._common_config.seed,
            **{**self._env_config}
        )
        return env
    
    def _create_agent(self, env: Env) -> agent.Agent:
        agent_type = self._agent_config["type"].lower()
        agent_instance = None
        
        if agent_type == "ppo":
            agent_instance = AgentFactory.create_agent(
                self._agent_config, 
                {"obs_shape": env.obs_shape, "num_actions": env.num_actions},
                self._common_config.num_envs, 
                self._common_config.device
            )
        elif agent_type == "rnd":
            agent_instance = AgentFactory.create_agent(
                self._agent_config, 
                {"obs_shape": env.obs_shape, "num_actions": env.num_actions},
                self._common_config.num_envs, 
                self._common_config.device
            )
        elif agent_type == "pretrained":
            agent_instance = AgentFactory.create_agent(
                self._agent_config,
                {"num_actions": env.num_actions},
                self._common_config.num_envs,
                self._common_config.device
            )
        elif agent_type == "dreamerv3":
            agent_instance = AgentFactory.create_agent(
                self._agent_config,
                {"obs_shape": env.obs_shape, "num_actions": env.num_actions},
                self._common_config.num_envs,
                self._common_config.device
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        # Load pretrained weights if available
        if self._pretrained is not None and hasattr(agent_instance, "_network"):
            agent_instance._network.load_state_dict(self._pretrained["model"], strict=False)
            
        return agent_instance

class MolRLInferenceFactory:
    @staticmethod
    def from_yaml(file_path: str) -> "MolRLInferenceFactory":
        """
        Create a `MolRLInferenceFactory` from a YAML file.
        """
        config_id, config = yaml_to_config_dict(file_path)
        return MolRLInferenceFactory(config_id, config)
    
    def __init__(self, id: str, config: dict):
        self._id = id
        self._config = config
        self._agent_config = self._config.get("Agent", dict())
        self._env_config = self._config.get("Env", dict())
        self._train_config = self._config.get("Train", dict())
        self._inference_config = self._config.get("Inference", dict())
        self._common_config = instance_from_dict(CommonConfig, self._train_config)
        self._pretrained = None
        
    def create_inference(self) -> Inference:
        self._inference_setup()
        
        try:
            env = self._create_env()
        except TypeError:
            raise ConfigParsingError("Invalid Env config. Missing arguments or wrong type.")
        try:
            agent = self._create_agent(env)
            agent = self._load_agent(agent)
            agent = agent.inference_agent(
                num_envs=env.num_envs,
                device=self._inference_config.get("device", self._common_config.device)
            )
        except TypeError:
            raise ConfigParsingError("Invalid Agent config. Missing arguments or wrong type.")
        except FileNotFoundError as e:
            raise ConfigParsingError(str(e))
        try:    
            if "refset_path" in self._inference_config:
                smiles_or_selfies_refset = util.load_smiles_or_selfies(self._inference_config["refset_path"])
            elif "refset_path" in self._train_config:
                smiles_or_selfies_refset = util.load_smiles_or_selfies(self._train_config["refset_path"])
            else:
                smiles_or_selfies_refset = None
            inference = instance_from_dict(Inference, {
                "env": env,
                "agent": agent,
                "id": self._id,
                "smiles_or_selfies_refset": smiles_or_selfies_refset,
                **self._inference_config,
            })
        except TypeError:
            raise ConfigParsingError("Invalid Train config. Missing arguments or wrong type.")
        return inference
        
    def _inference_setup(self):
        if "seed" in self._inference_config:
            util.seed(self._inference_config["seed"])
            
        if self._common_config.pretrained_path is not None:
            self._pretrained = torch.load(self._common_config.pretrained_path)
            
        util.logger.enable(self._id, enable_log_file=False)
        if "vocab_path" in self._env_config:
            vocab, max_str_len = util.load_vocab(self._env_config["vocab_path"])
            self._env_config["vocabulary"] = vocab
            self._env_config["max_str_len"] = max_str_len
        else:
            if os.path.exists(f"{util.logger.dir()}/vocab.json"):
                vocab, max_str_len = util.load_vocab(f"{util.logger.dir()}/vocab.json")
                self._env_config["vocabulary"] = vocab
                self._env_config["max_str_len"] = max_str_len
        util.logger.disable()
        
    def _create_env(self) -> Env:
        env = make_async_chem_env(
            num_envs=self._inference_config.get("num_envs", 1),
            seed=self._inference_config.get("seed", None),
            **{**self._env_config}
        )
        return env
    
    def _create_agent(self, env: Env) -> agent.Agent:
        agent_type = self._agent_config["type"].lower()
        num_envs = self._inference_config.get("num_envs", 1)
        device = self._inference_config.get("device", self._common_config.device)
        
        if agent_type == "ppo":
            agent_instance = AgentFactory.create_agent(
                self._agent_config,
                {"obs_shape": env.obs_shape, "num_actions": env.num_actions},
                num_envs,
                device
            )
        elif agent_type == "rnd":
            agent_instance = AgentFactory.create_agent(
                self._agent_config,
                {"obs_shape": env.obs_shape, "num_actions": env.num_actions},
                num_envs,
                device
            )
        elif agent_type == "pretrained":
            agent_instance = AgentFactory.create_agent(
                self._agent_config,
                {"num_actions": env.num_actions},
                num_envs,
                device
            )
        elif agent_type == "dreamerv3":
            agent_instance = AgentFactory.create_agent(
                self._agent_config,
                {"obs_shape": env.obs_shape, "num_actions": env.num_actions},
                num_envs,
                device
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        # Load pretrained weights if available
        if self._pretrained is not None and hasattr(agent_instance, "_network"):
            agent_instance._network.load_state_dict(self._pretrained["model"], strict=False)
            
        return agent_instance
        
    def _load_agent(self, agent: agent.Agent) -> agent.Agent:
        ckpt = self._inference_config.get("ckpt", "best")
        util.logger.enable(self._id, enable_log_file=False)
        
        if ckpt == "best":
            ckpt_path = f"{util.logger.dir()}/best_agent.pt"
        elif ckpt == "final":
            ckpt_path = f"{util.logger.dir()}/agent.pt"
        elif isinstance(ckpt, int):
            ckpt_path = f"{util.logger.dir()}/agent_ckpt/agent_{ckpt}.pt"

        util.logger.disable()
        
        try:
            state_dict = torch.load(ckpt_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
        
        agent.load_state_dict(state_dict["agent"])
        return agent
    
class MolRLPretrainFactory:
    @staticmethod
    def from_yaml(file_path: str) -> "MolRLPretrainFactory":
        """
        Create a `MolRLPretrainFactory` from a YAML file.
        """
        config_id, config = yaml_to_config_dict(file_path)
        return MolRLPretrainFactory(config_id, config)
    
    def __init__(self, id: str, config: dict):
        self._id = id
        self._config = config
        self._pretrain_config = self._config.get("Pretrain", dict())

    def create_pretrain(self) -> Pretrain:
        self._pretrain_setup()
        
        dataset = SelfiesDataset.from_txt(self._pretrain_config["dataset_path"])
        net = SelfiesPretrainedNet(vocab_size=dataset.tokenizer.vocab_size)
        
        try:
            pretrain = instance_from_dict(Pretrain, {
                "id": self._id,
                "net": net,
                "dataset": dataset,
                **self._pretrain_config,
            })
        except TypeError:
            raise ConfigParsingError("Invalid Pretrain config. Missing arguments or wrong type.")
        return pretrain
    
    def _pretrain_setup(self):
        util.logger.enable(self._id, enable_log_file=False)
        util.try_create_dir(util.logger.dir())
        config_to_save = {self._id: self._config}
        util.save_yaml(f"{util.logger.dir()}/config.yaml", config_to_save)
        self._log_dir = util.logger.dir()
        util.logger.disable()
        
        if "seed" in self._pretrain_config:
            util.seed(self._pretrain_config["seed"])