import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import pandas as pd
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt


@dataclass
class TrainingConfig:
    algorithm: str = "PPO"  # PPO, DQN, A2C
    total_timesteps: int = 100000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_update_interval: int = 10000
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.05


class TradingCallback(BaseCallback):
    """Custom callback for monitoring training progress."""
    
    def __init__(self, eval_env, eval_freq: int = 5000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.eval_rewards = []
        self.eval_returns = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate the agent
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=5, deterministic=True
            )
            
            self.eval_rewards.append(mean_reward)
            
            # Get portfolio return from evaluation
            obs = self.eval_env.reset()
            done = False
            initial_value = self.eval_env.envs[0].initial_balance
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.eval_env.step(action)
            
            final_value = info[0]['portfolio_value']
            portfolio_return = (final_value - initial_value) / initial_value
            self.eval_returns.append(portfolio_return)
            
            if self.verbose > 0:
                print(f"Eval at step {self.n_calls}: Mean reward: {mean_reward:.2f} ± {std_reward:.2f}, "
                      f"Portfolio return: {portfolio_return:.2%}")
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if hasattr(self.model, 'save'):
                    self.model.save(f"best_model_step_{self.n_calls}")
        
        return True


class RLTradingAgent:
    """
    Main RL trading agent that handles training and inference.
    Supports multiple algorithms: PPO, DQN, A2C.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model_name: str = "crypto_trading_agent",
        save_dir: str = "./models"
    ):
        self.config = config
        self.model_name = model_name
        self.save_dir = save_dir
        self.model = None
        self.training_env = None
        self.eval_env = None
        self.training_history = {
            'rewards': [],
            'returns': [],
            'episode_lengths': [],
            'training_loss': []
        }
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        logging.info(f"RLTradingAgent initialized with {config.algorithm}")
    
    def create_model(self, env):
        """Create the RL model based on configuration."""
        if self.config.algorithm == "PPO":
            self.model = PPO(
                "MlpPolicy",
                env,
                learning_rate=self.config.learning_rate,
                n_steps=2048,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                clip_range=self.config.clip_range,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                verbose=1,
                tensorboard_log=f"{self.save_dir}/tensorboard/"
            )
        
        elif self.config.algorithm == "DQN":
            self.model = DQN(
                "MlpPolicy",
                env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size,
                gamma=self.config.gamma,
                target_update_interval=self.config.target_update_interval,
                exploration_fraction=self.config.exploration_fraction,
                exploration_final_eps=self.config.exploration_final_eps,
                verbose=1,
                tensorboard_log=f"{self.save_dir}/tensorboard/"
            )
        
        elif self.config.algorithm == "A2C":
            self.model = A2C(
                "MlpPolicy",
                env,
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                verbose=1,
                tensorboard_log=f"{self.save_dir}/tensorboard/"
            )
        
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        logging.info(f"Created {self.config.algorithm} model")
    
    def train(
        self,
        train_env,
        eval_env=None,
        callback_freq: int = 5000
    ) -> Dict[str, Any]:
        """Train the RL agent."""
        self.training_env = DummyVecEnv([lambda: Monitor(train_env)])
        
        if eval_env is not None:
            self.eval_env = DummyVecEnv([lambda: Monitor(eval_env)])
        
        # Create model if not exists
        if self.model is None:
            self.create_model(self.training_env)
        
        # Create callback
        callback = None
        if self.eval_env is not None:
            callback = TradingCallback(
                eval_env=self.eval_env,
                eval_freq=callback_freq,
                verbose=1
            )
        
        logging.info(f"Starting training for {self.config.total_timesteps} timesteps")
        start_time = datetime.now()
        
        # Train the model
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callback,
            tb_log_name=f"{self.model_name}_{self.config.algorithm}"
        )
        
        training_time = datetime.now() - start_time
        logging.info(f"Training completed in {training_time}")
        
        # Save final model
        self.save_model()
        
        # Compile training results
        results = {
            'training_time': training_time.total_seconds(),
            'total_timesteps': self.config.total_timesteps,
            'algorithm': self.config.algorithm,
            'model_path': f"{self.save_dir}/{self.model_name}_final.zip"
        }
        
        if callback:
            results.update({
                'eval_rewards': callback.eval_rewards,
                'eval_returns': callback.eval_returns,
                'best_reward': callback.best_mean_reward
            })
        
        return results
    
    def evaluate(
        self,
        env,
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """Evaluate the trained agent."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        eval_env = DummyVecEnv([lambda: Monitor(env)])
        
        mean_reward, std_reward = evaluate_policy(
            self.model, eval_env, n_eval_episodes=n_episodes, deterministic=deterministic
        )
        
        # Get detailed episode results
        episode_results = []
        for episode in range(min(n_episodes, 5)):  # Detailed results for first 5 episodes
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            initial_value = eval_env.envs[0].initial_balance
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward[0]
            
            final_value = info[0]['portfolio_value']
            portfolio_return = (final_value - initial_value) / initial_value
            
            episode_results.append({
                'episode': episode,
                'reward': episode_reward,
                'portfolio_return': portfolio_return,
                'final_value': final_value,
                'total_trades': info[0]['total_trades'],
                'win_rate': info[0]['win_rate'],
                'max_drawdown': info[0]['max_drawdown']
            })
        
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'episode_results': episode_results,
            'average_return': np.mean([ep['portfolio_return'] for ep in episode_results]),
            'average_trades': np.mean([ep['total_trades'] for ep in episode_results]),
            'average_win_rate': np.mean([ep['win_rate'] for ep in episode_results])
        }
    
    def predict(self, observation, deterministic: bool = True):
        """Make a prediction using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def save_model(self, filename: Optional[str] = None):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        if filename is None:
            filename = f"{self.model_name}_final"
        
        model_path = f"{self.save_dir}/{filename}.zip"
        self.model.save(model_path)
        
        # Save configuration and metadata
        metadata = {
            'config': self.config,
            'model_name': self.model_name,
            'algorithm': self.config.algorithm,
            'saved_at': datetime.now().isoformat(),
            'training_history': self.training_history
        }
        
        metadata_path = f"{self.save_dir}/{filename}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logging.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, model_path: str):
        """Load a pre-trained model."""
        if self.config.algorithm == "PPO":
            self.model = PPO.load(model_path)
        elif self.config.algorithm == "DQN":
            self.model = DQN.load(model_path)
        elif self.config.algorithm == "A2C":
            self.model = A2C.load(model_path)
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        # Load metadata if available
        metadata_path = model_path.replace('.zip', '_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.training_history = metadata.get('training_history', {})
        
        logging.info(f"Model loaded from {model_path}")
    
    def plot_training_results(self, save_path: Optional[str] = None):
        """Plot training results and performance metrics."""
        if not self.training_history['rewards']:
            logging.warning("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        axes[0, 0].plot(self.training_history['rewards'])
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Plot returns
        axes[0, 1].plot(self.training_history['returns'])
        axes[0, 1].set_title('Portfolio Returns')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Return (%)')
        
        # Plot episode lengths
        axes[1, 0].plot(self.training_history['episode_lengths'])
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        
        # Plot training loss (if available)
        if self.training_history['training_loss']:
            axes[1, 1].plot(self.training_history['training_loss'])
            axes[1, 1].set_title('Training Loss')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Training plots saved to {save_path}")
        else:
            plt.show()


class EnsembleRLAgent:
    """
    Ensemble of multiple RL agents for improved performance and robustness.
    """
    
    def __init__(self, agents: List[RLTradingAgent], voting_strategy: str = "majority"):
        self.agents = agents
        self.voting_strategy = voting_strategy
        self.weights = np.ones(len(agents)) / len(agents)
        
        logging.info(f"EnsembleRLAgent created with {len(agents)} agents")
    
    def predict(self, observation, deterministic: bool = True):
        """Make ensemble prediction."""
        predictions = []
        
        for agent in self.agents:
            if agent.model is not None:
                action = agent.predict(observation, deterministic)
                predictions.append(action)
        
        if not predictions:
            return 0  # HOLD action if no agents available
        
        if self.voting_strategy == "majority":
            # Simple majority voting
            return np.bincount(predictions).argmax()
        
        elif self.voting_strategy == "weighted":
            # Weighted voting based on agent performance
            weighted_votes = np.zeros(3)  # 3 actions: HOLD, BUY, SELL
            for i, pred in enumerate(predictions):
                weighted_votes[pred] += self.weights[i]
            return weighted_votes.argmax()
        
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
    
    def update_weights(self, performance_scores: List[float]):
        """Update agent weights based on performance."""
        if len(performance_scores) != len(self.agents):
            raise ValueError("Number of scores must match number of agents")
        
        # Normalize scores to weights
        scores = np.array(performance_scores)
        scores = np.maximum(scores, 0)  # Ensure non-negative
        
        if scores.sum() > 0:
            self.weights = scores / scores.sum()
        else:
            self.weights = np.ones(len(self.agents)) / len(self.agents)
        
        logging.info(f"Updated ensemble weights: {self.weights}")


def create_multi_algorithm_ensemble(
    configs: List[TrainingConfig],
    model_names: List[str],
    save_dir: str = "./models"
) -> EnsembleRLAgent:
    """Create an ensemble of agents with different algorithms."""
    agents = []
    
    for config, name in zip(configs, model_names):
        agent = RLTradingAgent(config, name, save_dir)
        agents.append(agent)
    
    return EnsembleRLAgent(agents, voting_strategy="weighted")