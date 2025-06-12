import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt

from crypto.data_collector import BinanceDataCollector, DataPreprocessor
from crypto.rl_environment import CryptoTradingEnvironment
from crypto.rl_agents import RLTradingAgent, TrainingConfig, EnsembleRLAgent
from crypto.backtesting import BacktestEngine, BacktestConfig
from crypto.strategy import Strategy


class RLTrainingPipeline:
    """
    Complete pipeline for training RL agents for cryptocurrency trading.
    Handles data collection, preprocessing, training, evaluation, and deployment.
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        models_dir: str = "./models",
        results_dir: str = "./results",
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.data_collector = BinanceDataCollector(api_key, api_secret, data_dir)
        self.data_preprocessor = DataPreprocessor()
        self.agents = {}
        self.training_history = {}
        
        logging.info("RLTrainingPipeline initialized")
    
    def collect_training_data(
        self,
        symbols: List[str],
        intervals: List[str] = ['1h'],
        days_back: int = 365 * 2,  # 2 years of data
        update_existing: bool = True
    ):
        """Collect historical data for training."""
        logging.info(f"Collecting training data for {len(symbols)} symbols")
        
        self.data_collector.collect_and_store_data(
            symbols=symbols,
            intervals=intervals,
            days_back=days_back,
            update_existing=update_existing
        )
        
        logging.info("Data collection completed")
    
    def prepare_training_environment(
        self,
        symbol: str,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None,
        val_start: Optional[str] = None,
        val_end: Optional[str] = None,
        env_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[CryptoTradingEnvironment, CryptoTradingEnvironment]:
        """Prepare training and validation environments."""
        
        # Load and split data
        if val_start is None:
            # Automatic split if validation dates not provided
            train_data, val_data = self.data_collector.prepare_training_data(
                symbol=symbol,
                start_date=train_start,
                end_date=train_end,
                train_ratio=0.8
            )
        else:
            # Load separate train and validation data
            train_data = self.data_collector.load_historical_data(
                symbol=symbol,
                start_date=train_start,
                end_date=train_end
            )
            val_data = self.data_collector.load_historical_data(
                symbol=symbol,
                start_date=val_start,
                end_date=val_end
            )
        
        if train_data.empty or val_data.empty:
            raise ValueError(f"Insufficient data for {symbol}")
        
        # Preprocess data
        train_data = self.data_preprocessor.quality_check(train_data)
        val_data = self.data_preprocessor.quality_check(val_data)
        
        train_data = self.data_preprocessor.add_technical_features(train_data)
        val_data = self.data_preprocessor.add_technical_features(val_data)
        
        # Create environments
        env_config = env_config or {}
        
        train_env = CryptoTradingEnvironment(
            data=train_data,
            **env_config
        )
        
        val_env = CryptoTradingEnvironment(
            data=val_data,
            **env_config
        )
        
        logging.info(f"Training environment: {len(train_data)} steps")
        logging.info(f"Validation environment: {len(val_data)} steps")
        
        return train_env, val_env
    
    def train_single_agent(
        self,
        agent_name: str,
        symbol: str,
        config: TrainingConfig,
        train_env: CryptoTradingEnvironment,
        val_env: Optional[CryptoTradingEnvironment] = None,
        save_model: bool = True
    ) -> Dict[str, Any]:
        """Train a single RL agent."""
        
        logging.info(f"Training {config.algorithm} agent '{agent_name}' on {symbol}")
        
        # Create agent
        agent = RLTradingAgent(
            config=config,
            model_name=agent_name,
            save_dir=str(self.models_dir)
        )
        
        # Train agent
        start_time = datetime.now()
        training_results = agent.train(
            train_env=train_env,
            eval_env=val_env,
            callback_freq=5000
        )
        training_time = datetime.now() - start_time
        
        # Evaluate agent
        evaluation_results = {}
        if val_env is not None:
            evaluation_results = agent.evaluate(val_env, n_episodes=5)
        
        # Store agent and results
        self.agents[agent_name] = agent
        
        results = {
            'agent_name': agent_name,
            'symbol': symbol,
            'algorithm': config.algorithm,
            'training_time': training_time.total_seconds(),
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'config': config.__dict__
        }
        
        self.training_history[agent_name] = results
        
        # Save results
        if save_model:
            results_file = self.results_dir / f"{agent_name}_training_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        logging.info(f"Training completed for {agent_name}")
        return results
    
    def train_multiple_agents(
        self,
        symbol: str,
        agent_configs: List[Tuple[str, TrainingConfig]],
        env_config: Optional[Dict[str, Any]] = None,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train multiple agents with different configurations."""
        
        # Prepare environments
        train_env, val_env = self.prepare_training_environment(
            symbol=symbol,
            train_start=train_start,
            train_end=train_end,
            env_config=env_config
        )
        
        all_results = {}
        
        for agent_name, config in agent_configs:
            try:
                results = self.train_single_agent(
                    agent_name=agent_name,
                    symbol=symbol,
                    config=config,
                    train_env=train_env,
                    val_env=val_env
                )
                all_results[agent_name] = results
                
            except Exception as e:
                logging.error(f"Failed to train agent {agent_name}: {e}")
                continue
        
        return all_results
    
    def create_ensemble_agent(
        self,
        agent_names: List[str],
        ensemble_name: str = "ensemble"
    ) -> EnsembleRLAgent:
        """Create an ensemble from trained agents."""
        agents = [self.agents[name] for name in agent_names if name in self.agents]
        
        if not agents:
            raise ValueError("No valid agents found for ensemble")
        
        ensemble = EnsembleRLAgent(agents, voting_strategy="weighted")
        self.agents[ensemble_name] = ensemble
        
        logging.info(f"Created ensemble '{ensemble_name}' with {len(agents)} agents")
        return ensemble
    
    def backtest_agents(
        self,
        symbol: str,
        agent_names: Optional[List[str]] = None,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        backtest_config: Optional[BacktestConfig] = None
    ) -> Dict[str, Any]:
        """Backtest trained agents."""
        
        if agent_names is None:
            agent_names = list(self.agents.keys())
        
        # Load test data
        test_data = self.data_collector.load_historical_data(
            symbol=symbol,
            start_date=test_start,
            end_date=test_end
        )
        
        if test_data.empty:
            raise ValueError(f"No test data available for {symbol}")
        
        # Preprocess test data
        test_data = self.data_preprocessor.quality_check(test_data)
        test_data = self.data_preprocessor.add_technical_features(test_data)
        
        # Initialize backtest engine
        if backtest_config is None:
            backtest_config = BacktestConfig()
        
        engine = BacktestEngine(backtest_config)
        
        # Backtest each agent
        backtest_results = {}
        
        for agent_name in agent_names:
            if agent_name not in self.agents:
                logging.warning(f"Agent {agent_name} not found, skipping")
                continue
            
            try:
                agent = self.agents[agent_name]
                
                if isinstance(agent, RLTradingAgent):
                    results = engine.backtest_rl_agent(
                        agent=agent,
                        data=test_data,
                        symbol=symbol
                    )
                else:
                    # Handle ensemble agents
                    results = engine.backtest_rl_agent(
                        agent=agent,  # EnsembleRLAgent has predict method
                        data=test_data,
                        symbol=symbol
                    )
                
                backtest_results[agent_name] = results
                
                logging.info(f"Backtested {agent_name}: "
                           f"Return: {results['performance'].total_return:.2%}, "
                           f"Sharpe: {results['performance'].sharpe_ratio:.2f}")
                
            except Exception as e:
                logging.error(f"Failed to backtest agent {agent_name}: {e}")
                continue
        
        return backtest_results
    
    def compare_with_traditional_strategies(
        self,
        symbol: str,
        test_start: Optional[str] = None,
        test_end: Optional[str] = None,
        strategy_classes: Optional[List] = None
    ) -> pd.DataFrame:
        """Compare RL agents with traditional strategies."""
        
        if strategy_classes is None:
            # Import default strategies
            from crypto.strategy import RSI, MA, CMO
            from crypto.advanced_strategies import (
                MeanReversionStrategy, BreakoutStrategy, MomentumStrategy
            )
            strategy_classes = [RSI, MA, CMO, MeanReversionStrategy, BreakoutStrategy, MomentumStrategy]
        
        # Load test data
        test_data = self.data_collector.load_historical_data(
            symbol=symbol,
            start_date=test_start,
            end_date=test_end
        )
        
        if test_data.empty:
            raise ValueError(f"No test data available for {symbol}")
        
        # Preprocess test data
        test_data = self.data_preprocessor.quality_check(test_data)
        test_data = self.data_preprocessor.add_technical_features(test_data)
        
        # Initialize backtest engine
        backtest_config = BacktestConfig()
        engine = BacktestEngine(backtest_config)
        
        all_results = []
        
        # Backtest traditional strategies
        for strategy_class in strategy_classes:
            try:
                results = engine.backtest_traditional_strategy(
                    strategy_class=strategy_class,
                    data=test_data,
                    symbol=symbol
                )
                all_results.append(results)
                
            except Exception as e:
                logging.error(f"Failed to backtest {strategy_class.__name__}: {e}")
                continue
        
        # Backtest RL agents
        rl_results = self.backtest_agents(
            symbol=symbol,
            test_start=test_start,
            test_end=test_end,
            backtest_config=backtest_config
        )
        
        all_results.extend(rl_results.values())
        
        # Create comparison DataFrame
        comparison_df = engine.compare_strategies(all_results)
        
        # Save comparison results
        comparison_file = self.results_dir / f"{symbol}_strategy_comparison.csv"
        comparison_df.to_csv(comparison_file, index=False)
        
        # Plot results
        plot_file = self.results_dir / f"{symbol}_strategy_comparison.png"
        engine.plot_results(all_results, str(plot_file))
        
        logging.info(f"Strategy comparison completed for {symbol}")
        return comparison_df
    
    def hyperparameter_optimization(
        self,
        symbol: str,
        algorithm: str = "PPO",
        n_trials: int = 20,
        train_start: Optional[str] = None,
        train_end: Optional[str] = None
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna (optional dependency)."""
        try:
            import optuna
        except ImportError:
            logging.error("Optuna not installed. Run: pip install optuna")
            return {}
        
        def objective(trial):
            # Define hyperparameter search space
            if algorithm == "PPO":
                config = TrainingConfig(
                    algorithm="PPO",
                    learning_rate=trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
                    batch_size=trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                    n_epochs=trial.suggest_int('n_epochs', 5, 20),
                    gamma=trial.suggest_uniform('gamma', 0.9, 0.999),
                    gae_lambda=trial.suggest_uniform('gae_lambda', 0.8, 0.99),
                    clip_range=trial.suggest_uniform('clip_range', 0.1, 0.3),
                    ent_coef=trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
                    total_timesteps=20000  # Shorter for optimization
                )
            else:
                # Add other algorithms as needed
                config = TrainingConfig(algorithm=algorithm, total_timesteps=20000)
            
            # Prepare environment
            train_env, val_env = self.prepare_training_environment(
                symbol=symbol,
                train_start=train_start,
                train_end=train_end
            )
            
            # Train agent
            agent_name = f"optim_trial_{trial.number}"
            agent = RLTradingAgent(config, agent_name, str(self.models_dir))
            
            try:
                agent.train(train_env, val_env)
                eval_results = agent.evaluate(val_env, n_episodes=3)
                return eval_results['mean_reward']
            except Exception as e:
                logging.error(f"Trial {trial.number} failed: {e}")
                return -np.inf
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logging.info(f"Hyperparameter optimization completed. Best value: {best_value:.4f}")
        logging.info(f"Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'study': study
        }
    
    def save_agent(self, agent_name: str, filename: Optional[str] = None):
        """Save a trained agent."""
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")
        
        agent = self.agents[agent_name]
        if isinstance(agent, RLTradingAgent):
            model_path = agent.save_model(filename)
            logging.info(f"Agent {agent_name} saved to {model_path}")
        else:
            logging.warning(f"Cannot save ensemble agent {agent_name}")
    
    def load_agent(
        self,
        agent_name: str,
        model_path: str,
        algorithm: str
    ):
        """Load a pre-trained agent."""
        config = TrainingConfig(algorithm=algorithm)
        agent = RLTradingAgent(config, agent_name, str(self.models_dir))
        agent.load_model(model_path)
        
        self.agents[agent_name] = agent
        logging.info(f"Agent {agent_name} loaded from {model_path}")
    
    def get_training_summary(self) -> pd.DataFrame:
        """Get summary of all training sessions."""
        summary_data = []
        
        for agent_name, results in self.training_history.items():
            summary_data.append({
                'Agent': agent_name,
                'Symbol': results.get('symbol', 'Unknown'),
                'Algorithm': results.get('algorithm', 'Unknown'),
                'Training Time (s)': results.get('training_time', 0),
                'Mean Reward': results.get('evaluation_results', {}).get('mean_reward', 0),
                'Average Return': results.get('evaluation_results', {}).get('average_return', 0),
                'Average Win Rate': results.get('evaluation_results', {}).get('average_win_rate', 0)
            })
        
        return pd.DataFrame(summary_data)


# RL-based Strategy for integration with existing strategy system
class RLStrategy(Strategy):
    """
    RL-based trading strategy that can be used with the existing strategy framework.
    """
    
    def __init__(self, client, session, notifier, agent: RLTradingAgent, lookback_window: int = 30):
        super().__init__(client, session, notifier)
        self.agent = agent
        self.lookback_window = lookback_window
        self.price_history = []
        self.observation_buffer = []
        
    def trading_strategy(self, symbol, data):
        """Implement trading strategy using RL agent predictions."""
        if not data['x']:  # Only act on closed candles
            return
        
        # Update price history
        close_price = float(data['c'])
        self.price_history.append({
            'timestamp': data['t'],
            'open': float(data['o']),
            'high': float(data['h']),
            'low': float(data['l']),
            'close': close_price,
            'volume': float(data['v'])
        })
        
        # Need enough history to create observation
        if len(self.price_history) < self.lookback_window:
            return
        
        try:
            # Create observation from recent price history
            recent_data = pd.DataFrame(self.price_history[-self.lookback_window:])
            recent_data.set_index('timestamp', inplace=True)
            
            # Create mini environment for observation
            mini_env = CryptoTradingEnvironment(
                data=recent_data,
                lookback_window=self.lookback_window
            )
            
            observation = mini_env._get_observation()
            
            # Get action from RL agent
            action = self.agent.predict(observation, deterministic=True)
            
            # Execute action
            if action == 1:  # BUY
                logging.info(f"RL Agent BUY signal at price {close_price}")
                self.order('BUY', 0.01, symbol)  # Fixed size for now
            elif action == 2:  # SELL
                logging.info(f"RL Agent SELL signal at price {close_price}")
                self.order('SELL', 0.01, symbol)
            # action == 0 is HOLD, do nothing
            
        except Exception as e:
            logging.error(f"RL Strategy error: {e}")
            
    def set_agent(self, agent: RLTradingAgent):
        """Update the RL agent."""
        self.agent = agent