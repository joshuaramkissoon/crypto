#!/usr/bin/env python3
"""
Complete example of training and deploying RL agents for cryptocurrency trading.

This example demonstrates:
1. Data collection and preparation
2. Training multiple RL agents
3. Backtesting and evaluation
4. Comparison with traditional strategies
5. Deployment for live trading
"""

import logging
import pandas as pd
from datetime import datetime, timedelta

from crypto.rl_trainer import RLTrainingPipeline, TrainingConfig
from crypto.backtesting import BacktestConfig
from crypto.strategy_manager import StrategyManager


def main():
    """Complete RL trading pipeline example."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize pipeline
    pipeline = RLTrainingPipeline(
        data_dir="./data",
        models_dir="./models", 
        results_dir="./results"
    )
    
    # 1. COLLECT DATA
    logger.info("=== STEP 1: COLLECTING DATA ===")
    
    symbols = ['BTCUSDT', 'ETHUSDT']
    pipeline.collect_training_data(
        symbols=symbols,
        intervals=['1h'],
        days_back=730,  # 2 years
        update_existing=True
    )
    
    # 2. TRAIN MULTIPLE RL AGENTS
    logger.info("=== STEP 2: TRAINING RL AGENTS ===")
    
    symbol = 'BTCUSDT'
    
    # Define different agent configurations
    agent_configs = [
        ("btc_ppo_agent", TrainingConfig(
            algorithm="PPO",
            total_timesteps=100000,
            learning_rate=3e-4,
            batch_size=64,
            gamma=0.99
        )),
        ("btc_dqn_agent", TrainingConfig(
            algorithm="DQN", 
            total_timesteps=100000,
            learning_rate=1e-4,
            batch_size=32,
            gamma=0.95
        )),
        ("btc_a2c_agent", TrainingConfig(
            algorithm="A2C",
            total_timesteps=80000,
            learning_rate=7e-4,
            gamma=0.99
        ))
    ]
    
    # Train all agents
    training_results = pipeline.train_multiple_agents(
        symbol=symbol,
        agent_configs=agent_configs,
        train_start="2022-01-01",
        train_end="2023-06-01"
    )
    
    # Create ensemble
    ensemble = pipeline.create_ensemble_agent(
        agent_names=[name for name, _ in agent_configs],
        ensemble_name="btc_ensemble"
    )
    
    # 3. BACKTEST AGENTS
    logger.info("=== STEP 3: BACKTESTING ===")
    
    backtest_config = BacktestConfig(
        initial_capital=10000,
        commission=0.001,
        start_date="2023-06-01",
        end_date="2024-01-01"
    )
    
    backtest_results = pipeline.backtest_agents(
        symbol=symbol,
        test_start="2023-06-01",
        test_end="2024-01-01",
        backtest_config=backtest_config
    )
    
    # Print backtest results
    logger.info("Backtest Results:")
    for agent_name, result in backtest_results.items():
        perf = result['performance']
        logger.info(f"{agent_name}:")
        logger.info(f"  Total Return: {perf.total_return:.2%}")
        logger.info(f"  Sharpe Ratio: {perf.sharpe_ratio:.2f}")
        logger.info(f"  Max Drawdown: {perf.max_drawdown:.2%}")
        logger.info(f"  Win Rate: {perf.win_rate:.2%}")
    
    # 4. COMPARE WITH TRADITIONAL STRATEGIES
    logger.info("=== STEP 4: STRATEGY COMPARISON ===")
    
    comparison_df = pipeline.compare_with_traditional_strategies(
        symbol=symbol,
        test_start="2023-06-01", 
        test_end="2024-01-01"
    )
    
    logger.info("Strategy Comparison:")
    logger.info(comparison_df.to_string(index=False))
    
    # 5. HYPERPARAMETER OPTIMIZATION (Optional)
    logger.info("=== STEP 5: HYPERPARAMETER OPTIMIZATION ===")
    
    # Uncomment to run hyperparameter optimization
    # opt_results = pipeline.hyperparameter_optimization(
    #     symbol=symbol,
    #     algorithm="PPO",
    #     n_trials=10,
    #     train_start="2022-01-01",
    #     train_end="2023-06-01"
    # )
    
    # 6. DEPLOYMENT FOR LIVE TRADING
    logger.info("=== STEP 6: DEPLOYMENT EXAMPLE ===")
    
    # Example of how to deploy for live trading
    # This would be integrated with your existing trading system
    
    logger.info("To deploy the trained RL agent:")
    logger.info("1. The agents are saved in ./models/")
    logger.info("2. Use StrategyManager to deploy as RLStrategy")
    logger.info("3. The agent will make trading decisions in real-time")
    
    # Example deployment code (requires live trading setup):
    """
    from crypto.client import get_binance_client
    from crypto.session import Session
    from crypto.notifier import Notifier
    
    # Initialize trading components
    client = get_binance_client()
    session = Session()
    notifier = Notifier()
    
    # Create strategy manager
    strategy_manager = StrategyManager(client, notifier)
    
    # Load best performing agent
    best_agent = pipeline.agents['btc_ensemble']  # or whichever performed best
    
    # Create RL strategy configuration
    rl_config = {
        'agent': best_agent,
        'lookback_window': 30
    }
    
    # Deploy strategy
    strategy_id = strategy_manager.create_strategy(
        strategy_name='RLStrategy',
        base_asset='BTC',
        quote_asset='USDT',
        config=rl_config,
        custom_name='BTC_RL_Live'
    )
    
    # Start live trading
    strategy_manager.start_strategy(strategy_id)
    
    logger.info(f"RL strategy deployed and running: {strategy_id}")
    """
    
    # 7. MONITORING AND ANALYSIS
    logger.info("=== STEP 7: MONITORING ===")
    
    # Get training summary
    summary_df = pipeline.get_training_summary()
    logger.info("Training Summary:")
    logger.info(summary_df.to_string(index=False))
    
    # Save all results
    pipeline.save_agent('btc_ppo_agent')
    pipeline.save_agent('btc_dqn_agent') 
    pipeline.save_agent('btc_a2c_agent')
    
    logger.info("=== EXAMPLE COMPLETED ===")
    logger.info("Check ./results/ for detailed analysis and plots")
    logger.info("Check ./models/ for saved RL agents")


def quick_start_example():
    """Quick start example for immediate testing."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=== QUICK START EXAMPLE ===")
    
    # Initialize with minimal setup
    pipeline = RLTrainingPipeline()
    
    # Train a simple agent (faster for testing)
    config = TrainingConfig(
        algorithm="PPO",
        total_timesteps=10000,  # Reduced for quick testing
        learning_rate=3e-4
    )
    
    # This would need historical data already collected
    try:
        # Prepare environment
        train_env, val_env = pipeline.prepare_training_environment(
            symbol='BTCUSDT',
            train_start="2023-01-01",
            train_end="2023-06-01"
        )
        
        # Train agent
        results = pipeline.train_single_agent(
            agent_name="quick_test_agent",
            symbol='BTCUSDT', 
            config=config,
            train_env=train_env,
            val_env=val_env
        )
        
        logger.info(f"Quick training completed!")
        logger.info(f"Evaluation return: {results['evaluation_results'].get('average_return', 0):.2%}")
        
    except Exception as e:
        logger.error(f"Quick start failed: {e}")
        logger.info("Make sure to collect data first using the CLI:")
        logger.info("python crypto/rl_cli.py collect-data BTCUSDT --days-back 365")


def demo_traditional_vs_rl():
    """Demo comparing traditional strategies vs RL agents."""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=== TRADITIONAL vs RL COMPARISON DEMO ===")
    
    # This demo shows how RL agents can potentially outperform traditional strategies
    # by adapting to market conditions and learning complex patterns
    
    pipeline = RLTrainingPipeline()
    
    # Load or train a simple RL agent
    # (In practice, you'd use a well-trained agent)
    
    # Run comparison (needs data to be collected first)
    try:
        comparison_df = pipeline.compare_with_traditional_strategies(
            symbol='BTCUSDT',
            test_start="2023-06-01",
            test_end="2024-01-01"
        )
        
        logger.info("Performance Comparison:")
        logger.info(comparison_df[['Strategy', 'Total Return', 'Sharpe Ratio', 'Max Drawdown']].to_string(index=False))
        
    except Exception as e:
        logger.error(f"Comparison demo failed: {e}")
        logger.info("Ensure you have trained agents and collected data")


if __name__ == "__main__":
    # Run the complete example
    main()
    
    # Uncomment to run quick start instead
    # quick_start_example()
    
    # Uncomment to run comparison demo
    # demo_traditional_vs_rl()