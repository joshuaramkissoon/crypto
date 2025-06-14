#!/usr/bin/env python3
"""
Command-line interface for RL trading system.
Provides easy access to training, backtesting, and deployment of RL agents.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import List, Optional

from crypto.rl_trainer import RLTrainingPipeline, TrainingConfig
from crypto.backtesting import BacktestConfig
from crypto.rl_agents import RLTradingAgent


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('rl_trading.log')
        ]
    )


def cmd_collect_data(args):
    """Collect historical data command."""
    pipeline = RLTrainingPipeline(
        data_dir=args.data_dir,
        api_key=args.api_key,
        api_secret=args.api_secret
    )
    
    symbols = args.symbols if isinstance(args.symbols, list) else [args.symbols]
    intervals = args.intervals if isinstance(args.intervals, list) else [args.intervals]
    
    pipeline.collect_training_data(
        symbols=symbols,
        intervals=intervals,
        days_back=args.days_back,
        update_existing=not args.no_update
    )
    
    print(f"Data collection completed for {len(symbols)} symbols")


def cmd_train_agent(args):
    """Train RL agent command."""
    pipeline = RLTrainingPipeline(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )
    
    # Create training configuration
    config = TrainingConfig(
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gamma=args.gamma
    )
    
    # Prepare training environment
    train_env, val_env = pipeline.prepare_training_environment(
        symbol=args.symbol,
        train_start=args.train_start,
        train_end=args.train_end
    )
    
    # Train agent
    results = pipeline.train_single_agent(
        agent_name=args.agent_name,
        symbol=args.symbol,
        config=config,
        train_env=train_env,
        val_env=val_env
    )
    
    print(f"Training completed for agent '{args.agent_name}'")
    print(f"Final evaluation return: {results['evaluation_results'].get('average_return', 0):.2%}")


def cmd_train_ensemble(args):
    """Train ensemble of agents command."""
    pipeline = RLTrainingPipeline(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )
    
    # Create configurations for different algorithms
    algorithms = ['PPO', 'DQN', 'A2C'] if not args.algorithms else args.algorithms
    
    agent_configs = []
    for i, algorithm in enumerate(algorithms):
        agent_name = f"{args.ensemble_name}_{algorithm.lower()}"
        config = TrainingConfig(
            algorithm=algorithm,
            total_timesteps=args.timesteps,
            learning_rate=args.learning_rate
        )
        agent_configs.append((agent_name, config))
    
    # Train all agents
    results = pipeline.train_multiple_agents(
        symbol=args.symbol,
        agent_configs=agent_configs,
        train_start=args.train_start,
        train_end=args.train_end
    )
    
    # Create ensemble
    agent_names = [name for name, _ in agent_configs]
    ensemble = pipeline.create_ensemble_agent(agent_names, args.ensemble_name)
    
    print(f"Ensemble training completed with {len(agent_names)} agents")


def cmd_backtest(args):
    """Backtest agents command."""
    pipeline = RLTrainingPipeline(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )
    
    # Load agents if specified
    if args.agent_names:
        for agent_name in args.agent_names:
            model_path = Path(args.models_dir) / f"{agent_name}_final.zip"
            if model_path.exists():
                pipeline.load_agent(agent_name, str(model_path), args.algorithm)
            else:
                print(f"Warning: Model file not found for {agent_name}")
    
    # Run backtest
    backtest_config = BacktestConfig(
        initial_capital=args.initial_capital,
        commission=args.commission
    )
    
    results = pipeline.backtest_agents(
        symbol=args.symbol,
        agent_names=args.agent_names,
        test_start=args.test_start,
        test_end=args.test_end,
        backtest_config=backtest_config
    )
    
    print("Backtest Results:")
    print("-" * 50)
    for agent_name, result in results.items():
        perf = result['performance']
        print(f"{agent_name}:")
        print(f"  Total Return: {perf.total_return:.2%}")
        print(f"  Sharpe Ratio: {perf.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {perf.max_drawdown:.2%}")
        print(f"  Win Rate: {perf.win_rate:.2%}")
        print()


def cmd_compare_strategies(args):
    """Compare RL agents with traditional strategies."""
    pipeline = RLTrainingPipeline(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )
    
    # Load RL agents
    if args.agent_names:
        for agent_name in args.agent_names:
            model_path = Path(args.models_dir) / f"{agent_name}_final.zip"
            if model_path.exists():
                pipeline.load_agent(agent_name, str(model_path), args.algorithm)
    
    # Run comparison
    comparison_df = pipeline.compare_with_traditional_strategies(
        symbol=args.symbol,
        test_start=args.test_start,
        test_end=args.test_end
    )
    
    print("Strategy Comparison Results:")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    
    # Save to file
    output_file = f"{args.symbol}_strategy_comparison.csv"
    comparison_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


def cmd_optimize_hyperparams(args):
    """Optimize hyperparameters command."""
    pipeline = RLTrainingPipeline(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )
    
    results = pipeline.hyperparameter_optimization(
        symbol=args.symbol,
        algorithm=args.algorithm,
        n_trials=args.n_trials,
        train_start=args.train_start,
        train_end=args.train_end
    )
    
    if results:
        print("Hyperparameter Optimization Results:")
        print(f"Best Score: {results['best_value']:.4f}")
        print("Best Parameters:")
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")
        
        # Save results
        with open(f"{args.symbol}_hyperopt_results.json", 'w') as f:
            json.dump({k: v for k, v in results.items() if k != 'study'}, f, indent=2)


def cmd_deploy_agent(args):
    """Deploy agent for live trading (integration with existing system)."""
    print("To deploy an RL agent for live trading:")
    print()
    print("1. Load the trained agent:")
    print(f"   from crypto.rl_trainer import RLTrainingPipeline")
    print(f"   from crypto.rl_agents import TrainingConfig")
    print()
    print("2. Create pipeline and load agent:")
    print(f"   pipeline = RLTrainingPipeline()")
    print(f"   pipeline.load_agent('{args.agent_name}', '{args.model_path}', '{args.algorithm}')")
    print()
    print("3. Use with StrategyManager:")
    print(f"   from crypto.strategy_manager import StrategyManager")
    print(f"   from crypto.rl_trainer import RLStrategy")
    print()
    print("4. Create RL strategy instance:")
    print(f"   manager = StrategyManager(client, notifier)")
    print(f"   config = {{'agent': pipeline.agents['{args.agent_name}']}}")
    print(f"   strategy_id = manager.create_strategy('RLStrategy', '{args.base_asset}', '{args.quote_asset}', config)")
    print(f"   manager.start_strategy(strategy_id)")
    print()
    print("The RL agent is now running live!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="RL Trading System CLI")
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    parser.add_argument('--data-dir', default='./data', help='Data directory')
    parser.add_argument('--models-dir', default='./models', help='Models directory')
    parser.add_argument('--results-dir', default='./results', help='Results directory')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Data collection command
    collect_parser = subparsers.add_parser('collect-data', help='Collect historical data')
    collect_parser.add_argument('symbols', nargs='+', help='Trading pair symbols (e.g., BTCUSDT ETHUSDT)')
    collect_parser.add_argument('--intervals', nargs='+', default=['1h'], help='Kline intervals')
    collect_parser.add_argument('--days-back', type=int, default=730, help='Days of historical data')
    collect_parser.add_argument('--no-update', action='store_true', help="Don't update existing data")
    collect_parser.add_argument('--api-key', help='Binance API key')
    collect_parser.add_argument('--api-secret', help='Binance API secret')
    collect_parser.set_defaults(func=cmd_collect_data)
    
    # Train single agent command
    train_parser = subparsers.add_parser('train', help='Train RL agent')
    train_parser.add_argument('agent_name', help='Name for the agent')
    train_parser.add_argument('symbol', help='Trading pair symbol (e.g., BTCUSDT)')
    train_parser.add_argument('--algorithm', choices=['PPO', 'DQN', 'A2C'], default='PPO', help='RL algorithm')
    train_parser.add_argument('--timesteps', type=int, default=100000, help='Training timesteps')
    train_parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    train_parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    train_parser.add_argument('--train-start', help='Training start date (YYYY-MM-DD)')
    train_parser.add_argument('--train-end', help='Training end date (YYYY-MM-DD)')
    train_parser.set_defaults(func=cmd_train_agent)
    
    # Train ensemble command
    ensemble_parser = subparsers.add_parser('train-ensemble', help='Train ensemble of agents')
    ensemble_parser.add_argument('ensemble_name', help='Name for the ensemble')
    ensemble_parser.add_argument('symbol', help='Trading pair symbol')
    ensemble_parser.add_argument('--algorithms', nargs='+', choices=['PPO', 'DQN', 'A2C'], help='Algorithms to include')
    ensemble_parser.add_argument('--timesteps', type=int, default=50000, help='Training timesteps per agent')
    ensemble_parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    ensemble_parser.add_argument('--train-start', help='Training start date (YYYY-MM-DD)')
    ensemble_parser.add_argument('--train-end', help='Training end date (YYYY-MM-DD)')
    ensemble_parser.set_defaults(func=cmd_train_ensemble)
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Backtest agents')
    backtest_parser.add_argument('symbol', help='Trading pair symbol')
    backtest_parser.add_argument('--agent-names', nargs='+', help='Agent names to backtest')
    backtest_parser.add_argument('--algorithm', default='PPO', help='Algorithm (for loading)')
    backtest_parser.add_argument('--test-start', help='Test start date (YYYY-MM-DD)')
    backtest_parser.add_argument('--test-end', help='Test end date (YYYY-MM-DD)')
    backtest_parser.add_argument('--initial-capital', type=float, default=10000, help='Initial capital')
    backtest_parser.add_argument('--commission', type=float, default=0.001, help='Commission rate')
    backtest_parser.set_defaults(func=cmd_backtest)
    
    # Compare strategies command
    compare_parser = subparsers.add_parser('compare', help='Compare RL agents with traditional strategies')
    compare_parser.add_argument('symbol', help='Trading pair symbol')
    compare_parser.add_argument('--agent-names', nargs='+', help='RL agent names to include')
    compare_parser.add_argument('--algorithm', default='PPO', help='Algorithm (for loading)')
    compare_parser.add_argument('--test-start', help='Test start date (YYYY-MM-DD)')
    compare_parser.add_argument('--test-end', help='Test end date (YYYY-MM-DD)')
    compare_parser.set_defaults(func=cmd_compare_strategies)
    
    # Hyperparameter optimization command
    optim_parser = subparsers.add_parser('optimize', help='Optimize hyperparameters')
    optim_parser.add_argument('symbol', help='Trading pair symbol')
    optim_parser.add_argument('--algorithm', choices=['PPO', 'DQN', 'A2C'], default='PPO', help='RL algorithm')
    optim_parser.add_argument('--n-trials', type=int, default=20, help='Number of optimization trials')
    optim_parser.add_argument('--train-start', help='Training start date (YYYY-MM-DD)')
    optim_parser.add_argument('--train-end', help='Training end date (YYYY-MM-DD)')
    optim_parser.set_defaults(func=cmd_optimize_hyperparams)
    
    # Deploy agent command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy agent for live trading')
    deploy_parser.add_argument('agent_name', help='Agent name')
    deploy_parser.add_argument('model_path', help='Path to model file')
    deploy_parser.add_argument('base_asset', help='Base asset (e.g., BTC)')
    deploy_parser.add_argument('quote_asset', help='Quote asset (e.g., USDT)')
    deploy_parser.add_argument('--algorithm', default='PPO', help='Algorithm')
    deploy_parser.set_defaults(func=cmd_deploy_agent)
    
    # Parse arguments and run command
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_logging(args.verbose)
    
    try:
        args.func(args)
    except Exception as e:
        logging.error(f"Command failed: {e}")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == '__main__':
    main()