#!/usr/bin/env python3

import argparse
import logging
import json
import time
import signal
import sys
from typing import List, Optional
from datetime import datetime
from binance.client import Client
from crypto.environment import Environment
from crypto.strategy_manager import StrategyManager, StrategyStatus
from crypto.observability import LiveDashboard
from crypto.notifier import Notifier
import os


class CryptoTradingCLI:
    """
    Advanced CLI for managing crypto algorithmic trading strategies.
    Provides intuitive commands to create, start, stop, and monitor strategies.
    """
    
    def __init__(self):
        self.env = Environment()
        self.client = None
        self.strategy_manager = None
        self.dashboard = None
        self.notifier = None
        self._setup_client()
        self._setup_logging()
        self._setup_signal_handlers()
    
    def _setup_client(self):
        """Initialize Binance client."""
        try:
            api_key = self.env.get_binance_key('api')
            secret_key = self.env.get_binance_key('secret')
            self.client = Client(api_key, secret_key, testnet=not self.env.is_live)
            
            # Test connection
            self.client.ping()
            print(f"✓ Connected to Binance {'LIVE' if self.env.is_live else 'TESTNET'}")
            
        except Exception as e:
            print(f"✗ Failed to connect to Binance: {e}")
            sys.exit(1)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Reduce noise from other libraries
        logging.getLogger('websocket').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(sig, frame):
            print("\n🛑 Graceful shutdown initiated...")
            if self.strategy_manager:
                self.strategy_manager.stop_all_strategies()
            if self.dashboard:
                self.dashboard.stop_dashboard()
            print("✓ Shutdown complete")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _initialize_manager(self):
        """Initialize strategy manager if not already done."""
        if not self.strategy_manager:
            # Setup notifier if telegram key is available
            try:
                telegram_key = self.env.get_telegram_key()
                if telegram_key:
                    self.notifier = Notifier(telegram_key)
                    self.notifier._auth()
                    print("✓ Telegram notifications enabled")
            except:
                print("⚠ Telegram notifications not available")
            
            self.strategy_manager = StrategyManager(self.client, self.notifier)
            print("✓ Strategy manager initialized")
    
    def list_available_strategies(self):
        """List all available strategy classes."""
        self._initialize_manager()
        strategies = self.strategy_manager.get_available_strategies()
        
        print("\n📋 Available Trading Strategies:")
        print("=" * 50)
        
        strategy_descriptions = {
            'RSI': 'Simple RSI-based mean reversion',
            'MA': 'Moving average crossover',
            'CMO': 'Chande Momentum Oscillator',
            'MeanReversionStrategy': 'Bollinger Bands + RSI mean reversion',
            'BreakoutStrategy': 'Volume-confirmed price breakouts',
            'GridTradingStrategy': 'Grid trading for range-bound markets',
            'ArbitrageStrategy': 'Price difference arbitrage (simplified)',
            'MomentumStrategy': 'Multi-factor momentum trading',
            'BollingerBandsStrategy': 'Bollinger squeeze breakouts',
            'MACDStrategy': 'MACD crossover signals',
            'VolatilityBreakoutStrategy': 'ATR-based volatility breakouts',
            'TrendFollowingStrategy': 'Multi-timeframe trend following',
            'ScalpingStrategy': 'High-frequency scalping'
        }
        
        for i, strategy in enumerate(strategies, 1):
            description = strategy_descriptions.get(strategy, 'Custom strategy')
            print(f"{i:2d}. {strategy:<25} - {description}")
        
        print("=" * 50)
        print(f"Total: {len(strategies)} strategies available")
    
    def create_strategy(self, strategy_name: str, base_asset: str, quote_asset: str, 
                       custom_name: Optional[str] = None, config: Optional[dict] = None):
        """Create a new strategy instance."""
        self._initialize_manager()
        
        try:
            instance_id = self.strategy_manager.create_strategy(
                strategy_name=strategy_name,
                base_asset=base_asset,
                quote_asset=quote_asset,
                custom_name=custom_name,
                config=config
            )
            
            instance = self.strategy_manager.get_strategy_status(instance_id)
            print(f"\n✓ Strategy created successfully!")
            print(f"  ID: {instance_id}")
            print(f"  Name: {instance.name}")
            print(f"  Strategy: {instance.strategy_class}")
            print(f"  Symbol: {instance.symbol}")
            print(f"  Status: {instance.status.value}")
            
            return instance_id
            
        except Exception as e:
            print(f"✗ Failed to create strategy: {e}")
            return None
    
    def start_strategy(self, instance_id: str):
        """Start a strategy instance."""
        self._initialize_manager()
        
        try:
            success = self.strategy_manager.start_strategy(instance_id)
            if success:
                instance = self.strategy_manager.get_strategy_status(instance_id)
                print(f"\n🚀 Strategy started successfully!")
                print(f"  Name: {instance.name}")
                print(f"  Symbol: {instance.symbol}")
                print(f"  Started: {instance.started_at}")
            else:
                print(f"✗ Failed to start strategy {instance_id}")
        
        except Exception as e:
            print(f"✗ Error starting strategy: {e}")
    
    def stop_strategy(self, instance_id: str):
        """Stop a strategy instance."""
        self._initialize_manager()
        
        try:
            success = self.strategy_manager.stop_strategy(instance_id)
            if success:
                instance = self.strategy_manager.get_strategy_status(instance_id)
                print(f"\n🛑 Strategy stopped successfully!")
                print(f"  Name: {instance.name}")
                print(f"  Runtime: {(instance.stopped_at - instance.started_at).total_seconds()/3600:.1f} hours")
            else:
                print(f"✗ Failed to stop strategy {instance_id}")
        
        except Exception as e:
            print(f"✗ Error stopping strategy: {e}")
    
    def list_strategies(self, status_filter: Optional[str] = None):
        """List all strategy instances."""
        self._initialize_manager()
        
        strategies = self.strategy_manager.list_strategies()
        
        if status_filter:
            strategies = [s for s in strategies if s.status.value == status_filter.lower()]
        
        if not strategies:
            filter_text = f" (status: {status_filter})" if status_filter else ""
            print(f"\n📋 No strategies found{filter_text}")
            return
        
        print(f"\n📋 Strategy Instances{' (' + status_filter + ')' if status_filter else ''}:")
        print("=" * 100)
        
        for strategy in strategies:
            runtime = ""
            if strategy.started_at:
                end_time = strategy.stopped_at or datetime.now()
                runtime = f" | Runtime: {(end_time - strategy.started_at).total_seconds()/3600:.1f}h"
            
            error_text = ""
            if strategy.error_message:
                error_text = f" | Error: {strategy.error_message[:30]}..."
            
            print(f"ID: {strategy.id[:8]}... | {strategy.name:<20} | {strategy.strategy_class:<20} | "
                  f"{strategy.symbol:<10} | Status: {strategy.status.value.upper():<8}{runtime}{error_text}")
        
        print("=" * 100)
        print(f"Total: {len(strategies)} strategies")
    
    def show_strategy_performance(self, instance_id: str):
        """Show detailed performance metrics for a strategy."""
        self._initialize_manager()
        
        try:
            instance = self.strategy_manager.get_strategy_status(instance_id)
            if not instance:
                print(f"✗ Strategy {instance_id} not found")
                return
            
            performance = self.strategy_manager.get_strategy_performance(instance_id)
            
            print(f"\n📊 Performance Report: {instance.name}")
            print("=" * 60)
            print(f"Strategy: {instance.strategy_class}")
            print(f"Symbol: {instance.symbol}")
            print(f"Status: {instance.status.value.upper()}")
            print(f"Created: {instance.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if instance.started_at:
                print(f"Started: {instance.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if instance.stopped_at:
                print(f"Stopped: {instance.stopped_at.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if performance:
                print("\n📈 Trading Metrics:")
                print(f"  Total Trades: {performance.get('total_trades', 0)}")
                print(f"  Win Rate: {performance.get('win_rate', 0):.1f}%")
                print(f"  Net P&L: ${performance.get('net_profit_loss', 0):.2f}")
                print(f"  Total Commission: ${performance.get('total_commission', 0):.2f}")
                print(f"  Max Drawdown: {performance.get('max_drawdown', 0):.2f}%")
                print(f"  ROI: {performance.get('return_on_investment', 0):.2f}%")
                
                if performance.get('sharpe_ratio'):
                    print(f"  Sharpe Ratio: {performance.get('sharpe_ratio'):.2f}")
                
                print(f"\n⚡ Performance Metrics:")
                print(f"  Avg Trade Duration: {performance.get('avg_trade_duration', 0):.1f} seconds")
                print(f"  Trades/Hour: {performance.get('trades_per_hour', 0):.2f}")
                print(f"  Error Count: {performance.get('error_count', 0)}")
            else:
                print("\n📈 No performance data available yet")
            
            print("=" * 60)
            
        except Exception as e:
            print(f"✗ Error retrieving performance data: {e}")
    
    def start_dashboard(self):
        """Start the live monitoring dashboard."""
        self._initialize_manager()
        
        try:
            if not self.dashboard:
                self.dashboard = LiveDashboard(self.strategy_manager.metrics_collector)
            
            print("\n🖥️  Starting live dashboard...")
            print("   Press Ctrl+C to exit dashboard and return to CLI")
            self.dashboard.start_dashboard()
            
            # Keep the dashboard running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.dashboard.stop_dashboard()
                print("\n✓ Dashboard stopped")
        
        except Exception as e:
            print(f"✗ Error starting dashboard: {e}")
    
    def export_data(self, instance_id: str, format: str = 'json'):
        """Export strategy data."""
        self._initialize_manager()
        
        try:
            data = self.strategy_manager.export_strategy_data(instance_id, format)
            
            # Generate filename
            instance = self.strategy_manager.get_strategy_status(instance_id)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{instance.name}_{timestamp}.{format}"
            
            # Write to file
            with open(filename, 'w') as f:
                f.write(data)
            
            print(f"✓ Strategy data exported to: {filename}")
            
        except Exception as e:
            print(f"✗ Error exporting data: {e}")
    
    def interactive_mode(self):
        """Start interactive CLI mode."""
        print("\n🚀 Crypto Algo Trading CLI - Interactive Mode")
        print("Type 'help' for available commands or 'exit' to quit")
        
        while True:
            try:
                command = input("\n>>> ").strip().split()
                if not command:
                    continue
                
                cmd = command[0].lower()
                
                if cmd in ['exit', 'quit']:
                    break
                elif cmd == 'help':
                    self._show_interactive_help()
                elif cmd == 'list-strategies':
                    self.list_available_strategies()
                elif cmd == 'list':
                    status_filter = command[1] if len(command) > 1 else None
                    self.list_strategies(status_filter)
                elif cmd == 'create':
                    self._interactive_create_strategy()
                elif cmd == 'start':
                    if len(command) < 2:
                        print("Usage: start <strategy_id>")
                        continue
                    self.start_strategy(command[1])
                elif cmd == 'stop':
                    if len(command) < 2:
                        print("Usage: stop <strategy_id>")
                        continue
                    self.stop_strategy(command[1])
                elif cmd == 'performance' or cmd == 'perf':
                    if len(command) < 2:
                        print("Usage: performance <strategy_id>")
                        continue
                    self.show_strategy_performance(command[1])
                elif cmd == 'dashboard':
                    self.start_dashboard()
                elif cmd == 'export':
                    if len(command) < 2:
                        print("Usage: export <strategy_id> [format]")
                        continue
                    format_type = command[2] if len(command) > 2 else 'json'
                    self.export_data(command[1], format_type)
                elif cmd == 'stop-all':
                    self._stop_all_strategies()
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit")
            except Exception as e:
                print(f"Error: {e}")
    
    def _interactive_create_strategy(self):
        """Interactive strategy creation."""
        print("\n📝 Create New Strategy")
        print("-" * 30)
        
        # Show available strategies
        self.list_available_strategies()
        
        try:
            strategy_name = input("\nEnter strategy name: ").strip()
            base_asset = input("Enter base asset (e.g., BTC): ").strip().upper()
            quote_asset = input("Enter quote asset (e.g., USDT): ").strip().upper()
            
            custom_name = input("Enter custom name (optional): ").strip()
            if not custom_name:
                custom_name = None
            
            instance_id = self.create_strategy(strategy_name, base_asset, quote_asset, custom_name)
            
            if instance_id:
                start_now = input("\nStart strategy now? (y/N): ").strip().lower()
                if start_now in ['y', 'yes']:
                    self.start_strategy(instance_id)
        
        except KeyboardInterrupt:
            print("\nStrategy creation cancelled")
    
    def _stop_all_strategies(self):
        """Stop all running strategies."""
        self._initialize_manager()
        
        running_strategies = self.strategy_manager.get_running_strategies()
        if not running_strategies:
            print("No running strategies to stop")
            return
        
        print(f"\n🛑 Stopping {len(running_strategies)} running strategies...")
        success = self.strategy_manager.stop_all_strategies()
        
        if success:
            print("✓ All strategies stopped successfully")
        else:
            print("⚠ Some strategies may not have stopped properly")
    
    def _show_interactive_help(self):
        """Show help for interactive mode."""
        print("""
📚 Available Commands:

Strategy Management:
  list-strategies          List all available strategy classes
  list [status]           List strategy instances (optionally filter by status)
  create                  Interactive strategy creation
  start <strategy_id>     Start a strategy instance
  stop <strategy_id>      Stop a strategy instance
  stop-all               Stop all running strategies

Monitoring:
  performance <id>        Show detailed performance metrics
  dashboard              Start live monitoring dashboard
  export <id> [format]   Export strategy data (default: json)

General:
  help                   Show this help message
  exit/quit              Exit interactive mode

Examples:
  list running           List only running strategies
  start abc123           Start strategy with ID starting with abc123
  performance xyz789     Show performance for strategy xyz789
        """)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Crypto Algorithmic Trading CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s interactive                    Start interactive mode
  %(prog)s list-strategies               List available strategy classes
  %(prog)s create RSI BTC USDT           Create RSI strategy for BTC/USDT
  %(prog)s start abc123                  Start strategy with ID abc123
  %(prog)s dashboard                     Start monitoring dashboard
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Interactive mode
    subparsers.add_parser('interactive', help='Start interactive CLI mode')
    
    # List strategies
    subparsers.add_parser('list-strategies', help='List available strategy classes')
    
    # List instances
    list_parser = subparsers.add_parser('list', help='List strategy instances')
    list_parser.add_argument('--status', choices=['running', 'stopped', 'error', 'paused'], 
                            help='Filter by status')
    
    # Create strategy
    create_parser = subparsers.add_parser('create', help='Create new strategy instance')
    create_parser.add_argument('strategy', help='Strategy class name')
    create_parser.add_argument('base_asset', help='Base asset (e.g., BTC)')
    create_parser.add_argument('quote_asset', help='Quote asset (e.g., USDT)')
    create_parser.add_argument('--name', help='Custom strategy name')
    create_parser.add_argument('--config', help='JSON configuration string')
    
    # Start strategy
    start_parser = subparsers.add_parser('start', help='Start strategy instance')
    start_parser.add_argument('instance_id', help='Strategy instance ID')
    
    # Stop strategy
    stop_parser = subparsers.add_parser('stop', help='Stop strategy instance')
    stop_parser.add_argument('instance_id', help='Strategy instance ID')
    
    # Performance
    perf_parser = subparsers.add_parser('performance', help='Show strategy performance')
    perf_parser.add_argument('instance_id', help='Strategy instance ID')
    
    # Dashboard
    subparsers.add_parser('dashboard', help='Start live monitoring dashboard')
    
    # Export
    export_parser = subparsers.add_parser('export', help='Export strategy data')
    export_parser.add_argument('instance_id', help='Strategy instance ID')
    export_parser.add_argument('--format', default='json', choices=['json'], 
                              help='Export format')
    
    args = parser.parse_args()
    
    # Initialize CLI
    cli = CryptoTradingCLI()
    
    # Handle commands
    if not args.command or args.command == 'interactive':
        cli.interactive_mode()
    elif args.command == 'list-strategies':
        cli.list_available_strategies()
    elif args.command == 'list':
        cli.list_strategies(args.status)
    elif args.command == 'create':
        config = json.loads(args.config) if args.config else None
        cli.create_strategy(args.strategy, args.base_asset, args.quote_asset, 
                           args.name, config)
    elif args.command == 'start':
        cli.start_strategy(args.instance_id)
    elif args.command == 'stop':
        cli.stop_strategy(args.instance_id)
    elif args.command == 'performance':
        cli.show_strategy_performance(args.instance_id)
    elif args.command == 'dashboard':
        cli.start_dashboard()
    elif args.command == 'export':
        cli.export_data(args.instance_id, args.format)


if __name__ == '__main__':
    main()