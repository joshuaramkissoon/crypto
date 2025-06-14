import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import pickle
from crypto.strategy import Strategy
from crypto.rl_environment import CryptoTradingEnvironment
from crypto.rl_agents import RLTradingAgent
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    initial_capital: float = 10000.0
    commission: float = 0.001
    slippage: float = 0.0001
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    benchmark_symbol: str = "BTC"
    risk_free_rate: float = 0.02


@dataclass
class TradeRecord:
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    commission: float
    pnl: float
    portfolio_value: float


@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    calmar_ratio: float
    sortino_ratio: float
    var_95: float  # Value at Risk at 95% confidence
    cvar_95: float  # Conditional Value at Risk


class PortfolioTracker:
    """Track portfolio state during backtesting."""
    
    def __init__(self, initial_capital: float, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.cash = initial_capital
        self.positions = {}  # symbol -> quantity
        self.portfolio_history = []
        self.trade_history = []
        self.timestamps = []
        
    def execute_trade(
        self,
        timestamp: datetime,
        symbol: str,
        side: str,
        quantity: float,
        price: float
    ) -> bool:
        """Execute a trade and update portfolio."""
        trade_value = quantity * price
        commission_cost = trade_value * self.commission
        
        if side == "BUY":
            total_cost = trade_value + commission_cost
            if total_cost <= self.cash:
                self.cash -= total_cost
                self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                
                trade_record = TradeRecord(
                    timestamp=timestamp,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    commission=commission_cost,
                    pnl=0,  # Will be calculated on sell
                    portfolio_value=self.get_portfolio_value({symbol: price})
                )
                self.trade_history.append(trade_record)
                return True
            return False
        
        elif side == "SELL":
            if symbol in self.positions and self.positions[symbol] >= quantity:
                self.positions[symbol] -= quantity
                proceeds = trade_value - commission_cost
                self.cash += proceeds
                
                # Calculate PnL (simplified - assumes FIFO)
                pnl = proceeds - (quantity * price)  # Simplified
                
                trade_record = TradeRecord(
                    timestamp=timestamp,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    commission=commission_cost,
                    pnl=pnl,
                    portfolio_value=self.get_portfolio_value({symbol: price})
                )
                self.trade_history.append(trade_record)
                return True
            return False
        
        return False
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate current portfolio value."""
        position_value = sum(
            self.positions.get(symbol, 0) * price
            for symbol, price in prices.items()
        )
        return self.cash + position_value
    
    def update_portfolio_history(self, timestamp: datetime, prices: Dict[str, float]):
        """Update portfolio value history."""
        portfolio_value = self.get_portfolio_value(prices)
        self.portfolio_history.append(portfolio_value)
        self.timestamps.append(timestamp)


class BacktestEngine:
    """
    Comprehensive backtesting engine for both traditional strategies and RL agents.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results_cache = {}
        
        logging.info("BacktestEngine initialized")
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Prepare and validate data for backtesting."""
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' in data.columns:
                data.set_index('timestamp', inplace=True)
            elif 'date' in data.columns:
                data.set_index('date', inplace=True)
            else:
                raise ValueError("Data must have datetime index or timestamp column")
        
        # Filter by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add symbol column
        data['symbol'] = symbol
        
        logging.info(f"Data prepared: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
        return data
    
    def backtest_traditional_strategy(
        self,
        strategy_class,
        data: pd.DataFrame,
        symbol: str,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Backtest a traditional trading strategy."""
        # Prepare data
        data = self.prepare_data(data, symbol, self.config.start_date, self.config.end_date)
        
        # Initialize portfolio tracker
        portfolio = PortfolioTracker(self.config.initial_capital, self.config.commission)
        
        # Mock client and session for strategy
        class MockClient:
            pass
        
        class MockSession:
            def handle_order(self, order):
                return {'average_price': order['price'], 'net': order['quantity'] * order['price']}
        
        # Initialize strategy
        if strategy_params is None:
            strategy_params = {}
        
        strategy = strategy_class(MockClient(), MockSession(), None)
        
        # Apply strategy parameters
        for param, value in strategy_params.items():
            if hasattr(strategy, param):
                setattr(strategy, param, value)
        
        # Run backtest
        signals = []
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Create mock data format for strategy
            mock_data = {
                't': int(timestamp.timestamp() * 1000),
                'T': int(timestamp.timestamp() * 1000) + 60000,
                's': symbol,
                'i': '1m',
                'o': str(row['open']),
                'c': str(row['close']),
                'h': str(row['high']),
                'l': str(row['low']),
                'v': str(row['volume']),
                'x': True  # Assume candle is closed
            }
            
            # Store original order method
            original_order = strategy.order
            trades_executed = []
            
            def mock_order(side, quantity, symbol_arg, **kwargs):
                trade = {
                    'side': side,
                    'quantity': quantity,
                    'symbol': symbol_arg,
                    'price': row['close'],
                    'timestamp': timestamp
                }
                trades_executed.append(trade)
                return True, {'average_price': row['close'], 'net': quantity * row['close']}, None
            
            strategy.order = mock_order
            
            # Run strategy
            try:
                strategy.trading_strategy(symbol, mock_data)
            except Exception as e:
                logging.warning(f"Strategy error at {timestamp}: {e}")
            
            # Execute trades
            for trade in trades_executed:
                portfolio.execute_trade(
                    timestamp=trade['timestamp'],
                    symbol=trade['symbol'],
                    side=trade['side'],
                    quantity=trade['quantity'],
                    price=trade['price']
                )
            
            # Update portfolio history
            portfolio.update_portfolio_history(timestamp, {symbol: row['close']})
            
            # Record signals
            if trades_executed:
                signals.extend(trades_executed)
            
            # Restore original order method
            strategy.order = original_order
        
        # Calculate performance metrics
        performance = self._calculate_performance_metrics(
            portfolio, data, symbol
        )
        
        return {
            'performance': performance,
            'portfolio_history': portfolio.portfolio_history,
            'trade_history': [asdict(trade) for trade in portfolio.trade_history],
            'signals': signals,
            'timestamps': portfolio.timestamps,
            'strategy_name': strategy_class.__name__,
            'config': asdict(self.config)
        }
    
    def backtest_rl_agent(
        self,
        agent: RLTradingAgent,
        data: pd.DataFrame,
        symbol: str,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """Backtest a trained RL agent."""
        # Prepare data
        data = self.prepare_data(data, symbol, self.config.start_date, self.config.end_date)
        
        # Create trading environment
        env = CryptoTradingEnvironment(
            data=data,
            initial_balance=self.config.initial_capital,
            transaction_cost=self.config.commission
        )
        
        # Run episode
        obs = env.reset()
        done = False
        actions = []
        rewards = []
        portfolio_values = []
        
        while not done:
            action = agent.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            
            actions.append(action)
            rewards.append(reward)
            portfolio_values.append(info['portfolio_value'])
        
        # Get final results
        trade_history = env.get_trade_history()
        portfolio_history = env.get_portfolio_history()
        
        # Calculate performance metrics using portfolio history
        performance = self._calculate_rl_performance_metrics(
            portfolio_history, trade_history, data, env.initial_balance
        )
        
        return {
            'performance': performance,
            'portfolio_history': portfolio_history,
            'trade_history': trade_history,
            'actions': actions,
            'rewards': rewards,
            'timestamps': data.index.tolist(),
            'agent_name': agent.model_name,
            'algorithm': agent.config.algorithm,
            'config': asdict(self.config)
        }
    
    def _calculate_performance_metrics(
        self,
        portfolio: PortfolioTracker,
        data: pd.DataFrame,
        symbol: str
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        portfolio_values = np.array(portfolio.portfolio_history)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Basic metrics
        total_return = (portfolio_values[-1] - portfolio.initial_capital) / portfolio.initial_capital
        
        # Annualized return
        days = len(data)
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        
        # Sharpe ratio
        excess_returns = returns - self.config.risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown)
        
        # Trade statistics
        trades = portfolio.trade_history
        winning_trades = sum(1 for trade in trades if trade.side == 'SELL' and trade.pnl > 0)
        total_trades = len([t for t in trades if t.side == 'SELL'])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(trade.pnl for trade in trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in trades if trade.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.config.risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns) > 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_duration=0,  # TODO: Calculate
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            cvar_95=cvar_95
        )
    
    def _calculate_rl_performance_metrics(
        self,
        portfolio_history: List[float],
        trade_history: List[Dict[str, Any]],
        data: pd.DataFrame,
        initial_capital: float
    ) -> PerformanceMetrics:
        """Calculate performance metrics for RL agent."""
        portfolio_values = np.array(portfolio_history)
        if len(portfolio_values) == 0:
            portfolio_values = np.array([initial_capital])
        
        returns = np.diff(portfolio_values) / portfolio_values[:-1] if len(portfolio_values) > 1 else np.array([0])
        
        # Basic metrics
        total_return = (portfolio_values[-1] - initial_capital) / initial_capital
        
        # Annualized return
        days = len(data)
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        
        # Sharpe ratio
        excess_returns = returns - self.config.risk_free_rate / 252
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Trade statistics
        buy_trades = [t for t in trade_history if t.get('action') == 'BUY']
        sell_trades = [t for t in trade_history if t.get('action') == 'SELL']
        total_trades = len(buy_trades) + len(sell_trades)
        
        # Win rate (simplified)
        win_rate = 0.5  # Default, would need more sophisticated calculation
        profit_factor = 1.0  # Default
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.config.risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = np.mean(returns[returns <= var_95]) if len(returns) > 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            avg_trade_duration=0,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            var_95=var_95,
            cvar_95=cvar_95
        )
    
    def compare_strategies(
        self,
        results: List[Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """Compare performance of multiple strategies."""
        comparison_data = []
        
        for result in results:
            performance = result['performance']
            strategy_name = result.get('strategy_name', result.get('agent_name', 'Unknown'))
            
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return': performance.total_return,
                'Annualized Return': performance.annualized_return,
                'Volatility': performance.volatility,
                'Sharpe Ratio': performance.sharpe_ratio,
                'Max Drawdown': performance.max_drawdown,
                'Win Rate': performance.win_rate,
                'Profit Factor': performance.profit_factor,
                'Total Trades': performance.total_trades,
                'Calmar Ratio': performance.calmar_ratio,
                'Sortino Ratio': performance.sortino_ratio
            })
        
        df = pd.DataFrame(comparison_data)
        
        if save_path:
            df.to_csv(save_path, index=False)
            logging.info(f"Strategy comparison saved to {save_path}")
        
        return df
    
    def plot_results(
        self,
        results: Union[Dict[str, Any], List[Dict[str, Any]]],
        save_path: Optional[str] = None
    ):
        """Plot backtest results."""
        if isinstance(results, dict):
            results = [results]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        for result in results:
            portfolio_history = result['portfolio_history']
            timestamps = result.get('timestamps', range(len(portfolio_history)))
            strategy_name = result.get('strategy_name', result.get('agent_name', 'Strategy'))
            
            axes[0, 0].plot(timestamps, portfolio_history, label=strategy_name)
        
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Returns distribution
        for result in results:
            portfolio_history = np.array(result['portfolio_history'])
            if len(portfolio_history) > 1:
                returns = np.diff(portfolio_history) / portfolio_history[:-1]
                strategy_name = result.get('strategy_name', result.get('agent_name', 'Strategy'))
                axes[0, 1].hist(returns, bins=50, alpha=0.7, label=strategy_name)
        
        axes[0, 1].set_title('Returns Distribution')
        axes[0, 1].set_xlabel('Returns')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Drawdown
        for result in results:
            portfolio_history = np.array(result['portfolio_history'])
            peak = np.maximum.accumulate(portfolio_history)
            drawdown = (peak - portfolio_history) / peak
            timestamps = result.get('timestamps', range(len(portfolio_history)))
            strategy_name = result.get('strategy_name', result.get('agent_name', 'Strategy'))
            
            axes[1, 0].plot(timestamps, -drawdown * 100, label=strategy_name)
        
        axes[1, 0].set_title('Drawdown Over Time')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Performance metrics comparison
        metrics_data = []
        for result in results:
            performance = result['performance']
            strategy_name = result.get('strategy_name', result.get('agent_name', 'Strategy'))
            
            metrics_data.append({
                'Strategy': strategy_name,
                'Sharpe': performance.sharpe_ratio,
                'Calmar': performance.calmar_ratio,
                'Sortino': performance.sortino_ratio
            })
        
        if metrics_data:
            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.set_index('Strategy').plot(kind='bar', ax=axes[1, 1])
            axes[1, 1].set_title('Risk-Adjusted Performance Metrics')
            axes[1, 1].set_ylabel('Ratio')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Results plot saved to {save_path}")
        else:
            plt.show()
    
    def save_results(
        self,
        results: Dict[str, Any],
        filename: str,
        format: str = 'json'
    ):
        """Save backtest results to file."""
        # Convert numpy arrays and datetime objects to serializable format
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return asdict(obj)
            return obj
        
        if format.lower() == 'json':
            serializable_results = json.loads(
                json.dumps(results, default=convert_for_json)
            )
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
        
        elif format.lower() == 'pickle':
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logging.info(f"Results saved to {filename}")
    
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load backtest results from file."""
        if filename.endswith('.json'):
            with open(filename, 'r') as f:
                return json.load(f)
        elif filename.endswith('.pkl') or filename.endswith('.pickle'):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filename}")


class WalkForwardAnalysis:
    """
    Walk-forward analysis for robust strategy validation.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        train_period: int = 252,  # 1 year
        test_period: int = 63,   # 3 months
        step_size: int = 21      # 3 weeks
    ):
        self.data = data
        self.train_period = train_period
        self.test_period = test_period
        self.step_size = step_size
        
    def run_analysis(
        self,
        strategy_func: Callable,
        symbol: str,
        config: BacktestConfig
    ) -> List[Dict[str, Any]]:
        """Run walk-forward analysis."""
        results = []
        engine = BacktestEngine(config)
        
        start_idx = self.train_period
        
        while start_idx + self.test_period < len(self.data):
            # Define train and test periods
            train_start = start_idx - self.train_period
            train_end = start_idx
            test_start = start_idx
            test_end = start_idx + self.test_period
            
            train_data = self.data.iloc[train_start:train_end]
            test_data = self.data.iloc[test_start:test_end]
            
            # Run strategy on test data
            result = strategy_func(train_data, test_data, engine, symbol)
            result['train_period'] = (train_data.index[0], train_data.index[-1])
            result['test_period'] = (test_data.index[0], test_data.index[-1])
            
            results.append(result)
            
            # Move forward
            start_idx += self.step_size
        
        return results