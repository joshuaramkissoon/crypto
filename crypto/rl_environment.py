import numpy as np
import pandas as pd
import gym
from gym import spaces
import talib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque


class TradingAction(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


@dataclass
class TradingState:
    price_data: np.ndarray
    technical_indicators: np.ndarray
    portfolio_state: np.ndarray
    market_info: np.ndarray


class CryptoTradingEnvironment(gym.Env):
    """
    Cryptocurrency trading environment for reinforcement learning.
    
    State Space:
    - Historical price data (OHLCV)
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Portfolio information (balance, position, PnL)
    - Market conditions
    
    Action Space:
    - 0: HOLD (no action)
    - 1: BUY (enter long position or increase position)
    - 2: SELL (exit position or enter short position)
    
    Reward Function:
    - Based on portfolio return, Sharpe ratio, and transaction costs
    - Risk-adjusted returns with penalties for large drawdowns
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        transaction_cost: float = 0.001,
        max_position_size: float = 1.0,
        lookback_window: int = 30,
        reward_scaling: float = 100.0,
        risk_free_rate: float = 0.02
    ):
        super().__init__()
        
        self.data = data.copy()
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        self.reward_scaling = reward_scaling
        self.risk_free_rate = risk_free_rate
        
        # Prepare data and indicators
        self._prepare_data()
        
        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # Current position size (-1 to 1)
        self.entry_price = 0.0
        self.portfolio_value = initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.max_portfolio_value = initial_balance
        self.portfolio_history = []
        self.trade_history = []
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL
        
        # Observation space: price data + indicators + portfolio state
        obs_size = (
            5 +  # OHLCV
            len(self.technical_indicators.columns) +  # Technical indicators
            4    # Portfolio state: balance, position, portfolio_value, unrealized_pnl
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window, obs_size),
            dtype=np.float32
        )
        
        logging.info(f"CryptoTradingEnvironment initialized with {len(self.data)} steps")
    
    def _prepare_data(self):
        """Prepare price data and calculate technical indicators."""
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Calculate technical indicators
        self.technical_indicators = self._calculate_technical_indicators()
        
        # Normalize data
        self.normalized_data = self._normalize_data()
        
        # Calculate valid start index (after lookback window and indicator calculation)
        self.start_step = max(self.lookback_window, 200)  # 200 for longest indicator period
        self.end_step = len(self.data) - 1
        
        logging.info(f"Data prepared: {self.start_step} to {self.end_step} ({self.end_step - self.start_step} valid steps)")
    
    def _calculate_technical_indicators(self) -> pd.DataFrame:
        """Calculate comprehensive technical indicators."""
        indicators = pd.DataFrame(index=self.data.index)
        
        # Price data
        high = self.data['high'].values
        low = self.data['low'].values
        close = self.data['close'].values
        volume = self.data['volume'].values
        
        # Moving averages
        indicators['sma_10'] = talib.SMA(close, timeperiod=10)
        indicators['sma_20'] = talib.SMA(close, timeperiod=20)
        indicators['sma_50'] = talib.SMA(close, timeperiod=50)
        indicators['ema_12'] = talib.EMA(close, timeperiod=12)
        indicators['ema_26'] = talib.EMA(close, timeperiod=26)
        
        # RSI
        indicators['rsi'] = talib.RSI(close, timeperiod=14)
        
        # MACD
        macd, signal, histogram = talib.MACD(close)
        indicators['macd'] = macd
        indicators['macd_signal'] = signal
        indicators['macd_histogram'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
        indicators['bb_upper'] = bb_upper
        indicators['bb_middle'] = bb_middle
        indicators['bb_lower'] = bb_lower
        indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        # Stochastic
        slowk, slowd = talib.STOCH(high, low, close)
        indicators['stoch_k'] = slowk
        indicators['stoch_d'] = slowd
        
        # ATR (Average True Range)
        indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)
        
        # ADX (Average Directional Index)
        indicators['adx'] = talib.ADX(high, low, close, timeperiod=14)
        
        # Williams %R
        indicators['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
        
        # Commodity Channel Index
        indicators['cci'] = talib.CCI(high, low, close, timeperiod=14)
        
        # Rate of Change
        indicators['roc'] = talib.ROC(close, timeperiod=10)
        
        # On Balance Volume
        indicators['obv'] = talib.OBV(close, volume)
        
        # Money Flow Index
        indicators['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
        
        # TRIX
        indicators['trix'] = talib.TRIX(close, timeperiod=14)
        
        # Fill NaN values with forward fill then backward fill
        indicators = indicators.fillna(method='ffill').fillna(method='bfill')
        
        return indicators
    
    def _normalize_data(self) -> pd.DataFrame:
        """Normalize price data and indicators for better RL training."""
        normalized = pd.DataFrame(index=self.data.index)
        
        # Normalize OHLCV using percentage changes
        for col in ['open', 'high', 'low', 'close']:
            normalized[col] = self.data[col].pct_change().fillna(0)
        
        # Log-normalize volume
        normalized['volume'] = np.log1p(self.data['volume'])
        
        # Normalize technical indicators
        for col in self.technical_indicators.columns:
            values = self.technical_indicators[col].values
            if col in ['rsi', 'stoch_k', 'stoch_d', 'williams_r']:
                # Indicators already in 0-100 range, normalize to -1 to 1
                normalized[col] = (values - 50) / 50
            elif col == 'mfi':
                normalized[col] = (values - 50) / 50
            else:
                # Z-score normalization
                mean_val = np.nanmean(values)
                std_val = np.nanstd(values)
                if std_val > 0:
                    normalized[col] = (values - mean_val) / std_val
                else:
                    normalized[col] = np.zeros_like(values)
        
        return normalized.fillna(0)
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_step = self.start_step
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.portfolio_value = self.initial_balance
        self.total_trades = 0
        self.winning_trades = 0
        self.max_portfolio_value = self.initial_balance
        self.portfolio_history = []
        self.trade_history = []
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        if self.current_step >= self.end_step:
            return self._get_observation(), 0, True, self._get_info()
        
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute action
        reward = self._execute_action(action, current_price)
        
        # Update portfolio value
        self._update_portfolio_value(current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = (
            self.current_step >= self.end_step or
            self.portfolio_value <= self.initial_balance * 0.1  # Stop loss at 90% loss
        )
        
        return self._get_observation(), reward, done, self._get_info()
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """Execute trading action and calculate reward."""
        prev_portfolio_value = self.portfolio_value
        trade_executed = False
        
        if action == TradingAction.BUY.value:
            # Buy action: enter long or increase long position
            if self.position < self.max_position_size:
                trade_size = min(
                    (self.max_position_size - self.position) * 0.5,  # Partial position sizing
                    self.balance / current_price * (1 - self.transaction_cost)
                )
                
                if trade_size > 0:
                    cost = trade_size * current_price * (1 + self.transaction_cost)
                    if cost <= self.balance:
                        self.balance -= cost
                        self.position += trade_size
                        self.entry_price = current_price if self.position == trade_size else \
                                         (self.entry_price * (self.position - trade_size) + current_price * trade_size) / self.position
                        self.total_trades += 1
                        trade_executed = True
                        
                        self.trade_history.append({
                            'step': self.current_step,
                            'action': 'BUY',
                            'price': current_price,
                            'size': trade_size,
                            'balance': self.balance,
                            'position': self.position
                        })
        
        elif action == TradingAction.SELL.value:
            # Sell action: close long position or enter short
            if self.position > 0:
                # Close long position
                proceeds = self.position * current_price * (1 - self.transaction_cost)
                self.balance += proceeds
                
                # Track winning trades
                if current_price > self.entry_price:
                    self.winning_trades += 1
                
                self.position = 0.0
                self.entry_price = 0.0
                self.total_trades += 1
                trade_executed = True
                
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'price': current_price,
                    'size': self.position,
                    'balance': self.balance,
                    'position': self.position
                })
        
        # Calculate reward
        return self._calculate_reward(prev_portfolio_value, trade_executed)
    
    def _update_portfolio_value(self, current_price: float):
        """Update portfolio value based on current position and balance."""
        position_value = self.position * current_price
        self.portfolio_value = self.balance + position_value
        self.portfolio_history.append(self.portfolio_value)
        
        # Update max portfolio value for drawdown calculation
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value
    
    def _calculate_reward(self, prev_portfolio_value: float, trade_executed: bool) -> float:
        """Calculate reward based on portfolio performance and risk metrics."""
        # Portfolio return
        portfolio_return = (self.portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Base reward from portfolio return
        reward = portfolio_return * self.reward_scaling
        
        # Risk-adjusted reward using Sharpe ratio approximation
        if len(self.portfolio_history) > 1:
            returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
            if len(returns) > 10:  # Calculate Sharpe ratio with sufficient data
                excess_returns = returns - self.risk_free_rate / 252  # Daily risk-free rate
                if np.std(excess_returns) > 0:
                    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
                    reward += sharpe_ratio * 0.1  # Small bonus for good risk-adjusted returns
        
        # Drawdown penalty
        if self.max_portfolio_value > 0:
            drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            if drawdown > 0.1:  # Penalty for large drawdowns
                reward -= drawdown * 10
        
        # Transaction cost penalty
        if trade_executed:
            reward -= 0.01  # Small penalty for each trade to avoid overtrading
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        # Get price data window
        start_idx = max(0, self.current_step - self.lookback_window + 1)
        end_idx = self.current_step + 1
        
        # Price data (normalized)
        price_window = self.normalized_data.iloc[start_idx:end_idx][
            ['open', 'high', 'low', 'close', 'volume']
        ].values
        
        # Technical indicators (normalized)
        indicator_window = self.normalized_data.iloc[start_idx:end_idx][
            self.technical_indicators.columns
        ].values
        
        # Current price for portfolio calculations
        current_price = self.data.iloc[self.current_step]['close']
        
        # Portfolio state
        normalized_balance = self.balance / self.initial_balance - 1
        normalized_position = self.position / self.max_position_size
        normalized_portfolio = self.portfolio_value / self.initial_balance - 1
        unrealized_pnl = (current_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0
        
        portfolio_state = np.array([
            normalized_balance,
            normalized_position,
            normalized_portfolio,
            unrealized_pnl
        ])
        
        # Combine all features
        obs_window = np.concatenate([price_window, indicator_window], axis=1)
        
        # Ensure we have the correct window size
        if obs_window.shape[0] < self.lookback_window:
            # Pad with zeros if we don't have enough history
            padding = np.zeros((self.lookback_window - obs_window.shape[0], obs_window.shape[1]))
            obs_window = np.vstack([padding, obs_window])
        
        # Add portfolio state to each timestep
        portfolio_expanded = np.tile(portfolio_state, (self.lookback_window, 1))
        observation = np.concatenate([obs_window, portfolio_expanded], axis=1)
        
        return observation.astype(np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info for debugging and monitoring."""
        win_rate = self.winning_trades / max(1, self.total_trades)
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        
        drawdown = 0
        if self.max_portfolio_value > 0:
            drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        
        return {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'balance': self.balance,
            'position': self.position,
            'total_return': total_return,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'max_drawdown': drawdown,
            'trade_history': self.trade_history[-10:] if self.trade_history else []  # Last 10 trades
        }
    
    def render(self, mode='human'):
        """Render environment state."""
        if mode == 'human':
            info = self._get_info()
            print(f"Step: {info['step']}, Portfolio: ${info['portfolio_value']:.2f}, "
                  f"Return: {info['total_return']:.2%}, Trades: {info['total_trades']}, "
                  f"Win Rate: {info['win_rate']:.2%}")
    
    def get_portfolio_history(self) -> List[float]:
        """Get historical portfolio values."""
        return self.portfolio_history.copy()
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get complete trade history."""
        return self.trade_history.copy()