"""
Risk management module for algorithmic trading strategies.
Provides position sizing, stop-loss management, and portfolio risk controls.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import statistics


@dataclass
class RiskMetrics:
    """Risk metrics for a trading strategy or portfolio."""
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    var_95: float  # Value at Risk 95%
    volatility: float
    max_position_size: float
    total_exposure: float
    risk_score: float  # 0-100 scale


@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    recommended_size: float
    max_allowed_size: float
    risk_adjusted_size: float
    confidence_level: float
    warnings: List[str]


class RiskManager:
    """
    Comprehensive risk management system for algorithmic trading.
    Handles position sizing, stop-loss management, and portfolio-level risk controls.
    """
    
    def __init__(self, 
                 max_portfolio_risk: float = 0.02,
                 max_position_risk: float = 0.01,
                 max_drawdown_limit: float = 0.15,
                 var_confidence: float = 0.95):
        """
        Initialize risk manager.
        
        Args:
            max_portfolio_risk: Maximum portfolio risk per trade (2%)
            max_position_risk: Maximum position risk per trade (1%) 
            max_drawdown_limit: Maximum allowed drawdown (15%)
            var_confidence: VaR confidence level (95%)
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_drawdown_limit = max_drawdown_limit
        self.var_confidence = var_confidence
        
        # Track positions and performance
        self.positions: Dict[str, Dict] = {}
        self.portfolio_history: List[Dict] = []
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        
        # Risk limits
        self.position_limits: Dict[str, float] = {}
        self.strategy_limits: Dict[str, Dict] = {}
        
        logging.info("RiskManager initialized")
    
    def calculate_position_size(self,
                              strategy_id: str,
                              symbol: str,
                              entry_price: float,
                              stop_loss_price: float,
                              account_balance: float,
                              confidence: float = 1.0) -> PositionSizeResult:
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            entry_price: Planned entry price
            stop_loss_price: Stop loss price
            account_balance: Current account balance
            confidence: Strategy confidence level (0-1)
            
        Returns:
            PositionSizeResult with recommended position size
        """
        warnings = []
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        risk_percent = risk_per_share / entry_price
        
        # Portfolio risk-based sizing
        portfolio_risk_amount = account_balance * self.max_portfolio_risk
        portfolio_size = portfolio_risk_amount / risk_per_share
        
        # Position risk-based sizing
        position_risk_amount = account_balance * self.max_position_risk
        position_size = position_risk_amount / risk_per_share
        
        # Strategy performance-based adjustment
        strategy_multiplier = self._get_strategy_multiplier(strategy_id)
        
        # Confidence-based adjustment
        confidence_multiplier = min(confidence, 1.0)
        
        # Calculate recommended size
        base_size = min(portfolio_size, position_size)
        risk_adjusted_size = base_size * strategy_multiplier * confidence_multiplier
        
        # Apply hard limits
        max_allowed = self._get_max_position_size(strategy_id, symbol, account_balance)
        final_size = min(risk_adjusted_size, max_allowed)
        
        # Check for warnings
        if risk_percent > 0.05:  # 5% risk per trade
            warnings.append(f"High risk per share: {risk_percent:.2%}")
        
        if final_size < base_size * 0.5:
            warnings.append("Position size significantly reduced due to risk limits")
        
        if confidence < 0.7:
            warnings.append(f"Low confidence level: {confidence:.1%}")
        
        return PositionSizeResult(
            recommended_size=final_size,
            max_allowed_size=max_allowed,
            risk_adjusted_size=risk_adjusted_size,
            confidence_level=confidence,
            warnings=warnings
        )
    
    def check_position_risk(self,
                           strategy_id: str,
                           symbol: str,
                           position_size: float,
                           entry_price: float,
                           current_price: float) -> Tuple[bool, List[str]]:
        """
        Check if a position violates risk limits.
        
        Returns:
            Tuple of (is_within_limits, warnings)
        """
        warnings = []
        is_within_limits = True
        
        # Calculate current risk
        position_value = position_size * current_price
        unrealized_pnl = position_size * (current_price - entry_price)
        
        # Check position size limits
        max_position_value = self._get_max_position_value(strategy_id, symbol)
        if position_value > max_position_value:
            warnings.append(f"Position size exceeds limit: ${position_value:.2f} > ${max_position_value:.2f}")
            is_within_limits = False
        
        # Check drawdown limits
        if unrealized_pnl < 0:
            drawdown_percent = abs(unrealized_pnl) / position_value
            if drawdown_percent > self.max_position_risk * 2:  # 2x normal risk
                warnings.append(f"Position drawdown: {drawdown_percent:.2%}")
                is_within_limits = False
        
        return is_within_limits, warnings
    
    def update_position(self,
                       strategy_id: str,
                       symbol: str,
                       position_size: float,
                       entry_price: float,
                       current_price: float,
                       timestamp: datetime = None):
        """Update position information for risk tracking."""
        if timestamp is None:
            timestamp = datetime.now()
        
        position_key = f"{strategy_id}_{symbol}"
        
        self.positions[position_key] = {
            'strategy_id': strategy_id,
            'symbol': symbol,
            'position_size': position_size,
            'entry_price': entry_price,
            'current_price': current_price,
            'unrealized_pnl': position_size * (current_price - entry_price),
            'timestamp': timestamp
        }
    
    def close_position(self,
                      strategy_id: str,
                      symbol: str,
                      exit_price: float,
                      timestamp: datetime = None) -> float:
        """
        Close a position and record the realized P&L.
        
        Returns:
            Realized P&L
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        position_key = f"{strategy_id}_{symbol}"
        
        if position_key not in self.positions:
            logging.warning(f"Position {position_key} not found for closing")
            return 0.0
        
        position = self.positions[position_key]
        realized_pnl = position['position_size'] * (exit_price - position['entry_price'])
        
        # Record performance for strategy
        self.strategy_performance[strategy_id].append(realized_pnl)
        
        # Remove position
        del self.positions[position_key]
        
        logging.info(f"Closed position {position_key} with P&L: ${realized_pnl:.2f}")
        return realized_pnl
    
    def calculate_portfolio_risk(self) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics."""
        if not self.positions and not self.strategy_performance:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        # Calculate current drawdown
        total_unrealized = sum(pos['unrealized_pnl'] for pos in self.positions.values())
        total_exposure = sum(abs(pos['position_size'] * pos['current_price']) 
                           for pos in self.positions.values())
        
        # Calculate historical metrics
        all_returns = []
        for strategy_returns in self.strategy_performance.values():
            all_returns.extend(strategy_returns)
        
        if not all_returns:
            return RiskMetrics(0, 0, 0, 0, 0, 0, total_exposure, 0)
        
        # Max drawdown calculation
        cumulative_returns = []
        running_sum = 0
        for ret in all_returns:
            running_sum += ret
            cumulative_returns.append(running_sum)
        
        peak = cumulative_returns[0]
        max_drawdown = 0
        for value in cumulative_returns:
            if value > peak:
                peak = value
            drawdown = peak - value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Current drawdown
        current_peak = max(cumulative_returns) if cumulative_returns else 0
        current_value = cumulative_returns[-1] if cumulative_returns else 0
        current_drawdown = max(0, current_peak - current_value)
        
        # Volatility and Sharpe ratio
        volatility = statistics.stdev(all_returns) if len(all_returns) > 1 else 0
        mean_return = statistics.mean(all_returns) if all_returns else 0
        sharpe_ratio = (mean_return / volatility) if volatility > 0 else 0
        
        # VaR calculation (simplified)
        sorted_returns = sorted(all_returns)
        var_index = int(len(sorted_returns) * (1 - self.var_confidence))
        var_95 = abs(sorted_returns[var_index]) if var_index < len(sorted_returns) else 0
        
        # Risk score (0-100)
        risk_score = min(100, (current_drawdown / 1000) * 50 + (volatility * 1000) * 30 + (var_95 / 100) * 20)
        
        return RiskMetrics(
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe_ratio,
            var_95=var_95,
            volatility=volatility,
            max_position_size=max(abs(pos['position_size'] * pos['current_price']) 
                                for pos in self.positions.values()) if self.positions else 0,
            total_exposure=total_exposure,
            risk_score=risk_score
        )
    
    def should_halt_trading(self, strategy_id: str) -> Tuple[bool, str]:
        """
        Determine if trading should be halted for a strategy.
        
        Returns:
            Tuple of (should_halt, reason)
        """
        # Check strategy-specific performance
        if strategy_id in self.strategy_performance:
            recent_trades = self.strategy_performance[strategy_id][-10:]  # Last 10 trades
            
            if len(recent_trades) >= 5:
                # Check for excessive losses
                losing_trades = sum(1 for trade in recent_trades if trade < 0)
                if losing_trades >= 8:  # 8 out of last 10 trades
                    return True, "Excessive losing trades"
                
                # Check for large drawdown
                total_recent_pnl = sum(recent_trades)
                if total_recent_pnl < -1000:  # $1000 loss
                    return True, "Large recent losses"
        
        # Check portfolio-level risk
        portfolio_risk = self.calculate_portfolio_risk()
        
        if portfolio_risk.current_drawdown > self.max_drawdown_limit * 1000:
            return True, f"Portfolio drawdown exceeded: ${portfolio_risk.current_drawdown:.2f}"
        
        if portfolio_risk.risk_score > 80:
            return True, f"High risk score: {portfolio_risk.risk_score:.1f}"
        
        return False, ""
    
    def _get_strategy_multiplier(self, strategy_id: str) -> float:
        """Get performance-based multiplier for strategy position sizing."""
        if strategy_id not in self.strategy_performance:
            return 0.5  # Conservative for new strategies
        
        recent_trades = self.strategy_performance[strategy_id][-20:]  # Last 20 trades
        
        if len(recent_trades) < 5:
            return 0.5
        
        # Calculate win rate
        winning_trades = sum(1 for trade in recent_trades if trade > 0)
        win_rate = winning_trades / len(recent_trades)
        
        # Calculate average return
        avg_return = statistics.mean(recent_trades)
        
        # Calculate multiplier based on performance
        multiplier = 0.5  # Base multiplier
        
        if win_rate > 0.6:  # Good win rate
            multiplier += 0.3
        elif win_rate > 0.5:
            multiplier += 0.1
        
        if avg_return > 0:  # Profitable
            multiplier += 0.2
        
        return min(multiplier, 1.0)  # Cap at 1.0
    
    def _get_max_position_size(self, strategy_id: str, symbol: str, account_balance: float) -> float:
        """Get maximum allowed position size."""
        # Default limit: 10% of account balance
        default_limit = account_balance * 0.1
        
        # Strategy-specific limits
        if strategy_id in self.strategy_limits:
            strategy_limit = self.strategy_limits[strategy_id].get('max_position_size', default_limit)
            return min(default_limit, strategy_limit)
        
        return default_limit
    
    def _get_max_position_value(self, strategy_id: str, symbol: str) -> float:
        """Get maximum allowed position value."""
        return self.position_limits.get(f"{strategy_id}_{symbol}", 10000)  # $10k default
    
    def set_strategy_limits(self, strategy_id: str, limits: Dict[str, Any]):
        """Set custom limits for a strategy."""
        self.strategy_limits[strategy_id] = limits
        logging.info(f"Updated limits for strategy {strategy_id}: {limits}")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report."""
        portfolio_risk = self.calculate_portfolio_risk()
        
        # Position summary
        position_summary = []
        for pos_key, position in self.positions.items():
            position_summary.append({
                'key': pos_key,
                'symbol': position['symbol'],
                'size': position['position_size'],
                'value': position['position_size'] * position['current_price'],
                'unrealized_pnl': position['unrealized_pnl'],
                'entry_price': position['entry_price'],
                'current_price': position['current_price']
            })
        
        # Strategy performance summary
        strategy_summary = {}
        for strategy_id, trades in self.strategy_performance.items():
            if trades:
                strategy_summary[strategy_id] = {
                    'total_trades': len(trades),
                    'total_pnl': sum(trades),
                    'avg_pnl': statistics.mean(trades),
                    'win_rate': sum(1 for t in trades if t > 0) / len(trades),
                    'best_trade': max(trades),
                    'worst_trade': min(trades)
                }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_risk': {
                'max_drawdown': portfolio_risk.max_drawdown,
                'current_drawdown': portfolio_risk.current_drawdown,
                'sharpe_ratio': portfolio_risk.sharpe_ratio,
                'volatility': portfolio_risk.volatility,
                'total_exposure': portfolio_risk.total_exposure,
                'risk_score': portfolio_risk.risk_score
            },
            'positions': position_summary,
            'strategy_performance': strategy_summary,
            'risk_limits': {
                'max_portfolio_risk': self.max_portfolio_risk,
                'max_position_risk': self.max_position_risk,
                'max_drawdown_limit': self.max_drawdown_limit
            }
        }