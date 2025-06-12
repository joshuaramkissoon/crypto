import logging
import time
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import statistics
from threading import Lock
import psutil
import traceback


@dataclass
class TradeMetrics:
    """Metrics for individual trades."""
    timestamp: datetime
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    profit_loss: float
    strategy_id: str


@dataclass
class StrategyMetrics:
    """Comprehensive metrics for a strategy."""
    strategy_id: str
    strategy_name: str
    symbol: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    runtime_seconds: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Financial metrics
    total_profit_loss: float
    total_commission: float
    net_profit_loss: float
    max_drawdown: float
    return_on_investment: float
    sharpe_ratio: Optional[float]
    
    # Risk metrics
    max_position_size: float
    avg_position_size: float
    volatility: float
    
    # Performance metrics
    avg_trade_duration: float
    trades_per_hour: float
    
    # System metrics
    cpu_usage: float
    memory_usage: float
    error_count: int
    last_error: Optional[str]


class MetricsCollector:
    """Collects and stores comprehensive metrics for all trading strategies."""
    
    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self.lock = Lock()
        
        # Storage for metrics
        self.trade_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.strategy_metrics: Dict[str, StrategyMetrics] = {}
        self.system_metrics: Dict[str, Any] = {}
        self.error_logs: Dict[str, List[Dict]] = defaultdict(list)
        
        # Real-time data
        self.active_positions: Dict[str, Dict] = {}
        self.price_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Performance tracking
        self.start_time = datetime.now()
        self.last_metrics_update = {}
        
        logging.info("MetricsCollector initialized")
    
    def record_trade(self, strategy_id: str, trade_data: Dict[str, Any]):
        """Record a completed trade."""
        with self.lock:
            trade_metrics = TradeMetrics(
                timestamp=datetime.now(),
                symbol=trade_data.get('symbol', ''),
                side=trade_data.get('side', ''),
                quantity=float(trade_data.get('quantity', 0)),
                price=float(trade_data.get('price', 0)),
                commission=float(trade_data.get('commission', 0)),
                profit_loss=float(trade_data.get('profit_loss', 0)),
                strategy_id=strategy_id
            )
            
            self.trade_history[strategy_id].append(trade_metrics)
            self._update_strategy_metrics(strategy_id)
    
    def record_price_tick(self, symbol: str, price_data: Dict[str, Any]):
        """Record price tick data for analysis."""
        with self.lock:
            tick_data = {
                'timestamp': datetime.now(),
                'open': float(price_data.get('o', 0)),
                'high': float(price_data.get('h', 0)),
                'low': float(price_data.get('l', 0)),
                'close': float(price_data.get('c', 0)),
                'volume': float(price_data.get('v', 0))
            }
            self.price_data[symbol].append(tick_data)
    
    def record_error(self, strategy_id: str, error_message: str, error_type: str = "STRATEGY_ERROR"):
        """Record an error for a strategy."""
        with self.lock:
            error_record = {
                'timestamp': datetime.now(),
                'strategy_id': strategy_id,
                'error_type': error_type,
                'message': error_message,
                'traceback': traceback.format_exc()
            }
            self.error_logs[strategy_id].append(error_record)
            
            # Keep only last 100 errors per strategy
            if len(self.error_logs[strategy_id]) > 100:
                self.error_logs[strategy_id] = self.error_logs[strategy_id][-100:]
    
    def update_position(self, strategy_id: str, position_data: Dict[str, Any]):
        """Update current position for a strategy."""
        with self.lock:
            self.active_positions[strategy_id] = {
                'timestamp': datetime.now(),
                'symbol': position_data.get('symbol', ''),
                'side': position_data.get('side', ''),
                'size': float(position_data.get('size', 0)),
                'entry_price': float(position_data.get('entry_price', 0)),
                'current_price': float(position_data.get('current_price', 0)),
                'unrealized_pnl': float(position_data.get('unrealized_pnl', 0))
            }
    
    def _update_strategy_metrics(self, strategy_id: str):
        """Update comprehensive metrics for a strategy."""
        if strategy_id not in self.trade_history:
            return
        
        trades = list(self.trade_history[strategy_id])
        if not trades:
            return
        
        # Calculate trade statistics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.profit_loss > 0)
        losing_trades = sum(1 for t in trades if t.profit_loss < 0)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Financial metrics
        total_profit_loss = sum(t.profit_loss for t in trades)
        total_commission = sum(t.commission for t in trades)
        net_profit_loss = total_profit_loss - total_commission
        
        # Calculate drawdown
        running_pnl = 0
        peak = 0
        max_drawdown = 0
        for trade in trades:
            running_pnl += trade.profit_loss
            if running_pnl > peak:
                peak = running_pnl
            drawdown = peak - running_pnl
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Position size metrics
        position_sizes = [abs(t.quantity * t.price) for t in trades]
        max_position_size = max(position_sizes) if position_sizes else 0
        avg_position_size = statistics.mean(position_sizes) if position_sizes else 0
        
        # Calculate volatility
        returns = []
        for i in range(1, len(trades)):
            if trades[i-1].profit_loss != 0:
                returns.append(trades[i].profit_loss / abs(trades[i-1].profit_loss))
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0
        
        # Performance metrics
        if len(trades) >= 2:
            total_time = (trades[-1].timestamp - trades[0].timestamp).total_seconds()
            avg_trade_duration = total_time / len(trades)
            trades_per_hour = (len(trades) / (total_time / 3600)) if total_time > 0 else 0
        else:
            avg_trade_duration = 0
            trades_per_hour = 0
        
        # System metrics
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        # Error count
        error_count = len(self.error_logs.get(strategy_id, []))
        last_error = self.error_logs[strategy_id][-1]['message'] if error_count > 0 else None
        
        # ROI calculation (simplified)
        initial_capital = abs(trades[0].quantity * trades[0].price) if trades else 1000
        return_on_investment = (net_profit_loss / initial_capital) * 100 if initial_capital > 0 else 0
        
        # Sharpe ratio (simplified daily calculation)
        daily_returns = self._calculate_daily_returns(trades)
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns) if daily_returns else None
        
        # Update or create strategy metrics
        self.strategy_metrics[strategy_id] = StrategyMetrics(
            strategy_id=strategy_id,
            strategy_name=trades[0].strategy_id if trades else strategy_id,
            symbol=trades[0].symbol if trades else "",
            status="RUNNING",  # This should be updated from strategy manager
            start_time=trades[0].timestamp if trades else datetime.now(),
            end_time=None,
            runtime_seconds=(datetime.now() - trades[0].timestamp).total_seconds() if trades else 0,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_profit_loss=total_profit_loss,
            total_commission=total_commission,
            net_profit_loss=net_profit_loss,
            max_drawdown=max_drawdown,
            return_on_investment=return_on_investment,
            sharpe_ratio=sharpe_ratio,
            max_position_size=max_position_size,
            avg_position_size=avg_position_size,
            volatility=volatility,
            avg_trade_duration=avg_trade_duration,
            trades_per_hour=trades_per_hour,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            error_count=error_count,
            last_error=last_error
        )
    
    def _calculate_daily_returns(self, trades: List[TradeMetrics]) -> List[float]:
        """Calculate daily returns from trades."""
        if len(trades) < 2:
            return []
        
        daily_pnl = defaultdict(float)
        for trade in trades:
            date_key = trade.timestamp.date()
            daily_pnl[date_key] += trade.profit_loss
        
        return list(daily_pnl.values())
    
    def _calculate_sharpe_ratio(self, daily_returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from daily returns."""
        if len(daily_returns) < 2:
            return 0.0
        
        mean_return = statistics.mean(daily_returns)
        std_return = statistics.stdev(daily_returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio
        daily_risk_free = risk_free_rate / 365
        excess_return = mean_return - daily_risk_free
        sharpe = (excess_return / std_return) * (365 ** 0.5)
        
        return sharpe
    
    def get_strategy_metrics(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific strategy."""
        with self.lock:
            if strategy_id in self.strategy_metrics:
                return asdict(self.strategy_metrics[strategy_id])
            return None
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for all strategies."""
        with self.lock:
            return {
                'strategies': {sid: asdict(metrics) for sid, metrics in self.strategy_metrics.items()},
                'system': self._get_system_metrics(),
                'summary': self._get_summary_metrics(),
                'generated_at': datetime.now().isoformat()
            }
    
    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics."""
        return {
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'active_strategies': len(self.strategy_metrics),
            'total_trades': sum(len(trades) for trades in self.trade_history.values()),
            'total_errors': sum(len(errors) for errors in self.error_logs.values())
        }
    
    def _get_summary_metrics(self) -> Dict[str, Any]:
        """Get aggregated summary metrics."""
        if not self.strategy_metrics:
            return {}
        
        all_metrics = list(self.strategy_metrics.values())
        
        return {
            'total_strategies': len(all_metrics),
            'total_profit_loss': sum(m.total_profit_loss for m in all_metrics),
            'total_net_profit_loss': sum(m.net_profit_loss for m in all_metrics),
            'avg_win_rate': statistics.mean([m.win_rate for m in all_metrics]) if all_metrics else 0,
            'best_performing_strategy': max(all_metrics, key=lambda x: x.net_profit_loss).strategy_id if all_metrics else None,
            'worst_performing_strategy': min(all_metrics, key=lambda x: x.net_profit_loss).strategy_id if all_metrics else None
        }
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export all metrics in specified format."""
        data = self.get_all_metrics()
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")


class StrategyMonitor:
    """Real-time monitoring system for trading strategies."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.monitoring_threads: Dict[str, threading.Thread] = {}
        self.stop_events: Dict[str, threading.Event] = {}
        self.alert_thresholds = {
            'max_drawdown_percent': 10.0,
            'max_consecutive_losses': 5,
            'error_rate_threshold': 0.1,
            'memory_usage_threshold': 80.0,
            'cpu_usage_threshold': 80.0
        }
        self.alert_callbacks: List[callable] = []
        
        logging.info("StrategyMonitor initialized")
    
    def start_monitoring(self, strategy_id: str, strategy_instance):
        """Start monitoring a specific strategy."""
        if strategy_id in self.monitoring_threads:
            logging.warning(f"Already monitoring strategy {strategy_id}")
            return
        
        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=self._monitor_strategy,
            args=(strategy_id, strategy_instance, stop_event),
            daemon=True
        )
        
        self.stop_events[strategy_id] = stop_event
        self.monitoring_threads[strategy_id] = monitor_thread
        monitor_thread.start()
        
        logging.info(f"Started monitoring strategy {strategy_id}")
    
    def stop_monitoring(self, strategy_id: str):
        """Stop monitoring a specific strategy."""
        if strategy_id in self.stop_events:
            self.stop_events[strategy_id].set()
        
        if strategy_id in self.monitoring_threads:
            thread = self.monitoring_threads[strategy_id]
            thread.join(timeout=5)
            del self.monitoring_threads[strategy_id]
        
        self.stop_events.pop(strategy_id, None)
        logging.info(f"Stopped monitoring strategy {strategy_id}")
    
    def add_alert_callback(self, callback: callable):
        """Add a callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    def set_alert_threshold(self, metric: str, value: float):
        """Set alert threshold for a specific metric."""
        self.alert_thresholds[metric] = value
    
    def _monitor_strategy(self, strategy_id: str, strategy_instance, stop_event: threading.Event):
        """Monitor a strategy in a separate thread."""
        consecutive_losses = 0
        last_trade_count = 0
        
        while not stop_event.is_set():
            try:
                # Get current metrics
                metrics = self.metrics_collector.get_strategy_metrics(strategy_id)
                if not metrics:
                    time.sleep(5)
                    continue
                
                # Check for alerts
                alerts = []
                
                # Drawdown alert
                if metrics['max_drawdown'] > self.alert_thresholds['max_drawdown_percent']:
                    alerts.append({
                        'type': 'MAX_DRAWDOWN',
                        'message': f"Max drawdown exceeded: {metrics['max_drawdown']:.2f}%",
                        'severity': 'HIGH'
                    })
                
                # Consecutive losses alert
                if metrics['total_trades'] > last_trade_count:
                    last_trade = list(self.metrics_collector.trade_history[strategy_id])[-1]
                    if last_trade.profit_loss < 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    last_trade_count = metrics['total_trades']
                
                if consecutive_losses >= self.alert_thresholds['max_consecutive_losses']:
                    alerts.append({
                        'type': 'CONSECUTIVE_LOSSES',
                        'message': f"Consecutive losses: {consecutive_losses}",
                        'severity': 'MEDIUM'
                    })
                
                # System resource alerts
                if metrics['memory_usage'] > self.alert_thresholds['memory_usage_threshold']:
                    alerts.append({
                        'type': 'HIGH_MEMORY_USAGE',
                        'message': f"High memory usage: {metrics['memory_usage']:.1f}%",
                        'severity': 'MEDIUM'
                    })
                
                if metrics['cpu_usage'] > self.alert_thresholds['cpu_usage_threshold']:
                    alerts.append({
                        'type': 'HIGH_CPU_USAGE',
                        'message': f"High CPU usage: {metrics['cpu_usage']:.1f}%",
                        'severity': 'LOW'
                    })
                
                # Error rate alert
                if metrics['error_count'] > 0 and metrics['total_trades'] > 0:
                    error_rate = metrics['error_count'] / metrics['total_trades']
                    if error_rate > self.alert_thresholds['error_rate_threshold']:
                        alerts.append({
                            'type': 'HIGH_ERROR_RATE',
                            'message': f"High error rate: {error_rate:.2%}",
                            'severity': 'HIGH'
                        })
                
                # Trigger alerts
                for alert in alerts:
                    self._trigger_alert(strategy_id, strategy_instance, alert)
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logging.error(f"Error monitoring strategy {strategy_id}: {e}")
                time.sleep(10)
    
    def _trigger_alert(self, strategy_id: str, strategy_instance, alert: Dict[str, Any]):
        """Trigger an alert for a strategy."""
        alert_data = {
            'timestamp': datetime.now(),
            'strategy_id': strategy_id,
            'strategy_name': strategy_instance.name,
            'alert': alert
        }
        
        logging.warning(f"ALERT [{alert['type']}] {strategy_instance.name}: {alert['message']}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logging.error(f"Error in alert callback: {e}")


class LiveDashboard:
    """Real-time dashboard for monitoring all strategies."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.update_interval = 5  # seconds
        self.running = False
        self.dashboard_thread = None
    
    def start_dashboard(self):
        """Start the live dashboard in a separate thread."""
        if self.running:
            return
        
        self.running = True
        self.dashboard_thread = threading.Thread(target=self._run_dashboard, daemon=True)
        self.dashboard_thread.start()
        logging.info("Live dashboard started")
    
    def stop_dashboard(self):
        """Stop the live dashboard."""
        self.running = False
        if self.dashboard_thread:
            self.dashboard_thread.join(timeout=5)
        logging.info("Live dashboard stopped")
    
    def _run_dashboard(self):
        """Run the dashboard display loop."""
        while self.running:
            try:
                self._display_dashboard()
                time.sleep(self.update_interval)
            except Exception as e:
                logging.error(f"Dashboard error: {e}")
                time.sleep(self.update_interval)
    
    def _display_dashboard(self):
        """Display the current dashboard (console version)."""
        import os
        
        # Clear screen (works on most terminals)
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("=" * 80)
        print(" CRYPTO ALGO TRADING - LIVE DASHBOARD")
        print("=" * 80)
        print(f" Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        
        # System metrics
        system_metrics = self.metrics_collector._get_system_metrics()
        print(f" System - CPU: {system_metrics['cpu_usage']:.1f}% | Memory: {system_metrics['memory_usage']:.1f}% | Active Strategies: {system_metrics['active_strategies']}")
        print("-" * 80)
        
        # Strategy metrics
        for strategy_id, metrics in self.metrics_collector.strategy_metrics.items():
            print(f" {metrics.strategy_name} ({metrics.symbol})")
            print(f"   Status: {metrics.status} | Runtime: {metrics.runtime_seconds/3600:.1f}h")
            print(f"   Trades: {metrics.total_trades} | Win Rate: {metrics.win_rate:.1f}%")
            print(f"   P&L: ${metrics.net_profit_loss:.2f} | Drawdown: {metrics.max_drawdown:.2f}")
            if metrics.last_error:
                print(f"   Last Error: {metrics.last_error[:50]}...")
            print("-" * 80)
        
        if not self.metrics_collector.strategy_metrics:
            print(" No active strategies")
            print("-" * 80)
        
        print(" Press Ctrl+C to exit dashboard")
        print("=" * 80)