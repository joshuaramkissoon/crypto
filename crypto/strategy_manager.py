from typing import Dict, List, Optional, Any, Callable
from threading import Thread, Event, Lock
import logging
import time
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from crypto.strategy import Strategy
from crypto.algo import AlgoTrader
from crypto.observability import MetricsCollector, StrategyMonitor
from crypto.risk_manager import RiskManager
import uuid


class StrategyStatus(Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"


@dataclass
class StrategyInstance:
    id: str
    name: str
    strategy_class: str
    symbol: str
    base_asset: str
    quote_asset: str
    status: StrategyStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    error_message: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None


class StrategyManager:
    """
    Advanced strategy management system that allows running multiple strategies
    simultaneously with comprehensive monitoring and control capabilities.
    """
    
    def __init__(self, client, notifier=None):
        self.client = client
        self.notifier = notifier
        self.strategies: Dict[str, StrategyInstance] = {}
        self.running_traders: Dict[str, AlgoTrader] = {}
        self.strategy_threads: Dict[str, Thread] = {}
        self.stop_events: Dict[str, Event] = {}
        self.lock = Lock()
        
        # Initialize monitoring and risk management
        self.metrics_collector = MetricsCollector()
        self.strategy_monitor = StrategyMonitor(self.metrics_collector)
        self.risk_manager = RiskManager()
        
        # Strategy registry
        self.strategy_registry = {}
        self._register_built_in_strategies()
        
        logging.info("StrategyManager initialized")
    
    def _register_built_in_strategies(self):
        """Register all available strategy classes."""
        from crypto.advanced_strategies import (
            MeanReversionStrategy, 
            BreakoutStrategy, 
            GridTradingStrategy,
            ArbitrageStrategy,
            MomentumStrategy,
            BollingerBandsStrategy,
            MACDStrategy,
            VolatilityBreakoutStrategy,
            TrendFollowingStrategy,
            ScalpingStrategy
        )
        from crypto.strategy import RSI, MA, CMO
        from crypto.rl_trainer import RLStrategy
        
        strategies = [
            RSI, MA, CMO, MeanReversionStrategy, BreakoutStrategy, 
            GridTradingStrategy, ArbitrageStrategy, MomentumStrategy,
            BollingerBandsStrategy, MACDStrategy, VolatilityBreakoutStrategy,
            TrendFollowingStrategy, ScalpingStrategy, RLStrategy
        ]
        
        for strategy_cls in strategies:
            self.strategy_registry[strategy_cls.__name__] = strategy_cls
    
    def create_strategy(
        self, 
        strategy_name: str, 
        base_asset: str, 
        quote_asset: str,
        config: Optional[Dict[str, Any]] = None,
        custom_name: Optional[str] = None
    ) -> str:
        """
        Create a new strategy instance.
        
        Args:
            strategy_name: Name of the strategy class
            base_asset: Base asset symbol (e.g., 'BTC')
            quote_asset: Quote asset symbol (e.g., 'USDT')
            config: Optional configuration for the strategy
            custom_name: Optional custom name for the instance
            
        Returns:
            str: Unique strategy instance ID
        """
        if strategy_name not in self.strategy_registry:
            raise ValueError(f"Strategy '{strategy_name}' not found in registry")
        
        instance_id = str(uuid.uuid4())
        symbol = base_asset.upper() + quote_asset.upper()
        
        instance = StrategyInstance(
            id=instance_id,
            name=custom_name or f"{strategy_name}_{symbol}_{int(time.time())}",
            strategy_class=strategy_name,
            symbol=symbol,
            base_asset=base_asset.upper(),
            quote_asset=quote_asset.upper(),
            status=StrategyStatus.STOPPED,
            created_at=datetime.now(),
            config=config or {}
        )
        
        with self.lock:
            self.strategies[instance_id] = instance
        
        logging.info(f"Created strategy instance {instance.name} ({instance_id})")
        return instance_id
    
    def start_strategy(self, instance_id: str) -> bool:
        """
        Start a strategy instance.
        
        Args:
            instance_id: Strategy instance ID
            
        Returns:
            bool: True if started successfully
        """
        with self.lock:
            if instance_id not in self.strategies:
                logging.error(f"Strategy {instance_id} not found")
                return False
            
            instance = self.strategies[instance_id]
            
            if instance.status == StrategyStatus.RUNNING:
                logging.warning(f"Strategy {instance.name} is already running")
                return True
            
            try:
                # Create strategy class instance
                strategy_cls = self.strategy_registry[instance.strategy_class]
                
                # Create AlgoTrader with strategy
                trader = AlgoTrader(
                    client=self.client,
                    base_asset=instance.base_asset,
                    quote_asset=instance.quote_asset,
                    strategy=strategy_cls,
                    notifier=self.notifier
                )
                
                # Create stop event and thread
                stop_event = Event()
                thread = Thread(
                    target=self._run_strategy,
                    args=(instance_id, trader, stop_event),
                    daemon=True
                )
                
                # Store references
                self.running_traders[instance_id] = trader
                self.stop_events[instance_id] = stop_event
                self.strategy_threads[instance_id] = thread
                
                # Update instance status
                instance.status = StrategyStatus.RUNNING
                instance.started_at = datetime.now()
                instance.error_message = None
                
                # Start monitoring
                self.strategy_monitor.start_monitoring(instance_id, instance)
                
                # Start thread
                thread.start()
                
                logging.info(f"Started strategy {instance.name} ({instance_id})")
                return True
                
            except Exception as e:
                instance.status = StrategyStatus.ERROR
                instance.error_message = str(e)
                logging.error(f"Failed to start strategy {instance.name}: {e}")
                return False
    
    def stop_strategy(self, instance_id: str) -> bool:
        """
        Stop a strategy instance.
        
        Args:
            instance_id: Strategy instance ID
            
        Returns:
            bool: True if stopped successfully
        """
        with self.lock:
            if instance_id not in self.strategies:
                logging.error(f"Strategy {instance_id} not found")
                return False
            
            instance = self.strategies[instance_id]
            
            if instance.status != StrategyStatus.RUNNING:
                logging.warning(f"Strategy {instance.name} is not running")
                return True
            
            try:
                # Set status to stopping
                instance.status = StrategyStatus.STOPPING
                
                # Signal stop
                if instance_id in self.stop_events:
                    self.stop_events[instance_id].set()
                
                # Stop trader
                if instance_id in self.running_traders:
                    self.running_traders[instance_id].stop()
                
                # Wait for thread to finish
                if instance_id in self.strategy_threads:
                    thread = self.strategy_threads[instance_id]
                    thread.join(timeout=10)  # Wait up to 10 seconds
                
                # Stop monitoring
                self.strategy_monitor.stop_monitoring(instance_id)
                
                # Clean up
                self._cleanup_strategy(instance_id)
                
                # Update instance status
                instance.status = StrategyStatus.STOPPED
                instance.stopped_at = datetime.now()
                
                logging.info(f"Stopped strategy {instance.name} ({instance_id})")
                return True
                
            except Exception as e:
                instance.status = StrategyStatus.ERROR
                instance.error_message = str(e)
                logging.error(f"Failed to stop strategy {instance.name}: {e}")
                return False
    
    def pause_strategy(self, instance_id: str) -> bool:
        """Pause a running strategy."""
        with self.lock:
            if instance_id not in self.strategies:
                return False
            
            instance = self.strategies[instance_id]
            if instance.status == StrategyStatus.RUNNING:
                instance.status = StrategyStatus.PAUSED
                # Implementation depends on strategy support for pausing
                logging.info(f"Paused strategy {instance.name}")
                return True
            return False
    
    def resume_strategy(self, instance_id: str) -> bool:
        """Resume a paused strategy."""
        with self.lock:
            if instance_id not in self.strategies:
                return False
            
            instance = self.strategies[instance_id]
            if instance.status == StrategyStatus.PAUSED:
                instance.status = StrategyStatus.RUNNING
                logging.info(f"Resumed strategy {instance.name}")
                return True
            return False
    
    def remove_strategy(self, instance_id: str) -> bool:
        """Remove a strategy instance (must be stopped first)."""
        with self.lock:
            if instance_id not in self.strategies:
                return False
            
            instance = self.strategies[instance_id]
            if instance.status == StrategyStatus.RUNNING:
                logging.error(f"Cannot remove running strategy {instance.name}. Stop it first.")
                return False
            
            # Clean up any remaining resources
            self._cleanup_strategy(instance_id)
            
            # Remove from strategies
            del self.strategies[instance_id]
            
            logging.info(f"Removed strategy {instance.name} ({instance_id})")
            return True
    
    def get_strategy_status(self, instance_id: str) -> Optional[StrategyInstance]:
        """Get status of a specific strategy."""
        return self.strategies.get(instance_id)
    
    def list_strategies(self) -> List[StrategyInstance]:
        """List all strategy instances."""
        with self.lock:
            return list(self.strategies.values())
    
    def get_running_strategies(self) -> List[StrategyInstance]:
        """Get all currently running strategies."""
        with self.lock:
            return [s for s in self.strategies.values() if s.status == StrategyStatus.RUNNING]
    
    def stop_all_strategies(self) -> bool:
        """Stop all running strategies."""
        running_strategies = self.get_running_strategies()
        all_stopped = True
        
        for strategy in running_strategies:
            if not self.stop_strategy(strategy.id):
                all_stopped = False
        
        return all_stopped
    
    def get_strategy_performance(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a strategy."""
        return self.metrics_collector.get_strategy_metrics(instance_id)
    
    def get_all_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for all strategies."""
        return self.metrics_collector.get_all_metrics()
    
    def export_strategy_data(self, instance_id: str, format: str = 'json') -> str:
        """Export strategy data in specified format."""
        if instance_id not in self.strategies:
            raise ValueError(f"Strategy {instance_id} not found")
        
        instance = self.strategies[instance_id]
        performance = self.get_strategy_performance(instance_id)
        
        data = {
            'instance': asdict(instance),
            'performance': performance,
            'exported_at': datetime.now().isoformat()
        }
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _run_strategy(self, instance_id: str, trader: AlgoTrader, stop_event: Event):
        """Run a strategy in a separate thread."""
        try:
            # Start the trader
            trader.trade()
            
            # Keep running until stop event is set
            while not stop_event.is_set():
                time.sleep(1)
                
        except Exception as e:
            logging.error(f"Strategy {instance_id} crashed: {e}")
            with self.lock:
                if instance_id in self.strategies:
                    self.strategies[instance_id].status = StrategyStatus.ERROR
                    self.strategies[instance_id].error_message = str(e)
    
    def _cleanup_strategy(self, instance_id: str):
        """Clean up resources for a strategy."""
        # Remove from tracking dictionaries
        self.running_traders.pop(instance_id, None)
        self.stop_events.pop(instance_id, None)
        self.strategy_threads.pop(instance_id, None)
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy classes."""
        return list(self.strategy_registry.keys())
    
    def register_custom_strategy(self, strategy_class):
        """Register a custom strategy class."""
        if not issubclass(strategy_class, Strategy):
            raise ValueError("Strategy class must inherit from Strategy base class")
        
        self.strategy_registry[strategy_class.__name__] = strategy_class
        logging.info(f"Registered custom strategy: {strategy_class.__name__}")