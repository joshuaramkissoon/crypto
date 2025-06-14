import numpy as np
import talib
import logging
import time
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime, timedelta
from crypto.strategy import Strategy
from crypto.order import *
import statistics
import math


class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy using Bollinger Bands and RSI.
    Buys when price is oversold and below lower Bollinger Band.
    Sells when price is overbought and above upper Bollinger Band.
    """
    
    def __init__(self, client, session, notifier):
        super().__init__(client, session, notifier)
        self.bb_period = 20
        self.bb_std = 2.0
        self.rsi_period = 14
        self.rsi_oversold = 25
        self.rsi_overbought = 75
        self.trade_amount = 0.01
        self.position_size = 0
        self.entry_price = 0
        self.max_position_size = 0.05
        self.stop_loss_pct = 0.03  # 3% stop loss
        self.take_profit_pct = 0.06  # 6% take profit
        
        self.highs = deque(maxlen=self.bb_period + 10)
        self.lows = deque(maxlen=self.bb_period + 10)
        self.volumes = deque(maxlen=self.bb_period + 10)
        
        logging.info("MeanReversionStrategy initialized")
    
    def trading_strategy(self, symbol, data):
        if not data['x']:  # Only act on closed candles
            return
        
        close = float(data['c'])
        high = float(data['h'])
        low = float(data['l'])
        volume = float(data['v'])
        
        self.closes.append(close)
        self.highs.append(high)
        self.lows.append(low)
        self.volumes.append(volume)
        
        if len(self.closes) < self.bb_period:
            return
        
        # Calculate indicators
        closes_array = np.array(list(self.closes))
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            closes_array, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std
        )
        
        # RSI
        rsi = talib.RSI(closes_array, timeperiod=self.rsi_period)
        
        if len(bb_upper) == 0 or len(rsi) == 0:
            return
        
        current_bb_upper = bb_upper[-1]
        current_bb_lower = bb_lower[-1]
        current_rsi = rsi[-1]
        
        # Check for stop loss or take profit
        if self.position_size > 0:  # Long position
            profit_pct = (close - self.entry_price) / self.entry_price
            if profit_pct <= -self.stop_loss_pct or profit_pct >= self.take_profit_pct:
                logging.info(f"Closing long position - P&L: {profit_pct:.2%}")
                success, _, _ = self.order(SIDE_SELL, self.position_size, symbol)
                if success:
                    self.position_size = 0
                    self.entry_price = 0
                return
        elif self.position_size < 0:  # Short position
            profit_pct = (self.entry_price - close) / self.entry_price
            if profit_pct <= -self.stop_loss_pct or profit_pct >= self.take_profit_pct:
                logging.info(f"Closing short position - P&L: {profit_pct:.2%}")
                success, _, _ = self.order(SIDE_BUY, abs(self.position_size), symbol)
                if success:
                    self.position_size = 0
                    self.entry_price = 0
                return
        
        # Entry signals
        if self.position_size == 0:
            # Long signal: oversold RSI and price below lower BB
            if current_rsi < self.rsi_oversold and close < current_bb_lower:
                logging.info(f"Mean reversion BUY signal - RSI: {current_rsi:.1f}, Price below BB lower")
                success, _, _ = self.order(SIDE_BUY, self.trade_amount, symbol)
                if success:
                    self.position_size = self.trade_amount
                    self.entry_price = close
            
            # Short signal: overbought RSI and price above upper BB
            elif current_rsi > self.rsi_overbought and close > current_bb_upper:
                logging.info(f"Mean reversion SELL signal - RSI: {current_rsi:.1f}, Price above BB upper")
                success, _, _ = self.order(SIDE_SELL, self.trade_amount, symbol)
                if success:
                    self.position_size = -self.trade_amount
                    self.entry_price = close


class BreakoutStrategy(Strategy):
    """
    Breakout strategy using volume-confirmed price breakouts.
    Enters long when price breaks above resistance with high volume.
    Enters short when price breaks below support with high volume.
    """
    
    def __init__(self, client, session, notifier):
        super().__init__(client, session, notifier)
        self.lookback_period = 20
        self.volume_multiplier = 1.5
        self.trade_amount = 0.015
        self.position_size = 0
        self.entry_price = 0
        self.stop_loss_pct = 0.02
        self.take_profit_pct = 0.05
        
        self.highs = deque(maxlen=self.lookback_period + 10)
        self.lows = deque(maxlen=self.lookback_period + 10)
        self.volumes = deque(maxlen=self.lookback_period + 10)
        
        logging.info("BreakoutStrategy initialized")
    
    def trading_strategy(self, symbol, data):
        if not data['x']:
            return
        
        close = float(data['c'])
        high = float(data['h'])
        low = float(data['l'])
        volume = float(data['v'])
        
        self.closes.append(close)
        self.highs.append(high)
        self.lows.append(low)
        self.volumes.append(volume)
        
        if len(self.closes) < self.lookback_period:
            return
        
        # Calculate support and resistance levels
        recent_highs = list(self.highs)[-self.lookback_period:]
        recent_lows = list(self.lows)[-self.lookback_period:]
        recent_volumes = list(self.volumes)[-self.lookback_period:]
        
        resistance = max(recent_highs)
        support = min(recent_lows)
        avg_volume = sum(recent_volumes) / len(recent_volumes)
        
        # Check for stop loss or take profit
        if self.position_size != 0:
            if self.position_size > 0:  # Long position
                profit_pct = (close - self.entry_price) / self.entry_price
                if profit_pct <= -self.stop_loss_pct or profit_pct >= self.take_profit_pct:
                    logging.info(f"Closing long breakout position - P&L: {profit_pct:.2%}")
                    success, _, _ = self.order(SIDE_SELL, self.position_size, symbol)
                    if success:
                        self.position_size = 0
                        self.entry_price = 0
                    return
            else:  # Short position
                profit_pct = (self.entry_price - close) / self.entry_price
                if profit_pct <= -self.stop_loss_pct or profit_pct >= self.take_profit_pct:
                    logging.info(f"Closing short breakout position - P&L: {profit_pct:.2%}")
                    success, _, _ = self.order(SIDE_BUY, abs(self.position_size), symbol)
                    if success:
                        self.position_size = 0
                        self.entry_price = 0
                    return
        
        # Entry signals
        if self.position_size == 0:
            # Breakout above resistance with volume
            if high > resistance and volume > avg_volume * self.volume_multiplier:
                logging.info(f"Breakout BUY - Price: {close}, Resistance: {resistance}, Volume: {volume/avg_volume:.1f}x")
                success, _, _ = self.order(SIDE_BUY, self.trade_amount, symbol)
                if success:
                    self.position_size = self.trade_amount
                    self.entry_price = close
            
            # Breakdown below support with volume
            elif low < support and volume > avg_volume * self.volume_multiplier:
                logging.info(f"Breakdown SELL - Price: {close}, Support: {support}, Volume: {volume/avg_volume:.1f}x")
                success, _, _ = self.order(SIDE_SELL, self.trade_amount, symbol)
                if success:
                    self.position_size = -self.trade_amount
                    self.entry_price = close


class GridTradingStrategy(Strategy):
    """
    Grid trading strategy that places buy and sell orders at regular intervals
    around the current price to profit from price oscillations.
    """
    
    def __init__(self, client, session, notifier):
        super().__init__(client, session, notifier)
        self.grid_spacing = 0.01  # 1% spacing between grid levels
        self.num_grids = 5  # Number of grids on each side
        self.base_order_size = 0.01
        self.total_capital = 1000  # USD
        self.grid_orders = {}  # Track grid orders
        self.center_price = 0
        self.last_grid_update = 0
        self.grid_update_threshold = 0.05  # Update grid if price moves 5% from center
        
        logging.info("GridTradingStrategy initialized")
    
    def trading_strategy(self, symbol, data):
        if not data['x']:
            return
        
        close = float(data['c'])
        self.closes.append(close)
        
        # Initialize center price
        if self.center_price == 0:
            self.center_price = close
            self._setup_grid(symbol, close)
            return
        
        # Check if we need to update the grid
        price_change = abs(close - self.center_price) / self.center_price
        if price_change > self.grid_update_threshold:
            logging.info(f"Updating grid - price moved {price_change:.2%} from center")
            self.center_price = close
            self._setup_grid(symbol, close)
        
        # Check for grid order fills (simplified - in real implementation, 
        # you'd track actual order status via API)
        self._check_grid_fills(symbol, close)
    
    def _setup_grid(self, symbol, center_price):
        """Set up buy and sell grid levels."""
        self.grid_orders.clear()
        
        for i in range(1, self.num_grids + 1):
            # Buy levels (below center price)
            buy_price = center_price * (1 - i * self.grid_spacing)
            buy_level = f"buy_{i}"
            self.grid_orders[buy_level] = {
                'price': buy_price,
                'size': self.base_order_size * i,  # Larger orders further from price
                'side': 'BUY',
                'filled': False
            }
            
            # Sell levels (above center price)
            sell_price = center_price * (1 + i * self.grid_spacing)
            sell_level = f"sell_{i}"
            self.grid_orders[sell_level] = {
                'price': sell_price,
                'size': self.base_order_size * i,
                'side': 'SELL',
                'filled': False
            }
        
        logging.info(f"Grid setup complete - {len(self.grid_orders)} levels around {center_price}")
    
    def _check_grid_fills(self, symbol, current_price):
        """Check if any grid levels have been hit."""
        for level, order_info in self.grid_orders.items():
            if order_info['filled']:
                continue
            
            # Check if price has crossed this level
            if order_info['side'] == 'BUY' and current_price <= order_info['price']:
                # Price hit buy level
                logging.info(f"Grid BUY triggered at {order_info['price']}")
                success, _, _ = self.order(SIDE_BUY, order_info['size'], symbol)
                if success:
                    order_info['filled'] = True
            
            elif order_info['side'] == 'SELL' and current_price >= order_info['price']:
                # Price hit sell level
                logging.info(f"Grid SELL triggered at {order_info['price']}")
                success, _, _ = self.order(SIDE_SELL, order_info['size'], symbol)
                if success:
                    order_info['filled'] = True


class ArbitrageStrategy(Strategy):
    """
    Simple arbitrage strategy that looks for price differences between
    multiple trading pairs or exchanges (simplified version).
    """
    
    def __init__(self, client, session, notifier):
        super().__init__(client, session, notifier)
        self.min_profit_threshold = 0.005  # 0.5% minimum profit
        self.max_position_size = 0.1
        self.price_history = {}
        self.last_arbitrage_check = 0
        self.check_interval = 10  # seconds
        
        logging.info("ArbitrageStrategy initialized")
    
    def trading_strategy(self, symbol, data):
        # This is a simplified arbitrage example
        # In reality, you'd need to monitor multiple exchanges
        close = float(data['c'])
        current_time = time.time()
        
        # Store price data
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=10)
        
        self.price_history[symbol].append({
            'price': close,
            'timestamp': current_time
        })
        
        # Check for arbitrage opportunities periodically
        if current_time - self.last_arbitrage_check > self.check_interval:
            self._check_arbitrage_opportunity(symbol)
            self.last_arbitrage_check = current_time
    
    def _check_arbitrage_opportunity(self, symbol):
        """Look for arbitrage opportunities (simplified)."""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 5:
            return
        
        recent_prices = [p['price'] for p in list(self.price_history[symbol])[-5:]]
        price_volatility = (max(recent_prices) - min(recent_prices)) / min(recent_prices)
        
        # If there's significant price volatility, there might be arbitrage opportunities
        if price_volatility > self.min_profit_threshold:
            current_price = recent_prices[-1]
            avg_price = sum(recent_prices) / len(recent_prices)
            
            if current_price < avg_price * (1 - self.min_profit_threshold):
                logging.info(f"Potential arbitrage BUY opportunity - Current: {current_price}, Avg: {avg_price}")
                self.order(SIDE_BUY, 0.01, symbol)
            elif current_price > avg_price * (1 + self.min_profit_threshold):
                logging.info(f"Potential arbitrage SELL opportunity - Current: {current_price}, Avg: {avg_price}")
                self.order(SIDE_SELL, 0.01, symbol)


class MomentumStrategy(Strategy):
    """
    Momentum strategy using multiple timeframes and indicators.
    Combines Price momentum, Volume momentum, and RSI momentum.
    """
    
    def __init__(self, client, session, notifier):
        super().__init__(client, session, notifier)
        self.short_period = 10
        self.long_period = 30
        self.momentum_threshold = 0.02
        self.volume_period = 20
        self.trade_amount = 0.012
        self.position_size = 0
        self.entry_price = 0
        
        self.volumes = deque(maxlen=self.long_period + 10)
        self.timestamps = deque(maxlen=self.long_period + 10)
        
        logging.info("MomentumStrategy initialized")
    
    def trading_strategy(self, symbol, data):
        if not data['x']:
            return
        
        close = float(data['c'])
        volume = float(data['v'])
        timestamp = float(data['t'])
        
        self.closes.append(close)
        self.volumes.append(volume)
        self.timestamps.append(timestamp)
        
        if len(self.closes) < self.long_period:
            return
        
        # Calculate momentum indicators
        closes_array = np.array(list(self.closes))
        
        # Price momentum
        short_ma = talib.SMA(closes_array, timeperiod=self.short_period)
        long_ma = talib.SMA(closes_array, timeperiod=self.long_period)
        
        if len(short_ma) == 0 or len(long_ma) == 0:
            return
        
        price_momentum = (short_ma[-1] - long_ma[-1]) / long_ma[-1]
        
        # Volume momentum
        volumes_array = np.array(list(self.volumes))
        current_volume = volumes_array[-1]
        avg_volume = np.mean(volumes_array[-self.volume_period:])
        volume_momentum = (current_volume - avg_volume) / avg_volume
        
        # RSI momentum
        rsi = talib.RSI(closes_array, timeperiod=14)
        rsi_momentum = (rsi[-1] - 50) / 50  # Normalized RSI momentum
        
        # Combined momentum score
        momentum_score = (price_momentum + volume_momentum * 0.3 + rsi_momentum * 0.2)
        
        logging.info(f"Momentum scores - Price: {price_momentum:.3f}, Volume: {volume_momentum:.3f}, RSI: {rsi_momentum:.3f}, Combined: {momentum_score:.3f}")
        
        # Position management
        if self.position_size != 0:
            # Exit conditions
            profit_pct = (close - self.entry_price) / self.entry_price if self.position_size > 0 else (self.entry_price - close) / self.entry_price
            
            # Exit if momentum reverses or profit/loss targets hit
            if ((self.position_size > 0 and momentum_score < -0.01) or 
                (self.position_size < 0 and momentum_score > 0.01) or
                profit_pct >= 0.04 or profit_pct <= -0.02):
                
                side = SIDE_SELL if self.position_size > 0 else SIDE_BUY
                amount = abs(self.position_size)
                
                success, _, _ = self.order(side, amount, symbol)
                if success:
                    logging.info(f"Momentum exit - P&L: {profit_pct:.2%}")
                    self.position_size = 0
                    self.entry_price = 0
        
        # Entry conditions
        elif abs(momentum_score) > self.momentum_threshold:
            if momentum_score > self.momentum_threshold:
                # Strong positive momentum - go long
                logging.info(f"Strong positive momentum - BUY signal: {momentum_score:.3f}")
                success, _, _ = self.order(SIDE_BUY, self.trade_amount, symbol)
                if success:
                    self.position_size = self.trade_amount
                    self.entry_price = close
            
            elif momentum_score < -self.momentum_threshold:
                # Strong negative momentum - go short
                logging.info(f"Strong negative momentum - SELL signal: {momentum_score:.3f}")
                success, _, _ = self.order(SIDE_SELL, self.trade_amount, symbol)
                if success:
                    self.position_size = -self.trade_amount
                    self.entry_price = close


class BollingerBandsStrategy(Strategy):
    """
    Advanced Bollinger Bands strategy with squeeze detection and breakout trading.
    """
    
    def __init__(self, client, session, notifier):
        super().__init__(client, session, notifier)
        self.bb_period = 20
        self.bb_std = 2.0
        self.squeeze_threshold = 0.02
        self.trade_amount = 0.01
        self.position_size = 0
        self.entry_price = 0
        self.in_squeeze = False
        
        logging.info("BollingerBandsStrategy initialized")
    
    def trading_strategy(self, symbol, data):
        if not data['x']:
            return
        
        close = float(data['c'])
        self.closes.append(close)
        
        if len(self.closes) < self.bb_period:
            return
        
        closes_array = np.array(list(self.closes))
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            closes_array, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std
        )
        
        if len(bb_upper) == 0:
            return
        
        current_upper = bb_upper[-1]
        current_lower = bb_lower[-1]
        current_middle = bb_middle[-1]
        
        # Calculate band width (squeeze detection)
        band_width = (current_upper - current_lower) / current_middle
        
        # Detect squeeze conditions
        was_in_squeeze = self.in_squeeze
        self.in_squeeze = band_width < self.squeeze_threshold
        
        # Squeeze breakout signals
        if was_in_squeeze and not self.in_squeeze:
            # Squeeze ended - look for breakout direction
            if close > current_middle:
                logging.info(f"Bollinger squeeze breakout BUY - Band width: {band_width:.4f}")
                success, _, _ = self.order(SIDE_BUY, self.trade_amount, symbol)
                if success:
                    self.position_size = self.trade_amount
                    self.entry_price = close
            else:
                logging.info(f"Bollinger squeeze breakout SELL - Band width: {band_width:.4f}")
                success, _, _ = self.order(SIDE_SELL, self.trade_amount, symbol)
                if success:
                    self.position_size = -self.trade_amount
                    self.entry_price = close
        
        # Position management
        if self.position_size != 0:
            # Exit at opposite band or middle line
            if self.position_size > 0 and (close >= current_upper or close <= current_middle):
                success, _, _ = self.order(SIDE_SELL, self.position_size, symbol)
                if success:
                    profit_pct = (close - self.entry_price) / self.entry_price
                    logging.info(f"BB long exit - P&L: {profit_pct:.2%}")
                    self.position_size = 0
                    self.entry_price = 0
            
            elif self.position_size < 0 and (close <= current_lower or close >= current_middle):
                success, _, _ = self.order(SIDE_BUY, abs(self.position_size), symbol)
                if success:
                    profit_pct = (self.entry_price - close) / self.entry_price
                    logging.info(f"BB short exit - P&L: {profit_pct:.2%}")
                    self.position_size = 0
                    self.entry_price = 0


class MACDStrategy(Strategy):
    """
    MACD strategy with signal line crossovers and histogram analysis.
    """
    
    def __init__(self, client, session, notifier):
        super().__init__(client, session, notifier)
        self.fast_period = 12
        self.slow_period = 26
        self.signal_period = 9
        self.trade_amount = 0.01
        self.position_size = 0
        self.entry_price = 0
        self.last_macd_signal = None
        
        logging.info("MACDStrategy initialized")
    
    def trading_strategy(self, symbol, data):
        if not data['x']:
            return
        
        close = float(data['c'])
        self.closes.append(close)
        
        if len(self.closes) < self.slow_period + self.signal_period:
            return
        
        closes_array = np.array(list(self.closes))
        
        # Calculate MACD
        macd, signal, histogram = talib.MACD(
            closes_array, 
            fastperiod=self.fast_period, 
            slowperiod=self.slow_period, 
            signalperiod=self.signal_period
        )
        
        if len(macd) < 2 or len(signal) < 2:
            return
        
        current_macd = macd[-1]
        current_signal = signal[-1]
        prev_macd = macd[-2]
        prev_signal = signal[-2]
        current_histogram = histogram[-1]
        
        # Detect crossovers
        bullish_crossover = prev_macd <= prev_signal and current_macd > current_signal
        bearish_crossover = prev_macd >= prev_signal and current_macd < current_signal
        
        # Position management
        if self.position_size != 0:
            # Exit on opposite signal
            if (self.position_size > 0 and bearish_crossover) or (self.position_size < 0 and bullish_crossover):
                side = SIDE_SELL if self.position_size > 0 else SIDE_BUY
                amount = abs(self.position_size)
                
                success, _, _ = self.order(side, amount, symbol)
                if success:
                    profit_pct = (close - self.entry_price) / self.entry_price if self.position_size > 0 else (self.entry_price - close) / self.entry_price
                    logging.info(f"MACD exit - P&L: {profit_pct:.2%}")
                    self.position_size = 0
                    self.entry_price = 0
        
        # Entry signals
        elif self.position_size == 0:
            if bullish_crossover and current_histogram > 0:
                logging.info(f"MACD bullish crossover - BUY signal")
                success, _, _ = self.order(SIDE_BUY, self.trade_amount, symbol)
                if success:
                    self.position_size = self.trade_amount
                    self.entry_price = close
            
            elif bearish_crossover and current_histogram < 0:
                logging.info(f"MACD bearish crossover - SELL signal")
                success, _, _ = self.order(SIDE_SELL, self.trade_amount, symbol)
                if success:
                    self.position_size = -self.trade_amount
                    self.entry_price = close


class VolatilityBreakoutStrategy(Strategy):
    """
    Volatility breakout strategy using Average True Range (ATR).
    """
    
    def __init__(self, client, session, notifier):
        super().__init__(client, session, notifier)
        self.atr_period = 14
        self.atr_multiplier = 2.0
        self.trade_amount = 0.01
        self.position_size = 0
        self.entry_price = 0
        self.stop_loss_atr_multiplier = 1.5
        
        self.highs = deque(maxlen=self.atr_period + 10)
        self.lows = deque(maxlen=self.atr_period + 10)
        
        logging.info("VolatilityBreakoutStrategy initialized")
    
    def trading_strategy(self, symbol, data):
        if not data['x']:
            return
        
        close = float(data['c'])
        high = float(data['h'])
        low = float(data['l'])
        
        self.closes.append(close)
        self.highs.append(high)
        self.lows.append(low)
        
        if len(self.closes) < self.atr_period:
            return
        
        # Calculate ATR
        closes_array = np.array(list(self.closes))
        highs_array = np.array(list(self.highs))
        lows_array = np.array(list(self.lows))
        
        atr = talib.ATR(highs_array, lows_array, closes_array, timeperiod=self.atr_period)
        
        if len(atr) == 0:
            return
        
        current_atr = atr[-1]
        prev_close = self.closes[-2] if len(self.closes) > 1 else close
        
        # Calculate breakout levels
        upper_breakout = prev_close + (current_atr * self.atr_multiplier)
        lower_breakout = prev_close - (current_atr * self.atr_multiplier)
        
        # Position management with ATR-based stops
        if self.position_size != 0:
            if self.position_size > 0:  # Long position
                stop_loss = self.entry_price - (current_atr * self.stop_loss_atr_multiplier)
                if close <= stop_loss:
                    logging.info(f"ATR stop loss hit for long position")
                    success, _, _ = self.order(SIDE_SELL, self.position_size, symbol)
                    if success:
                        self.position_size = 0
                        self.entry_price = 0
                    return
            else:  # Short position
                stop_loss = self.entry_price + (current_atr * self.stop_loss_atr_multiplier)
                if close >= stop_loss:
                    logging.info(f"ATR stop loss hit for short position")
                    success, _, _ = self.order(SIDE_BUY, abs(self.position_size), symbol)
                    if success:
                        self.position_size = 0
                        self.entry_price = 0
                    return
        
        # Entry signals
        if self.position_size == 0:
            if high > upper_breakout:
                logging.info(f"Volatility breakout BUY - ATR: {current_atr:.4f}, Breakout: {upper_breakout:.4f}")
                success, _, _ = self.order(SIDE_BUY, self.trade_amount, symbol)
                if success:
                    self.position_size = self.trade_amount
                    self.entry_price = close
            
            elif low < lower_breakout:
                logging.info(f"Volatility breakout SELL - ATR: {current_atr:.4f}, Breakout: {lower_breakout:.4f}")
                success, _, _ = self.order(SIDE_SELL, self.trade_amount, symbol)
                if success:
                    self.position_size = -self.trade_amount
                    self.entry_price = close


class TrendFollowingStrategy(Strategy):
    """
    Multi-timeframe trend following strategy using EMA crossovers and trend strength.
    """
    
    def __init__(self, client, session, notifier):
        super().__init__(client, session, notifier)
        self.fast_ema = 21
        self.slow_ema = 55
        self.trend_ema = 200
        self.trade_amount = 0.015
        self.position_size = 0
        self.entry_price = 0
        self.min_trend_strength = 0.01
        
        logging.info("TrendFollowingStrategy initialized")
    
    def trading_strategy(self, symbol, data):
        if not data['x']:
            return
        
        close = float(data['c'])
        self.closes.append(close)
        
        if len(self.closes) < self.trend_ema:
            return
        
        closes_array = np.array(list(self.closes))
        
        # Calculate EMAs
        fast_ema = talib.EMA(closes_array, timeperiod=self.fast_ema)
        slow_ema = talib.EMA(closes_array, timeperiod=self.slow_ema)
        trend_ema = talib.EMA(closes_array, timeperiod=self.trend_ema)
        
        if len(fast_ema) < 2 or len(slow_ema) < 2 or len(trend_ema) == 0:
            return
        
        current_fast = fast_ema[-1]
        current_slow = slow_ema[-1]
        current_trend = trend_ema[-1]
        prev_fast = fast_ema[-2]
        prev_slow = slow_ema[-2]
        
        # Determine trend direction
        trend_direction = 1 if close > current_trend else -1
        trend_strength = abs(close - current_trend) / current_trend
        
        # EMA crossovers
        bullish_cross = prev_fast <= prev_slow and current_fast > current_slow
        bearish_cross = prev_fast >= prev_slow and current_fast < current_slow
        
        # Position management
        if self.position_size != 0:
            # Exit on opposite crossover or trend change
            if (self.position_size > 0 and (bearish_cross or trend_direction < 0)) or \
               (self.position_size < 0 and (bullish_cross or trend_direction > 0)):
                
                side = SIDE_SELL if self.position_size > 0 else SIDE_BUY
                amount = abs(self.position_size)
                
                success, _, _ = self.order(side, amount, symbol)
                if success:
                    profit_pct = (close - self.entry_price) / self.entry_price if self.position_size > 0 else (self.entry_price - close) / self.entry_price
                    logging.info(f"Trend following exit - P&L: {profit_pct:.2%}")
                    self.position_size = 0
                    self.entry_price = 0
        
        # Entry signals
        elif self.position_size == 0 and trend_strength > self.min_trend_strength:
            if bullish_cross and trend_direction > 0:
                logging.info(f"Trend following BUY - Trend strength: {trend_strength:.3f}")
                success, _, _ = self.order(SIDE_BUY, self.trade_amount, symbol)
                if success:
                    self.position_size = self.trade_amount
                    self.entry_price = close
            
            elif bearish_cross and trend_direction < 0:
                logging.info(f"Trend following SELL - Trend strength: {trend_strength:.3f}")
                success, _, _ = self.order(SIDE_SELL, self.trade_amount, symbol)
                if success:
                    self.position_size = -self.trade_amount
                    self.entry_price = close


class ScalpingStrategy(Strategy):
    """
    High-frequency scalping strategy for small, quick profits.
    """
    
    def __init__(self, client, session, notifier):
        super().__init__(client, session, notifier)
        self.trade_amount = 0.005
        self.position_size = 0
        self.entry_price = 0
        self.min_profit_ticks = 0.0005  # 0.05% minimum profit
        self.max_loss_ticks = 0.0003    # 0.03% maximum loss
        self.ema_period = 9
        self.last_trade_time = 0
        self.min_trade_interval = 30  # seconds between trades
        
        logging.info("ScalpingStrategy initialized")
    
    def trading_strategy(self, symbol, data):
        close = float(data['c'])
        current_time = time.time()
        
        self.closes.append(close)
        
        if len(self.closes) < self.ema_period:
            return
        
        # Calculate short-term EMA
        closes_array = np.array(list(self.closes))
        ema = talib.EMA(closes_array, timeperiod=self.ema_period)
        
        if len(ema) == 0:
            return
        
        current_ema = ema[-1]
        
        # Quick position management
        if self.position_size != 0:
            if self.position_size > 0:  # Long position
                profit_pct = (close - self.entry_price) / self.entry_price
                if profit_pct >= self.min_profit_ticks or profit_pct <= -self.max_loss_ticks:
                    success, _, _ = self.order(SIDE_SELL, self.position_size, symbol)
                    if success:
                        logging.info(f"Scalp long exit - P&L: {profit_pct:.4%}")
                        self.position_size = 0
                        self.entry_price = 0
                        self.last_trade_time = current_time
            
            else:  # Short position
                profit_pct = (self.entry_price - close) / self.entry_price
                if profit_pct >= self.min_profit_ticks or profit_pct <= -self.max_loss_ticks:
                    success, _, _ = self.order(SIDE_BUY, abs(self.position_size), symbol)
                    if success:
                        logging.info(f"Scalp short exit - P&L: {profit_pct:.4%}")
                        self.position_size = 0
                        self.entry_price = 0
                        self.last_trade_time = current_time
        
        # Entry signals (only if enough time has passed since last trade)
        elif current_time - self.last_trade_time > self.min_trade_interval:
            price_vs_ema = (close - current_ema) / current_ema
            
            # Quick mean reversion scalp signals
            if price_vs_ema < -0.0005:  # Price below EMA
                logging.info(f"Scalp BUY signal - Price {price_vs_ema:.4%} below EMA")
                success, _, _ = self.order(SIDE_BUY, self.trade_amount, symbol)
                if success:
                    self.position_size = self.trade_amount
                    self.entry_price = close
            
            elif price_vs_ema > 0.0005:  # Price above EMA
                logging.info(f"Scalp SELL signal - Price {price_vs_ema:.4%} above EMA")
                success, _, _ = self.order(SIDE_SELL, self.trade_amount, symbol)
                if success:
                    self.position_size = -self.trade_amount
                    self.entry_price = close