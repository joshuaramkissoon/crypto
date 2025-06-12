import pandas as pd
import numpy as np
from binance.client import Client
from binance import ThreadedWebSocketManager
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import time
import pickle
import os
from pathlib import Path
import sqlite3
import threading
from collections import deque


class BinanceDataCollector:
    """
    Collects historical and real-time cryptocurrency data from Binance.
    Stores data locally for RL training and backtesting.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        data_dir: str = "./data"
    ):
        self.client = Client(api_key, api_secret) if api_key and api_secret else Client()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Database for storing historical data
        self.db_path = self.data_dir / "crypto_data.db"
        self._init_database()
        
        # Real-time data streaming
        self.twm = None
        self.real_time_data = {}
        self.streaming_active = False
        
        logging.info("BinanceDataCollector initialized")
    
    def _init_database(self):
        """Initialize SQLite database for storing historical data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create table for OHLCV data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open_time TEXT NOT NULL,
                close_time TEXT NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume REAL NOT NULL,
                quote_volume REAL NOT NULL,
                trades_count INTEGER NOT NULL,
                taker_buy_base_volume REAL NOT NULL,
                taker_buy_quote_volume REAL NOT NULL,
                UNIQUE(symbol, timestamp)
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON ohlcv_data(symbol, timestamp)')
        
        conn.commit()
        conn.close()
        
        logging.info("Database initialized")
    
    def get_historical_data(
        self,
        symbol: str,
        interval: str = '1h',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data from Binance.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
            start_date: Start date in format 'YYYY-MM-DD' or datetime
            end_date: End date in format 'YYYY-MM-DD' or datetime
            limit: Maximum number of records to retrieve
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Get klines from Binance
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_date,
                end_str=end_date,
                limit=limit
            )
            
            if not klines:
                logging.warning(f"No data retrieved for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            columns = [
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count',
                'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
            ]
            
            df = pd.DataFrame(klines, columns=columns)
            
            # Convert data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                             'taker_buy_base_volume', 'taker_buy_quote_volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['trades_count'] = pd.to_numeric(df['trades_count'], errors='coerce')
            
            # Convert timestamps
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Set index to open_time
            df.set_index('open_time', inplace=True)
            
            # Drop unnecessary column
            df.drop('ignore', axis=1, inplace=True)
            
            # Add symbol information
            df['symbol'] = symbol
            df['interval'] = interval
            
            logging.info(f"Retrieved {len(df)} records for {symbol} ({interval})")
            return df
            
        except Exception as e:
            logging.error(f"Error retrieving historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def save_historical_data(self, df: pd.DataFrame, symbol: str):
        """Save historical data to database."""
        if df.empty:
            logging.warning("No data to save")
            return
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            for _, row in df.iterrows():
                # Insert or replace data
                conn.execute('''
                    INSERT OR REPLACE INTO ohlcv_data 
                    (symbol, timestamp, open_time, close_time, open_price, high_price, 
                     low_price, close_price, volume, quote_volume, trades_count,
                     taker_buy_base_volume, taker_buy_quote_volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    int(row.name.timestamp()),
                    row.name.isoformat(),
                    row['close_time'].isoformat(),
                    row['open'],
                    row['high'],
                    row['low'],
                    row['close'],
                    row['volume'],
                    row['quote_volume'],
                    row['trades_count'],
                    row['taker_buy_base_volume'],
                    row['taker_buy_quote_volume']
                ))
            
            conn.commit()
            logging.info(f"Saved {len(df)} records for {symbol} to database")
            
        except Exception as e:
            logging.error(f"Error saving data to database: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def load_historical_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load historical data from database."""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM ohlcv_data WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(int(pd.to_datetime(start_date).timestamp()))
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(int(pd.to_datetime(end_date).timestamp()))
        
        query += " ORDER BY timestamp"
        
        try:
            df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                logging.warning(f"No historical data found for {symbol}")
                return df
            
            # Convert timestamp back to datetime
            df['open_time'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('open_time', inplace=True)
            
            # Rename columns to match expected format
            df.rename(columns={
                'open_price': 'open',
                'high_price': 'high',
                'low_price': 'low',
                'close_price': 'close'
            }, inplace=True)
            
            # Select relevant columns
            columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades_count']
            df = df[columns]
            
            logging.info(f"Loaded {len(df)} records for {symbol} from database")
            return df
            
        except Exception as e:
            logging.error(f"Error loading data from database: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def collect_and_store_data(
        self,
        symbols: List[str],
        intervals: List[str] = ['1h'],
        days_back: int = 365,
        update_existing: bool = True
    ):
        """
        Collect and store historical data for multiple symbols and intervals.
        
        Args:
            symbols: List of trading pair symbols
            intervals: List of intervals to collect
            days_back: Number of days of historical data to collect
            update_existing: Whether to update existing data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        for symbol in symbols:
            for interval in intervals:
                try:
                    logging.info(f"Collecting data for {symbol} ({interval})")
                    
                    # Check if we need to update existing data
                    if update_existing:
                        existing_data = self.load_historical_data(symbol)
                        if not existing_data.empty:
                            # Get data from last record onwards
                            last_date = existing_data.index[-1]
                            start_date = max(start_date, last_date)
                    
                    # Get historical data in chunks (Binance limit is 1000 records)
                    current_start = start_date
                    all_data = []
                    
                    while current_start < end_date:
                        # Calculate chunk end date
                        if interval == '1m':
                            chunk_end = current_start + timedelta(days=1)  # ~1440 minutes
                        elif interval == '5m':
                            chunk_end = current_start + timedelta(days=3)  # ~864 records
                        elif interval == '15m':
                            chunk_end = current_start + timedelta(days=10)  # ~960 records
                        elif interval == '1h':
                            chunk_end = current_start + timedelta(days=40)  # ~960 records
                        elif interval == '4h':
                            chunk_end = current_start + timedelta(days=160)  # ~960 records
                        elif interval == '1d':
                            chunk_end = current_start + timedelta(days=1000)  # 1000 records
                        else:
                            chunk_end = current_start + timedelta(days=30)  # Default
                        
                        chunk_end = min(chunk_end, end_date)
                        
                        # Get data chunk
                        chunk_data = self.get_historical_data(
                            symbol=symbol,
                            interval=interval,
                            start_date=current_start.strftime('%Y-%m-%d'),
                            end_date=chunk_end.strftime('%Y-%m-%d'),
                            limit=1000
                        )
                        
                        if not chunk_data.empty:
                            all_data.append(chunk_data)
                            self.save_historical_data(chunk_data, symbol)
                        
                        # Move to next chunk
                        current_start = chunk_end
                        
                        # Rate limiting
                        time.sleep(0.1)
                    
                    if all_data:
                        combined_data = pd.concat(all_data)
                        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                        logging.info(f"Collected total {len(combined_data)} records for {symbol} ({interval})")
                    
                except Exception as e:
                    logging.error(f"Error collecting data for {symbol} ({interval}): {e}")
                    continue
    
    def start_real_time_stream(
        self,
        symbols: List[str],
        callback: Optional[callable] = None
    ):
        """Start real-time data streaming."""
        if self.streaming_active:
            logging.warning("Real-time streaming already active")
            return
        
        self.twm = ThreadedWebSocketManager()
        self.twm.start()
        
        # Initialize data storage for each symbol
        for symbol in symbols:
            self.real_time_data[symbol] = deque(maxlen=1000)
        
        def handle_socket_message(msg):
            """Handle incoming WebSocket messages."""
            try:
                symbol = msg['s']
                if symbol in self.real_time_data:
                    # Parse kline data
                    kline_data = {
                        'symbol': symbol,
                        'timestamp': pd.to_datetime(msg['k']['t'], unit='ms'),
                        'open': float(msg['k']['o']),
                        'high': float(msg['k']['h']),
                        'low': float(msg['k']['l']),
                        'close': float(msg['k']['c']),
                        'volume': float(msg['k']['v']),
                        'is_closed': msg['k']['x']
                    }
                    
                    self.real_time_data[symbol].append(kline_data)
                    
                    # Call user callback if provided
                    if callback:
                        callback(kline_data)
                        
            except Exception as e:
                logging.error(f"Error processing real-time data: {e}")
        
        # Start streams for all symbols
        for symbol in symbols:
            stream_name = f"{symbol.lower()}@kline_1m"
            self.twm.start_kline_socket(
                callback=handle_socket_message,
                symbol=symbol.lower(),
                interval='1m'
            )
        
        self.streaming_active = True
        logging.info(f"Started real-time streaming for {len(symbols)} symbols")
    
    def stop_real_time_stream(self):
        """Stop real-time data streaming."""
        if self.twm:
            self.twm.stop()
            self.twm = None
        
        self.streaming_active = False
        logging.info("Stopped real-time streaming")
    
    def get_real_time_data(self, symbol: str, last_n: int = 100) -> List[Dict]:
        """Get recent real-time data for a symbol."""
        if symbol not in self.real_time_data:
            return []
        
        return list(self.real_time_data[symbol])[-last_n:]
    
    def prepare_training_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for RL training by splitting into train/validation sets.
        
        Returns:
            Tuple of (train_data, validation_data)
        """
        # Load historical data
        data = self.load_historical_data(symbol, start_date, end_date)
        
        if data.empty:
            logging.error(f"No data available for {symbol}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Split data
        split_idx = int(len(data) * train_ratio)
        train_data = data.iloc[:split_idx].copy()
        val_data = data.iloc[split_idx:].copy()
        
        logging.info(f"Training data: {len(train_data)} records")
        logging.info(f"Validation data: {len(val_data)} records")
        
        return train_data, val_data
    
    def create_multi_asset_dataset(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        align_timestamps: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Create a multi-asset dataset for portfolio optimization or ensemble training.
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        datasets = {}
        
        for symbol in symbols:
            data = self.load_historical_data(symbol, start_date, end_date)
            if not data.empty:
                datasets[symbol] = data
        
        if align_timestamps and len(datasets) > 1:
            # Find common time range
            all_indices = [df.index for df in datasets.values()]
            common_start = max(idx.min() for idx in all_indices)
            common_end = min(idx.max() for idx in all_indices)
            
            # Align all datasets to common timeframe
            for symbol in datasets:
                datasets[symbol] = datasets[symbol].loc[common_start:common_end]
            
            logging.info(f"Aligned {len(datasets)} datasets to common timeframe: {common_start} to {common_end}")
        
        return datasets
    
    def get_market_data_summary(self, symbols: List[str]) -> pd.DataFrame:
        """Get market data summary for multiple symbols."""
        summary_data = []
        
        for symbol in symbols:
            try:
                # Get 24hr ticker data
                ticker = self.client.get_ticker(symbol=symbol)
                
                # Get recent klines for additional metrics
                recent_data = self.get_historical_data(symbol, '1d', limit=30)
                
                volatility = 0
                avg_volume = 0
                if not recent_data.empty:
                    returns = recent_data['close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(365)  # Annualized volatility
                    avg_volume = recent_data['volume'].mean()
                
                summary_data.append({
                    'symbol': symbol,
                    'price': float(ticker['lastPrice']),
                    'change_24h': float(ticker['priceChangePercent']),
                    'volume_24h': float(ticker['volume']),
                    'high_24h': float(ticker['highPrice']),
                    'low_24h': float(ticker['lowPrice']),
                    'volatility_30d': volatility,
                    'avg_volume_30d': avg_volume
                })
                
            except Exception as e:
                logging.error(f"Error getting market data for {symbol}: {e}")
                continue
        
        return pd.DataFrame(summary_data)


class DataPreprocessor:
    """
    Preprocesses raw market data for RL training.
    Handles normalization, feature engineering, and data quality checks.
    """
    
    def __init__(self):
        self.scalers = {}
        self.feature_columns = []
    
    def add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis features to the dataset."""
        # This is already handled in the RL environment
        # But we can add additional features here if needed
        
        # Price-based features
        df['price_change'] = df['close'].pct_change()
        df['price_momentum'] = df['close'] / df['close'].shift(10) - 1
        df['volatility'] = df['price_change'].rolling(20).std()
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Bollinger Bands position
        bb_period = 20
        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        bb_std = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + 2 * bb_std
        df['bb_lower'] = df['bb_middle'] - 2 * bb_std
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Market microstructure
        df['spread'] = df['high'] - df['low']
        df['spread_ratio'] = df['spread'] / df['close']
        
        return df.fillna(method='ffill').fillna(0)
    
    def quality_check(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform data quality checks and cleaning."""
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        # Check for gaps in data
        time_diff = df.index.to_series().diff()
        expected_freq = time_diff.mode()[0] if not time_diff.empty else pd.Timedelta(hours=1)
        large_gaps = time_diff > expected_freq * 2
        
        if large_gaps.any():
            logging.warning(f"Found {large_gaps.sum()} large gaps in data")
        
        # Remove outliers (price jumps > 20%)
        price_changes = df['close'].pct_change().abs()
        outliers = price_changes > 0.2
        
        if outliers.any():
            logging.warning(f"Found {outliers.sum()} potential outliers")
            # Option to remove or cap outliers
            # df = df[~outliers]
        
        # Ensure positive prices and volumes
        df = df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1)]
        df = df[df['volume'] >= 0]
        
        logging.info(f"Data quality check completed. Final dataset: {len(df)} records")
        return df