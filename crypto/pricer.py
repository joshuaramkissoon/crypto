import websocket
import json
from crypto.constants import SOCKET_BASE
from crypto.environment import Environment
from binance.client import Client
import concurrent.futures
import threading
from pprint import pprint
import logging
from enum import Enum
from datetime import datetime
import time
import random
from binance.exceptions import BinanceAPIException, BinanceRequestException

class StreamType(Enum):
    KLINE = 0
    ORDER_BOOK = 1

class PriceStream:
    '''
    Robust price streaming for asset pairs with automatic reconnection, rate limiting,
    and comprehensive error handling. Streams KLine/Candlestick data with updates
    every ~2 seconds.
    '''

    def __init__(self, base_asset, quote_asset='usdt', interval='1m', log_ticks=False, strategy=None, session=None, notifier=None, client=None, close_callback=None, stream_type=StreamType.KLINE, max_reconnect_attempts=10, reconnect_delay=5):
        '''
        Initialize a price stream for an asset pair.
        Parameters
        ----------
        base_asset: String, ticker for asset
        quote_asset: String, ticker for reference asset (defaults to USDT)
        interval: String, interval for candlestick (minute (m), hour (h), day (d)). Defaults to 1 minute
        log_ticks: Bool, set to True to log close prices for every tick
        strategy: class, algo-trading strategy (must be a subclass of Strategy)
        client: Binance client object
        '''
        self.base_asset = base_asset
        self.quote_asset = quote_asset
        self.interval = interval
        self.stream_type = stream_type
        self.symbol = base_asset.upper() + quote_asset.upper()
        self.socket_uri = self.__make_socket_uri(stream_type, base_asset, quote_asset, interval)
        
        # Connection management
        self.ws = None
        self.is_running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.last_ping_time = None
        self.connection_thread = None
        
        # Rate limiting (Binance allows 5 messages per second)
        self.message_times = []
        self.max_messages_per_second = 4  # Conservative limit
        
        # Static class variables for callbacks
        PriceStream.log_ticks = log_ticks
        PriceStream.close_callback = close_callback
        PriceStream.instance = self  # Store instance for static methods
        
        # Initialize trading strategy
        if strategy:
            assert client, 'To use a strategy, PriceStream object must be initialised with a Binance Client.'
            PriceStream.strategy = strategy(client, session, notifier)
        else:
            PriceStream.strategy = None
        
        logging.info(f'PriceStream initialized for {self.symbol} with {stream_type.name} stream')
    
    def run(self):
        """Start the price stream with automatic reconnection."""
        self.is_running = True
        self.connection_thread = threading.Thread(target=self._run_with_reconnect)
        self.connection_thread.daemon = True
        self.connection_thread.start()
        logging.info(f'PriceStream started for {self.symbol}')
    
    def _run_with_reconnect(self):
        """Run the WebSocket with automatic reconnection logic."""
        while self.is_running and self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                self._create_websocket()
                self.ws.run_forever(
                    ping_interval=20,  # Send ping every 20 seconds
                    ping_timeout=10,   # Wait 10 seconds for pong
                    ping_payload='ping'
                )
            except Exception as e:
                logging.error(f'WebSocket error for {self.symbol}: {e}')
                if self.is_running:
                    self._handle_reconnection()
            
            if not self.is_running:
                break
                
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logging.error(f'Max reconnection attempts reached for {self.symbol}. Stopping stream.')
            self._notify_failure()
    
    def _create_websocket(self):
        """Create a new WebSocket connection."""
        self.ws = websocket.WebSocketApp(
            self.socket_uri,
            on_open=self.on_open,
            on_close=self.on_close,
            on_message=self.on_message,
            on_error=self.on_error,
            on_ping=self.on_ping,
            on_pong=self.on_pong
        )
    
    def _handle_reconnection(self):
        """Handle reconnection with exponential backoff."""
        if not self.is_running:
            return
            
        self.reconnect_attempts += 1
        delay = min(self.reconnect_delay * (2 ** (self.reconnect_attempts - 1)), 60)  # Max 60 seconds
        delay += random.uniform(0, 5)  # Add jitter
        
        logging.warning(f'Reconnecting {self.symbol} stream in {delay:.1f}s (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})')
        time.sleep(delay)
    
    def _notify_failure(self):
        """Notify about stream failure."""
        if PriceStream.close_callback:
            try:
                PriceStream.close_callback()
            except Exception as e:
                logging.error(f'Error in close callback: {e}')
    
    def _check_rate_limit(self):
        """Check if we're within rate limits (5 messages/second for Binance)."""
        current_time = time.time()
        # Remove messages older than 1 second
        self.message_times = [t for t in self.message_times if current_time - t < 1.0]
        
        if len(self.message_times) >= self.max_messages_per_second:
            logging.warning(f'Rate limit approached for {self.symbol}. Messages in last second: {len(self.message_times)}')
            return False
        
        self.message_times.append(current_time)
        return True

    def stop(self):
        """Gracefully stop the price stream."""
        logging.info(f'Stopping PriceStream for {self.symbol}')
        self.is_running = False
        if self.ws:
            self.ws.close()
        if self.connection_thread and self.connection_thread.is_alive():
            self.connection_thread.join(timeout=5)
        logging.info(f'PriceStream stopped for {self.symbol}')
    
    def on_open(self, ws):
        """Handle WebSocket connection opened."""
        self.reconnect_attempts = 0  # Reset on successful connection
        self.last_ping_time = time.time()
        logging.info(f'PriceStream connection opened for {self.symbol}')
        

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closed."""
        logging.info(f'PriceStream connection closed for {self.symbol}. Code: {close_status_code}, Message: {close_msg}')
        if not self.is_running and PriceStream.close_callback:
            try:
                PriceStream.close_callback()
            except Exception as e:
                logging.error(f'Error in close callback: {e}')
    
    def on_error(self, ws, error):
        """Handle WebSocket errors."""
        logging.error(f'PriceStream WebSocket error for {self.symbol}: {error}')
    
    def on_ping(self, ws, message):
        """Handle ping from server."""
        self.last_ping_time = time.time()
        logging.debug(f'Received ping for {self.symbol}')
    
    def on_pong(self, ws, message):
        """Handle pong from server."""
        logging.debug(f'Received pong for {self.symbol}')


    def on_message(self, ws, message):
        """Handle incoming price tick messages with error handling and rate limiting."""
        try:
            # Check rate limits
            if not self._check_rate_limit():
                return
            
            json_message = json.loads(message)
            
            # Validate message structure
            if 's' not in json_message:
                logging.warning(f'Invalid message format for {self.symbol}: missing symbol')
                return
                
            symbol = json_message['s']
            
            # Handle kline data
            if 'k' in json_message:
                data = json_message['k']
                
                # Validate kline data
                required_fields = ['c', 'o', 'h', 'l', 'v']
                if not all(field in data for field in required_fields):
                    logging.warning(f'Invalid kline data for {symbol}: missing required fields')
                    return
                
                if PriceStream.log_ticks:
                    logging.info(f'{symbol} Close: {data["c"]} Volume: {data["v"]}')
                
                # Call trading strategy with error handling
                if PriceStream.strategy:
                    try:
                        PriceStream.strategy.trading_strategy(symbol, data)
                    except Exception as e:
                        logging.error(f'Strategy error for {symbol}: {e}')
                        # Don't stop the stream for strategy errors
            
            # Handle order book data
            elif 'b' in json_message and 'a' in json_message:
                # Book ticker data
                if PriceStream.log_ticks:
                    logging.info(f'{symbol} Bid: {json_message["b"]} Ask: {json_message["a"]}')
            
        except json.JSONDecodeError as e:
            logging.error(f'JSON decode error for {self.symbol}: {e}')
        except Exception as e:
            logging.error(f'Message processing error for {self.symbol}: {e}')
            # Don't reconnect for message processing errors

    
    def __make_socket_uri(self, stream_type: StreamType, base_asset: str, quote_asset: str, interval=None):
        symbol = base_asset.lower() + quote_asset.lower()
        if stream_type == StreamType.KLINE:
            if not interval:
                raise Exception('KLine stream needs an interval')
            return SOCKET_BASE + '/ws/{}@kline_{}'.format(symbol, interval)
        if stream_type == StreamType.ORDER_BOOK:
            return SOCKET_BASE + '/ws/{}@bookTicker'.format(symbol)


class Pricer:
    def __init__(self, client=None):
        if client:
            self.client = client
        else:
            env = Environment()
            api_key = env.get_binance_key('api')
            secret_key = env.get_binance_key('secret')
            self.client = Client(api_key, secret_key, testnet = not env.is_live)
        if not self.client:
            raise Exception('Could not initialise Pricer object with client.')
    
    def get_average_price(self, symbol: str):
        return (symbol, self.client.get_avg_price(symbol=symbol))
    
    def get_average_prices(self, symbols: list):
        '''
        Gets the average price for a list of symbols using multithreading.
        Returns
        -------
        Dictionary {symbol: price}
        '''
        prices = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            res = [executor.submit(self.get_average_price, s) for s in symbols]
            for task in concurrent.futures.as_completed(res):
                try:
                    result = task.result()
                    symbol, price = result[0], float(result[1].get('price'))
                    prices[symbol] = price
                except (BinanceAPIException, BinanceRequestException) as e:
                    logging.error(f'Binance API error for {result[0]}: {e}')
                except Exception as e:
                    logging.error(f'Unexpected error getting price for {result[0] if result else "unknown"}: {e}')
        return prices