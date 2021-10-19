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

class StreamType(Enum):
    KLINE = 0
    ORDER_BOOK = 1

class PriceStream:
    '''
    Streams price for an asset pair as KLine/Candlestick data. The stream pushes updates
    to the current candlestick every second.
    '''

    def __init__(self, base_asset, quote_asset='usdt', interval='1m', log_ticks=False, strategy=None, session=None, notifier=None, client=None, close_callback=None, stream_type=StreamType.KLINE):
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
        socket = self.__make_socket_uri(stream_type, base_asset, quote_asset, interval)
        self.ws = websocket.WebSocketApp(socket, on_open=PriceStream.on_open, on_close=PriceStream.on_close, on_message=PriceStream.on_message)
        self.symbol = base_asset.upper() + quote_asset.upper()
        PriceStream.log_ticks = log_ticks
        PriceStream.close_callback = close_callback
        # Initialise trading strategy class with client object
        if strategy:
            assert client, 'To use a strategy, PriceStream object must be initialised with a Binance Client.'
            PriceStream.strategy = strategy(client, session, notifier)
        else:
            PriceStream.strategy = None
    
    def run(self):
        _thread = threading.Thread(target=self.ws.run_forever)
        _thread.start()

    def stop(self):
        self.ws.close()
    
    def on_open(ws):
        logging.info('PriceStream connection opened')
        

    def on_close(ws, *args):
        if PriceStream.close_callback:
            PriceStream.close_callback()
        logging.info('PriceStream connection closed')


    def on_message(ws, message):
        '''
        Method called when a new price tick is received, usually every 2 seconds.
        '''
        json_message = json.loads(message)
        symbol, data = json_message['s'], json_message['k']
        if PriceStream.log_ticks:
            logging.info('{} Close: {}'.format(symbol, data['c']))
        # Call the trading strategy function with price data            
        if PriceStream.strategy:
            PriceStream.strategy.trading_strategy(symbol, data)

    
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
                except Exception as e:
                    print(e)
        return prices