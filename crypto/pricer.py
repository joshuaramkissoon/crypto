import websocket
import json
from crypto.constants import SOCKET_BASE
import concurrent.futures
import threading
from pprint import pprint
import logging

class PriceStream:
    '''
    Streams price for an asset pair as KLine/Candlestick data. The stream pushes updates
    to the current candlestick every second.
    '''

    def __init__(self, base_asset, quote_asset='usdt', interval='1m', log_ticks=False, strategy=None, session=None, notifier=None, client=None, close_callback=None):
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
        socket = self.__make_socket_uri(base_asset, quote_asset, interval)
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
        wst = threading.Thread(target=self.ws.run_forever)
        wst.daemon = True
        wst.start()

    def stop(self):
        self.ws.close()
    
    def on_open(ws):
        logging.info('PriceStream connection opened')

    def on_close(ws, *args):
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

    def __make_socket_uri(self, base_asset, quote_asset, interval):
        symbol = base_asset.lower() + quote_asset.lower()
        return SOCKET_BASE + '/ws/{}@kline_{}'.format(symbol, interval)

class Pricer:
    def __init__(self, client):
        self.client = client
    
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