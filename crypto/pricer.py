import websocket
import json
from crypto.constants import SOCKET_BASE
import concurrent.futures
from pprint import pprint
import logging

class PriceStream:
    '''
    Streams price for an asset pair as KLine/Candlestick data. The stream pushes updates
    to the current candlestick every second.
    '''

    def __init__(self, base_asset, quote_asset='usdt', interval='1m', strategy=None, client=None):
        '''
        Initialize a price stream for an asset pair.
        Parameters
        ----------
        base_asset: String, ticker for asset
        quote_asset: String, ticker for reference asset (defaults to USDT)
        interval: String, interval for candlestick (minute (m), hour (h), day (d)). Defaults to 1 minute
        '''
        socket = self.__make_socket_uri(base_asset, quote_asset, interval)
        self.symbol = base_asset.upper() + quote_asset.upper()
        self.ws = websocket.WebSocketApp(socket, on_open=PriceStream.on_open, on_close=PriceStream.on_close, on_message=PriceStream.on_message)
        # Initialise trading strategy class with client object
        PriceStream.strategy = strategy(client)
    
    def run(self):
        self.ws.run_forever()
    
    def on_open(ws):
        logging.info('PriceStream connection opened')

    def on_close(ws):
        logging.info('PriceStream connection closed')

    def on_message(ws, message):
        '''
        Method called when a new price tick is received, usually every 2 seconds.
        '''
        json_message = json.loads(message)
        symbol, data = json_message['s'], json_message['k']
        # Call the trading strategy function with price data            
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
                result = task.result()
                symbol, price = result[0], float(result[1].get('price'))
                prices[symbol] = price
        return prices