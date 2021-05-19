import websocket
import json
from crypto.constants import SOCKET_BASE
import concurrent.futures
from pprint import pprint

class PriceStream:
    '''
    Streams price for an asset pair as KLine/Candlestick data. The stream pushes updates
    to the current candlestick every second.
    '''

    def __init__(self, base_asset, quote_asset='usdt', interval='1m'):
        '''
        Initialize a price stream for an asset pair.
        Parameters
        ----------
        asset: String, ticker for asset
        ref_asset: String, ticker for reference asset (defaults to USDT)
        interval: String, interval for candlestick (minute (m), hour (h), day (d)). Defaults to 1 minute
        '''
        socket = self.__make_socket_uri(base_asset, quote_asset, interval)
        self.ws = websocket.WebSocketApp(socket, on_open=PriceStream.on_open, on_close=PriceStream.on_close, on_message=PriceStream.on_message)
    
    def run(self, f):
        self.strategy = f
        self.ws.run_forever()
    
    def on_open(ws):
        print('Connection opened')

    def on_close(ws):
        print('Connection closed')

    def on_message(ws, message):
        json_message = json.loads(message)
        data = json_message['k']
        o = data['o']
        c = data['c']
        # print('Open: {} Close: {}'.format(o,c))
        PriceStream.handle_tick(o, c)

    def handle_tick(open, close):
        '''
        Implement a trading strategy here. Function params can be changed and these changed should 
        be reflected in the on_message function body.
        '''
        print(open, close)
        if open > close:
            print('Sell')
        else:
            print('Buy')

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