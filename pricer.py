import websocket
import json
from constants import SOCKET_BASE
from pprint import pprint

class PriceStream:
    '''
    Streams price for an asset pair as KLine/Candlestick data. The stream pushes updates
    to the current candlestick every second.
    '''

    def __init__(self, asset, ref_asset='usdt', interval='1m'):
        '''
        Initialize a price stream for an asset pair.
        Parameters
        ----------
        asset: String, ticker for asset
        ref_asset: String, ticker for reference asset (defaults to USDT)
        interval: String, interval for candlestick (minute (m), hour (h), day (d)). Defaults to 1 minute
        '''
        socket = self.__make_socket_uri(asset, ref_asset, interval)
        self.ws = websocket.WebSocketApp(socket, on_open=PriceStream.on_open, on_close=PriceStream.on_close, on_message=PriceStream.on_message)
    
    def run(self):
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
        print('Open: {} Close: {}'.format(o,c))

    def __make_socket_uri(self, asset, ref_asset, interval):
        symbol = asset.lower() + ref_asset.lower()
        return SOCKET_BASE + '/ws/{}@kline_{}'.format(symbol, interval)