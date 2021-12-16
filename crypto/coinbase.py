import time, cbpro, requests
from pprint import pprint
from datetime import datetime
from crypto.helpers import make_url

class CoinbaseOrderbookStream(cbpro.WebsocketClient):
    def __init__(self, base, quote, on_snapshot_handler=None, on_update_handler=None, orderbook=None):
        super().__init__()
        self.products = [base + '-' + quote]
        self.url = "wss://ws-feed.pro.coinbase.com/"
        self.channels=["level2"]
        self.orderbook = orderbook
        self.on_snapshot_handler = on_snapshot_handler
        self.on_update_handler = on_update_handler
    
    def on_open(self):
        pass

    def on_message(self, msg):
        handler = None
        if msg['type'] == 'snapshot':
            handler = self.on_snapshot_handler
        elif msg['type'] == 'l2update':
            handler = self.on_update_handler
        if handler is not None:
            handler(self.orderbook, msg)

class Coinbase:
    base_url = 'https://api.pro.coinbase.com/'

    def get_products():
        '''Get all products supported by Coinbase.'''
        url = make_url(Coinbase.base_url, 'products')
        res = requests.get(url)
        if res.status_code != 200:
            raise Exception('Could not fetch products')
        return res.json()

    def is_valid_product(base, quote):
        '''Checks if a product pair is valid on Coinbase.'''
        product = f'{base}-{quote}'
        products = Coinbase.get_products()
        filtered = list(filter(lambda x: x['id'] == product, products))
        return len(filtered) != 0