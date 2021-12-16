from crypto.coinbase import CoinbaseOrderbookStream, Coinbase
from crypto.helpers import weighted_average
from collections import OrderedDict

class Orderbook:
    def __init__(self, base, quote):
        self.base = base
        self.quote = quote
        self.stream = CoinbaseOrderbookStream(base, quote, on_snapshot_handler=Orderbook._handle_snapshot_message, on_update_handler=Orderbook._handle_update_message, orderbook=self)
        self.orderbook = {'asks': {}, 'bids': {}}

    def _handle_snapshot_message(self, data):
        # Bids
        bids = data['bids'] # List of tuple (price, quantity)
        self.orderbook['bids'] = {float(b[0]): float(b[1]) for b in bids}
        # Asks
        asks = data['asks']
        self.orderbook['asks'] = {float(a[0]): float(a[1]) for a in asks}
    
    def _update_book(self, side, price, quantity):
        _type = None
        if side == 'buy':
            _type = 'bids'
        elif side == 'sell':
            _type = 'asks'
        if quantity == 0:
            # Remove price level
            if self.orderbook[_type].get(price):
                del self.orderbook[_type][price]
        else:
            self.orderbook[_type][price] = quantity

    def _handle_update_message(self, data):
        for change in data['changes']:
            # change: [side, price, quantity]
            self._update_book(change[0], float(change[1]), float(change[2]))
    
    def get_top_asks(self, n):
        '''Lowest n prices people are looking to sell base asset for quote asset.'''
        _sorted = OrderedDict(sorted(self.orderbook['asks'].items()))
        return list(_sorted.items())[:n]

    def get_top_bids(self, n):
        '''Highest n prices people are looking to buy base asset with quote asset.'''
        _sorted = OrderedDict(sorted(self.orderbook['bids'].items(), reverse=True))
        return list(_sorted.items())[:n]
    
    def get_exchange_rate_from_quote_amt(self, side, quote_amt):
        def quote_amount(pair):
            return pair[0]*pair[1]
        
        remainder = quote_amt
        base_amt = 0
        fills = []
        book = None
        if side == 'buy':
            book = self.get_top_asks(5)
        elif side == 'sell':
            book = self.get_top_bids(5)
        if not book:
            print('No book')
            return None
        curr = 0
        while remainder:
            try:
                if quote_amount(book[curr]) >= remainder:
                    qty = remainder/book[curr][0]
                    base_amt += qty
                    fills.append((book[curr][0], qty))
                    remainder = 0
                else:
                    fills.append(book[curr])
                    remainder -= quote_amount(book[curr])
                    base_amt += book[curr][1]
                curr += 1
            except Exception as e:
                print(e)
                return None
        return (weighted_average(fills), base_amt)

    def get_exchange_rate(self, side, qty):
        remainder = qty
        fills = []
        book = None
        if side == 'buy':
            book = self.get_top_asks(5)
        elif side == 'sell':
            book = self.get_top_bids(5)
        if not book:
            print('No book')
            return None
        curr = 0
        amt = 0
        while remainder:
            try:
                if book[curr][1] >= remainder:
                    fills.append((book[curr][0], remainder))
                    amt += remainder*book[curr][0]
                    remainder = 0
                else:
                    fills.append((book[curr][0], book[curr][1]))
                    amt += book[curr][1]*book[curr][0]
                    remainder -= book[curr][1]
                curr += 1
            except Exception as e:
                print(e)
                return None
        return weighted_average(fills), amt

    def start(self):
        self.stream.start()

class OrderbookManager:
    def __init__(self):
        self.managed_book = {}

    def add_pair(self, base, quote):
        if not Coinbase.is_valid_product(base, quote):
            raise Exception('Invalid Coinbase product')
        ob = Orderbook(base, quote)
        ob.start()
        self.managed_book[base+quote] = ob

    def get_exchange_rate(self, base, quote, side, quantity):
        return self.managed_book[base+quote].get_exchange_rate(side, quantity)
    
    def get_exchange_rate_from_quote_amt(self, base, quote, side, quote_amount):
        return self.managed_book[base+quote].get_exchange_rate_from_quote_amt(side, quote_amount)