from crypto.environment import Environment
from coinbase.wallet.client import Client

class TriangularArbitrage:
    def __init__(self):
        self.environment = Environment()
        keys = self.environment.get_coinbase_keys()
        self.client = Client(keys['api-key'], keys['secret-key'])

    def get_rates(self, ticker):
        rates = self.client.get_exchange_rates(currency=ticker)
        return rates['rates']

    def arbitrage(self):
        eth = TriangularArbitrage().get_rates('ETH')
        r3 = float(eth.get('BTC')) # From BTC -> ETH
        btc = {k: 1/float(v) for k,v in TriangularArbitrage().get_rates('BTC').items()}
        for asset in eth:
            r1 = float(eth.get(asset))
            r2 = float(btc.get(asset))
            p = r1*r2*r3
            print('{}: {:.4f}'.format(asset, p))

    def get_exchange_rate(self, base, quote, quantity):
        '''
        Get the exchange rate between two assets.

        - Buy the base asset with the quote asset
        '''
        pass

# from crypto.arbitrage import TriangularArbitrage

# TriangularArbitrage().arbitrage()

from crypto.orderbook import Orderbook, OrderbookManager
from crypto.coinbase import Coinbase

# wsClient = WebsocketClient()
# wsClient.start()

# cos = CoinbaseOrderbookStream(['ETH-USD'])
# cos.start()

# manager = OrderbookManager()
# manager.add_pair('LINK', 'BTC')
# manager.add_pair('LINK', 'ETH')
# manager.add_pair('ETH', 'BTC')
# manager.add_pair('BTC', 'USD')
# manager.add_pair('USD', 'ETH')

# orderbook = Orderbook('ETH', 'BTC')
# orderbook.start()

import time

# while True:
    # time.sleep(1)
#     # x = manager.get_exchange_rate('ETH', 'USD', 1)
    # amt_btc = manager.get_exchange_rate('ETH', 'BTC', 1) # Amount of BTC you get if you buy BTC with 1 ETH
    # print(amt_btc)
    
    # r2 = manager.get_exchange_rate('LINK', 'BTC', 1)
    # print(r1)
    # r1 = manager.get_exchange_rate('ETH', 'BTC', 1)
    # print(r1)
    # if rate := orderbook.get_exchange_rate(1):
        # print(f'Exchange rate for 1: {rate}')
    # if rate := orderbook.get_exchange_rate(5):
    #     print(f'Exchange rate for 5: {round(rate, 2)}')

# import requests, pprint


# res = requests.get(url)
# pprint.pprint(res.json())

'''
If we want to buy, look at the orderbook asks
If we want to sell, look at the orderbook bids
'''
# from pprint import pprint
# products = [('BTC', 'USD'), ('LTC', 'BTC'), ('USD', 'LTC'), ('LTC', 'USD')]
# Buy BTC with USD
# Buy LTC with BTC
# Sell LTC for USD

# for p in products:
    # print(Coinbase.is_valid_product(p[0], p[1]))

# ob = Orderbook('BTC', 'USD')
# ob.start()

# while True:
#     time.sleep(1)
#     pprint(ob.get_top_asks(5))

# ob = Orderbook('LTC', 'BTC')
# ob.start()

# while True:
#     time.sleep(1)
#     pprint(ob.get_top_asks(5))

# ob = Orderbook('LTC', 'USD')
# ob.start()

# while True:
#     time.sleep(1)
#     pprint(ob.get_top_bids(5))

# start_usd = 100
# manager = OrderbookManager()
# manager.add_pair('BTC', 'USD')
# manager.add_pair('LTC', 'BTC')
# manager.add_pair('LTC', 'USD')
# amt_btc, amt_ltc, end_usd = 0, 0, 0
# while True:
#     time.sleep(1)
#     print('---------------------')
#     if res := manager.get_exchange_rate_from_quote_amt('BTC', 'USD', 'buy', start_usd):
#         rate, amt_btc = res
#         print(f'BTCUSD Rate: {rate} Amount of BTC: {amt_btc}')
#     if res := manager.get_exchange_rate_from_quote_amt('LTC', 'BTC', 'buy', amt_btc):
#         rate, amt_ltc = res
#         print(f'LTCBTC Rate: {rate} Amount of LTC: {amt_ltc}')
#     if res := manager.get_exchange_rate('LTC', 'USD', 'sell', amt_ltc):
#         rate, end_usd = res
#         print(f'LTCUSD Rate: {rate} Amount of USD: {end_usd}')
    