import os
from pprint import pprint
from binance.client import Client
from pricer import PriceStream

api_key = os.environ.get('binance_api')
api_secret = os.environ.get('binance_secret')

client = Client(api_key, api_secret)
account = client.get_account()
balances = account['balances']
holdings = list(filter(lambda d: float(d['free']) != 0, balances))
pprint(holdings)


stream = PriceStream('btc')
stream.run()