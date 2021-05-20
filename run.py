from binance.client import Client
from binance.enums import *
from crypto import *
from pprint import pprint
import concurrent.futures


config_path = 'config-local.yaml'
is_live_account = True       # Set to True for live trading
env = Environment(config_path, is_live=is_live_account)

api_key = env.get_variable('api')
secert_key = env.get_variable('secret')

client = Client(api_key, secert_key, testnet = not is_live_account)

account = Account(client)
balances = account.get_portfolio()
pprint(balances)

stream = PriceStream('ETH')
# Change PriceStream handle_tick method to implement strategy
stream.run()