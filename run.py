from binance.client import Client
from binance.enums import *
from crypto import Environment
from pprint import pprint
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

config_path = 'config-local.yaml'
is_live_account = True       # Set to True for live trading
env = Environment(config_path, is_live=is_live_account)

api_key = env.get_variable('api')
secert_key = env.get_variable('secret')

client = Client(api_key, secert_key, testnet = not is_live_account)

# Check user balances
account = Account(client)
balances = account.get_portfolio()
pprint(balances)

from crypto.algo import AlgoTrader
from crypto.strategy import RSI, MA

bot = AlgoTrader(client, 'ETH', 'GBP', strategy=RSI)
bot.trade()