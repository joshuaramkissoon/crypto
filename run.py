from binance.client import Client
from binance.enums import *
from crypto import Environment, Account, OrderExecutor
from crypto.mobile import MobileClient
from pprint import pprint
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

env = Environment()
api_key = env.get_binance_key('api')
secret_key = env.get_binance_key('secret')

client = Client(api_key, secret_key, testnet = not env.is_live)

MobileClient(client).start()
