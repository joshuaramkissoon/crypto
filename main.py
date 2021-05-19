import os
from pprint import pprint
from binance.client import Client
from binance.enums import *
from pricer import PriceStream
from environment import Environment
from order import OrderExecutor


config_path = 'config-local.yaml'
is_live_account = False
env = Environment(config_path, is_live=is_live_account)

api_key = env.get_variable('api')
secert_key = env.get_variable('secret')

client = Client(api_key, secert_key, testnet = not is_live_account)


# account = client.get_account()
# pprint(account)
# balances = account['balances']
# holdings = list(filter(lambda d: float(d['free']) != 0, balances))
# pprint(holdings)
# orders = client.get_all_orders(symbol='ETHGBP')
# pprint(orders)

exec = OrderExecutor(client)
symbol = 'ETHUSDT'
prec, min_notional = exec.prepare_order(symbol)
# exec.create_order(
#     SIDE_BUY,
#     'min',
#     'ETHGBP',
#     lot_step=prec
# )