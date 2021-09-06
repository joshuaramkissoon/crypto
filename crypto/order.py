from binance.enums import *
from crypto.constants import AssetFilter
import numpy as np
from pprint import pprint

class OrderExecutor:

    MIN_NOTIONAL = 10.01

    def __init__(self, client):
        self.client = client

    def prepare_order(self, symbol):
        '''
        Gets step precision for LOT_SIZE. Other parameters can be preloaded here before an order is created.
        '''
        try:
            info = self.client.get_symbol_info(symbol=symbol).get('filters')
            lot_size_filter = self.__get_filter(info, AssetFilter.lot_size)
            min_notional = self.__get_min_notional(self.__get_filter(info, AssetFilter.min_notional))
            step_precision = self.__get_lot_precision(lot_size_filter)
            return step_precision, min_notional
        except Exception as e:
            print('Could not prepare order: ', e)

    def __get_filter(self, filter_arr: list, filter_name) -> dict:
        try:
            return list(filter(lambda d: d['filterType'] == filter_name, filter_arr))[0]
        except Exception as e:
            raise Exception(e)
    
    def __get_lot_precision(self, info: dict) -> int:
        '''Gets the lot step precision.'''
        lot_step = float(info.get('stepSize'))
        return np.abs(np.log10(lot_step)).astype(int)

    def __get_min_notional(self, info: dict) -> int:
        '''
        Gets the minimum notional value of an order.
        notional_value = price * quantity
        '''
        return float(info.get('minNotional'))
    
    def create_order(self, side, quantity, symbol, lot_step=None, order_type=ORDER_TYPE_MARKET):
        '''
        Creates a live asset order.
        
        Parameters
        ----------
        side: Binance enum, SIDE_BUY or SIDE_SELL
        quantity: float, quantity of base asset to order. Must satisfy filter conditions. See https://github.com/joshuaramkissoon/crypto#executing-orders for more info.
        symbol: str, asset pair (e.g. ETHGBP, ETHUSDT)
        lot_step: Optional float, step precision for LOT_SIZE filter
        order_type: Binance enum
        '''
        if quantity == 'min':
            try:
                # Get average price for symbol
                price = self.client.get_avg_price(symbol=symbol).get('price')
                # Calculate minimum amount based on Binance filters (assuming MIN_NOTIONAL=10)
                min_qty = round(self.MIN_NOTIONAL / float(price), lot_step)
                print(min_qty, min_qty*float(price))
            except Exception as e:
                print('Order not executed: ', e)
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity if quantity != 'min' else min_qty
            )
            pprint(order)
            return order
        except Exception as e:
            print('Order not executed: ', e)
            return False