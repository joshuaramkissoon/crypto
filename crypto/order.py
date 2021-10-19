from binance.enums import *
from crypto.constants import AssetFilter, DEFAULT_CURRENCY
from crypto.pricer import Pricer
import numpy as np
from pprint import pprint

class Commission:
    def __init__(self, quantity, asset, fiat_value=None):
        self.quantity = quantity
        self.asset = asset
        self.fiat_value = fiat_value

class Order:
    def __init__(self, symbol: str, side: str, fills: list, quantity: float, cum_quote_qty: float, client=None):
        self.symbol = symbol
        self.side = side
        self.fills = fills
        self.quantity = quantity
        self.cum_quote_qty = cum_quote_qty
        self.average_price = self.get_average_price()
        self.client = client

    def __eq__(self, other):
        return (
            self.symbol == other.symbol 
            and self.side == other.side
            and self.fills == other.fills
            and self.quantity == other.quantity
            and self.cum_quote_qty == other.cum_quote_qty
        )

    def get_average_price(self) -> float:
        '''Return the average execution price of an order.'''
        total_qty = sum([float(f['qty']) for f in self.fills])
        fraction = lambda x: x/total_qty
        weighted_avg = sum([fraction(float(f['qty']))*float(f['price']) for f in self.fills])
        return round(weighted_avg, 2)

    def get_total_commission(self):
        return sum([float(f['commission']) for f in self.fills])
        
    # def get_commission(self):
    #     # Don't use this function
    #     if not self.fills:
    #         return None
    #     qty = sum([float(f['commission']) for f in self.fills])
    #     commissionAsset = self.fills[0]['commissionAsset']
    #     fiat_val = None
    #     if self.client and commissionAsset != DEFAULT_CURRENCY:
    #         # Convert commission asset to default currency
    #         rate = Pricer(self.client).get_average_price(commissionAsset+DEFAULT_CURRENCY)
    #         fiat_val = rate*qty
    #     return Commission(qty, commissionAsset, fiat_val)

    def get_asset(self):
        if not self.fills:
            return None
        return self.fills[0]['commissionAsset']

    def get_result(self):
        '''
        Returns the net after this order. Value is positive if side was SELL, negative if side was BUY.
        It takes into account the fiat value of the commission.
        '''
        res = {}
        res['side'] = self.side
        res['asset'] = self.get_asset() # Result of order; This is the asset that you obtained
        if self.side == 'BUY':
            amt = sum([float(f['qty']) - float(f['commission']) for f in self.fills])
        elif self.side == 'SELL':
            amt = sum([float(f['qty'])*float(f['price']) - float(f['commission']) for f in self.fills])
        else:
            raise Exception('Invalid side')
        res['amount'] = amt
        return res

        

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

        Throws
        ------
        Throws Exception if order creation not successful. Must be handled by caller.
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
        order = self.client.create_order(
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity if quantity != 'min' else min_qty
        )
        if (status := order.get('status')) and status == 'FILLED':
            # Add support for partial fills
            pprint(order)
            return order
        raise Exception('Order not filled')