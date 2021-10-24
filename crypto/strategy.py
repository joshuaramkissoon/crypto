from crypto.order import *
from crypto.account import Account
import logging, talib, pprint
import numpy as np

class Strategy:
    '''
    Superclass for all strategy implementations.
    Parameters
    ----------
    client - Binance Client object
    '''
    def __init__(self, client, session, notifier):
        self.client = client
        self.exec = OrderExecutor(client)
        self.session = session
        self.notifier = notifier
        self.closes = []

    def _create_order_message(self, side, order_type, quantity, symbol, order_result_dict):
        '''Formats a message update to user about successful order execution.'''
        avg, net = order_result_dict['average_price'], order_result_dict['net']
        s = f'*Order Executed:*\nPair Symbol: {symbol}\nSide: {side}\nOrder Type: {order_type}\nQuantity: {quantity}\nAverage Execution Price: {avg}\nNet Trade Spend: {net}'
        return s

    def order(self, side, quantity, symbol, lot_step=None, order_type=ORDER_TYPE_MARKET):
        '''
        Creates a live order based on parameters specified. See OrderExecutor class for more docs.
        '''
        is_successful = False
        error = None
        logging.info('Executing order: {} ({}) {} {}'.format(side, order_type, quantity, symbol))
        try:
            order = self.exec.create_order(side, quantity, symbol, lot_step=None, order_type=order_type)
            order_result = self.session.handle_order(order)
            update_msg = self._create_order_message(side, order_type, quantity, symbol, order_result)
            is_successful = True
        except Exception as e:
            error = str(e)
            is_successful = False
            update_msg = 'Order Execution Failed: {} ({}) {} {}\nError: {}'.format(side, order_type, quantity, symbol, str(e))
        logging.info(update_msg)
        if self.notifier and self.notifier.is_auth:
            self.notifier.update(update_msg, parse_markdown=is_successful)
        return is_successful, order_result, error


    def trading_strategy(self, symbol, data):
        '''
        Function that implements trading strategy. This method is called on receipt of every price tick (usually every 2 seconds). Subclasses must override this method to implement a specific strategy.
        
        Parameters
        ----------
        symbol (str): Asset pair symbol e.g. ETHGBP
        data: {
            "t": 123400000, // Kline start time
            "T": 123460000, // Kline close time
            "s": "BNBBTC",  // Symbol
            "i": "1m",      // Interval
            "f": 100,       // First trade ID
            "L": 200,       // Last trade ID
            "o": "0.0010",  // Open price
            "c": "0.0020",  // Close price
            "h": "0.0025",  // High price
            "l": "0.0015",  // Low price
            "v": "1000",    // Base asset volume
            "n": 100,       // Number of trades
            "x": false,     // Is this kline closed?
            "q": "1.0000",  // Quote asset volume
            "V": "500",     // Taker buy base asset volume
            "Q": "0.500",   // Taker buy quote asset volume
            "B": "123456"   // Ignore
        }
        '''
        logging.info('Open: {} \t Close: {}'.format(data['o'], data['c']))

class RSI(Strategy):
    '''Simple implementation of RSI(14) to test end to end algo trading functionality'''

    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    TRADE_AMOUNT = 0.004
    
    def __init__(self, client, session, notifier):
        self.in_position = False
        super().__init__(client, session, notifier)

    def trading_strategy(self, symbol, data):
        is_closed, close = data['x'], data['c']
        if is_closed:
            logging.info('Candle closed at {}'.format(close))
            self.closes.append(float(close))

            if len(self.closes) > self.RSI_PERIOD:
                np_closes = np.array(self.closes)
                rsi = talib.RSI(np_closes, self.RSI_PERIOD)
                last = rsi[-1]
                # print('Current RSI: {}'.format(last))

                if last > self.RSI_OVERBOUGHT:
                    if self.in_position:
                        logging.info('-----------SELL-------------')
                        self.order(
                            SIDE_SELL,
                            self.TRADE_AMOUNT,
                            symbol
                        )
                        self.in_position = False
                    else:
                        logging.info('Overbought but not in position, no action taken')
                if last < self.RSI_OVERSOLD:
                    if self.in_position:
                        logging.info('Oversold but already in position, no action taken')
                    else:
                        logging.info('-----------BUY-------------')
                        self.order(
                            SIDE_BUY,
                            self.TRADE_AMOUNT,
                            symbol
                        )
                        self.in_position = True
                if last > self.RSI_OVERSOLD and last < self.RSI_OVERBOUGHT:
                    print(logging.info('Not oversold or overbought. No action taken.'))

class MA(Strategy):
    
    def __init__(self, client, session, notifier):
        self.in_position = False
        self.count = 0
        super().__init__(client, session, notifier)

    def trading_strategy(self, symbol, data):
        if self.count < 2:
            success, error = self.order(SIDE_BUY, 0.006, symbol)
            if success:
                self.count += 1
            else:
                print('Error executing order: ', error)



class CMO(Strategy):

    PERIOD = 15
    TRADE_FRAC = 0.2
    TRADE_USDT = 20
    TRADE_AMOUNT = 0.006
    CAPITAL = 100 # USD
    OVERBOUGHT = 50
    OVERSOLD = -50

    def __init__(self, client, session, notifier):
        super().__init__(client, session, notifier)
        self.ups = []
        self.downs = []
        self.cmos = []
        self.last_close = None
        self.in_position = False
        self.buy_prices = []

    @property
    def can_trade(self):
        return len(self.ups) == self.PERIOD and len(self.downs) == self.PERIOD
    
    def trading_strategy(self, symbol, data):
        if data['x']:
            # Candle closed
            if self.last_close is None:
                self.last_close = float(data['x'])
                return
            self._handle_close(float(data['c']))
            if self.can_trade:
                cmo = self.calculate_cmo()
                self.cmos.append(cmo)
                # Check cmo against overbought and oversold threshold values, buy/sell accordingly
                if cmo > self.OVERBOUGHT and self.in_position and float(data['c']) > self.buy_prices[0]:
                    # Sell
                    success, _, _ = self.order(SIDE_SELL, self.TRADE_AMOUNT, symbol)
                    if success:
                        self.in_position = False
                        self.buy_prices.pop(0)
                    return
                if cmo < self.OVERSOLD and not self.in_position:
                    # Buy
                    success, result, _ = self.order(SIDE_BUY, self.TRADE_AMOUNT, symbol)
                    if success:
                        self.in_position = True
                        self.buy_prices.append(result['average_price'])
                    return


    def _handle_close(self, close: float):
        # Check if current close is higher or lower than last close
        direction = None
        if close > self.last_close:
            direction = 'up'
        elif close < self.last_close:
            direction = 'down'
        else:
            return
        self._add_close(close, direction)
        self.last_close = close

    def calculate_cmo(self):
        return 100*(sum(self.ups) - sum(self.downs))/(sum(self.ups) + sum(self.downs))

    def _add_close(self, close: float, direction: str):
        if len(self.ups) == self.PERIOD:
            self.ups.pop(0)
        if len(self.downs) == self.PERIOD:
            self.downs.pop(0)
        
        _list_close = None # List that gets appended with new close
        _list_zero = None # List that gets appended with zero
        if direction == 'up':
            _list_close = self.ups
            _list_zero = self.downs
        elif direction == 'down':
            _list_close = self.downs
            _list_zero = self.ups
        else:
            raise Exception('Invalid direction')
        # Append new close
        _list_close.append(close)
        _list_zero.append(0)