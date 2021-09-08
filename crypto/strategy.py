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
        self.account = Account(client)
        self.start_val = self.account.get_portfolio_value()
        logging.info('Account value: ${}'.format(self.start_val))
        self.exec = OrderExecutor(client)
        self.session = session
        self.notifier = notifier
        self.closes = []

    def _create_order_message(self, side, order_type, quantity, symbol, order_result_dict):
        '''Formats a message update to user about successful order execution.'''
        avg, net = order_result_dict['average_price'], order_result_dict['net']
        s = f'Order Executed:\nPair Symbol: {symbol}\nSide: {side}\nOrder Type: {order_type}\nQuantity: {quantity}\nAverage Execution Price: {avg}\nNet Trade Spend: {net}'
        return s

    def order(self, side, quantity, symbol, lot_step=None, order_type=ORDER_TYPE_MARKET):
        '''
        Creates a live order based on parameters specified. See OrderExecutor class for more docs.
        '''
        logging.info('Executing order: {} ({}) {} {}'.format(side, order_type, quantity, symbol))
        try:
            order = self.exec.create_order(side, quantity, symbol, lot_step=None, order_type=order_type)
            order_result = self.session.handle_order(order)
            update_msg = self._create_order_message(side, order_type, quantity, symbol, order_result)
        except Exception as e:
            update_msg = 'Order Execution Failed: {} ({}) {} {}\nError: {}'.format(side, order_type, quantity, symbol, str(e))
            logging.info(update_msg)
        self.notifier.update(update_msg)


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
    def __init__(self, client):
        super().__init__(client)

    def trading_strategy(self, symbol, data):
        print('Trading strategy from ', type(self).__name__)
        logging.info(data['c'])


    