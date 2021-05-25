from crypto.order import *
import logging, talib, pprint
import numpy as np

class Strategy:
    '''
    Superclass for all strategy implementations.
    Parameters
    ----------
    client - Binance Client object
    '''
    def __init__(self, client):
        self.client = client
        self.exec = OrderExecutor(client)
        self.closes = []

    def order(self, side, quantity, symbol, lot_step=None, order_type=ORDER_TYPE_MARKET):
        '''
        Creates a live order based on parameters specified.
        '''
        logging.info('Executing order: {} ({}) {} {}'.format(side, order_type, quantity, symbol))
        self.exec.create_order(side, quantity, symbol, lot_step=None, order_type=ORDER_TYPE_MARKET)

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

    RSI_PERIOD = 4
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    TRADE_SYMBOL = 'ETHGBP'
    TRADE_AMOUNT = 0.008
    
    def __init__(self, client):
        super().__init__(client)

    def trading_strategy(self, symbol, data):
        in_position = False
        is_closed, close = data['x'], data['c']
        if is_closed:
            print('Candle closed at {}'.format(close))
            self.closes.append(float(close))

            if len(self.closes) > self.RSI_PERIOD:
                np_closes = np.array(self.closes)
                rsi = talib.RSI(np_closes, self.RSI_PERIOD)
                print(rsi)
                last = rsi[-1]
                print('Current RSI: {}'.format(last))

                if last > self.RSI_OVERBOUGHT:
                    if in_position:
                        logging.info('SELL')
                    else:
                        logging.info('Overbought but not in position, no action taken')
                if last < self.RSI_OVERSOLD:
                    if in_position:
                        logging.info('Oversold but already in position, no action taken')
                    else:
                        logging.info('BUY')

class MA(Strategy):
    def __init__(self, client):
        super().__init__(client)

    def trading_strategy(self, symbol, data):
        print('Trading strategy from ', type(self).__name__)
        logging.info(data['c'])


    