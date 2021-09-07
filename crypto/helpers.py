import logging
import threading
import time
from crypto.environment import Environment
import crypto.strategy

def format_runtime(t: float):
    '''Converts a session runtime in seconds to an appropriate unit and returns the string.'''
    SECOND_THRESHOLD = 60
    MINUTE_THRESHOLD = 60 * SECOND_THRESHOLD
    is_second = lambda x: x < SECOND_THRESHOLD
    is_minute = lambda x: x >= SECOND_THRESHOLD and x < MINUTE_THRESHOLD
    is_hour = lambda x: x >= MINUTE_THRESHOLD
    if is_second(t):
        return f'{t} seconds'
    elif is_minute(t):
        t = round(t/60, 2)
        return f'{t} minutes'
    elif is_hour(t):
        t = round(t/3600, 2)
        return f'{t} hours'

def currency(c: float):
    '''Format currency.'''
    return '£{:0,.2f}'.format(c).replace('£-', '-£')

def get_strategy(s: str):
    '''Return the Strategy subclass with name matching the string s.'''
    env = Environment()
    strategies = env.get_root_variable('strategy')
    if not strategies:
        raise Exception('No strategies found in config file.')
    return getattr(crypto.strategy, s)

class SessionTracker:
    '''
    Class that keeps track of an algo-trading session (Profit/Loss, positions etc.).
    '''
    
    def __init__(self):
        self.profit = 0
        self.start_time = time.perf_counter()
        self.is_running = True

    def stop(self):
        self.is_running = False
        self.runtime = round(time.perf_counter()-self.start_time, 2)

    def handle_order(self, order):
        '''
        Handles an order result by calculating the net amount paid/received.
        '''
        side = order['side']
        net = self.aggregate_fills(order['fills'])
        if side == 'BUY':
            self.profit -= net
        elif side == 'SELL':
            self.profit += self.commission(net)
        logging.info('Profit for session: {}'.format(self.profit))

    def aggregate_fills(self, fills: list) -> float:
        '''
        Gets the aggregate result of all the fills in an order. 
        '''
        return sum([float(f['price'])*float(f['qty']) for f in fills])

    def commission(self, value: float, commission_percent=0.1) -> float:
        '''
        Calculates value after commission is subtracted.
        Parameters
        ----------
        commission_percent: Percent commission to use (Default is 0.1%)
        '''
        return (1-0.01*commission_percent)*value

    def get_session_info(self):
        logging.info('Session runtime: {} seconds'.format(self.get_session_runtime()))
        logging.info('Profit for session: {}'.format(self.profit))

    def get_session_runtime(self):
        if self.is_running:
            t = round(time.perf_counter()-self.start_time, 2)
            return format_runtime(t)
        else:
            return format_runtime(self.runtime)