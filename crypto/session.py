import logging
import time
from crypto.helpers import format_runtime

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

        Returns
        -------
        dict: Containing order information (average price, net spend etc.)
        '''
        side, fills = order['side'], order['fills']
        net = self.aggregate_fills(fills)
        if side == 'BUY':
            self.profit -= net
        elif side == 'SELL':
            self.profit += self.commission(net)
        logging.info('Profit for session: {}'.format(self.profit))
        return {
            'net': net,
            'average_price': self.get_average_price(fills)
        }

    def aggregate_fills(self, fills: list) -> float:
        '''
        Gets the aggregate result of all the fills in an order. 
        '''
        return round(sum([float(f['price'])*float(f['qty']) for f in fills]), 2)

    def get_average_price(self, fills: list) -> float:
        '''Return the average execution price of an order.'''
        total_qty = sum([float(f['qty']) for f in fills])
        fraction = lambda x: x/total_qty
        weighted_avg = sum([fraction(float(f['qty']))*float(f['price']) for f in fills])
        return round(weighted_avg, 2)

    def commission(self, value: float, commission_percent=0.1) -> float:
        '''
        Calculates value after commission is subtracted.
        Parameters
        ----------
        commission_percent: Percent commission to use (Default is 0.1%)
        '''
        return (1-0.01*commission_percent)*value

    def get_session_info(self):
        logging.info('Session runtime: {}'.format(self.get_session_runtime()))
        logging.info('Profit for session: {}'.format(self.profit))

    def get_session_runtime(self):
        if self.is_running:
            t = round(time.perf_counter()-self.start_time, 2)
            return format_runtime(t)
        else:
            return format_runtime(self.runtime)