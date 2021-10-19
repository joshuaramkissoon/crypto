import logging, time
from crypto.helpers import format_runtime
from crypto.order import Order

class SessionTracker:
    '''
    Class that keeps track of an algo-trading session (Profit/Loss, positions etc.).
    '''
    
    def __init__(self, client, start_fiat=0):
        self.client = client
        self.profit = 0
        self.start_time = time.perf_counter()
        self.is_running = True
        self.orders = []
        self.fiat = start_fiat
        self.crypto_amt = 0

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
        order = Order(
            order['symbol'], 
            order['side'], 
            order['fills'], 
            float(order['executedQty']), 
            float(order['cummulativeQuoteQty']), 
            client=self.client
        )
        self.orders.append(order)
        order_result = order.get_result()
        net = self.aggregate_fills(order.fills)
        if order_result['side'] == 'BUY':
            self.crypto_amt += order_result['amount']
            self.fiat -= order.cum_quote_qty
            self.profit -= order.cum_quote_qty
        else:
            self.fiat += order_result['amount']
            self.crypto_amt -= order.quantity
            commission = order.get_total_commission()
            self.profit += (order.cum_quote_qty - commission)
            net -= commission
        logging.info('Profit for session: {}'.format(self.profit))
        return {
            'net': net if order.side == 'SELL' else -net,
            'average_price': order.get_average_price()
        }

    def aggregate_fills(self, fills: list) -> float:
        '''
        Gets the aggregate result of all the fills in an order. 
        '''
        return sum([float(f['price'])*float(f['qty']) for f in fills])

    def get_session_info(self):
        logging.info('Session runtime: {}'.format(self.get_session_runtime()))
        logging.info('Profit for session: {}'.format(self.profit))

    def get_session_runtime(self):
        if self.is_running:
            t = round(time.perf_counter()-self.start_time, 2)
            return format_runtime(t)
        else:
            return format_runtime(self.runtime)