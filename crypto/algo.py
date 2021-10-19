from crypto.account import Account
from crypto.environment import Environment
from crypto.notifier import Notifier
from crypto.pricer import PriceStream
from crypto.session import SessionTracker
from crypto.strategy import Strategy
from crypto.helpers import currency
import logging, schedule, threading

class AlgoTrader:
    def __init__(
        self, 
        client, 
        base_asset: str, 
        quote_asset: str,
        strategy: Strategy = None,
        price_interval: str = '1m', 
        log_ticks: bool = False, 
        default_notifications: bool = False,
        notifier: Notifier = None, 
        account: Account = None
    ):
        self.client = client
        self.session = SessionTracker(client)
        self.account = account if account else Account(client)
        self.symbol = base_asset.upper() + quote_asset.upper()
        if default_notifications and not notifier:
            # Create default Notifier
            notifier = Notifier(Environment().get_telegram_key())
            notifier._auth()
        elif default_notifications and notifier:
            raise Exception('Default notifications can\'t be set if a Notifier object is provided.')
        self.price_stream = PriceStream(
            base_asset, 
            quote_asset=quote_asset, 
            interval=price_interval,
            log_ticks=log_ticks, 
            strategy=strategy, 
            session=self.session,
            notifier=notifier,
            client=client,
            close_callback=self.session.get_session_info
        )
    
    def trade(self):
        logging.info('Trading started for pair: {}'.format(self.symbol))
        self.start_val = self.account.get_portfolio_value()
        logging.info('Account value: ${}'.format(self.start_val))
        self.price_stream.run()

    def stop(self):
        self.price_stream.stop()
        session_info = self._get_session_info()
        runtime, profit, orders = str(session_info['runtime']), session_info['profit'], session_info['orders']
        summary_msg = f'Trading Stopped:\nSession Runtime: {runtime}\nProfit: {profit}'
        print(summary_msg)
        print('--------Trades--------')
        for order in orders:
            print(f'Symbol: {order.symbol}\tSide: {order.side}\tQuantity: {order.quantity}\tAverage Price: {order.get_average_price()}')
    
    def _get_session_info(self):
        return {
            'profit': currency(self.session.profit),
            'runtime': self.session.get_session_runtime(),
            'orders': self.session.orders
        }