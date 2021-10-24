from crypto.account import Account
from crypto.environment import Environment
from crypto.notifier import Notifier
from crypto.pricer import PriceStream
from crypto.session import SessionTracker
from crypto.strategy import Strategy
from crypto.helpers import currency, TelegramHelpers
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
        self.base_asset = base_asset
        self.quote_asset = quote_asset
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
        self.session_info = self.get_session_info()
        stop_msg = self._create_stop_message()
        logging.info(stop_msg.replace('*', ''))
    
    def get_session_info(self):
        return {
            'profit': currency(self.session.profit),
            'runtime': self.session.get_session_runtime(),
            'orders': self.session.orders
        }

    def _create_stop_message(self, markdown=False):
        runtime, profit, orders = self.session_info['runtime'], self.session_info['profit'], self.session_info['orders']
        summary_msg = f'*Trading Stopped*:\nSession Runtime: {runtime}\nProfit: {profit}\n\n'
        balances_msg = self.create_balance_update_message()
        trades_msg = None
        if not orders:
            trades_msg = 'No trades yet'
        else:
            trades_msg = '*Trades:*'
            for i, order in enumerate(orders):
                trades_msg += f'\n\n*{i}.* {TelegramHelpers.create_order_msg(order, markdown=markdown)}'
        msg = summary_msg + balances_msg + trades_msg
        return msg.replace('*', '') if not markdown else msg

    def get_balance_update(self):
        '''Return the current base and quote balances in user's account.'''
        base_balance = self.account.get_asset_balance(self.base_asset)
        quote_balance = self.account.get_asset_balance(self.quote_asset)
        total_balance = lambda dct: float(dct['free']) + float(dct['locked'])
        total_base_balance = total_balance(base_balance)
        total_quote_balance = total_balance(quote_balance)
        return {'base': total_base_balance, 'quote': total_quote_balance}

    def create_balance_update_message(self, markdown=False):
        balances = self.get_balance_update()
        base_balance, quote_balance = balances['base'], balances['quote']
        balances_msg = f'*Balances:*\n\n{self.base_asset}: {base_balance}\n{self.quote_asset}: {quote_balance}\n\n'
        return balances_msg.replace('*', '') if not markdown else balances_msg