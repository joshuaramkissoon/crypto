from crypto import PriceStream, Account
from crypto.environment import Environment
from crypto.helpers import SessionTracker
import crypto.strategy
import logging, schedule

class AlgoTrader:
    def __init__(self, client, base_asset, quote_asset, strategy, price_interval='1m', log_ticks=False, notifier=None):
        self.client = client
        self.session = SessionTracker()
        self.account = Account(client)
        self.symbol = base_asset.upper() + quote_asset.upper()
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
        self.price_stream.run()

    def stop(self):
        self.price_stream.stop()
    