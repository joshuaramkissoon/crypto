from crypto import PriceStream, Account
import logging, schedule

class AlgoTrader:
    def __init__(self, client, base_asset, quote_asset, strategy, price_interval='1m'):
        self.client = client
        self.account = Account(client)
        self.start_value = self.account.get_portfolio_value()
        self.price_stream = PriceStream(
            base_asset, 
            quote_asset=quote_asset, 
            interval=price_interval, 
            strategy=strategy, 
            client=client
        )
    
    def trade(self):
        logging.info('Trading started')
        logging.info('Account value: ${}'.format(self.start_value))
        self.price_stream.run()

    