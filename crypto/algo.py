from crypto import PriceStream, Account
import logging, schedule

class AlgoTrader:
    def __init__(self, client, base_asset, quote_asset, strategy, price_interval='1m'):
        self.client = client
        self.account = Account(client)
        self.symbol = base_asset.upper() + quote_asset.upper()
        self.price_stream = PriceStream(
            base_asset, 
            quote_asset=quote_asset, 
            interval=price_interval, 
            strategy=strategy, 
            client=client
        )
    
    def trade(self):
        logging.info('Trading started for pair: {}'.format(self.symbol))
        self.price_stream.run()

    