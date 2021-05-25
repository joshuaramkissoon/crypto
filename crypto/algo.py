from crypto import PriceStream

class AlgoTrader:
    def __init__(self, client, base_asset, quote_asset, strategy, price_interval='1m'):
        self.client = client
        self.price_stream = PriceStream(base_asset, quote_asset=quote_asset, interval=price_interval, strategy=strategy, client=client)

    def trade(self):
        self.price_stream.run()