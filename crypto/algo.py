from crypto import PriceStream, OrderExecutor

class AlgoTrader:
    def __init__(self, client, base_asset, quote_asset, trading_strategy, price_interval='1m'):
        self.client = client
        self.price_stream = PriceStream(base_asset, quote_asset=quote_asset, interval=price_interval, trading_strategy=trading_strategy)

    def trade(self):
        self.price_stream.run()