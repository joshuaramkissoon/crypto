# Crypto Trading
This project uses the Binance API to get real-time streams for cryptocurrencies and make trades.

## Pre-requisites
- Python 3
- [Binance account](https://www.binance.com/en)
- [Binance API Key and Secret Key](https://www.binance.com/en/my/settings/api-management)

## Usage
1. Clone the repo to your local machine
2. Install required dependencies from `requirements.txt` file
    ```
    cd path/to/crypto
    pip install -r requirements.txt
    ```
3. Update `config.yaml` file with your Binance API and Secret keys. Demo keys can be generated [here](https://testnet.binance.vision/).
4. Set `is_live_account` to `True` if you want to connect to your live account and be able to live trade. Set to `False` otherwise.
5. You can access account data using the `Account` class, and run a continuous algo-trading bot with the `AlgoTrader` class.

### Streaming real-time prices
Real-time prices can be streamed using webhooks and the `PriceStream` class handles this. A `PriceStream` object is initialised with the following parameters:
- `base_asset` - Ticker/Symbol for asset (e.g. ETH)
- `quote_asset` - Ticker/Symbol for reference asset (e.g. USDT)
- `interval` - Interval for candlestick 
- `strategy` - Optional class (subclass of Strategy) that implements trading strategy
- `client` - Binenace client object. Required if strategy is used.

```python3
stream = PriceStream(base_asset='ETH', quote_asset='GBP', interval='1m')
stream.run()
```

### Executing orders
Orders have requirements, called _filters_, that need to be satisfied for the order to be successful. More info on all filters can be found [here](https://sammchardy.github.io/binance-order-filters/). The important filters are `MIN_NOTIONAL` and `LOT_SIZE`. These filters can be found by getting the symbol info for a pair.

```python
info = client.get_symbol_info('ETHGBP')
```

`MIN_NOTIONAL` defines the minimum value calculated in the quote asset for a symbol. If you are trading `ETHGBP`, the quote symbol is `GBP` and the notional value is calculated as:
```
notional_value = price * quantity
```
The minimum quantity allowed for the symbol is calculated using this value:
```python
min_quantity = MIN_NOTIONAL / price
```

This quantity must have precision specified by the `LOT_SIZE` precision. The `OrderExecutor` class has a method `prepare_order` that returns the lot size step precision and the `MIN_NOTIONAL` value. The `create_order` method is used to place trades.

```python3
exec = OrderExecutor(client)
exec.create_order(side=SIDE_BUY, quantity=0.1, symbol='ETHGBP', order_type=ORDER_TYPE_MARKET)
```

If the order is executed, a response will be printed showing the order details.

### Creating a trading strategy
A trading strategy is created by subclassing the `Strategy` class (in `strategy.py`). This base class handles logic for accessing account details and executing orders. Your implementation only needs to have a basic `__init__` and to override the `trading_strategy` method:

```python3
class MovingAverage(Strategy):
    def __init__(self, client):
        super().__init__(client)

    def trading_strategy(self, symbol, data):
        '''
        Implement your logic to handle price ticks here.
        '''
        close = data['c']
        if is_low(close):
            # Create a buy order
            self.order(SIDE_BUY, 0.1, 'ETHUSDT')
        elif is_high(close):
            # Create a sell order
            self.order(SIDE_SELL, 0.1, 'ETHUSDT')
```

The `trading_strategy` method is called every time a new tick is received (usually every 2 seconds). The method receives 2 arguments:
- `symbol` - Asset pair of interest
- `data` - Price tick received

`data` has the following model:
```python3
data: {
    "t": 123400000, # Candlestick start time
    "T": 123460000, # Candlestick close time
    "s": "BNBBTC",  # Symbol
    "i": "1m",      # Interval
    "f": 100,       # First trade ID
    "L": 200,       # Last trade ID
    "o": "0.0010",  # Open price
    "c": "0.0020",  # Close price
    "h": "0.0025",  # High price
    "l": "0.0015",  # Low price
    "v": "1000",    # Base asset volume
    "n": 100,       # Number of trades
    "x": False,     # Is this candlestick closed?
    "q": "1.0000",  # Quote asset volume
    "V": "500",     # Taker buy base asset volume
    "Q": "0.500",   # Taker buy quote asset volume
    "B": "123456"   # Ignore
}
```

There's no need to override the `order` method from the base `Strategy` class.

### Implementing a trading strategy
The `AlgoTrader` class can be used to run a trading strategy. It opens a real-time stream of price data for an asset pair and a user-defined class handles the price data received on every tick, deciding whether to make a trade or not. 

It is initialised with a `Client` object, base and quote asset symbols and a class implementation of a trading strategy. Start trading by calling the `trade` method.

```python3
bot = AlgoTrader(client, 'ETH', 'USDT', strategy=MovingAverage)
bot.trade()
```
