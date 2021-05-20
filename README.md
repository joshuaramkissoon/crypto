# Crypto Trading
This project uses the Binance API to get real-time streams for cryptocurrencies and make trades.

## Pre-requisites
- Python 3
- [Binance account](https://www.binance.com/en)
- [Binance API Key and Secret Key](https://www.binance.com/en/my/settings/api-management)

## Usage
1. Clone the repo to your local machine
2. Install required dependencies
    - python-binance
    - websocket-client
    - numpy
    ```
    pip install python-binance
    pip install websocket-client
    pip install numpy
    ```
3. Update `config.yaml` file with your Binance API and Secret keys. Demo keys can be generated [here](https://testnet.binance.vision/).
4. Set `is_live_account` to `True` if you want to connect to your live account and be able to live trade. Set to `False` otherwise.
5. You can access account data using the `Account` class, and run a continuous algo-trading bot with the `AlgoTrader` class.

### Streaming real-time prices
Real-time prices can be streamed using webhooks and the `PriceStream` class handles this. A `PriceStream` object is initialised with the following parameters:
- `base_asset` - Ticker/Symbol for asset (e.g. ETH)
- `quote_asset` - Ticker/Symbol for reference asset (e.g. USDT)
- `interval` - Interval for candlestick 
- `trading_strategy` - Optional function that is called everytime a tick is received. Used to analyse price data and trade accordingly

```python3
stream = PriceStream(base_asset='ETH', quote_asset='GBP', interval='1m')
stream.run()
```

### Executing orders
Orders have requirements, called _filters_, that need to be satisfied for the order to be successful. More info on all filters can be found [here](https://sammchardy.github.io/binance-order-filters/). The important filters are `MIN_NOTIONAL` and `LOT_SIZE`. These filters can be found by getting the symbol info for a pair.

```python
info = client.get_symbol_info('ETHGBP)
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

### Implementing a trading strategy
The `AlgoTrader` class can be used to run a trading strategy. It streams real-time price data and calls a user-defined function on receipt of every tick. It is initialised with a `Client` object, base and quote asset symbols and a trading strategy. Trading is started by calling the `trade` method.

```python3
bot = AlgoTrader(client, 'ETH', 'USDT', trading_strategy=trading_strategy)
bot.trade()
```