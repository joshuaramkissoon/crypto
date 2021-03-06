# Crypto Algo Trading

This library can be used to create, run and monitor algorithmic trading strategies for cryptocurrencies.
It can be linked to a user's Binance account and allows for order execution in live and test environments.

## Key Features

- Stream real-time prices of asset-pairs
- Execute live buy or sell orders
- Implement a trading strategy and run an `AlgoTrader` using this strategy
- Receive trading alerts with Telegram
    - Order execution
- Dynamic remote trading using a Telegram Bot

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

If the order is executed, the response will show the fill(s) for the order and other related metadata.

```
{'clientOrderId': 'X0NWa1gWz3HeuL3OLphzDQ',
 'cummulativeQuoteQty': '10.85634000',
 'executedQty': '0.00600000',
 'fills': [{'commission': '0.00000600',
            'commissionAsset': 'ETH',
            'price': '1809.39000000',
            'qty': '0.00600000',
            'tradeId': 10410826}],
 'orderId': 188166352,
 'orderListId': -1,
 'origQty': '0.00600000',
 'price': '0.00000000',
 'side': 'BUY',
 'status': 'FILLED',
 'symbol': 'ETHGBP',
 'timeInForce': 'GTC',
 'transactTime': 1622578783778,
 'type': 'MARKET'}
```

Note: All orders have a 0.1% commission.

### Creating a trading strategy

Create a trading strategy by subclassing the `Strategy` class (in `strategy.py`). This base class handles logic for accessing account details and executing orders. Your implementation only needs to have a basic `__init__` and to override the `trading_strategy` method:

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

In the implementation of the `trading_strategy` method, buy or sell orders can be created using the base class' `order()` method. This method takes a `side`, `amount` and `symbol`, handles the execution of the order and updates the current trading session or handles exceptions if the order wasn't executed successfully. There's no need to override the `order` method from the base `Strategy` class.

### Implementing a trading strategy

After creating a strategy, the `AlgoTrader` class can be used to run it. The `AlgoTrader` opens a real-time stream of price data for an asset pair and the strategy handles incoming tick data. It is initialised with a `Client` object, base and quote asset symbols and a trading strategy class. Start trading by calling the `trade` method.

```python3
bot = AlgoTrader(client, 'ETH', 'USDT', strategy=MovingAverage)
bot.trade()
```

### Remote Trading using Telegram

The `MobileClient` can be used to manage remote trading by speaking to a Telegram Bot. Trading can be started using a particular strategy
and different commands can be used to control the trading session. The bot's username is `@jkmr_crypto_bot`.

```python3
# Control or manage trading by texting the CryptoBot
MobileClient(client).start()
```

#### How to use the bot

You can find information on how to use the bot below or text `/info` to `CryptoBot`.

#### Start Trading

Use the `/start` command in the telegram chat and provide parameters separated by spaces as `key:value` to start trading. 
The `strategy` parameter is case-sensitive and must correspond to a class in the `strategy.py` module. This must also be under `strategy` in `config.yaml`. 
Your `access code` will be shown when you start a `MobileClient` object from your Python script.

**Parameters needed:**

- **Base Asset** (key: `base`) (E.g. `base:ETH`)
- **Quote asset** (key: `quote`) (E.g. `quote:GBP`)
- **Strategy** (key: `strategy`) (E.g. `strategy:RSI`)
- **Access Code** (key: `code`) (E.g. `code:1111`)

**Example:**

```
/start base:ETH quote:GBP strategy:RSI code:1111
```

This will start trading `ETH/GBP` using the user's `RSI` strategy. The code provided registers the trading session to the correct user, ensuring no
other users can get access to this session. Alerts will be sent when orders are executed successfully and if they fail.

**Order Executed Alert:**

```
Order Executed:
Pair Symbol: ETH/GBP
Side: BUY
Order Type: MARKET
Quantity: 0.005
Average Execution Price: ??2278.83
Net Trade Spend: ??11.39
```

#### Stop Trading

Text `stop` to `CryptoBot` to stop trading. A summary of the trading session will be sent back to you.

#### Update Trading

Text `update` to `CryptoBot` to get an update on the trading session. Net profit, number of trades placed etc. will be sent back to you.

#### Querying an Account

Queries can be sent to `CryptoBot` to get account information. Each query must contain the authentication code. 
There are two types of queries that can be used:

- **Amount** - Used to get the amount or balance of an asset in a user's wallet

```
/query amount:ETH code:1111
```

The bot will respond with the balance of `ETH` in the current user's wallet: the `free` and `locked` amount.

- **Value** - Used to get the value of an asset in a specific currency

```
/query value:ETH/GBP code:1111
```

The response will be the value of the user's `free` and `locked` `ETH` holdings in `GBP`.
