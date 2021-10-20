# Commission

Assume we are trading `ETHGBP` and that commission rate is 0.1% (this will change with trade frequency on Binance).

## Buying

If you buy 1 ETH for £2000.00, you pay £2000 but only receive 0.999 ETH since Binance takes 0.001 ETH commission.

Looking at a sample buy order result:

```
{
    'clientOrderId': 'X0NWa1gWz3HeuL3OLphzDW',
    'cummulativeQuoteQty': '10.85634000',
    'executedQty': '0.00600000',
    'fills': [
        {
            'commission': '0.00000600',
            'commissionAsset': 'ETH',
            'price': '1809.39000000',
            'qty': '0.00600000',
            'tradeId': 10410827
        }
    ],
    'orderId': 188166352,
    'orderListId': -1,
    'origQty': '0.00600000',
    'price': '0.00000000',
    'side': 'BUY',
    'status': 'FILLED',
    'symbol': 'ETHGBP',
    'timeInForce': 'GTC',
    'transactTime': 1622578783771,
    'type': 'MARKET'
}
```

We bought 0.006 ETH at a rate of 1 ETH = £1809.39 so we paid £10.85634 (`cumulativeQuoteQty`). The amount of ETH we actually receive is the sum of each `fill`'s quantity less that `fill`'s commission. 
In this case it will be `0.006 - 0.000006 = 0.005994 ETH` that we get as a result of the trade.

## Selling

If you sell 1 ETH for £2000.00, you will have 1 less ETH but you will only receive £1998.00 since Binance takes £2 commission.

Looking at a sample sell order result:

```
{
    'clientOrderId': '1dipROdV3qJpySjUTM5pZA',
    'cummulativeQuoteQty': '16.37220000',
    'executedQty': '0.00600000',
    'fills': [
        {
            'commission': '0.01637220',
            'commissionAsset': 'GBP',
            'price': '2728.70000000',
            'qty': '0.00600000',
            'tradeId': 12605422
        }
    ],
    'orderId': 317643959,
    'orderListId': -1,
    'origQty': '0.00600000',
    'price': '0.00000000',
    'side': 'SELL',
    'status': 'FILLED',
    'symbol': 'ETHGBP',
    'timeInForce': 'GTC',
    'transactTime': 1634309618382,
    'type': 'MARKET'
}
```

We sold 0.006 ETH at a rate of 1 ETH = £2728.70 so we should receive £16.3722 but the commission is £0.0163722. The amount of GBP we receive from the trade is the sum of each fill's `(qty * price) - commission`. 
In this case it will be `16.3722 - 0.0163722 = £16.3558 or £16.35`.