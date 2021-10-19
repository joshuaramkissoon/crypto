import pytest
from binance.client import Client
from crypto.environment import Environment
from crypto.order import Order
from crypto.session import SessionTracker

class MockClient:
    def __init__(self):
        pass

def make_fill(price, qty):
    return {'price': price, 'qty': qty}

@pytest.fixture
def mock_client():
    return MockClient()

@pytest.fixture
def session(mock_client):
    return SessionTracker(mock_client)

# ---- Fills ----

@pytest.fixture(name='BuyFills')
def buy_fills():
    fills = {
        'fills': [
            {
                'commission': '0.00000600',
                'commissionAsset': 'ETH',
                'price': '1809.39000000',
                'qty': '0.00600000',
                'tradeId': 10410826
            }
        ]
    }
    return fills

@pytest.fixture(name='BuyBaseQty')
def buy_base_qty():
    return 0.006

@pytest.fixture(name='BuyCommissionQty')
def buy_commission_qty():
    return 0.000006

@pytest.fixture(name='SellFills')
def sell_fills():
    fills = {
        'fills': [
            {
                'commission': '0.01637220',
                'commissionAsset': 'GBP',
                'price': '2728.70000000',
                'qty': '0.00600000',
                'tradeId': 12605422
            }
        ]
    }
    return fills

@pytest.fixture(name='SellQuoteQty')
def sell_quote_qty():
    '''Sell 0.006 ETH at a rate of 1 ETH = 2728.70 GBP.'''
    return 16.3722

@pytest.fixture(name='SellCommissionQty')
def sell_commission_qty(SellQuoteQty):
    return 0.001*SellQuoteQty

@pytest.fixture(name='order_fills_1')
def order_fills():
    return [
        {
            'fills': [
                make_fill(3000, 1),
                make_fill(5000, 1)
            ],
            'average_price': 4000
        },
        {
            'fills': [
                make_fill(3000, 1)
            ],
            'average_price': 3000
        },
        {
            'fills': [
                make_fill(3000, 1),
                make_fill(4000, 4),
                make_fill(4500, 1)
            ],
            'average_price': 3916.67
        }
    ]

@pytest.fixture(name='BuyOrder')
def buy_order_object(BuyFills, mock_client):
    order = Order(
                symbol='ETHGBP', 
                side='BUY', 
                fills=BuyFills['fills'],
                quantity=0.006,
                cum_quote_qty=10.85634,
                client=mock_client
            )
    return order

@pytest.fixture(name='BuyOrderResult')
def buy_order_result(BuyBaseQty, BuyCommissionQty):
    return {
        'side': 'BUY',
        'asset': 'ETH',
        'amount': BuyBaseQty - BuyCommissionQty
    }

@pytest.fixture(name='BuyOrders')
def buy_orders(order_fills_1, mock_client):
    orders = []
    for case in order_fills_1:
        orders.append(
            Order(
                symbol='ETHGBP', 
                side='BUY', 
                fills=case['fills'],
                quantity=1.0,
                cum_quote_qty=2000.0,
                client=mock_client
            )
        )
    return orders

@pytest.fixture(name='BuyOrderDict')
def buy_order_dict():
    return {
        'clientOrderId': 'X0NWa1gWz3HeuL3OLphzDQ',
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
        'type': 'MARKET'
    }

# ---- Sell Orders ----

@pytest.fixture(name='SellOrder')
def sell_order_object(SellFills, mock_client):
    order = Order(
                symbol='ETHGBP', 
                side='SELL', 
                fills=SellFills['fills'],
                quantity=0.006,
                cum_quote_qty=16.3722,
                client=mock_client
            )
    return order

@pytest.fixture(name='SellOrderResult')
def sell_order_result(SellQuoteQty, SellCommissionQty):
    return {
        'side': 'SELL',
        'asset': 'GBP',
        'amount': SellQuoteQty - SellCommissionQty
    }

@pytest.fixture(name='SellOrderDict')
def sell_order_dict():
    return {
        'clientOrderId': '1dipROdV3qJpySjUTM5pZA',
        'cummulativeQuoteQty': '16.37220000',
        'executedQty': '0.00600000',
        'fills': [{'commission': '0.01637220',
                    'commissionAsset': 'GBP',
                    'price': '2728.70000000',
                    'qty': '0.00600000',
                    'tradeId': 12605422}],
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