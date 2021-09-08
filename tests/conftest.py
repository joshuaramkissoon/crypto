import pytest
from crypto.session import SessionTracker

def make_fill(price, qty):
    return {'price': price, 'qty': qty}

@pytest.fixture
def session():
    return SessionTracker()

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