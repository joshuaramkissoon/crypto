import pytest
from crypto.order import Order
from crypto.pricer import Pricer

def test_get_average_price(BuyOrders, order_fills_1):
    '''Test the get_average_price method for an Order.'''
    for idx, case in enumerate(order_fills_1):
        expected_avg = case['average_price']
        assert BuyOrders[idx].get_average_price() == pytest.approx(expected_avg)

def test_get_buy_commission(mocker, BuyOrder, BuyCommissionQty):
    mocker.patch('crypto.pricer.Pricer.get_average_price', return_value=2.0)
    com = BuyOrder.get_commission()
    assert com.quantity == BuyCommissionQty
    assert com.asset == 'ETH'
    assert com.fiat_value == 2*BuyCommissionQty # Mocked rate to be 1 so fiat value should be equal to the qty

def test_buy_order_get_result(BuyOrder, BuyOrderResult):
    assert BuyOrder.get_result() == BuyOrderResult

def test_sell_order_get_result(SellOrder, SellOrderResult):
    assert SellOrder.get_result() == SellOrderResult
    