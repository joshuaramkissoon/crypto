import pytest

def test_handle_buy_order(session, BuyOrderDict, BuyOrder, BuyOrderResult):
    assert session.profit == 0
    assert session.orders == []
    assert session.crypto_amt == 0

    result = session.handle_order(BuyOrderDict)
    cqt = float(BuyOrderDict['cummulativeQuoteQty'])
    expected_res = {
        'net': pytest.approx(-cqt),
        'average_price': 1809.39
    }
    
    assert result == expected_res
    assert session.profit == expected_res['net']
    assert session.fiat == expected_res['net']
    assert session.orders == [BuyOrder]
    assert session.crypto_amt == BuyOrderResult['amount']

def test_handle_sell_order(mocker, session, SellOrderDict, SellOrder):
    assert session.profit == 0
    assert session.orders == []
    assert session.crypto_amt == 0

    result = session.handle_order(SellOrderDict)
    cqt = float(SellOrderDict['cummulativeQuoteQty'])
    net = 0.999*cqt # The net gain from this sell order is 99.9% of the sell amount (because of 0.1% commission)
    expected_res = {
        'net': pytest.approx(net),
        'average_price': 2728.70
    }
    assert result == expected_res
    assert session.profit == expected_res['net']
    assert session.fiat == expected_res['net']
    assert session.orders == [SellOrder]
    assert session.crypto_amt == -SellOrder.quantity

# TODO: Test order with multiple fills