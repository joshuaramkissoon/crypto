import pytest
from crypto.helpers import get_strategy, currency, format_runtime
from crypto.strategy import RSI, MA
from crypto.mobile import TGHelpers

@pytest.mark.parametrize(
    'strategy_str,strategy',
    [
        ('RSI', RSI),
        ('MA', MA)
    ]
)
def test_get_valid_strategy(strategy_str, strategy):
    _strategy = get_strategy(strategy_str)
    assert type(_strategy) == type(strategy)

@pytest.mark.parametrize(
    'strategy_str',
    ['rsi', 'ma', 'Volume']
)
def test_get_invalid_strategy_throws(strategy_str):
    with pytest.raises(AttributeError):
        _strategy = get_strategy(strategy_str)

@pytest.mark.parametrize(
    'input,expected',
    [
        (240, '£240.00'),
        (-240, '-£240.00'),
        (-240.123, '-£240.12'),
        (240.123, '£240.12'),
        (0, '£0.00')
    ]
)
def test_currency_formatter(input, expected):
    assert currency(input) == expected

@pytest.mark.parametrize(
    'input,expected',
    [
        (15, '15 seconds'),
        (80, '1.33 minutes'),
        (3600, '1.0 hours'),
        (13680, '3.8 hours')
    ]
)
def test_runtime_formatter(input, expected):
    assert format_runtime(input) == expected

def test_parse_start_parts_valid():
    input = '/start base:eth quote:usdt strategy:rsi code:9999'
    expected = {
        'base': 'ETH',
        'quote': 'USDT',
        'strategy': 'rsi',
        'code': '9999'
    }
    assert TGHelpers.parse_start_parts(input.split(' ')) == expected

def test_parse_start_parts_insufficient_params():
    input = '/start base:eth quote:usdt strategy:rsi'
    with pytest.raises(Exception) as e:
        TGHelpers.parse_start_parts(input.split(' '))
        assert str(e) == 'Insufficient parameters provided. Run /info for more info on how to start trading.'

@pytest.mark.parametrize(
    'input',
    [
        '/start base:eth quote:usdt strategy-rsi',
        '/start base:eth quote:usdt strategy:rsi:extra',
    ]
)
def test_parse_start_parts_invalid_params(input):
    with pytest.raises(Exception) as e:
        TGHelpers.parse_start_parts(input.split(' '))
        assert str(e) == 'Invalid parameter format. Run /info for more info on how to provide parameters.'