import logging
import threading
import time
from crypto.environment import Environment
import crypto.strategy

def format_runtime(t: float):
    '''Converts a session runtime in seconds to an appropriate unit and returns the string.'''
    SECOND_THRESHOLD = 60
    MINUTE_THRESHOLD = 60 * SECOND_THRESHOLD
    is_second = lambda x: x < SECOND_THRESHOLD
    is_minute = lambda x: x >= SECOND_THRESHOLD and x < MINUTE_THRESHOLD
    is_hour = lambda x: x >= MINUTE_THRESHOLD
    if is_second(t):
        return f'{t} seconds'
    elif is_minute(t):
        t = round(t/60, 2)
        return f'{t} minutes'
    elif is_hour(t):
        t = round(t/3600, 2)
        return f'{t} hours'

def currency(c: float):
    '''Format currency.'''
    return '£{:0,.2f}'.format(c).replace('£-', '-£')

def get_strategy(s: str):
    '''Return the Strategy subclass with name matching the string s.'''
    env = Environment()
    strategies = env.get_root_variable('strategy')
    if not strategies:
        raise Exception('No strategies found in config file.')
    return getattr(crypto.strategy, s)