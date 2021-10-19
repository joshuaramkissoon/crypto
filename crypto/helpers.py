import logging, threading, time, random
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

def weighted_average(l: list):
    '''Returns the weighted average of a list of tuples (value, amount). The sum of amounts dont need to be 1, their ratios will be used.'''
    total = sum(list(map(lambda x: x[1], l)))
    weight = lambda x: float(x/total)
    avg = sum([x[0]*weight(x[1]) for x in l])
    return avg

def make_url(base, route):
    return base + route

class TelegramHelpers:

    VALID_QUERY_TYPES = ['value', 'amount']

    def parse_start_parts(parts: list) -> dict:
        if len(parts) < 5:
            raise Exception('Insufficient parameters provided. Run /info for more info on how to start trading.')
        parts = parts[1:]
        dct = {}
        for p in parts:
            try:
                k, v = p.split(':')
                dct[k.lower()] = v.upper() if k.lower() != 'strategy' else v
            except Exception as e:
                raise Exception('Invalid parameter format. Run /info for more info on how to provide parameters.')
        return dct

    def generate_key() -> int:
        '''
        Generate a 4 digit code that must be sent with a start command. This ensure that the telegram sender
        is authorised to access the Binance Client.
        '''
        return random.randint(1111, 9999)

    def parse_query(q: str) -> dict:
        '''
        Parse a query message. (E.g. /query amount:ETH or /query value:ETH/GBP)
        '''
        parts = q.split(' ')
        if len(parts) != 3:
            return {}
        _, query_parts, auth_parts = parts
        query_split = [s.strip() for s in query_parts.split(':')]
        _, auth = [s.strip() for s in auth_parts.split(':')]
        return {'type': query_split[0], 'params': query_split[1], 'auth': auth}

    def _get_value_query_params(s: str) -> dict:
        parts = [p.strip() for p in s.split('/')]
        if len(parts) != 2:
            raise Exception('Invalid value query parameters.')
        return {'asset': parts[0], 'currency': parts[1]}
    
    def is_valid_query(query_dict: dict) -> bool:
        '''Check that a message received from Telegram is a valid query.'''
        if 'auth' not in query_dict:
            return False, 'No authorisation code provided.'
        try:
            query = query_dict['type']
            if query == 'amount':
                return True, None
            if query == 'value':
                # Make sure valid params provided (ASSET/CURRENCY_CODE)
                params = query_dict['params']
                value_query_params = TelegramHelpers._get_value_query_params(params)
                return 'asset' in value_query_params and 'currency' in value_query_params, 'Invalid value query parameters provided.'
            return False, 'Unrecognized query.'
        except Exception as e:
            return False, str(e)