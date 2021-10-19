import yaml
import logging
from enum import Enum

class Environment:
    '''Loads environment variables from a config file.'''
    class Constants(Enum):
        BINANCE_LIVE_KEY = 'binance-live'
        BINANCE_DEMO_KEY = 'binance-test'
        IS_LIVE_ACCOUNT_KEY = 'is-live-account'
        CONFIG_PATH_KEY = 'config-path'
        DEFAULT_CONFIG_PATH = 'config.yaml'
        TELEGRAM_KEY = 'telegram'

    config = None

    def __init__(self, path='config.yaml'):
        constants = self.Constants
        self.config = self.load_file(path)
        self.is_live = self.get_root_variable(constants.IS_LIVE_ACCOUNT_KEY.value)
        if (config_path := self.get_root_variable(constants.CONFIG_PATH_KEY.value)) != constants.DEFAULT_CONFIG_PATH.value:
            self.config = self.load_file(config_path)

    def load_file(self, path: str) -> dict:
        '''Load yaml file from path.'''
        try:
            with open(path, 'r') as stream:
                return yaml.safe_load(stream)
        except Exception as e:
            raise Exception(f'Could not load config file: {str(e)}')

    def _get(self, var: str):
        '''Get a variable by key from the config file loaded.'''
        if not self.config:
            raise Exception('No configuration file found.')
        return self.config.get(var)
    
    def get_binance_key(self, key_type: str):
        '''
        Returns a Binance API or Secret key.
        Parameters
        ----------
        var: String. Options: "api", "secret"
        '''
        if key_type == 'api' or key_type == 'secret':
            parent_var = self.Constants.BINANCE_LIVE_KEY if self.is_live else self.Constants.BINANCE_DEMO_KEY
            return self._get(parent_var.value).get(key_type+'-key')
        raise Exception(f'Invalid binance key type: {key_type}')
    
    def get_coinbase_keys(self):
        api = self._get('coinbase').get('api-key')
        secret = self._get('coinbase').get('secret-key')
        return {'api-key': api, 'secret-key': secret}

    def get_root_variable(self, var: str):
        '''
        Returns a root environment variable.
        Parameters
        ----------
        var: String. Key of variable
        '''
        if (_var := self._get(var)) is not None:
            return _var
        raise Exception(f'Variable {var} not found in config file.')

    def get_telegram_key(self):
        if tg := self.get_root_variable(self.Constants.TELEGRAM_KEY.value):
            return tg
        raise Exception(f'Telegram key not found in config file.')