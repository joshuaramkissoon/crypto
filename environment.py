import yaml

class Environment:
    '''
    Loads environment from a config file with format:
    binance-live:
        api-key: ...
        secret-key: ...
    binance-test:
        api-key: ...
        secret-key: ...
    '''

    BINANCE_LIVE = 'binance-live'
    BINANCE_DEMO = 'binance-test'

    def __init__(self, path, is_live):
        self.is_live = is_live
        with open(path, 'r') as stream:
            try:
                self.config = yaml.safe_load(stream)
            except Exception as e:
                print('Could not load config file: ', e)

    def get_variable(self, var):
        '''
        Returns an environment variable.
        Parameters
        ----------
        var: String. Options: "api", "secret"
        '''
        assert self.config is not None
        parent_var = self.BINANCE_LIVE if self.is_live else self.BINANCE_DEMO
        return self.config.get(parent_var).get(var+'-key')