from crypto.pricer import Pricer
from pprint import pprint

class Account:
    def __init__(self, client):
        self.client = client
        self.account = None
        self.holdings = None

    def get_account(self):
        return self.client.get_account()

    def get_asset_balance(self, asset):
        return self.client.get_asset_balance(asset=asset)

    def get_account_balances(self, include_zero=False):
        '''
        Gets the balances for an account: the quantity for each asset in a portfolio
        include_zero: bool, whether to include all assets including those with quantity = 0
        '''
        if not self.account:
            self.account = self.get_account()
        self.balances = self.account['balances']
        if include_zero:
            return self.balances
        else:
            self.holdings = list(filter(lambda d: float(d['free']) + float(d['locked']) != 0, self.balances))
            return self.holdings
    
    def get_portfolio(self):
        return self.get_account_balances()

    def get_portfolio_value(self):
        '''Gets the current value of a portfolio in USD.'''
        if not self.holdings:
            self.holdings = self.get_account_balances()
        symbols = [d['asset'] + 'USDT' for d in self.holdings]
        prices = Pricer(self.client).get_average_prices(symbols)
        val = 0
        for dct in self.holdings:
            amt = float(dct['free']) + float(dct['locked'])
            price = prices[dct['asset']+'USDT']
            val += amt*price
        return round(val, 2)

        


    