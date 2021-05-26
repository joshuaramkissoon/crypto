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
        alts = [d['asset'] + 'USDT' for d in self.holdings if 'USD' not in d['asset']] # alt coin symbols (non USD)
        stables = [d['asset'] for d in self.holdings if 'USD' in d['asset']] # Coins with USD like BUSD, USDT
        prices = Pricer(self.client).get_average_prices(alts)
        val = 0
        for holding in self.holdings:
            asset = holding['asset']
            amt = float(holding['free']) + float(holding['locked'])
            if 'USD' in asset:
                # stable coin, no conversion needed
                val += amt
            else:
                if price := prices.get(holding['asset']+'USDT'):
                    val += amt*price
        return round(val, 2)

        


    