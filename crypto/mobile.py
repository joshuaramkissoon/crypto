import logging
import telegram
from telegram.ext import MessageHandler, Filters, Updater, CommandHandler
from crypto.algo import AlgoTrader
from crypto.account import Account
from crypto.environment import Environment
from crypto.helpers import currency, get_strategy, TelegramHelpers
import crypto.strategy

class UnauthorisedUser(Exception):
    def __init__(self, msg='User not authorised to perform this operation.'):
        super().__init__(self, msg)

class MobileClient:
    '''
    MobileClient handles communication between a user and an algorithm via a TelegramNotifier.

    Parameters
    ----------
    client: Binance Client object
    '''
    def __init__(self, client):
        self.client = client
        self.account = Account(self.client)
        self.environment = Environment()
        self.api_key = self.environment.get_root_variable('telegram')
        MobileClient.is_running = False
        self.algo = None
    
    def start(self):
        '''Start listening on mobile client.'''
        self.tg = TelegramNotifier(self.api_key, self)

    def start_algo(self, params: dict):
        '''
        Prepare an algorithm and start trading using the parameters specified.
        
        Parameters
        ----------
        params: dict, parameters for running the algorithm (keys required: base, quote, strategy)
        '''
        base, quote, strategy = params['base'], params['quote'], params['strategy']
        if self.algo is not None:
            del self.algo
        try:
            self.algo = AlgoTrader(
                self.client, 
                base, 
                quote, 
                strategy=get_strategy(strategy),
                notifier=self.tg,
                account=self.account
            )
            self.algo.trade()
            MobileClient.is_running = True
            pair = base + '/' + quote
            update_msg = (
                f'Session Started:\nTrading {pair} using {strategy} strategy.\nAccount Value: ${self.algo.start_val}\n\n'
                f'{self.algo.create_balance_update_message(markdown=True)}'
            )
            self.tg.update(update_msg)
        except AttributeError:
            update_msg = f'Trading not started: Strategy {strategy} not found. The strategy must be in the config file and defined in the strategy file.'
            self.tg.update(update_msg)

    def stop_trading(self):
        self.algo.stop()
        MobileClient.is_running = False
        self.tg.update(self.algo._create_stop_message(markdown=True))

    def _get_asset_amount(self, asset: str) -> dict:
        if balance := self.account.get_asset_balance(asset):
            return balance
        raise Exception('Asset not found.')


class TelegramNotifier:

    INFO_MESSAGE = '''
        *Crypto Trading Bot:*\n\n*Start Trading*:
Use the /start command and provide parameters separated by spaces as _key:value_ to start trading. The *strategy* parameter is case-sensitive and must correspond to a class in the `strategy.py` module. Your *access code* will be shown when you start a `MobileClient` object from your Python script.\n\n_Parameters needed:_ 
        • Base asset (key: base) (base:ETH)
        • Quote asset (key: quote) (quote:GBP)
        • Strategy (key: strategy) (strategy:RSI)
        • Access Code (key: code) (code:1111)\n\n_Example:_
        /start base:ETH quote:GBP strategy:RSI code:1111\n\n*Stop Trading:*
        • Text *Stop* to stop trading.\n\n*Get a Trading Update:*
        • Text *Update* to get information about the current trading session like net profit, trades placed etc. 
    '''

    def __init__(self, api_key, mobile_client):
        self.mobile_client = mobile_client
        self.user_id = None
        self._provide_access_code()
        TelegramNotifier.bot = telegram.Bot(token=api_key)
        self.setup_handlers(api_key)
        self.parse_mode = telegram.ParseMode.MARKDOWN

    def _provide_access_code(self):
        self.unique_key = TelegramHelpers.generate_key()
        logging.info(f'Unique access code: {self.unique_key}. Provide this when executing a /start command. Text /info to CryptoBot to find out how to use the /start command.')

    # Properties

    @property
    def is_auth(self):
        return self.user_id is not None
    
    # Decorators

    def authorised(func):
        '''Decorator for functions that require authorisation to be called.'''
        def wrapper(self, update, context):
            chat_id = update['message']['chat']['id']
            if not self.user_id or self.user_id != chat_id:
                msg = 'You are not authorised to perform this operation.'
                TelegramNotifier.bot.send_message(text=msg, chat_id=chat_id)
                raise UnauthorisedUser
            func(self, update, context)
        return wrapper

    def active_required(func):
        '''Decorator for functions that require an active trading session to be called.'''
        def wrapper(self, update, context):
            chat_id = update['message']['chat']['id']
            if not self.mobile_client.is_running or chat_id != self.user_id:
                context.bot.send_message(chat_id=update.effective_chat.id, text='No trading session in progress.', parse_mode=self.parse_mode)
            else:
                func(self, update, context)
        return wrapper

    # Class Methods

    def setup_handlers(self, api_key):
        '''Setup message handlers.'''
        updater = Updater(token=api_key, use_context=True)
        # Start handler
        start_handler = CommandHandler('start', self.start_handler)
        updater.dispatcher.add_handler(start_handler)
        # Info handler
        info_handler = CommandHandler('info', self.info_handler)
        updater.dispatcher.add_handler(info_handler)
        # Query handler
        query_handler = CommandHandler('query', self.query_handler)
        updater.dispatcher.add_handler(query_handler)
        # Message handler
        message_handler = MessageHandler(Filters.text & (~Filters.command), self.message_handler)
        updater.dispatcher.add_handler(message_handler)
        updater.start_polling()

    def start_handler(self, update, context):
        '''Handler for start command. Start trading if parameters provided are valid.'''
        user_id = update['message']['chat']['id']
        parts = update.message.text.split(' ')
        try:
            parsed_parts = TelegramHelpers.parse_start_parts(parts)
            if self._is_authorised_start(user_id, parsed_parts.get('code')):
                logging.info('User authorised.')
                self.mobile_client.start_algo(parsed_parts)
            else:
                context.bot.send_message(chat_id=update.effective_chat.id, text='Invalid access code.')
        except Exception as e:
            context.bot.send_message(chat_id=update.effective_chat.id, text=f'Trading not started: {e}')

    def info_handler(self, update, context):
        '''Handler for info command.'''
        context.bot.send_message(chat_id=update.effective_chat.id, text=TelegramNotifier.INFO_MESSAGE, parse_mode=self.parse_mode)

    def query_handler(self, update, context):
        '''
        Handler for query command. A query message should specify a query type and an asset and other fields if required.
        
        Query Types
        -----------
        amount: Get the amount of an asset (e.g. amount:ETH returns the amount of ETH in the wallet)
        value: Get the value of an asset in a specified currency (e.g. value:ETH/GBP returns the value of ETH in the wallet in GBP (£))
        '''
        msg = update.message.text.lower()
        query_dict = TelegramHelpers.parse_query(msg)
        if not query_dict.get('auth'):
            context.bot.send_message(chat_id=update.effective_chat.id, text='No access code provided.')
            return
        # Authenticate user
        user_id = update['message']['chat']['id']
        if not self._is_authorised_start(user_id, query_dict.get('auth')):
            context.bot.send_message(chat_id=update.effective_chat.id, text='Invalid access code.')
            return
        response = None
        valid, error_msg = TelegramHelpers.is_valid_query(query_dict)
        if valid:
            _type = query_dict['type']
            params = query_dict['params']
            if _type == 'amount':
                response = self._handle_amount_query(params)
            if _type == 'value':
                print(TelegramHelpers._get_value_query_params(params))
                _response = 'Value query not implemented yet.'
            context.bot.send_message(chat_id=update.effective_chat.id, text=response, parse_mode=self.parse_mode)
        else:
            context.bot.send_message(chat_id=update.effective_chat.id, text=f'Invalid query parameters. {error_msg}', parse_mode=self.parse_mode)
        
    @active_required
    @authorised
    def message_handler(self, update, context):
        '''Handler for incoming messages (not commands).'''
        msg = update.message.text.lower()
        if msg == 'stop':
            self._handle_stop_message(update, context)        
        if msg == 'update':
            self._handle_update_message(update, context)

    def _handle_stop_message(self, update, context):
        '''Stop trading.'''
        self.mobile_client.stop_trading()
    
    def _handle_update_message(self, update, context):
        '''Provide session update.'''
        session_info = self.mobile_client.algo.get_session_info()
        runtime, profit, orders = session_info['runtime'], session_info['profit'], session_info['orders']
        summary_msg = f'*Session Update:*\nRuntime: {runtime}\nProfit: {profit}\n\n'
        balances_msg = self.mobile_client.algo.create_balance_update_message(markdown=True)
        trades_msg = None
        if not orders:
            trades_msg = 'No trades yet.'
        else:
            trades_msg = '*Trades:*'
            for i, order in enumerate(orders):
                trades_msg += f'\n\n{i}. {TelegramHelpers.create_order_msg(order, markdown=True)}'
        context.bot.send_message(chat_id=update.effective_chat.id, text=summary_msg+balances_msg+trades_msg, parse_mode=self.parse_mode)

    def _handle_amount_query(self, asset: str):
        '''Respond with amount of asset in user's wallet.'''
        _asset = asset.upper()
        res = None
        try:
            balance = self.mobile_client._get_asset_amount(_asset)
            free, locked = balance['free'], balance['locked']
            res = f'*{_asset} Balance:*\n\n*Free*: {free}\n*Locked*: {locked}'
        except Exception as e:
            res = f'Could not get balance for asset: {asset}'
        return res
            

    def _is_authorised_start(self, user_id, unique_key):
        '''Checks start message for unique key code and validates it. If valid, set the object's user_id for future reference.'''
        if not unique_key or self.unique_key != int(unique_key):
            return False
        self.user_id = user_id
        return True

    def update(self, message, parse_markdown=True):
        '''Send a message to the registered user.'''
        if not self.user_id:
            raise Exception('No user registered, could not send update.')
        TelegramNotifier.bot.send_message(text=message,chat_id=self.user_id, parse_mode=self.parse_mode if parse_markdown else None)