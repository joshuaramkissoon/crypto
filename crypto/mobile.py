import logging
import random
import telegram
from telegram.ext import MessageHandler, Filters, Updater, CommandHandler
from crypto.algo import AlgoTrader
from crypto.environment import Environment
from crypto.helpers import currency, get_strategy
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
                notifier=self.tg
            )
            self.algo.trade()
            MobileClient.is_running = True
            pair = base + '/' + quote
            update_msg = f'Session Started:\nTrading {pair} using {strategy} strategy.\nAccount Value: ${self.algo.price_stream.strategy.start_val}'
            self.tg.update(update_msg)
        except AttributeError:
            update_msg = f'Trading not started: Strategy {strategy} not found. The strategy must be in the config file and defined in the strategy file.'
            self.tg.update(update_msg)

    def stop_trading(self):
        self.algo.stop()
        MobileClient.is_running = False
        session_info = self._get_session_info()
        runtime, profit = str(session_info['runtime']), session_info['profit']
        summary_msg = f'Trading Stopped:\nSession Runtime: {runtime}\nProfit: {profit}'
        self.tg.update(summary_msg)

    def _get_session_info(self):
        return {'profit': currency(self.algo.session.profit), 'runtime': self.algo.session.get_session_runtime()}


class TelegramNotifier:

    INFO_MESSAGE = '''
        Crypto Trading Bot:\n\nStart Trading:
        - Use /start command
        - Provide parameters delim by space as key:value\n\nParameters needed: 
        - Base asset (key: base) (base:ETH)
        - Quote asset (key: quote) (quote:GBP)
        - Strategy (key: strategy) (strategy:RSI)
        - Code (key: code) (code:1111)\n\nExample:
        /start base:ETH quote:GBP strategy:RSI code:1111\n\nStop Trading:
        - Text "stop"
    '''

    def __init__(self, api_key, mobile_client):
        self.mobile_client = mobile_client
        self.user_id = None
        self._provide_access_code()
        TelegramNotifier.bot = telegram.Bot(token=api_key)
        self.setup_handlers(api_key)

    def _provide_access_code(self):
        self.unique_key = TGHelpers.generate_key()
        logging.info(f'Unique access code: {self.unique_key}. Provide this when executing a \\start command. Text \\info to CryptoBot to find out how to use the \\start command.')
    
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
                context.bot.send_message(chat_id=update.effective_chat.id, text='No trading session in progress.')
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
        # Message handler
        message_handler = MessageHandler(Filters.text & (~Filters.command), self.message_handler)
        updater.dispatcher.add_handler(message_handler)
        updater.start_polling()

    def start_handler(self, update, context):
        '''Handler for start command. Start trading if parameters provided are valid.'''
        user_id = update['message']['chat']['id']
        parts = update.message.text.split(' ')
        try:
            parsed_parts = TGHelpers.parse_start_parts(parts)
            if self._is_authorised_start(user_id, parsed_parts.get('code')):
                logging.info('User authorised.')
                self.mobile_client.start_algo(parsed_parts)
            else:
                raise UnauthorisedUser
        except UnauthorisedUser:
            context.bot.send_message(chat_id=update.effective_chat.id, text='Invalid access code.')
        except Exception as e:
            context.bot.send_message(chat_id=update.effective_chat.id, text=f'Trading not started: {e}')

    def info_handler(update, context):
        '''Handler for info command.'''
        context.bot.send_message(chat_id=update.effective_chat.id, text=TelegramNotifier.INFO_MESSAGE)
        
    @active_required
    @authorised
    def message_handler(self, update, context):
        '''Handler for incoming messages (not commands).'''
        msg = update.message.text.lower()
        if msg == 'stop':
            self.handle_stop_message(update, context)        
        if msg == 'update':
            self.handle_update_message(update, context)

    def handle_stop_message(self, update, context):
        '''Stop trading.'''
        self.mobile_client.stop_trading()
    
    def handle_update_message(self, update, context):
        '''Provide session update.'''
        session_info = self.mobile_client._get_session_info()
        runtime, profit = session_info['runtime'], session_info['profit']
        summary_msg = f'Session Update:\nRuntime: {runtime}\nProfit: {profit}'
        context.bot.send_message(chat_id=update.effective_chat.id, text=summary_msg)

    def _is_authorised_start(self, user_id, unique_key):
        '''Checks start message for unique key code and validates it. If valid, set the object's user_id for future reference.'''
        if not unique_key or self.unique_key != int(unique_key):
            return False
        self.user_id = user_id
        return True

    def update(self, message):
        '''Send a message to the registered user.'''
        if not self.user_id:
            raise Exception('No user registered, could not send update.')
        TelegramNotifier.bot.send_message(text=message,chat_id=self.user_id)


class TGHelpers:
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