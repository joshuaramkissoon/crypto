import logging
import random
import telegram
from telegram.ext import MessageHandler, Filters, Updater, CommandHandler
from crypto.algo import AlgoTrader
from crypto.environment import Environment
from crypto.helpers import currency
import crypto.strategy

class UnauthorisedUser(Exception):
    def __init__(self, msg='User not authorised to perform this operation.'):
        super().__init__(self, msg)

class MobileClient:
    def __init__(self, client):
        self.client = client
        self.environment = Environment()
        self.api_key = self.environment.get_root_variable('telegram')
        MobileClient.is_running = False
        self.algo = None
    
    def start(self):
        '''
        Start listening on mobile client. Optionally, trading can be started automatically if correct keyword args are passed.
        
        Parameters
        ----------
        **kwargs - 
        - base: str, Base asset
        - quote: str, Quote asset
        - strategy: str, Name of strategy to run
        '''
        self.tg = TelegramNotifier(self.api_key, self)

    def prepare_algo(self, params):
        def get_strategy(s):
            strategies = self.environment.get_root_variable('strategy')
            if not strategies:
                raise Exception('No strategies found in config file.')
            return getattr(crypto.strategy, s.upper())
        if self.algo is not None:
            del self.algo
        self.algo = AlgoTrader(
            self.client, 
            params['base'], 
            params['quote'], 
            strategy=get_strategy(params['strategy']),
            notifier=self.tg
        )
        self.algo.trade()
        MobileClient.is_running = True
        base, quote, strategy = params['base'], params['quote'], params['strategy']
        update_msg = f'Session Started:\nTrading {base+quote} using {strategy} strategy.\nAccount Value: ${self.algo.price_stream.strategy.start_val}'
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

    user_id = None

    def __init__(self, api_key, mobile_client):
        TelegramNotifier.mobile_client = mobile_client
        TelegramNotifier.unique_key = self.generate_key()
        TelegramNotifier.bot = telegram.Bot(token=api_key)
        self.setup_handlers(api_key)

    def _is_authorised_start(user_id, unique_key):
        '''Checks start message for unique key code and validates it. If valid, set the object's user_id for future reference.'''
        if not unique_key or TelegramNotifier.unique_key != int(unique_key):
            return False
        TelegramNotifier.user_id = user_id
        return True

    def authorised(func):
        '''Wrapper for functions that require authorisation to be called.'''
        def wrapper(*args, **kwargs):
            chat_id = args[1]['message']['chat']['id']
            if not TelegramNotifier.user_id or TelegramNotifier.user_id != chat_id:
                msg = 'You are not authorised to perform this operation.'
                TelegramNotifier.bot.send_message(text=msg, chat_id=chat_id)
                raise UnauthorisedUser
            func(*args, **kwargs)
        return wrapper
    
    def generate_key(self) -> int:
        '''
        Generate a 4 digit code that must be sent with a start command. This ensure that the telegram sender
        is authorised to access the Binance Client.
        '''
        key = random.randint(1111, 9999)
        logging.info(f'Unique access code: {key}. Provide this when executing a \\start command. Text \\info to CryptoBot to find out how to use the \\start command.')
        return key

    def setup_handlers(self, api_key):
        updater = Updater(token=api_key, use_context=True)
        start_handler = CommandHandler('start', TelegramNotifier.start_handler)
        updater.dispatcher.add_handler(start_handler)
        info_handler = CommandHandler('info', TelegramNotifier.info_handler)
        updater.dispatcher.add_handler(info_handler)
        message_handler = MessageHandler(Filters.text & (~Filters.command), TelegramNotifier.message_handler)
        updater.dispatcher.add_handler(message_handler)
        updater.start_polling()

    def update(self, message):
        '''Send a message to the recipient.'''
        TelegramNotifier.bot.send_message(text=message,chat_id=self.user_id)
    
    @classmethod
    def start_handler(cls, update, context):
        user_id = update['message']['chat']['id']
        parts = update.message.text.split(' ')
        try:
            parsed_parts = TGHelpers.parse_start_parts(parts)
            if TelegramNotifier._is_authorised_start(user_id, parsed_parts.get('code')):
                logging.info('User authorised.')
                cls.mobile_client.prepare_algo(parsed_parts)
            else:
                raise UnauthorisedUser
        except UnauthorisedUser:
            context.bot.send_message(chat_id=update.effective_chat.id, text='Invalid access code.')
        except Exception as e:
            context.bot.send_message(chat_id=update.effective_chat.id, text=f'Trading not started: {e}')

    def info_handler(update, context):
        context.bot.send_message(chat_id=update.effective_chat.id, text=TelegramNotifier.INFO_MESSAGE)

    def active_required(handler):
        def wrapper(cls, update, context):
            chat_id = update['message']['chat']['id']
            if not cls.mobile_client.is_running or chat_id != TelegramNotifier.user_id:
                context.bot.send_message(chat_id=update.effective_chat.id, text='No trading session in progress.')
            else:
                handler(cls, update, context)
        return wrapper
        
    @classmethod
    @active_required
    @authorised
    def message_handler(cls, update, context):
        msg = update.message.text.lower()
        if msg == 'stop':
            cls.handle_stop_message(cls, update, context)        
        if msg == 'update':
            cls.handle_update_message(cls, update, context)

    def handle_stop_message(cls, update, context):
        cls.mobile_client.stop_trading()
    
    def handle_update_message(cls, update, context):
        session_info = cls.mobile_client._get_session_info()
        runtime, profit = session_info['runtime'], session_info['profit']
        summary_msg = f'Session Update:\nRuntime: {runtime}\nProfit: {profit}'
        context.bot.send_message(chat_id=update.effective_chat.id, text=summary_msg)

class TGHelpers:
    def parse_start_parts(parts) -> dict:
        if len(parts) < 5:
            raise Exception('Insufficient parameters provided. Run /info for more info on how to start trading.')
        parts = parts[1:]
        dct = {}
        for p in parts:
            try:
                k, v = p.split(':')
                dct[k.lower()] = v.upper()
            except Exception as e:
                raise Exception('Invalid parameter format. Run /info for more info on how to provide parameters.')
        return dct