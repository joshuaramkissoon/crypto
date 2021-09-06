'''
Module that handles notifications for alerts. Currently supporting Telegram.
'''
import logging
import random
import telegram
from telegram.ext import MessageHandler, Filters, Updater, CommandHandler
from crypto.algo import AlgoTrader
import crypto.strategy
from crypto.environment import Environment

class Notifier:

    INFO_MESSAGE = '''
        Crypto Trading Bot:\n\nStart Trading:
        - Use /start command
        - Provide parameters delim by space as key:value\n\nParameters needed: 
        - Base asset (key: base) (base:ETH)
        - Quote asset (key: quote) (quote:GBP)
        - Strategy (key: strategy) (strategy:RSI)\n\nExample:
        /start base:ETH quote:GBP strategy:RSI\n\nStop Trading:
        - Text "stop"
    '''

    def __init__(self, client=None, first_name=None, last_name=None, algo=None, session=None):
        Notifier.client = client
        Notifier.environment = Environment()
        Notifier.api_key = Notifier.environment.get_root_variable('telegram')
        Notifier.bot = telegram.Bot(token=Notifier.api_key)
        Notifier.algo = algo
        Notifier.session = session
        Notifier.unique_key = self.generate_key()
        self.setup_handlers()

    def setup_handlers(self):
        updater = Updater(token=Notifier.api_key, use_context=True)
        start_handler = CommandHandler('start', Notifier.start_handler)
        updater.dispatcher.add_handler(start_handler)
        info_handler = CommandHandler('info', Notifier.info_handler)
        updater.dispatcher.add_handler(info_handler)
        message_handler = MessageHandler(Filters.text & (~Filters.command), Notifier.message_handler)
        updater.dispatcher.add_handler(message_handler)
        updater.start_polling()

    def get_recipient_id(self, first_name, last_name):
        updates = self.bot.get_updates(limit=50)
        update = [u for u in updates if u['message']['chat']['first_name'] == first_name and u['message']['chat']['last_name'] == last_name]
        if update:
            self.recipient_id = update[0]['message']['chat']['id']
        else:
            raise Exception('Recipient ID not found')

    def update(message):
        Notifier.bot.send_message(text=message,chat_id=Notifier.recipient_id)

    def start_handler(update, context):
        if not Notifier.client:
            Notifier.update(f'Trading not started: No client provided to notifier object.')
            raise Exception('No client provided.')
        Notifier.recipient_id = update['message']['chat']['id']
        parts = update.message.text.split(' ')
        try:
            dct = Notifier.parse_start_parts(parts)
            Notifier.prepare_algo(dct)
            base, quote, strategy = dct['base'], dct['quote'], dct['strategy']
            update_msg = f'Session Started: Trading {base+quote} using {strategy} strategy.'
            Notifier.update(update_msg)
            Notifier.algo.trade()
        except Exception as e:
            Notifier.update(f'Trading not started: {e}')
    
    @classmethod
    def prepare_algo(cls, params:dict):
        def get_strategy(s):
            strategies = Notifier.environment.get_root_variable('strategy')
            if not strategies:
                raise Exception('No strategies found in config file.')
            return getattr(crypto.strategy, 'RSI')
        Notifier.algo = AlgoTrader(
            Notifier.client, 
            params['base'], 
            params['quote'], 
            strategy=get_strategy(params['strategy']),
            notifier=cls
        )
    
    def parse_start_parts(parts) -> dict:
        if len(parts) < 4:
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

    def info_handler(update, context):
        context.bot.send_message(chat_id=update.effective_chat.id, text=Notifier.INFO_MESSAGE)

    def message_handler(update, context):
        msg = update.message.text.lower()
        if msg == 'stop':
            Notifier.handle_stop_message(update, context)        
        if msg == 'update':
            Notifier.handle_update_message(update, context) 

    def handle_stop_message(update, context):
        Notifier.stop_trading()
        session_info = Notifier.get_session_info()
        runtime, profit = str(session_info['runtime']), str(session_info['profit'])
        summary_msg = f'Trading Stopped:\nSession Runtime: {runtime}\nSession Profit: {profit}'
        context.bot.send_message(chat_id=update.effective_chat.id, text=summary_msg)
    
    def handle_update_message(update, context):
        session_info = Notifier.get_session_info()
        runtime, profit = session_info['runtime'], str(session_info['profit'])
        summary_msg = f'Session Update:\Runtime: {runtime}\nProfit: £{profit}'
        context.bot.send_message(chat_id=update.effective_chat.id, text=summary_msg)
    
    def get_session_info():
        return {'profit': Notifier.session.profit, 'runtime': Notifier.session.get_session_runtime()}

    def stop_trading():
        Notifier.algo.stop()

    