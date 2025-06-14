from binance.enums import *
from crypto.constants import AssetFilter, DEFAULT_CURRENCY
from crypto.pricer import Pricer
import numpy as np
from pprint import pprint
import logging
import time
import random
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from binance.exceptions import BinanceAPIException, BinanceRequestException
from typing import Optional, Dict, Any, List

class Commission:
    def __init__(self, quantity, asset, fiat_value=None):
        self.quantity = quantity
        self.asset = asset
        self.fiat_value = fiat_value

class Order:
    def __init__(self, symbol: str, side: str, fills: list, quantity: float, cum_quote_qty: float, client=None):
        self.symbol = symbol
        self.side = side
        self.fills = fills
        self.quantity = quantity
        self.cum_quote_qty = cum_quote_qty
        self.average_price = self.get_average_price()
        self.client = client
        self.net = None

    def __eq__(self, other):
        return (
            self.symbol == other.symbol 
            and self.side == other.side
            and self.fills == other.fills
            and self.quantity == other.quantity
            and self.cum_quote_qty == other.cum_quote_qty
        )

    def get_average_price(self) -> float:
        '''Return the average execution price of an order.'''
        if not self.fills:
            return 0.0
        total_qty = sum([float(f['qty']) for f in self.fills])
        if total_qty == 0:
            return 0.0
        fraction = lambda x: x/total_qty
        weighted_avg = sum([fraction(float(f['qty']))*float(f['price']) for f in self.fills])
        return round(weighted_avg, 2)

    def get_total_commission(self):
        if not self.fills:
            return 0.0
        return sum([float(f['commission']) for f in self.fills])
        
    def get_commission(self):
        if not self.fills:
            return None
        qty = sum([float(f['commission']) for f in self.fills])
        commissionAsset = self.fills[0]['commissionAsset']
        fiat_val = None
        if self.client and commissionAsset != DEFAULT_CURRENCY:
            # Convert commission asset to default currency
            try:
                pricer = Pricer(self.client)
                rate_data = pricer.get_average_price(commissionAsset+DEFAULT_CURRENCY)
                rate = float(rate_data[1]['price'])
                fiat_val = rate*qty
            except Exception as e:
                logging.warning(f'Could not convert commission to {DEFAULT_CURRENCY}: {e}')
        return Commission(qty, commissionAsset, fiat_val)

    def get_asset(self):
        if not self.fills:
            return None
        return self.fills[0]['commissionAsset']

    def get_result(self) -> Dict[str, Any]:
        """
        Calculate the net result of this order accounting for commissions.
        
        Returns:
            Dict with 'side', 'asset', 'amount', 'gross_amount', 'commission_cost'
        """
        if not self.fills:
            return {'side': self.side, 'asset': None, 'amount': 0, 'gross_amount': 0, 'commission_cost': 0}
        
        res = {
            'side': self.side,
            'asset': self.get_asset(),
            'commission_cost': self.get_total_commission()
        }
        
        try:
            if self.side == 'BUY':
                # For BUY orders, return the net base asset received
                gross_amount = sum([float(f['qty']) for f in self.fills])
                commission_in_base = sum([float(f['commission']) for f in self.fills 
                                        if f['commissionAsset'] == self.symbol.replace(self.symbol[-4:], '')])
                net_amount = gross_amount - commission_in_base
                res['gross_amount'] = gross_amount
                res['amount'] = net_amount
            elif self.side == 'SELL':
                # For SELL orders, return the net quote asset received
                gross_amount = sum([float(f['qty']) * float(f['price']) for f in self.fills])
                commission_in_quote = sum([float(f['commission']) for f in self.fills 
                                         if f['commissionAsset'] == self.symbol[-4:]])
                net_amount = gross_amount - commission_in_quote
                res['gross_amount'] = gross_amount
                res['amount'] = net_amount
            else:
                raise ValueError(f'Invalid order side: {self.side}')
                
        except (ValueError, KeyError, TypeError) as e:
            logging.error(f'Error calculating order result: {e}')
            res.update({'amount': 0, 'gross_amount': 0})
        
        return res


class OrderExecutor:
    """
    Enhanced order executor with comprehensive error handling, retry logic,
    and risk management features.
    """
    
    MIN_NOTIONAL = 10.01
    MAX_RETRY_ATTEMPTS = 3
    RETRY_DELAY_BASE = 1  # seconds

    def __init__(self, client, max_retry_attempts: int = 3, enable_dry_run: bool = False):
        self.client = client
        self.max_retry_attempts = max_retry_attempts
        self.enable_dry_run = enable_dry_run
        self.order_history = []
        self.failed_orders = []
        
        # Rate limiting for orders (Binance spot: 10 orders per second, 100,000 orders per 24hr)
        self.order_times = []
        self.max_orders_per_second = 5  # Conservative limit
        
        logging.info(f'OrderExecutor initialized. Dry run: {enable_dry_run}')

    def prepare_order(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Gets trading rules and filters for a symbol with retry logic.
        
        Returns:
            Dict with 'step_precision', 'min_notional', 'min_qty', 'max_qty', 'tick_size'
        """
        for attempt in range(self.max_retry_attempts):
            try:
                symbol_info = self.client.get_symbol_info(symbol=symbol)
                if not symbol_info:
                    raise Exception(f'Symbol {symbol} not found')
                
                filters = symbol_info.get('filters', [])
                if not filters:
                    raise Exception(f'No filters found for {symbol}')
                
                lot_size_filter = self._get_filter(filters, AssetFilter.lot_size)
                min_notional_filter = self._get_filter(filters, AssetFilter.min_notional)
                price_filter = self._get_filter(filters, AssetFilter.price_filter)
                
                result = {
                    'step_precision': self._get_lot_precision(lot_size_filter),
                    'min_notional': self._get_min_notional(min_notional_filter),
                    'min_qty': float(lot_size_filter.get('minQty', 0)) if lot_size_filter else 0,
                    'max_qty': float(lot_size_filter.get('maxQty', float('inf'))) if lot_size_filter else float('inf'),
                    'step_size': float(lot_size_filter.get('stepSize', 0)) if lot_size_filter else 0,
                    'tick_size': float(price_filter.get('tickSize', 0)) if price_filter else 0,
                    'min_price': float(price_filter.get('minPrice', 0)) if price_filter else 0,
                    'max_price': float(price_filter.get('maxPrice', float('inf'))) if price_filter else float('inf')
                }
                
                logging.info(f'Order preparation successful for {symbol}: {result}')
                return result
                
            except (BinanceAPIException, BinanceRequestException) as e:
                logging.error(f'Binance API error preparing order for {symbol} (attempt {attempt + 1}): {e}')
                if attempt < self.max_retry_attempts - 1:
                    delay = self.RETRY_DELAY_BASE * (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(delay)
            except Exception as e:
                logging.error(f'Unexpected error preparing order for {symbol}: {e}')
                break
        
        logging.error(f'Failed to prepare order for {symbol} after {self.max_retry_attempts} attempts')
        return None

    def _get_filter(self, filter_arr: List[Dict], filter_name: str) -> Optional[Dict]:
        """
        Get a specific filter from the symbol filters.
        """
        try:
            matching_filters = [f for f in filter_arr if f.get('filterType') == filter_name]
            return matching_filters[0] if matching_filters else None
        except Exception as e:
            logging.error(f'Error getting filter {filter_name}: {e}')
            return None

    def _get_lot_precision(self, info: Optional[Dict]) -> int:
        """
        Calculate the decimal precision for lot sizes.
        """
        if not info:
            return 8  # Default precision
        try:
            step_size = float(info.get('stepSize', 0.00000001))
            if step_size <= 0:
                return 8
            return max(0, int(np.abs(np.log10(step_size))))
        except (ValueError, TypeError):
            return 8

    def _get_min_notional(self, info: Optional[Dict]) -> float:
        """
        Get the minimum notional value for an order.
        """
        if not info:
            return self.MIN_NOTIONAL
        try:
            return float(info.get('minNotional', self.MIN_NOTIONAL))
        except (ValueError, TypeError):
            return self.MIN_NOTIONAL

    def create_order(self, side: str, quantity: float, symbol: str, 
                    order_type: str = ORDER_TYPE_MARKET, 
                    price: Optional[float] = None,
                    time_in_force: str = TIME_IN_FORCE_GTC,
                    stop_price: Optional[float] = None,
                    validate_only: bool = False) -> Optional[Dict[str, Any]]:
        """
        Create an order with comprehensive validation, error handling, and retry logic.
        
        Args:
            side: SIDE_BUY or SIDE_SELL
            quantity: Base asset quantity (or 'min' for minimum order)
            symbol: Trading pair (e.g., 'BTCUSDT')
            order_type: Order type (MARKET, LIMIT, etc.)
            price: Price for limit orders
            time_in_force: Time in force for limit orders
            stop_price: Stop price for stop orders
            validate_only: If True, only validate without placing order
            
        Returns:
            Order response dict if successful, None if failed
        """
        if self.enable_dry_run:
            logging.info(f'DRY RUN: Would create {side} order for {quantity} {symbol}')
            return self._create_mock_order(side, quantity, symbol, order_type)
        
        # Rate limiting check
        if not self._check_order_rate_limit():
            logging.warning('Order rate limit reached, delaying order')
            time.sleep(0.2)  # Brief delay
        
        # Prepare order with symbol filters
        order_info = self.prepare_order(symbol)
        if not order_info:
            logging.error(f'Failed to prepare order for {symbol}')
            return None
        
        # Validate and adjust quantity
        adjusted_quantity = self._validate_and_adjust_quantity(
            quantity, symbol, order_info, side, price
        )
        if adjusted_quantity is None:
            return None
        
        # Create order parameters
        order_params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': adjusted_quantity
        }
        
        # Add parameters based on order type
        if order_type in [ORDER_TYPE_LIMIT, ORDER_TYPE_STOP_LOSS_LIMIT, ORDER_TYPE_TAKE_PROFIT_LIMIT]:
            if price is None:
                logging.error(f'Price required for {order_type} order')
                return None
            order_params['price'] = self._round_price(price, order_info['tick_size'])
            order_params['timeInForce'] = time_in_force
        
        if order_type in [ORDER_TYPE_STOP_LOSS, ORDER_TYPE_STOP_LOSS_LIMIT]:
            if stop_price is None:
                logging.error(f'Stop price required for {order_type} order')
                return None
            order_params['stopPrice'] = self._round_price(stop_price, order_info['tick_size'])
        
        # Validate only mode
        if validate_only:
            logging.info(f'Order validation successful for {symbol}: {order_params}')
            return {'status': 'VALIDATED', 'params': order_params}
        
        # Execute order with retry logic
        return self._execute_order_with_retry(order_params)
    
    def _validate_and_adjust_quantity(self, quantity: Any, symbol: str, 
                                    order_info: Dict, side: str, 
                                    price: Optional[float] = None) -> Optional[float]:
        """
        Validate and adjust order quantity according to symbol filters.
        """
        try:
            # Handle 'min' quantity
            if quantity == 'min':
                if not price:
                    # Get current market price
                    avg_price_data = self.client.get_avg_price(symbol=symbol)
                    price = float(avg_price_data['price'])
                
                # Calculate minimum quantity for minimum notional
                min_notional = order_info['min_notional']
                quantity = (min_notional / price) * 1.01  # Add 1% buffer
            
            quantity = float(quantity)
            
            # Validate quantity constraints
            if quantity < order_info['min_qty']:
                logging.warning(f'Quantity {quantity} below minimum {order_info["min_qty"]} for {symbol}')
                quantity = order_info['min_qty']
            
            if quantity > order_info['max_qty']:
                logging.error(f'Quantity {quantity} exceeds maximum {order_info["max_qty"]} for {symbol}')
                return None
            
            # Round to step size
            step_size = order_info['step_size']
            if step_size > 0:
                quantity = self._round_to_step(quantity, step_size)
            
            # Validate notional value
            if not price:
                avg_price_data = self.client.get_avg_price(symbol=symbol)
                price = float(avg_price_data['price'])
            
            notional_value = quantity * price
            if notional_value < order_info['min_notional']:
                logging.error(f'Notional value {notional_value} below minimum {order_info["min_notional"]} for {symbol}')
                return None
            
            logging.info(f'Validated quantity: {quantity} for {symbol} (notional: {notional_value:.2f})')
            return quantity
            
        except Exception as e:
            logging.error(f'Error validating quantity for {symbol}: {e}')
            return None
    
    def _round_to_step(self, value: float, step_size: float) -> float:
        """
        Round value to the nearest step size.
        """
        try:
            return float(Decimal(str(value)).quantize(
                Decimal(str(step_size)), rounding=ROUND_DOWN
            ))
        except (InvalidOperation, ValueError):
            # Fallback to simple rounding
            return round(value / step_size) * step_size
    
    def _round_price(self, price: float, tick_size: float) -> float:
        """
        Round price to the nearest tick size.
        """
        if tick_size <= 0:
            return price
        return self._round_to_step(price, tick_size)
    
    def _check_order_rate_limit(self) -> bool:
        """
        Check if we're within order rate limits.
        """
        current_time = time.time()
        # Remove orders older than 1 second
        self.order_times = [t for t in self.order_times if current_time - t < 1.0]
        
        if len(self.order_times) >= self.max_orders_per_second:
            return False
        
        self.order_times.append(current_time)
        return True
    
    def _execute_order_with_retry(self, order_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Execute order with retry logic and comprehensive error handling.
        """
        symbol = order_params['symbol']
        
        for attempt in range(self.max_retry_attempts):
            try:
                logging.info(f'Attempting to create order for {symbol}: {order_params}')
                
                order = self.client.create_order(**order_params)
                
                if not order:
                    raise Exception('Empty order response')
                
                order_status = order.get('status')
                logging.info(f'Order created for {symbol}: Status={order_status}, OrderId={order.get("orderId")}')
                
                # Handle different order statuses
                if order_status in ['FILLED', 'PARTIALLY_FILLED']:
                    self.order_history.append(order)
                    logging.info(f'Order executed successfully for {symbol}')
                    return order
                elif order_status in ['NEW', 'PENDING_CANCEL']:
                    logging.info(f'Order placed but not filled for {symbol}: {order_status}')
                    return order
                else:
                    logging.warning(f'Unexpected order status for {symbol}: {order_status}')
                    return order
                
            except BinanceAPIException as e:
                error_code = getattr(e, 'code', None)
                error_message = getattr(e, 'message', str(e))
                logging.error(f'Binance API error for {symbol} (attempt {attempt + 1}): Code={error_code}, Message={error_message}')
                
                # Handle specific error codes
                if error_code in [-1013, -1021]:  # Precision or order rate errors
                    logging.error(f'Order precision/rate error for {symbol}. Not retrying.')
                    break
                elif error_code == -2010:  # Insufficient balance
                    logging.error(f'Insufficient balance for {symbol} order')
                    break
                elif error_code == -1001:  # Internal error - can retry
                    if attempt < self.max_retry_attempts - 1:
                        delay = self.RETRY_DELAY_BASE * (2 ** attempt) + random.uniform(0, 1)
                        logging.info(f'Retrying order for {symbol} in {delay:.1f}s')
                        time.sleep(delay)
                        continue
                
                self.failed_orders.append({'params': order_params, 'error': str(e), 'attempt': attempt + 1})
                
            except BinanceRequestException as e:
                logging.error(f'Binance request error for {symbol} (attempt {attempt + 1}): {e}')
                if attempt < self.max_retry_attempts - 1:
                    delay = self.RETRY_DELAY_BASE * (2 ** attempt)
                    time.sleep(delay)
                    continue
                
            except Exception as e:
                logging.error(f'Unexpected error creating order for {symbol}: {e}')
                self.failed_orders.append({'params': order_params, 'error': str(e), 'attempt': attempt + 1})
                break
        
        logging.error(f'Failed to create order for {symbol} after {self.max_retry_attempts} attempts')
        return None
    
    def _create_mock_order(self, side: str, quantity: float, symbol: str, order_type: str) -> Dict[str, Any]:
        """
        Create a mock order for dry run mode.
        """
        try:
            price_data = self.client.get_avg_price(symbol=symbol)
            price = float(price_data['price'])
        except:
            price = 100.0  # Fallback price
        
        return {
            'symbol': symbol,
            'orderId': int(time.time() * 1000),  # Mock order ID
            'clientOrderId': f'mock_{int(time.time())}',
            'transactTime': int(time.time() * 1000),
            'price': str(price),
            'origQty': str(quantity),
            'executedQty': str(quantity),
            'cummulativeQuoteQty': str(quantity * price),
            'status': 'FILLED',
            'timeInForce': 'GTC',
            'type': order_type,
            'side': side,
            'fills': [{
                'price': str(price),
                'qty': str(quantity),
                'commission': str(quantity * 0.001),  # Mock 0.1% commission
                'commissionAsset': symbol[-4:] if len(symbol) >= 4 else 'USDT'
            }]
        }
    
    def get_order_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of executed orders.
        """
        return self.order_history.copy()
    
    def get_failed_orders(self) -> List[Dict[str, Any]]:
        """
        Get the history of failed orders.
        """
        return self.failed_orders.copy()
    
    def clear_history(self):
        """
        Clear order and error history.
        """
        self.order_history.clear()
        self.failed_orders.clear()
        logging.info('Order history cleared')