"""
AI Trading Advancements - Timezone Utilities

This module provides timezone utilities specifically for financial markets,
ensuring all timestamps use Eastern Standard Time (EST/EDT) - the timezone
of NYSE and NASDAQ.

Author: AI Trading Development Team  
Date: August 31, 2025
Version: 1.0.0
"""

from datetime import datetime, timezone, time
from typing import Optional, Union
import pytz


# Eastern timezone (handles EST/EDT automatically)
EASTERN_TZ = pytz.timezone('America/New_York')

# Market hours in Eastern time
MARKET_OPEN_TIME = time(9, 30)  # 9:30 AM ET
MARKET_CLOSE_TIME = time(16, 0)  # 4:00 PM ET

# Pre-market and after-hours
PREMARKET_OPEN_TIME = time(4, 0)   # 4:00 AM ET
AFTERHOURS_CLOSE_TIME = time(20, 0)  # 8:00 PM ET


def now_eastern() -> datetime:
    """
    Get current time in Eastern timezone.
    
    Returns:
        datetime: Current time in Eastern timezone (EST/EDT)
    """
    return datetime.now(EASTERN_TZ)


def to_eastern(dt: datetime) -> datetime:
    """
    Convert datetime to Eastern timezone.
    
    Args:
        dt: Datetime to convert (can be naive or timezone-aware)
        
    Returns:
        datetime: Datetime in Eastern timezone
    """
    if dt.tzinfo is None:
        # Assume naive datetime is in UTC and convert
        dt = dt.replace(tzinfo=timezone.utc)
    
    return dt.astimezone(EASTERN_TZ)


def from_eastern_to_utc(dt: datetime) -> datetime:
    """
    Convert Eastern datetime to UTC.
    
    Args:
        dt: Datetime in Eastern timezone
        
    Returns:
        datetime: Datetime in UTC
    """
    if dt.tzinfo is None:
        # Assume naive datetime is Eastern
        dt = EASTERN_TZ.localize(dt)
    
    return dt.astimezone(timezone.utc)


def eastern_timestamp() -> float:
    """
    Get current Eastern time as Unix timestamp.
    
    Returns:
        float: Unix timestamp adjusted for Eastern timezone
    """
    return now_eastern().timestamp()


def format_eastern_time(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S %Z") -> str:
    """
    Format datetime in Eastern timezone.
    
    Args:
        dt: Datetime to format
        format_str: Format string (default includes timezone)
        
    Returns:
        str: Formatted datetime string in Eastern timezone
    """
    eastern_dt = to_eastern(dt)
    return eastern_dt.strftime(format_str)


def is_market_hours(dt: Optional[datetime] = None) -> bool:
    """
    Check if given time is during regular market hours (9:30 AM - 4:00 PM ET).
    
    Args:
        dt: Datetime to check (defaults to current Eastern time)
        
    Returns:
        bool: True if during market hours
    """
    if dt is None:
        dt = now_eastern()
    else:
        dt = to_eastern(dt)
    
    # Check if it's a weekday (Monday=0, Sunday=6)
    if dt.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check time
    current_time = dt.time()
    return MARKET_OPEN_TIME <= current_time <= MARKET_CLOSE_TIME


def is_extended_hours(dt: datetime = None) -> bool:
    """
    Check if given time is during extended hours (pre-market or after-hours).
    
    Args:
        dt: Datetime to check (defaults to current Eastern time)
        
    Returns:
        bool: True if during extended hours
    """
    if dt is None:
        dt = now_eastern()
    else:
        dt = to_eastern(dt)
    
    # Check if it's a weekday
    if dt.weekday() >= 5:
        return False
    
    current_time = dt.time()
    
    # Pre-market: 4:00 AM - 9:30 AM ET
    is_premarket = PREMARKET_OPEN_TIME <= current_time < MARKET_OPEN_TIME
    # After-hours: 4:00 PM - 8:00 PM ET  
    is_afterhours = MARKET_CLOSE_TIME < current_time <= AFTERHOURS_CLOSE_TIME
    
    return is_premarket or is_afterhours


def next_market_open(dt: datetime = None) -> datetime:
    """
    Get the next market open time.
    
    Args:
        dt: Reference datetime (defaults to current Eastern time)
        
    Returns:
        datetime: Next market open time in Eastern timezone
    """
    if dt is None:
        dt = now_eastern()
    else:
        dt = to_eastern(dt)
    
    # Start with today
    next_open = dt.replace(hour=9, minute=30, second=0, microsecond=0)
    
    # If current time is past market open today, or it's weekend, move to next business day
    if dt.time() >= MARKET_OPEN_TIME and dt.weekday() < 5:
        next_open = next_open.replace(day=next_open.day + 1)
    
    return next_open


def next_market_close(dt: Optional[datetime] = None) -> datetime:
    """
    Get the next market close time.
    
    Args:
        dt: Reference datetime (defaults to current Eastern time)
        
    Returns:
        datetime: Next market close time in Eastern timezone
    """
    if dt is None:
        dt = now_eastern()
    else:
        dt = to_eastern(dt)
    
    # If it's before market close today and weekday, use today
    if dt.time() = 5:
        next_close = next_close.replace(day=next_close.day + 1)
    
    return next_close


def market_session_type(dt: Optional[datetime] = None) -> str:
    """
    Determine the current market session type.
    
    Args:
        dt: Datetime to check (defaults to current Eastern time)
        
    Returns:
        str: 'REGULAR', 'PREMARKET', 'AFTERHOURS', or 'CLOSED'
    """
    if dt is None:
        dt = now_eastern()
    else:
        dt = to_eastern(dt)
    
    if dt.weekday() >= 5:  # Weekend
        return 'CLOSED'
    
    current_time = dt.time()
    
    if MARKET_OPEN_TIME  datetime:
    """
    Get the start of the trading day (4:00 AM ET) for a given date.
    
    Args:
        dt: Reference date (defaults to current Eastern time)
        
    Returns:
        datetime: Start of trading day in Eastern timezone
    """
    if dt is None:
        dt = now_eastern()
    else:
        dt = to_eastern(dt)
    
    return dt.replace(hour=4, minute=0, second=0, microsecond=0)


def trading_day_end(dt: Optional[datetime] = None) -> datetime:
    """
    Get the end of the trading day (8:00 PM ET) for a given date.
    
    Args:
        dt: Reference date (defaults to current Eastern time)
        
    Returns:
        datetime: End of trading day in Eastern timezone
    """
    if dt is None:
        dt = now_eastern()
    else:
        dt = to_eastern(dt)
    
    return dt.replace(hour=20, minute=0, second=0, microsecond=0)


def validate_market_timestamp(dt: datetime) -> bool:
    """
    Validate that a timestamp is within reasonable market data bounds.
    
    Args:
        dt: Datetime to validate
        
    Returns:
        bool: True if timestamp is valid for market data
    """
    eastern_dt = to_eastern(dt)
    
    # Check if it's not too far in the future (max 1 day ahead)
    if eastern_dt > now_eastern().replace(day=now_eastern().day + 1):
        return False
    
    # Check if it's not too far in the past (min 10 years ago)
    min_date = now_eastern().replace(year=now_eastern().year - 10)
    if eastern_dt  pytz.BaseTzInfo:
    """
    Get the market timezone object.
    
    Returns:
        pytz.BaseTzInfo: Eastern timezone object
    """
    return EASTERN_TZ


if __name__ == "__main__":
    # Timezone utility demonstration
    print("\n=== AI Trading Timezone Utilities Demo ===")
    
    current_time = now_eastern()
    print(f"[DATA] Current Eastern time: {format_eastern_time(current_time)}")
    
    utc_time = from_eastern_to_utc(current_time)
    print(f"[DATA] Same time in UTC: {utc_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    print(f"[DATA] Market session: {market_session_type()}")
    print(f"[DATA] Is market hours: {is_market_hours()}")
    print(f"[DATA] Is extended hours: {is_extended_hours()}")
    
    next_open = next_market_open()
    next_close = next_market_close()
    
    print(f"[DATA] Next market open: {format_eastern_time(next_open)}")
    print(f"[DATA] Next market close: {format_eastern_time(next_close)}")
    
    print(f"\n[SUCCESS] All times are in Eastern timezone (NYSE/NASDAQ)")
