#!/usr/bin/env python3
"""
Enhanced CLI runner for crypto algorithmic trading.
This is the main entry point for the trading system.
"""

import sys
import os

# Add the crypto package to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto.cli import main

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)