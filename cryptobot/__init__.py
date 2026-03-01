"""
CryptoBot — Trading Bot Educativo

Uso básico:
    from cryptobot import CryptoBot

    bot = CryptoBot(symbol="BTC")
    bot.fetch_data()
    bot.create_features()
    bot.detect_regime()
    bot.recommend_strategies()
"""

from .bot import CryptoBot

__version__ = "0.1.0"
__all__ = ["CryptoBot"]
