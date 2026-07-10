"""One-time calibration helper: fetch real market data from Massive (formerly
Polygon.io -- polygon.io/docs now redirect to massive.com/docs) to sanity-check
the peg-related fields in configs/currencies/*.yaml and
src/economy/macro_state.py's default peg_reference_rates.

This is NOT part of any simulation code path. It never runs during a
simulation and simulation_runner.py does not import it -- Phase 1 stays
deterministic and reproducible regardless of live market conditions. Run it
manually, read the printed report, and hand-edit the YAML configs yourself if
you want to incorporate the real numbers -- it does not write any files.

Usage:
    pip install -e ".[market-data]"
    python scripts/calibrate_currency_configs.py
"""

import os
import sys

import requests
from dotenv import load_dotenv

from src.utils.constants import REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

BASE_URL = "https://api.massive.com"
API_KEY = os.getenv("POLYGON_API_KEY") or os.getenv("Polygon_API_KEY")

# USD-pegged stablecoins: peg_error is deviation from $1.00.
USD_STABLECOIN_TICKERS = {"USDC": "X:USDCUSD", "USDT": "X:USDTUSD", "DAI": "X:DAIUSD"}
# Gold-backed tokens: peg_error is deviation from the live XAU/USD spot rate, not from $1.
GOLD_TOKEN_TICKERS = {"PAXG": "X:PAXGUSD", "XAUT": "X:XAUTUSD"}
# Reference peg rates used to seed src/economy/macro_state.py's peg_reference_rates.
FOREX_TICKERS = {"EUR": "C:EURUSD", "XAU": "C:XAUUSD"}


def _get(path: str, params: dict) -> dict:
    response = requests.get(f"{BASE_URL}{path}", params={**params, "apiKey": API_KEY}, timeout=10)
    response.raise_for_status()
    return response.json()


def fetch_forex_rate(peg: str) -> float | None:
    ticker = FOREX_TICKERS[peg]
    data = _get(f"/v2/snapshot/locale/global/markets/forex/tickers/{ticker}", {})
    return data.get("ticker", {}).get("lastQuote", {}).get("a")


def fetch_crypto_price(ticker: str) -> float | None:
    data = _get("/v2/snapshot/locale/global/markets/crypto/tickers", {"tickers": ticker})
    tickers = data.get("tickers", [])
    return tickers[0]["lastTrade"]["p"] if tickers else None


def main() -> None:
    if not API_KEY:
        sys.exit("Set POLYGON_API_KEY (or Polygon_API_KEY) in .env before running this script")

    print("=== Reference peg rates (compare against src/economy/macro_state.py defaults) ===")
    xau_usd_rate = None
    for peg, ticker in FOREX_TICKERS.items():
        try:
            rate = fetch_forex_rate(peg)
        except requests.RequestException as exc:
            print(f"{peg}/USD ({ticker}): request failed -- {exc}")
            continue
        if peg == "XAU":
            xau_usd_rate = rate
        print(f"{peg}/USD: {rate}")

    print("\n=== USD-pegged stablecoin deviation (compare against configs/currencies/*.yaml peg_error) ===")
    for symbol, ticker in USD_STABLECOIN_TICKERS.items():
        try:
            price = fetch_crypto_price(ticker)
        except requests.RequestException as exc:
            print(f"{symbol}: request failed -- {exc}")
            continue
        if price is None:
            print(f"{symbol}: no data returned for {ticker}")
            continue
        print(f"{symbol}: last price ${price:.4f}, implied peg_error: {abs(price - 1.0):.4f}")

    print("\n=== Gold-backed token deviation from live XAU/USD spot ===")
    for symbol, ticker in GOLD_TOKEN_TICKERS.items():
        try:
            price = fetch_crypto_price(ticker)
        except requests.RequestException as exc:
            print(f"{symbol}: request failed -- {exc}")
            continue
        if price is None:
            print(f"{symbol}: no data returned for {ticker}")
            continue
        if xau_usd_rate:
            print(f"{symbol}: last price ${price:.2f}, implied peg_error: {abs(price - xau_usd_rate) / xau_usd_rate:.4f}")
        else:
            print(f"{symbol}: last price ${price:.2f} (XAU/USD rate unavailable, can't compute peg_error)")

    print("\nThis script is advisory only -- it has not modified any config files.")


if __name__ == "__main__":
    main()
