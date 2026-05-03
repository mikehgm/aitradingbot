import os
import json
import sqlite3
import requests
import pandas as pd
import ccxt
import logging
from datetime import datetime
from dotenv import load_dotenv

# Configure logging to both file and console with automatic timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables from .env file
load_dotenv()

# Configuration list for multiple exchanges
TRADING_CONFIG = [
    {"exchange_id": "bitso", "symbol": "XRP/MXN", "timeframe": "1h"},
    {"exchange_id": "binance", "symbol": "XRP/USDT", "timeframe": "1h"}
]

def setup_database():
    logging.info("Initializing SQLite database...")
    db_name = os.getenv('DB_NAME', 'paper_trading.db')
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                exchange TEXT,
                symbol TEXT,
                price REAL,
                sma_14 REAL,
                rsi_14 REAL,
                action TEXT,
                reasoning TEXT
            )
        ''')
        conn.commit()

def get_market_data(exchange_id, symbol, timeframe, limit=1000):
    logging.info(f"Connecting to {exchange_id.upper()}...")
    
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class()
    
    logging.info(f"Fetching data for {symbol} - {timeframe}")

    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    df['SMA_14'] = df['close'].rolling(window=14).mean()
    
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    df = df.dropna()
    
    return df 

def get_ai_decision(df):
    logging.info("Processing market data for AI decision...")
    last_data = df.tail(15).to_dict(orient='records')

    for row in last_data:
        row['timestamp'] = str(row['timestamp'])

    json_data = json.dumps(last_data, indent=2)

    prompt = f"""
    You are an expert quantitative trading bot. Your goal is to analyze historical data and make a decision.
    
    Here are the last 15 periods, focusing on the 'close' price, Moving Average ('SMA_14'), and Relative Strength Index ('RSI_14'):
    {json_data}
    
    Decision rules:
    - BUY SIGNAL: If 'close' crosses above 'SMA_14' AND 'RSI_14' is below 30 (oversold) or trending upwards from a low point.
    - SELL SIGNAL: If 'close' falls below 'SMA_14' OR 'RSI_14' is above 70 (overbought).
    - HOLD SIGNAL: If 'RSI_14' is between 40 and 60 with no clear SMA crossover, or if signals contradict each other.
    
    RESPOND ONLY WITH A VALID JSON. Do not add text outside the brackets.
    Format:
    {{
        "action": "BUY", "SELL", or "HOLD",
        "reasoning": "Technical explanation in under 20 words referencing SMA and RSI"
    }}
    """

    groq_key = os.getenv('GROQ_API_KEY')
    if not groq_key:
        logging.error("GROQ_API_KEY not found in .env file.")
        return {"action": "HOLD", "reasoning": "Missing Groq API Key."}

    url = os.getenv('GROQ_API_URL', 'https://api.groq.com/openai/v1/chat/completions')
    
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system", 
                "content": "You are a quantitative trading bot. Always output strictly valid JSON."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() 
        ai_text = response.json()['choices'][0]['message']['content']
        return json.loads(ai_text)
    except Exception:
        logging.error("Cloud AI Query failed: Network error or invalid response.")
        return {"action": "HOLD", "reasoning": "Security fallback due to Cloud API error."}

def log_trade(exchange_id, symbol, current_price, current_sma, current_rsi, decision):
    logging.info("Saving decision to database...")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db_name = os.getenv('DB_NAME', 'paper_trading.db')
    
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO trade_logs (timestamp, exchange, symbol, price, sma_14, rsi_14, action, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (current_time, exchange_id, symbol, current_price, current_sma, current_rsi, decision.get('action'), decision.get('reasoning')))
        conn.commit()

def execute_trade(exchange_id, symbol, action, current_price, test_mode=True):
    # Security validation against AI hallucinations
    if action not in ["BUY", "SELL"]:
        logging.info(f"No trade action required for {symbol} on {exchange_id} (Action: {action}).")
        return False

    logging.info(f"Initiating {action} sequence for {symbol} on {exchange_id}...")

    try:
        exchange_class = getattr(ccxt, exchange_id)
        api_key = os.getenv(f"{exchange_id.upper()}_API_KEY")
        api_secret = os.getenv(f"{exchange_id.upper()}_API_SECRET")
        
        if not api_key or not api_secret:
             logging.error(f"Missing API credentials for {exchange_id.upper()}.")
             return False

        exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        
        # Risk management: Calculate amount based on wallet balance
        base_currency = symbol.split('/')[0]
        quote_currency = symbol.split('/')[1]
        risk_percentage = 0.05
        amount = 0
        
        if test_mode:
            logging.info(f"[TEST MODE] Skipping live balance fetch. Using simulated risk amount.")
            # Set a fixed simulated amount for testing purposes
            amount = 10.0
        else:
            balance = exchange.fetch_balance()
            if action == "BUY":
                available_quote = balance.get(quote_currency, {}).get('free', 0.0)
                spend_amount = available_quote * risk_percentage
                if current_price > 0:
                    amount = spend_amount / current_price
            elif action == "SELL":
                available_base = balance.get(base_currency, {}).get('free', 0.0)
                amount = available_base * risk_percentage

        if amount <= 0:
            logging.warning(f"Trade aborted: Amount is zero. Check {exchange_id} wallet balance.")
            return False

        logging.info(f"Calculated trade size: {amount:.4f} {base_currency} ({risk_percentage*100}% risk)")

        if test_mode:
             logging.info(f"[TEST MODE] Simulated {action} of {amount:.4f} {symbol} successful.")
             return True

        # LIVE EXECUTION
        side = action.lower() 
        order = exchange.create_market_order(symbol, side, amount)
        logging.info(f"[LIVE MODE] Order executed successfully. ID: {order.get('id')}")
        return True

    except ccxt.InsufficientFunds:
        logging.error(f"Insufficient funds on {exchange_id} to complete {action}.")
    except Exception:
        logging.error(f"Execution failed on {exchange_id}: API error or connection timeout.")
        
    return False

if __name__ == "__main__":
    setup_database()
    
    for config in TRADING_CONFIG:
        exchange_id = config["exchange_id"]
        symbol = config["symbol"]
        timeframe = config["timeframe"]
        
        logging.info(f"--- Starting analysis for {exchange_id.upper()} ({symbol}) ---")
        
        try:
            data = get_market_data(exchange_id, symbol, timeframe, 1000)
            
            current_price = data.iloc[-1]['close']
            current_sma = data.iloc[-1]['SMA_14']
            current_rsi = data.iloc[-1]['RSI_14']
            
            decision = get_ai_decision(data)
            action = decision.get('action')
            
            logging.info(f"AI Decision: {action}")
            logging.info(f"Reasoning: {decision.get('reasoning')}")
            
            log_trade(exchange_id, symbol, current_price, current_sma, current_rsi, decision)
            
            # Execute trade with test_mode=True for safety
            execute_trade(exchange_id, symbol, action, current_price, test_mode=True)
            
        except Exception:
            logging.error(f"Critical error during {exchange_id.upper()} analysis cycle.")
            
    logging.info("Run completed successfully for all configured exchanges.\n")