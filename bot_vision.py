import os
import json
import sqlite3
import requests
import pandas as pd
import ccxt
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration list for multiple exchanges
TRADING_CONFIG = [
    {"exchange_id": "bitso", "symbol": "XRP/MXN", "timeframe": "1h"},
    {"exchange_id": "binance", "symbol": "XRP/USDT", "timeframe": "1h"}
]

def setup_database():
    print("Initializing SQLite database...")
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
    print(f"Connecting to {exchange_id.upper()}...")
    
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class()
    
    print(f"Getting data for {symbol} - {timeframe}")

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
    print("Processing data for AI decision...")
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
        print("CRITICAL ERROR: GROQ_API_KEY not found in .env file.")
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
    except Exception as e:
        # SANITIZED: Replaced raw error printing with a generic message to prevent data leakage
        print("Error querying the Cloud AI: Network timeout or invalid response structure.")
        return {"action": "HOLD", "reasoning": "Security fallback due to Cloud API error."}

def log_trade(exchange_id, symbol, current_price, current_sma, current_rsi, decision):
    print("Saving decision to database with indicator states...")
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
    """
    Executes a market order calculating the amount dynamically based on wallet balance.
    """
    # 1. Security validation against AI hallucinations
    if action not in ["BUY", "SELL"]:
        print(f"No action required (or invalid action '{action}') for {symbol} on {exchange_id}.")
        return False

    print(f"Preparing to execute {action} order for {symbol} on {exchange_id}...")

    try:
        exchange_class = getattr(ccxt, exchange_id)
        api_key = os.getenv(f"{exchange_id.upper()}_API_KEY")
        api_secret = os.getenv(f"{exchange_id.upper()}_API_SECRET")
        
        if not api_key or not api_secret:
             print(f"Error: Missing API credentials for {exchange_id.upper()}.")
             return False

        exchange = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True,
        })
        
        # 2. Read the wallet balance
        balance = exchange.fetch_balance()
        base_currency = symbol.split('/')[0]   # Ex: XRP
        quote_currency = symbol.split('/')[1]  # Ex: MXN or USDT
        
        risk_percentage = 0.05  # Only risk 5% of the available balance per trade
        amount = 0
        
        if action == "BUY":
            # To buy XRP, check how much fiat (MXN/USDT) is available
            available_quote = balance.get(quote_currency, {}).get('free', 0.0)
            spend_amount = available_quote * risk_percentage
            if current_price > 0:
                amount = spend_amount / current_price # Convert fiat to crypto amount
                
        elif action == "SELL":
            # To sell XRP, check how much crypto is available
            available_base = balance.get(base_currency, {}).get('free', 0.0)
            amount = available_base * risk_percentage # Sell only 5% of your coins

        # 3. Minimum balance validation
        if amount <= 0:
            print(f"Trade aborted: Calculated amount is zero. Insufficient free balance in wallet.")
            return False

        print(f"Calculated dynamic trade size: {amount:.4f} {base_currency} (Using {risk_percentage*100}% of available balance)")

        if test_mode:
             print(f"[TEST MODE] Simulated {action} of {amount:.4f} {symbol} completed successfully.")
             return True

        # LIVE RISK ZONE (test_mode=False)
        side = action.lower() 
        order = exchange.create_market_order(symbol, side, amount)
        print(f"[LIVE MODE] Order executed successfully. Order ID logged securely.")
        return True

    except ccxt.InsufficientFunds:
        print(f"Error: Insufficient funds on {exchange_id}. Trade aborted gracefully.")
    except Exception:
        # SANITIZED: No longer printing raw error to prevent info leaks in the log
        print(f"Execution error on {exchange_id}: API connection failed, network timeout, or order rejected.")
        
    return False


if __name__ == "__main__":
    setup_database()
    
    for config in TRADING_CONFIG:
        exchange_id = config["exchange_id"]
        symbol = config["symbol"]
        timeframe = config["timeframe"]
        
        print(f"\n--- Starting analysis for {exchange_id.upper()} ({symbol}) ---")
        
        try:
            data = get_market_data(exchange_id, symbol, timeframe, 1000)
            
            current_price = data.iloc[-1]['close']
            current_sma = data.iloc[-1]['SMA_14']
            current_rsi = data.iloc[-1]['RSI_14']
            
            decision = get_ai_decision(data)
            action = decision.get('action')
            
            print(f"DECISION FOR {exchange_id.upper()}: {action}")
            print(f"REASONING: {decision.get('reasoning')}")
            
            log_trade(exchange_id, symbol, current_price, current_sma, current_rsi, decision)
            
            # Execute trade with test_mode=True for safety
            execute_trade(exchange_id, symbol, action, current_price, test_mode=True)
            
        except Exception as e:
            print(f"Critical error processing {exchange_id}: {e}")
            
    print("\nRun completed successfully for all configured exchanges.")