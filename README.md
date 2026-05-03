# Autonomous Quantitative AI Trading Bot

## Overview
This project is an autonomous quantitative trading bot built with Python. It retrieves live market data from multiple cryptocurrency exchanges (Bitso and Binance), calculates technical indicators (SMA, RSI), and leverages a Large Language Model (Llama 3.1 via Groq API) to make buy, sell, or hold decisions based on market context. It logs all historical data and decisions to a local SQLite database for auditing. 

**Security & Risk Management:** The bot features dynamic position sizing, automatically calculating trade amounts based on a defined risk percentage (e.g., 5% of available wallet balance) to protect capital during execution. Error logs are sanitized to prevent API key or header leakage in production environments.

## Architecture and Tech Stack
* Language: Python 3
* Market Data & Execution: CCXT Library
* AI Engine: Llama 3.1 8B (via Groq API)
* Database: SQLite
* Infrastructure: Designed for Linux Cloud Deployment (e.g., Cron Jobs)

## Prerequisites
* Python 3.8+
* API keys for Bitso and Binance
* API key for Groq
* A Linux server is recommended for 24/7 automated execution

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YourUsername/YourRepo.git
cd YourRepo
```

2. Create a virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Configuration
Create a `.env` file in the root directory of the project and add your specific credentials and configurations. Ensure this file is added to your `.gitignore` and never committed to version control.

```env
BITSO_API_KEY=your_bitso_key
BITSO_API_SECRET=your_bitso_secret
BINANCE_API_KEY=your_binance_key
BINANCE_API_SECRET=your_binance_secret
GROQ_API_KEY=your_groq_key
DB_NAME=paper_trading.db
GROQ_API_URL=https://api.groq.com/openai/v1/chat/completions
```

## Usage
Run the bot manually to verify connectivity and logic:
```bash
python bot_vision.py
```

By default, the script executes in test mode (`test_mode=True`), meaning it will simulate trades and log decisions to the database without executing real market orders. 

To deploy with real funds, update the `execute_trade` function call in the main block of the script:
```python
execute_trade(exchange_id, symbol, action, current_price, test_mode=False)
```
*Note: The bot uses dynamic position sizing. Ensure the `risk_percentage` variable inside the `execute_trade` function (defaulted to 0.05 or 5%) is configured according to your personal risk management strategy before going live.*

## Automation
To run the bot continuously (e.g., every hour), configure a Cron job on your Linux server using absolute paths:

```bash
0 * * * * cd /path/to/project && /path/to/project/venv/bin/python bot_vision.py >> /path/to/project/bot.log 2>&1
```

## Disclaimer
This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. The authors and all affiliates assume no responsibility for your trading results.