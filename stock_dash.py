{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "fe29182d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import yfinance as yf\n",
    "import pandas as pd\n",
    "\n",
    "# Function to fetch historical stock data\n",
    "def get_stock_data(ticker):\n",
    "    stock = yf.Ticker(ticker)\n",
    "    hist = stock.history(period=\"1mo\", interval=\"1h\")  # hourly data for last month\n",
    "    return hist\n",
    "\n",
    "def predict_direction(data):\n",
    "    # Simple rule: if last hour close > previous hour close, it's going up\n",
    "    last_close = data['Close'][-1]\n",
    "    prev_close = data['Close'][-2]\n",
    "    if last_close > prev_close:\n",
    "        return 'Up'\n",
    "    else:\n",
    "        return 'Down'\n",
    "\n",
    "from sklearn.linear_model import LinearRegression\n",
    "import numpy as np\n",
    "\n",
    "def forecast_prices(data, steps=1):\n",
    "    # Prepare the data for forecasting (simple linear regression example)\n",
    "    data['Time'] = np.arange(len(data))\n",
    "    X = data[['Time']]\n",
    "    y = data['Close']\n",
    "\n",
    "    model = LinearRegression()\n",
    "    model.fit(X, y)\n",
    "\n",
    "    future_time = np.arange(len(data), len(data) + steps).reshape(-1, 1)\n",
    "    predictions = model.predict(future_time)\n",
    "    return predictions\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "6d744196",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Dashboard layout\n",
    "st.title('Stock Market Prediction Dashboard')\n",
    "\n",
    "# Input section\n",
    "ticker = st.text_input('Enter stock ticker (e.g., AAPL, TSLA):', 'AAPL')\n",
    "data = get_stock_data(ticker)\n",
    "\n",
    "# Show stock data\n",
    "st.subheader(f\"Stock Data for {ticker}\")\n",
    "st.write(data.tail())\n",
    "\n",
    "# Predict stock direction\n",
    "direction = predict_direction(data)\n",
    "st.write(f\"Predicted direction for the next hour: **{direction}**\")\n",
    "\n",
    "# Forecast prices\n",
    "forecast_hours = forecast_prices(data, steps=1)\n",
    "forecast_days = forecast_prices(data, steps=24)\n",
    "\n",
    "st.write(f\"Predicted price for the next hour: **{forecast_hours[0]:.2f}**\")\n",
    "st.write(f\"Predicted price for the next day: **{forecast_days[-1]:.2f}**\")\n",
    "\n",
    "# Plot stock data\n",
    "st.subheader('Stock Price Over Time')\n",
    "fig, ax = plt.subplots()\n",
    "ax.plot(data.index, data['Close'], label='Close Price')\n",
    "ax.set_xlabel('Date')\n",
    "ax.set_ylabel('Close Price')\n",
    "st.pyplot(fig)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
