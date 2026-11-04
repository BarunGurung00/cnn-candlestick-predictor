# cnn-candlestick-predictor

# CNN Candlestick Pattern Predictor

A lightweight deep learning framework that converts financial time-series market data (OHLCV) into normalized visual feature vectors and trains a Convolutional Neural Network (CNN) to predict future price directions.

## Project Structure

```text
cnn-candlestick-predictor/
├── README.md         # Project description, installation steps, and usage guide
├── main.py           # Orchestration entry point: builds model, trains, and evaluates
└── utils.py          # Helper routines for data ingestion and tensor construction

## Pre-requisite
git clone https://github.com/BarunGurung00/cnn-candlestick-predictor.git
cd cnn-candlestick-predictor
pip install yfinance tensorflow pandas numpy
python main.py
