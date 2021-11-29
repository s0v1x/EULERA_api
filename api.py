from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ta import trend
from yahooquery import Ticker
import pandas as pd
import numpy as np
from yahoo_fin import stock_info as si
import joblib
from datetime import datetime
from pytz import timezone





app = FastAPI()

class StockIn(BaseModel):
    ticker: str


AAPL_model = joblib.load("AAPL_model.pkl")
cl_list = ["EMA_10", "EMA_20", "EMA_5", "SMA_10", "SMA_20", "SMA_5"]

def compute_EMA_SMA(df):
    df["EMA_5"] = trend.ema_indicator(df.close, 5)
    df["SMA_5"] = trend.sma_indicator(df.close, 5)

    df["EMA_10"] = trend.ema_indicator(df.close, 10)
    df["SMA_10"] = trend.sma_indicator(df.close, 10)

    df["EMA_20"] = trend.ema_indicator(df.close, 20)
    df["SMA_20"] = trend.sma_indicator(df.close, 20)

    return df


@app.get("/test")
def test():
    return {"test": "A"}


@app.post("/predict",  status_code=200)
def get_prediction(payload: StockIn):
    ticker = payload.ticker
    dt = datetime.now(timezone("America/New_York"))
    dt = dt.replace(tzinfo=None)
    dt = dt.strftime("%Y-%m-%d %H:%M:%S")
    data = dt +'\t'+'POST'+'\t/PREDICT'+'\t'+ticker+'\n'
    print(data)
    
    if ticker != 'AAPL':
        raise HTTPException(status_code=400, detail="Forecasting is not availabale for " + ticker + "...")

    df = Ticker(ticker).history(period="1mo", interval="1d", adj_timezone=True)
    df = df.loc[ticker]
    status = si.get_market_status()
    if status == "REGULAR":
        df = df.iloc[: len(df) - 1]
    df["date"] = pd.to_datetime(df.index).strftime("%Y-%m-%d")
    df.index = np.arange(0, len(df))

    df = compute_EMA_SMA(df)
    exo_data = df.iloc[-1]
    df[cl_list] = df[cl_list].shift(1)

    fc, conf_int = AAPL_model.predict(
        n_periods=1, X=pd.DataFrame(exo_data[cl_list]).T, return_conf_int=True
    )

    response_object = {"ticker": ticker, "forecast": fc[0], "CI": {"min": conf_int[0][0], "max": conf_int[0][1]}}
    return response_object




@app.post("/update",  status_code=200)
def update_model(payload: StockIn):

    ticker = payload.ticker
    dt = datetime.now(timezone("America/New_York"))
    dt = dt.replace(tzinfo=None)
    dt = dt.strftime("%Y-%m-%d %H:%M:%S")
    data = dt +'\t'+'POST'+'\t/UPDATE'+'\t'+ticker+'\n'
    print(data)

    df = Ticker(ticker).history(period="1mo", interval="1d", adj_timezone=True)
    df = df.loc[ticker]
    df["date"] = pd.to_datetime(df.index).strftime("%Y-%m-%d")
    df.index = np.arange(0, len(df))
    
    df = compute_EMA_SMA(df)
    df[cl_list] = df[cl_list].shift(1)

    AAPL_model.update([df.close.iloc[-1]], X=pd.DataFrame(df[cl_list].iloc[-1]).T)
    return 'NAAAAAADI'

