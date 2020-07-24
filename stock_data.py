import yfinance as yf
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from datetime import date
import matplotlib as plt

st.write("""
# Predict stock prices with Hades
This app predicts the Stock´s Price of any Company!
""")

days = []
months = []
years = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]
for i in range(1,32):
    days.append(i)
    if i<13:
        months.append(i)


st.sidebar.header('User Input Parameters')

add_selectbox = st.sidebar.selectbox(
    "What company or cryptocurrency are you interested in?",
    ("AAPL", "GOOGL", "MELI","GGAL", "TSLA", "BTC-USD", "ETH-USD", "ETH-BTC", "USDARS=X")
)
day_0 = st.sidebar.selectbox(
    "Initial day",
    days
)

month_0 = st.sidebar.selectbox(
    "Initial month",
    months
)
year_0 = st.sidebar.selectbox(
    "Initial year",
    years
)

start = f"{year_0}-{month_0}-{day_0}"
today = date.today()
input_data = {"company" : add_selectbox,
                "day_0" : day_0,
                "month_0" : month_0,
                "year_0" : year_0} 
features = pd.DataFrame(input_data, index=[0])

tickerData = yf.Ticker(input_data["company"])
tickerDf = tickerData.history(period='1d', start=start, end=today)

st.write(f"""
## {input_data["company"]} Closing Price
""")
st.line_chart(tickerDf.Close)
st.write(f"""
## {input_data["company"]} Volume Price
""")
st.line_chart(tickerDf.Volume)


st.write(f"""
## {input_data["company"]} Prediction Closing Price
""")


chart_data = pd.DataFrame(
    tickerDf.Close
    )
st.line_chart(chart_data)


arr = yf.download("AAPL", start="2020-04-25", end="2020-04-30")


