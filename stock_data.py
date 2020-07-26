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



st.sidebar.header('User Input Parameters')

add_selectbox = st.sidebar.selectbox(
    "What company or cryptocurrency are you interested in?",
    ("AAPL", "GOOGL", "MELI","GGAL", "TSLA", "BTC-USD", "ETH-USD", "ETH-BTC", "USDARS=X")
)



today = date.today()
input_data = {"company" : add_selectbox,
                } 
features = pd.DataFrame(input_data, index=[0])


# Neural Network model to predict the stock market 

# Libraries
import pandas_datareader as web
import numpy as np
import math 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from datetime import datetime
from datetime import timedelta, date


# Input data
stock = input_data["company"]
start = "2000-01-01"
end = datetime.today()
tomorrow = pd.to_datetime(end) + pd.DateOffset(days=1)
previous_days = 1
datetime.today()

# Import the stock dataset
df = web.DataReader(stock, data_source = "yahoo", start = start, end = end)

# Dataset only with Close column
data = df.filter(["Close"]) # New dataset with only "Close" column
dataset = data.values # transform the data into a numpy array

#Graphs
st.write(f"""
## {input_data["company"]} Closing Price
""")
fig = plt.figure(figsize = (16,8))
plt.title(f"${stock} Close Price History", size = 30)
plt.plot(df["Close"])
plt.xlabel("Date", fontsize = 18)
plt.ylabel("Close Price", fontsize = 18)
plt.legend([f"${stock} Close Price"], fontsize=20)
plt.grid()
plt.grid(linestyle='-', linewidth='0.5', color='red')
#plt.show()
st.pyplot()

st.write(f"""
## {input_data["company"]} Volume Price
""")
fig = plt.figure(figsize = (16,8))
plt.title(f"${stock} Volume", size = 30)
plt.plot(df["Volume"])
plt.xlabel("Date", fontsize = 18)
plt.ylabel("Volume", fontsize = 18)
plt.legend([f"${stock} Volume"], fontsize=20)
plt.grid()
plt.grid(linestyle='-', linewidth='0.5', color='red')
#plt.show()
st.pyplot()

# Decide the amount of instances for train and test
train_examples = math.ceil(len(dataset)*0.8) # amount of train instances
test_explames =  len(dataset) - train_examples# amount of test instances

# Generate the train and test dataset
train_data = dataset[0:train_examples, :]
test_data = dataset[train_examples-previous_days: , : ]

# Dataset scaled (values between 0-1)
scaler = MinMaxScaler(feature_range = (0,1))
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.fit_transform(test_data)

    # Split the train set into inputs (x) and outputs (y)
x_train = []
y_train = []
for i in range(previous_days, len(train_data_scaled)):
    x_train.append(train_data_scaled[i-previous_days:i, 0])
    y_train.append(train_data_scaled[i,0])

    # COnvert into numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

        # Split the test set into inputs(x) and outputs(y)
x_test = []
y_test = scaler.fit_transform(dataset[train_examples: , : ])
for i in range(previous_days, len(test_data_scaled)):
    x_test.append(test_data_scaled[i-previous_days:i, 0])

    # COnvert into numpy arrays
x_test = np.array(x_test)
y_test = np.array(y_test)

        # Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Deep Learning model
model = Sequential() #creation of the model
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1))) # input layer
model.add(LSTM(50, return_sequences=False)) # hidden layer
model.add(Dense(25)) # hidden layer
model.add(Dense(1)) # output layer

# Compile the neural network
model.compile(optimizer = "adam", loss = "mean_squared_error")

# Train the model
model.fit(x_train, y_train, batch_size = 1, epochs = 2,verbose=0)

# TOMORROW PRICE: 2019-12-18
df_future = df.filter(["Close"])
last_60_days = df_future[-previous_days:].values
last_60_days_scaled = scaler.transform(last_60_days)
x_future = []
x_future.append(last_60_days_scaled)
x_future = np.array(x_future)
x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))
price_future = model.predict(x_future)
price_future = scaler.inverse_transform(price_future)
print(F"${stock} Close Price - Tomorrow {tomorrow} - Model: $ {((price_future[0][0]))}")
price = price_future[0][0]
#print(df)

st.write(f"""
## {input_data["company"]} Price Predicted for Tomorrow
 {price}""")












