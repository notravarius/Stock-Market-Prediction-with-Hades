import yfinance as yf
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from datetime import date
import matplotlib as plt
from termcolor import colored


st.write("""
# Beat the market with Hades
This app tries to predict the future price of any company, cryptocurrency or currency and its future trend. 
This model only takes historical price values ​​as input. In the coming days, we will be incorporating fundamental analysis variables into the model.
That said, and considering the market as an unpredictable phenomenon, this model should not be taken as an investment strategy. \n

............................................................................................................................................................ \n

**Input: **History data from yahoo finance library \n
**Output**: Future Price of any company or currency \n
**Model**: Neural Network with LSTM architecture \n
**Libraries:** yfinance, pandas, tensorflow and keras, matplotlib, sklearn, numpy.

............................................................................................................................................................

**Choose the company** you want in the **left panel**. You can also change the parameters that train the neuroanl network. Then click on the button below
""")

st.sidebar.header('User Input Parameters')

add_selectbox = st.sidebar.selectbox(
    "What company or cryptocurrency are you interested in?",
    ("AAPL", "GOOGL", "MELI","GGAL", "TSLA","MSFT", "AMZN", "FB", "V", "JNJ", "MA", "JPM", "NVDA", "INTC", "NFLX", "KO", "ADBE", "PYPL", "PEP", "TM", "ORCL", "CVX", "MCD", "IBM", "SNEJF", "BTC-USD", "ETH-USD", "ETH-BTC", "USDARS=X"),
    
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

days_prev = []

for i in range(1,500):
    days_prev.append(i)
    

# Input data
stock = input_data["company"]
start = "2000-01-01"
end = datetime.today()
tomorrow = pd.to_datetime(end) + pd.DateOffset(days=1)
previous_days = st.sidebar.selectbox(
    "How many days used for train the model?",
    days_prev,
    19
)  

post_days = st.sidebar.selectbox(
    "How many days do you want to predict?",
    days_prev,
    9
)
days_before_in_graph = st.sidebar.selectbox(
    "How many days do you want to see in the graph with the predition?",
    days_prev,
    100
)



button = st.button("Click here to get the prediction!")

if button:    
    with st.spinner('Wait for it...'):
        # Import the stock dataset
        df = web.DataReader(stock, data_source = "yahoo", start = start, end = end)

        # Dataset only with Close column
        data = df.filter(["Close"]) # New dataset with only "Close" column
        dataset = data.values # transform the data into a numpy array

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
            if i < len(train_data_scaled)-post_days:
                x_train.append(train_data_scaled[i-previous_days:i, 0])
                y_train.append(train_data_scaled[i:i+post_days,0])
            
            # COnvert into numpy arrays
        x_train = np.array(x_train)
        y_train = np.array(y_train)

            # Split the test set into inputs(x) and outputs(y)
        x_test = []
        y_test = []
        #y_test = scaler.fit_transform(dataset[train_examples: , : ])
        for i in range(previous_days, len(test_data_scaled)):
            if i < len(test_data_scaled)-post_days:
                x_test.append(test_data_scaled[i-previous_days:i, 0])
                y_test.append(test_data_scaled[i:i+post_days,0])
                
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
        model.add(Dense(post_days)) # output layer

        # Compile the neural network
        model.compile(optimizer = "adam", loss = "mean_squared_error")

        # Train the model
        model.fit(x_train, y_train, batch_size = 1, epochs = 1,verbose=0)

        #Evaluate the model with the test set
        prediction_test = model.predict(x_test) # output of the model
        prediction_test = scaler.inverse_transform(prediction_test) # output of the model 
        y_test = scaler.inverse_transform(y_test)
        RMSE = np.sqrt(np.mean((prediction_test- y_test)**2)) 

        # FUTURE PRICE PREDICTION
        df_future = df.filter(["Close"])
        last_60_days = df_future[-previous_days:].values
        last_60_days_scaled = scaler.transform(last_60_days)
        x_future = []
        x_future.append(last_60_days_scaled)
        x_future = np.array(x_future)
        x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))
        price_future = model.predict(x_future)
        price_future = scaler.inverse_transform(price_future)


        new_dataset = []
        dates = []
        for i in range(0,len(dataset)):
            new_dataset.append(dataset[i][0])
        for j in range(0,post_days):
            new_dataset.append(price_future[0][j])
        for k in range(0,len(new_dataset)):
            dates.append(k)

        # Success message   
        st.success("Price predicted succsessfully!")    
        # Displaying the Top 10 similar profiles

        #Graphs
        st.write(f"""
        ## **{input_data["company"]} Closing Price**
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
        ## **{input_data["company"]} Volume Price**
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

        st.write(f"""
        ## **{input_data["company"]} Price Predicted for the next {post_days} days**:
        **Last Price Values**:
        """)
        st.dataframe(df["Close"].tail(), width=300, height=900)

        st.write(f"""**Future Stock Price:**""")
        st.write(f"The percentages of each day indicate how much the price will be compared to today.")

        total_percentages = []

        for j in range(0,post_days):
            percent = ((price_future[0][j])-(last_60_days[previous_days-1][0]))*100/(price_future[0][j])
            total_percentages.append(percent)
            if percent < 0:
                st.write(f"""{pd.to_datetime(end)+pd.DateOffset(days=j-1)}: ${price_future[0][j]} ({round(percent,2)}%)""")
            else:
                st.write(f"""{pd.to_datetime(end)+pd.DateOffset(days=j-1)}: ${price_future[0][j]} **({round(percent,2)}%)**""")
            

        st.write(f"""**Graph**""")

        plt.figure(figsize = (16,8))
        plt.title(f"${stock} Close Price History with {post_days} days future prediction", size = 30)
        plt.plot(dates[len(new_dataset)-post_days-days_before_in_graph:len(new_dataset)-post_days], new_dataset[len(new_dataset)-post_days-days_before_in_graph:len(new_dataset)-post_days], c="b",label="Close Price Data")
        plt.plot(dates[len(new_dataset)-post_days-1:],new_dataset[len(new_dataset)-post_days-1:], c="g",label=f"Future Close Price for the next {post_days} days")
        plt.xlabel("Date", fontsize = 18)
        plt.ylabel("Close Price", fontsize = 18)
        plt.legend([f"${stock} Close Price"], fontsize=20)
        plt.grid()
        plt.grid(linestyle='-', linewidth='0.5', color='red')
        plt.legend( prop={'size': 20})         
        st.pyplot()

        posittive_trend = 0
        negative_trend = 0

        for i in total_percentages:
            if i < 0:
                negative_trend = negative_trend + 1
            else:
                posittive_trend = posittive_trend + 1

        if posittive_trend > negative_trend:
            st.write(f"**This stock shows a positive trend, it could be a good investment!**")
        else:
            st.write(f"**This stock doesn´t show a positive trend.**")

        
















