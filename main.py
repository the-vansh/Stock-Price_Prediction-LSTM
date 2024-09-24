import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

st.title('STOCK TREND PREDICTION AND ANALYSIS')
start = st.text_input('ENTER THE STARTING DATE IF YOU WANT','2015-01-01')
end = st.text_input('ENTER THE ENDING DATE DATE IF YOU WANT','2024-01-01')


user_input=st.text_input('ENTER THE NAME OF THE STOCK COMPANY','RELIANCE.NS')
df = yf.download(user_input, start=start, end=end)
st.write(df.describe())

st.subheader('OPENING PRICE VS TIME CHART')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Open)
st.pyplot(fig)

st.subheader('CLOSING PRICE VS TIME CHART')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('100 DAYS AVERAGE OF OPENING VALUES')
ma100_O=df.Open.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100_O)
plt.plot(df.Open)
st.pyplot(fig)

st.subheader('100 DAYS AVERAGE OF CLOSING VALUES')
ma100=df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('200 DAYS AVERAGE OF OPENGING VALUES')
ma200_O=df.Open.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma200_O)
plt.plot(df.Open)
st.pyplot(fig)

st.subheader('200 DAYS AVERAGE OF CLOSING VALUES')
ma200=df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

#Starting of trainging and testing
#FOR CLOSING VALUES
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
#FOR OPENING VALUES
data_training_O = pd.DataFrame(df['Open'][0:int(len(df)*0.70)])
data_test_O = pd.DataFrame(df['Open'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_test.shape)
print(data_training_O.shape)
print(data_test_O.shape)

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training) #for closing
data_training_array_O = scaler.fit_transform(data_training_O) #for opening


#load model

model = load_model('keras_model.h5')#for closing values

model1 = load_model('keras_model1.h5')# for opening values

past_100_days = data_training.tail(100)
past_100_days_O = data_training_O.tail(100)
final_df = pd.concat([past_100_days, data_test], ignore_index=True)
final_df_O = pd.concat([past_100_days_O, data_test_O], ignore_index=True)

input_data = scaler.fit_transform(final_df)
input_data_O = scaler.fit_transform(final_df_O)

# for closing values
x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

# for opening values
x_test_O=[]
y_test_O=[]

for i in range(100,input_data_O.shape[0]):
  x_test_O.append(input_data_O[i-100:i])
  y_test_O.append(input_data_O[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)
x_test_O,y_test_O = np.array(x_test_O),np.array(y_test_O)

y_predict=model.predict(x_test)# prediction for opening values
y_predict_O=model1.predict(x_test_O)#predicction for closing values

#finding the scale factor
scaler = scaler.scale_
scale_factor = 1/scaler[0]

#getting original values for closing
y_predict = y_predict*scale_factor
y_test=y_test*scale_factor
#getting original values for opening
y_predict_O = y_predict_O*scale_factor
y_test_O=y_test_O*scale_factor

st.subheader('PREDICTION VS ORIGINAL FOR OPENING VALUES')

fig3 = plt.figure(figsize=(12,6))
plt.plot(y_test_O,'b',label='original price')
plt.plot(y_predict_O,'r',label='predicted price')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)

st.subheader('PREDICTION VS ORIGINAL FOR CLOSING VALUES')
fig2= plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='original price')
plt.plot(y_predict,'r',label='predicted price')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
