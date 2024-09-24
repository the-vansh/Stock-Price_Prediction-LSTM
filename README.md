# Stock Price Prediction using LSTM

This project uses Long Short-Term Memory (LSTM) neural networks to predict stock prices based on 20 years of historical stock data (open and close prices). It also includes a web-based application built with Streamlit, allowing users to interact with the model and make stock price predictions for selected companies.

## Features

- **Stock Price Prediction**: Predict future stock prices using an LSTM model trained on 20 years of historical data.
- **Interactive Web App**: A user-friendly interface built with Streamlit for users to interact with the application and predict stock prices.
- **Yahoo Finance Data**: Fetch real-time stock data from Yahoo Finance by entering the stock symbol.
- **Model Persistence**: The trained LSTM model is saved and loaded to make predictions quickly and efficiently.

## Dataset

The LSTM model is trained using 20 years of stock data, specifically the opening and closing prices of various stocks. The data is fetched from Yahoo Finance.

## How It Works

1. **Model Training**:
   - The LSTM model is trained using historical stock data (open and close prices) over the last 20 years.
   - Once trained, the model is saved for future use.
   
2. **Web App**:
   - The Streamlit web app provides an interface for users to input a stock symbol (e.g., AAPL for Apple) and predict future stock prices.
   - The app fetches real-time data from Yahoo Finance to analyze and make predictions using the trained model.

## Requirements

Make sure the following dependencies are installed:

```bash
numpy
pandas
tensorflow
yfinance
matplotlib
streamlit
```

## Running the Application
```
streamlit run main.py
```

## Contributors

- **Vansh Sani** - Creator and Maintainer


## Contact

For any issues or inquiries, please contact [Vansh Saini](mailto:contactvanshsaini@gmail.com).

---

Thank you for exploring the project! We hope you find it exciting and useful. If you have any questions or feedback, please feel free to reach out.

