# Stock-Price-Prediction-using-Bidirectional-LSTM

In this notebook, Evalys profit on investing Nvdia Stock over 3 years has been calculated. stock market data, particularly for NVDA stocks has been  examined. The evaly orders information and stock information has been retrieved using pandas, and different aspects has been visualized. Risk analysis based on historical performance will be conducted, and future stock prices will be predicted using the Bidirectional Long Short Term Memory (Bi-LSTM) method. At the end, based on the predicted price, cumulative profit of Evaly has been calculated for 3 years.


#Data Processing and Visualization Overview for Evaly datasets

1. Data Ingestion: JSON data for Evaly products was read and converted into a DataFrame.

2. Revenue Calculation: The average revenue per order was computed by taking the mean of the offer_price for each product.

3. Weekly Revenue: Weekly revenue was calculated based on the number of orders and average revenue per order.

4. Liquid Money Accumulation:  2-month delay is incorporated to account for the time required to accumulate liquid money. The revenue for each week is aggregated to reflect the delayed liquid money accumulation.

5. Total Liquid Money: The total liquid money at the end of 3 years was calculated by summing the aggregated weekly revenue.

6. Visualization: Statistical graphs were plotted to illustrate orders, revenue, and cumulative liquid money for better understanding.

# Stock-Price-Prediction-using-Bidirectional-LSTM

In this notebook, stock market data, particularly for NVDA stocks, will be examined. The stock information will be retrieved using pandas, and different aspects will be visualized. Risk analysis based on historical performance will be conducted, and future stock prices will be predicted using the Bidirectional Long Short Term Memory (Bi-LSTM) method.

### Trend Analysis (Feature Engineering)
Before building the model, data analysis was performed on NVDA stock market trends. The closing price and stock volume were plotted to observe trends over the years. A moving average was calculated and plotted to smooth out short-term fluctuations.

#### Risk Analysis
To conduct a risk analysis, the daily return of the stock was calculated. Daily return measures the day-to-day performance of stocks, comparing the price of the stock at today's closure with the previous day's closure. A positive daily return indicates appreciation in stock price on a daily comparison.

#### Correlation between Different Stocks' Closing Prices
To analyze stock returns, daily percentage returns of stocks are compared to check correlation. Although only one stock's data (NVDA) is available, it is interesting to note that the NVDA stock is positively correlated with itself, indicating low risk and low return.

## Model Selection and Training
We use the Bi-Directional LSTM model to capture dependencies in both forward and backward directions in the time series data. The model is trained on the preprocessed data to predict future stock prices. After dropping null values and cleaning data the dataset has been splitted into training and testingset. Here, 90% of the data has been used for training and testing and prediction has been done by using 10% of the data.

#### Model Building

Input Layer: input_layer = Input(shape=input_shape, name="input") where input_shape is a tuple (timesteps, features).<br>

Bidirectional LSTM Layers: Used for capturing dependencies in both forward and backward directions. return_sequences=True in the first LSTM layer returns sequences for the next LSTM layer <br>

Dropout Layer: Used dropout=0.5 in the LSTM layer to avoid model overfitting <br>

Dense Layers: The Dense layers add fully connected layers to the network. The final Dense layer outputs a single value, typically used for regression tasks.<br>

x_train is a 3D numpy array where the second dimension represents the number of timesteps, and the third dimension represents the number of features. Input_shape is adjusted as needed for my data. <br>

## Evaluation and Prediction
The model's performance is evaluated using metrics like Mean Squared Error (MSE) and Root Mean Squared Error (RMSE). Predictions are made on the test dataset. Finally, Visualizing the actual historical closing prices alongside the predicted values to analyze the model's accuracy. We can see the loss has been decreased over epochs. And the model is not overfitting.

## Investment and Profit Calculation based on Stock price Prediction

1. Determine Months: Approximate the number of months by dividing total weeks by 4.

2. Monthly Investment: Calculate the fixed monthly investment by dividing total liquid money by the number of months.

3. Predict Stock Prices: Generate monthly predicted stock prices between the last real and predicted prices.

4. Initialize Investment Tracking: Set up an array to track cumulative investment values over time.

5. Compute Investment Growth: Adjust the monthly investment by predicted stock prices and update cumulative investment values.

6. Calculate Profit: Subtract the initial liquid money from the final investment value to determine the net profit or loss.

7. Cumulative Profit: Then computes the cumulative sum of these monthly profits, resulting in an array that shows how the profit accumulates over time

# Visualization over time

Finally a graph has been plotted showing investment, profit and predicted stock price over time
