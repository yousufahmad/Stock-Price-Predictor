# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:33:08 2019
@author: yousuf
"""

import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression

import pandas as pd

import pandas_datareader.data as web

import matplotlib.pyplot as plt


'''
predictLinear takes in a ticker, a start date, and the number
of days in the future and computes a prediction of the stock 
price in the future using linear regression
'''
def predictLinear(ticker, start_date, days_in_future):
    end = datetime.now()
    #Retrieves stock data using Pandas Datareader
    df = web.DataReader(ticker, "yahoo", start_date, end)
    df.to_csv(ticker + "_history.csv")
    
    #Retrieve close values of the stock for every single day
    close_vals = df['Close'].values
    
    #Make a list of numbers that correspond to a date
    #ex: 0 -> 1/1/2017, 1 -> 1/2/2017
    dates = np.arange(len(df))
    
    plt.plot(dates, close_vals)
    
    #Generate matrix to feed into linear regression algorithm
    Mat = np.zeros((len(dates), 2))
    
    #first column is a vector of ones
    Mat[:, 0] = np.ones(len(dates))
    
    #second column is our dates(x dates)
    Mat[:, 1] = dates
    
    #Generate linear regression model
    model = LinearRegression().fit(Mat, close_vals)
    coeffs = model.coef_
    intercept = model.intercept_
    
    #graphing data
    a = np.linspace(0, len(Mat), 10000)
    b = model.intercept_ + coeffs[1]*a
    
    plt.title('Linear Regression Model for ' + ticker + 'starting at ' + 
              start_date.strftime('%m-%d-%y'))
    plt.ylabel('Price ($)')
    plt.xlabel('Date')
    
    plt.plot(dates, close_vals, color='b')
    plt.plot(a, b, color='r')
    plt.show()
    
    
    #Compute prediction using computed coefficients
    # y = b + ax
    # x is the number of days in the future
    # b is the intercept
    # a is coeffs[1]
    # y is the prediction
    prediction = intercept + coeffs[1] * (len(dates) + days_in_future - 1)
    
    return prediction




ticker = input("Enter a list of tickers separated by commas: ")
ticker_array = ticker.split(', ')
print(ticker_array)
start_date = input("Enter a date (MM-DD-YYYY): ")
start_date = datetime.strptime(start_date, '%m-%d-%Y') 
days_in_future = int(input("Enter the number of days in the future: "))

for elem in ticker_array:
    prediction = predictLinear(elem, start_date, days_in_future)
    print(elem + " price in " + str(days_in_future) + " days will be $" 
          + str(round(prediction, 2)) + " according to this model")
