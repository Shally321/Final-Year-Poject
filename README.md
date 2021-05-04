# My Final Project
3rd year Empirical Investigation

# Description
Exploring the use of Supervised Machine Learning models for predicting stock price movement. Trading over an emerging (MSCI EM) stock market index using the EEM dataset, and an established (S&P500) stock market index using the SPY dataset, across two financial crises (Global Financial Crisis (2008) & Covid-19 Pandemic (2020)). Measuring the predictive accuracy and cumulative return performance before and during the crises. 

# Running the program
Within Algo.py, select the task you want to perform by setting the control number (on line 394):
1. for cross validation
2. for feature importance
3. for trade signals
4. for crisis analysis: Trading signals + PnL

- Select the market to train/test on by commenting out the mareket you don't want (on line 399/400)
- If control = 4, to change the length of the trading window, define the months to trade on line 437

# General Approach 

![Trading_algo_flowchart](/uploads/026f51531e3b75197e81930a86d343b0/Trading_algo_flowchart.png)

# Programming Environment
- Python version 3.5 was used for the development of the software. 
- Anaconda virtual environment was used.
- JetBrain's PyCharm IDE was used. 

# Libraries used
- Pandas
- MatplotLib
- Scikit-learn
- Ta-lib
- NumPy

# Datasets
Taken from YahooFinance:
- EEM: consists of historical data from the MSCI EM index, starting in April 2003. 
- SPY: consists of historical data from the S&P 500 index, starting in January 1993.

# Future Work
Model Training
-	Trying other ensemble models such as a Voting Regressor, or constructing a hybridised model. 
-	Spend more time on hyper-parameter testing, employing Randomised search or Bayesian optimisation. 
-	Use of extended set of attributes, perhaps an automated feature selection process. An experiment was done in this study whereby a function was created to add different features to the models to improve its trading performance. 
-	Use of Natural Language Processing techniques to extract features from public textual data like financial news and twitter posts in order to identify and predict potential trends. 
-	Improve the Neural Network model’s performance by using more data, more features and different network architectures like RNN and CNN. 
-	Test other major crises, such as the Great Depression or the Dot-Com bearish market. 
-	Optimise the training and trading window sizes.

Trading Strategy
-	Optimise and adjust the trading entry thresholds for each machine learning model, as opposed to the current simple threshold. 
-	Implement model rotation whereby only the best performing strategies on the last training window are used for the current trading period.
-	Improve exit strategy through holding an open position for shorter or longer than a one day period.
-	Add risk management features, such as: Stop-loss, Take-profit, Hedging, Diversification.
-	Report additional trade statistics, such as: Drawdown, VAR ratio, Sharpe ratio.

Real-time Trading 
- Once satisfied with the model’s performance from back-testing on historical data, the next natural step could be to try and trade on the actual market, demo or live. This is possible by integrating the trading signals with ha particular API to trade and mange orders on a given trading platform. 
