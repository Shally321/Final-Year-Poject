"""Import the libraries"""
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import talib as ta

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

""" Defining markets """
MARKET_SPY = "SPY"
MARKET_EM = "EEM"

"""Accurate date for the visualisation"""
date_parser = lambda x: pd.datetime.strptime(x, "%d/%m/%Y")

"""
    Makes features out of raw data set (Yahoo Finance format) 
    INPUT: <MARKET>.csv
    OUTPUT:<MARKET>_features.csv
"""
def prepare_features(market):
    """Store the data, Set the Date as te index"""
    fn_source     = market + ".csv"
    fn_target     = market + "_features.csv"
    data          = pd.read_csv(fn_source, index_col = "Date", parse_dates = True, date_parser= date_parser)

    # variables to hold numpy array series
    data["Date"]  = data.index
    high          = data["High"].values
    low           = data["Low"].values
    close         = data["Close"].values
    open          = data["Open"].values
    volume        = data["Volume"].values

    """Technical Indicators:"""
    # Simple Moving Average for 5,10,20 days interval:
    data["SMA5"]  = ta.SMA(close, timeperiod = 5)
    data["SMA10"] = ta.SMA(close, timeperiod = 10)
    data["SMA20"] = ta.SMA(close, timeperiod=20)

    # Exponentially Moving Average for 5,10,20 days interval:
    data["EMA5"]  = ta.EMA(close, timeperiod = 5)
    data["EMA10"] = ta.EMA(close, timeperiod = 10)
    data["EMA20"] = ta.EMA(close, timeperiod=20)

    # Double Exponentially Moving Average for 5,10,20 days interval
    data["DEMA5"]  = ta.DEMA(close, timeperiod = 5)
    data["DEMA10"] = ta.DEMA(close, timeperiod = 10)
    data["DEMA20"] = ta.DEMA(close, timeperiod = 20)

    # Weighted Moving Average for 5,10,20 days interval
    data["WMA5"]  = ta.WMA(close, timeperiod = 5)
    data["WMA10"] = ta.WMA(close, timeperiod = 10)
    data["WMA20"] = ta.WMA(close, timeperiod = 20)

    # Moving Average Convergence Divergence:
    _, data["MACDS1"], _ = ta.MACD(close, fastperiod=10, slowperiod=20, signalperiod = 5)
    _, data["MACDS2"], _ = ta.MACD(close, fastperiod=5,  slowperiod=10, signalperiod = 2)

    # Relative Strength Index for 5,10,20 days interval:
    data["RSI5"]  = ta.RSI(close, timeperiod = 5)
    data["RSI10"] = ta.RSI(close, timeperiod = 10)
    data["RSI20"] = ta.RSI(close, timeperiod = 20)

    # Average Directional Movement Index for 5,10,20 days interval:
    data["ADX5"]  = ta.ADX(high, low, close, timeperiod = 5)
    data["ADX10"] = ta.ADX(high, low, close, timeperiod = 10)
    data["ADX20"] = ta.ADX(high, low, close, timeperiod = 20)

    # AROON momentum indicator:
    data["AROON_DOWN"], data["AROON_UP"] = ta.AROON(high, low)

    # Momentum:
    data["MOM"]   = ta.MOM(close)

    # Larry William's R%:
    data["WILLR"] = ta.WILLR(high, low, close)

    # Commodity Channel Index:
    data["CCI"]   = ta.CCI(high, low, close)

    # Accumulation/Distribution Oscillator:
    data["ADOSC"] = ta.ADOSC(high, low, close, volume.astype(np.double), fastperiod = 10, slowperiod = 20)
    data["ADOSC"].fillna(0.0,inplace = True)

    # Stochastic K% + D%:
    data["STOCH_K"], data["STOCH_D"] = ta.STOCH(high, low, close, fastk_period = 5, slowk_period = 10,slowd_period = 10, slowd_matype = 0)

    # Volume
    data["Volume"]      = data["Volume"]/ 1000.0
    data["Volume_SMA5"] = ta.SMA(data["Volume"].values, timeperiod = 5)

    # TIME FEATUREs
    data["day"]   = data.index.day
    data["dow"]   = data.index.dayofweek
    data["month"] = data.index.month

    """ Storing Previous Days EOD Returns 1,2,3,4,5,6,7 days ago  """
    # Return FEATURES
    data["Ret0"] = data["Close"].pct_change()
    data["Ret1"] = data["Ret0"].shift(1)
    data["Ret2"] = data["Ret0"].shift(2)
    data["Ret3"] = data["Ret0"].shift(3)
    data["Ret4"] = data["Ret0"].shift(4)
    data["Ret5"] = data["Ret0"].shift(5)
    data["Ret6"] = data["Ret0"].shift(6)
    data["Ret7"] = data["Ret0"].shift(7)
    # TARGET is tomorrow return
    data["RetT"] = data["Ret0"].shift(-1)

    # Calculating HOLIDAYS
    cal      = USFederalHolidayCalendar()
    holidays = cal.holidays(start="1990-01-01", end="2031-03-01").to_pydatetime()

    # Calculating next / previous day slider
    prev_day = data["Date"].shift(1)
    next_day = data["Date"].shift(-1)

    data["after_hol"]  = [pd.to_datetime(d) in holidays for d in list(prev_day)]
    data["before_hol"] = [pd.to_datetime(d) in holidays for d in list(next_day)]

    # Take a look at the data on command line
    print("--> ", data.head(1)) # The start of the data
    print("== ", data.tail(1))  # The most recent / end of the data
    print("++ ", data.head())

    # Clean The Data
    data = data.iloc[-NUM_DAYS:-1] # Most Recent N days of data

    # Apply standard normal scaler to ADOSC
    scaler = StandardScaler()
    data["ADOSC"]   = scaler.fit_transform(data[["ADOSC"]])
    #data["Volume"] = scaler.fit_transform(data[["VOLUME"]])

    """ Feature Enrichment """
    # For Prediction models with interest in relative values rather than abosolute levels of moving averages for example
    # SMA Ratio 5 to 10 days
    data["SMA_5_10"]  = (data["SMA5"] - data["SMA10"])/ data["SMA10"]
    # EMA Ratio 5 to 10 days
    data["EMA_5_10"]  = (data["EMA5"] - data["EMA10"])/ data["EMA10"]
    # DEMA Ratio 5 to 10 days
    data["DEMA_5_10"] = (data["DEMA5"] - data["DEMA10"])/ data["DEMA10"]
    # RSI Ratio 5 to 10 days
    data["RSI_5_10"]  = (data["RSI5"] - data["RSI10"])/ data["RSI10"]
    # ADX Ratio 5 to 10 days
    data["ADX_5_10"]  = (data["ADX5"] - data["ADX10"])/ data["ADX10"]

    # This feature will give an indication if current volume spikes comparing to last 5 days average
    data["VOL_5"]     = (data["Volume"] - data["Volume_SMA5"])/ data["Volume_SMA5"]
    # composite feature of Volume and PRice Return
    data["VOL_RET"]   = data["Ret0"] * data["VOL_5"]

    # saving the features files
    data.to_csv(fn_target, date_format=  "%d/%m/%Y")
    print("saved data.csv")

    return data


# FIX OTHERWISE THEY GET RANDOM AND MODEL PRODUCE RANDOM RESULTS
# FEATURES :
# TECHNICAL INDICATORS provided by TA LIB
INDICATORS = ["SMA5","SMA10","SMA20", 'SMA_5_10',         \
              'EMA5', 'EMA10', 'EMA20', 'EMA_5_10',       \
              "DEMA5", "DEMA10", "DEMA20",'DEMA_5_10',    \
              'ADX5', "ADX10", "ADX20", "ADX_5_10",       \
              'RSI5', 'RSI10', "RSI20", "RSI_5_10",       \
              'WMA5', 'WMA10', 'WMA20',                   \
              'MACDS1', 'MACDS2',                         \
              'STOCH_K', 'STOCH_D',                       \
              'Volume', 'VOL_5', 'VOL_RET', 'Volume_SMA5',\
              "AROON_UP",'AROON_DOWN',                    \
              'WILLR',\
              'ADOSC',\
              'MOM',  \
              'CCI']

# PREVIOUS 1,2,3,4,5,6,7 days ago daily returns
RETS           = ["Ret0", "Ret1", "Ret2", "Ret3", "Ret4", "Ret5", "Ret6", "Ret7"]
# TIME RELATED FEATURES
EXTRA_FEATURES = ["dow", "day", "month", "before_hol", "after_hol"]

ALL_FEATURES   = RETS + INDICATORS + EXTRA_FEATURES

# Ignore first 100 days as indicators are not completely evaluated yet
WARMUP_DAYS    = 100
# Take the LAST NUM_DAYS of price history
NUM_DAYS       = 10000
# Goal is to predict next day price movement
TARGET         = ["RetT"]
# Columns to be removed
REMOVE_COLS    = ["Date", "Date.1"]
date_parser    = lambda x: pd.datetime.strptime(x, "%d/%m/%Y")

"""
Class Data
responsible for 
 *) loading the features data set
 *) Making X, y
 *) Splitting into validation, train, test set
 
 
INPUT: <MARKET>_features.csv file
"""
class Data:
    def __init__(self, market, split=False):
        fn = market + "_features.csv"
        self.data = pd.read_csv(fn,
                                infer_datetime_format=True,
                                index_col="Date",
                                parse_dates=True,
                                date_parser=date_parser)
        # don't need no Nans
        self.data = self.data.iloc[WARMUP_DAYS:-1]
        global ALL_FEATURES
        self.features = ALL_FEATURES
        if self.features is None:
            self.features = list(set(self.data.columns) - set(TARGET + REMOVE_COLS))
        print(self.features)
        print(self.data.columns)

        # Storing features and targets
        self.y = self.data[TARGET]

        # Splitting up the training data (70%) and the test data (30%)
        self.X_train   = None
        self.X_test    = None
        self.y_train   = None
        self.y_test    = None

        self.mkt_train = None
        self.mkt_test  = None

        # Split if split was requested
        if split:
            self.X_train, self.X_test, self.y_train, self.y_test   = train_test_split(self.data,
                                                                                      self.y,
                                                                                      test_size=0.3,
                                                                                      shuffle=False)
            self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(self.X_train,
                                                                                      self.y_train,
                                                                                      test_size=0.3,
                                                                                      shuffle=False)

            self.mkt_train = self.X_train[["Close"]]
            self.mkt_test  = self.X_test[["Close"]]

            self.X_train   = self.X_train[self.features]
            self.X_test    = self.X_test[self.features]
            self.X_valid   = self.X_valid[self.features]

        # When plotting the market we plot the Close prices
        self.mkt = self.data[["Close"]]
        self.X   = self.data[self.features]

        # Apply min max scaler
        self.X[self.X.columns] = MinMaxScaler().fit_transform(self.X[self.X.columns])

        if split:
            self.X_train[self.X.columns] = MinMaxScaler().fit_transform(self.X_train[self.X.columns])
            self.X_valid[self.X.columns] = MinMaxScaler().fit_transform(self.X_valid[self.X.columns])
            self.X_test[self.X.columns]  = MinMaxScaler().fit_transform(self.X_test[self.X.columns])

    """ Useful stats about the data """
    def __str__(self):
        return "DATA: X train: {0} y_ train {1} x_test {2} y test {3} ".format(self.X_train.shape,
                                                                               self.y_train.shape,
                                                                               self.X_test.shape,
                                                                               self.y_test.shape)

# if you run this script SPY and EM feature files will be generated
if __name__ == "__main__":
    data = prepare_features(MARKET_SPY)
    data = prepare_features(MARKET_EM)