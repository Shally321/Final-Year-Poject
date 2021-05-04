# Description:
    # Development of machine learning regressive models,
    # using technical indicators and other features as initial attributes,
    # to learn, predict and execute trades before and during periods of crisis (Recession & Covid),
    # and to compare the models perfomrance against an established market index(S&P500) and an emerging market (MCSI EM)

"""Import the libraries"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import datetime
from datetime import timedelta

# My project packages
from models import optimize, run_model, feature_importance, cross_val
from prepare_data import Data, MARKET_SPY, MARKET_EM
from plots import plot_pnl, plot_signals, plot_signals2, plot_pnls

"""
    TRADE SIMULATION
    ENTER: 
       Long trades:   if next day price move prediction is positive and at least 0.08%
       Short trades:  if next day price move prediction is negative and at least -0.08%
    EXIT: at the end of each day
    RISK MANAGEMENT: E.g.STOP LOSS: TBD in future 
    
    RETURNS: add "pos" and "pnl" columns to the data.X_test df
"""
def trade(y_predict, data):

    data.X_test["pnl"]  = 0
    data.X_test["RetT"] = data.y_test["RetT"]
    data.X_test["RetP"] = y_predict

    positions = []

    # Threshold check to avoid poor trading
    for pred in y_predict:
        pos = 0
        if pred > 0.0008:
            pos = 1
        elif pred < -0.0008:
            pos = -1
        positions.append(pos)

    data.X_test["pos"] = positions                                 # np.sign(y_predict)
    data.X_test["pnl"] = data.X_test["pos"] * data.X_test["RetT"]

    return data.X_test["pnl"].sum()

"""
    Calculates trade stats on trading results: 
    INPUT: X_test 
    CALCULATES:
        1) Num trades/ long/ short
        2) Avg TRade Pnl
        3) Trade PNl STD
        
        TBD for future work:
        4) Drwadown 
        5) VAR 
"""
def trade_stats(model_name, X_test, df = None,verbose = False):

    if df is None:
        df = pd.DataFrame(columns=["ModelName", "Buys", "Sells", "NumTrades",
                                   "AvgTradePnl", "PnlStd", "TotalPnl"])

    """Calculations for Trading Statistics"""
    pos_trades       = (X_test["pnl"] > 0).sum()
    neg_trades       = (X_test["pnl"] < 0).sum()
    trade_pnls       = X_test[X_test["pos"] != 0]["pnl"]

    pos_long_trades  = ((X_test["pnl"] > 0) & (X_test["pos"] > 0)).sum()
    neg_long_trades  = ((X_test["pnl"] < 0) & (X_test["pos"] > 0)).sum()

    pos_short_trades = ((X_test["pnl"] > 0) & (X_test["pos"] < 0)).sum()
    neg_short_trades = ((X_test["pnl"] < 0) & (X_test["pos"] < 0)).sum()

    long_trades      = (X_test["pos"] > 0).sum()
    short_trades     = (X_test["pos"] < 0).sum()

    up_days          = (X_test["RetT"] > 0).sum()
    down_days        = (X_test["RetT"] < 0).sum()
    num_trades       = long_trades + short_trades
    pnl_std          = np.std(trade_pnls)

    """Display Trading Statistics"""
    if verbose:
        print("POS TRADES : ", pos_trades)
        print("NEG TRADES : ", neg_trades)

        print("POS LONG TRADES : ", pos_long_trades)
        print("NEG LONG TRADES : ", neg_long_trades)

        print("POS SHORT TRADES : ", pos_short_trades)
        print("NEG SHORT TRADES : ", neg_short_trades)

        print("LONG TRADES : ", long_trades)
        print("SHORT TRADES : ", short_trades)

        print("UP DAYS : ", up_days)
        print("DOWN DAYS : ", down_days)

    pnl = X_test["pnl"].cumsum()[-1]

    # Append a new row (model/period trade results)
    df = df.append(
        {"ModelName":   model_name,
         "Buys":        long_trades,
         "Sells":       short_trades,
         "NumTrades":   num_trades,
         "AvgTradePnl": pnl / num_trades,
         "PnlStd":      pnl_std,
         "TotalPnl":    pnl},
        ignore_index=True)
    return df

"""
    Add underlying benchmark mArket specific stats
"""
def add_market_stats(X_test, stats, market):
    pnl   = X_test["RetT"].cumsum()[-1]
    stats = stats.append(
        {
            "ModelName":   market,
            "Buys":        0,
            "Sells":       0,
            "NumTrades":   0,
            "AvgTradePnl": 0,
            "PnlStd":      0,
            "TotalPnl":    pnl
        },
        ignore_index=True)
    return stats

"""
    Run one model/ trading strategy only
    Plot pnl
    Print model metrics
    Print trade stats
"""
def run_one_model(data, model = "NN"):
    """Execution of the Models"""
    [mdl, y_predict] = run_model(data, model)

    """Execution of the Accuracy Metrics"""
    metrics(data.y_test, y_predict)

    trade(y_predict, data.X_test, data.y_test)
    trade_stats(data.X_test)
    plot_pnl(data.X_test, model)
    plt.show()

    """Execution of the Accuracy Metrics"""
    metrics(data.y_test, y_predict)
    print(data.X_test.head())

"""
    Run multiple models
    Adds Average and Voting model results
    Adds GOD results (god knows what market will do tomorrow!)
"""
def run_mult_models(data, models = ["KNN", "RF"]):
    mdls = []
    X_test_orig       = data.X_test.copy(deep=True)
    y_predict_average = None

    returns = {}
    avg_ret = None
    for model_name in models:
        data.X_test = X_test_orig.copy(deep=True)

        [mdl, y_predict] = run_model(data, model_name)
        mdls.append(mdl)

        trade(y_predict, data.X_test, data.y_test)

        trade_stats(data.X_test)

        returns[model_name] = data.X_test["pnl"]

        # AVERAGE PREDICTION
        if y_predict_average is None:
            y_predict_average = y_predict
        else:
            y_predict_average += y_predict.reshape((y_predict.shape[0],1))

        # AVERAGE PNL
        if avg_ret is None:
            avg_ret = data.X_test["pnl"].copy()
        else:
            avg_ret += data.X_test["pnl"]

    # I implemeted very basic version of Voting Regressor
    if True:
        data.X_test = X_test_orig.copy(deep=True)
        y_predict_average /= len(models)

        trade(y_predict_average, data.X_test, data.y_test)

        returns["Voting"] = data.X_test["pnl"]

    # I could not use VotingRegressor() as it is not year supported in my version of sklearn package
    if False:
        vr = VotingRegressor(estimators=zip(models, mdls))
        vr.fit(data.X_train, data.y_train)
        data.X_test = X_test_orig.copy(deep=True)
        y_repdict = vr.predict(data.X_test)
        trade(y_predict, data.X_tet, data.y_tet)
        returns["Voting"] = data.X_test["pnl"]

    # Adding a GOD like model performance
    if True:
        godPnl = np.abs(data.X_test["RetT"])
        returns["GOD"] = godPnl

    # Keeping the underlying market itself
    returns["SP500"] = data.X_test["RetT"]
    returns["AVG"]   = avg_ret / len(models)

    # Plot all the models on a single plot
    plot_pnls(returns)

"""
    Extra analysis
"""
def analyze(data):
    analyse_lr(data.X_train, data.y_train)


"""
    find nearest date (id) in the list of dates
"""
def get_nearest(dates, date):
    delta = [abs((x-date).days) for x in dates]
    res   = np.argmin(delta)
    return res

"""
    Runs a given model over given schedule
    RETURNS merged_data and combined metrics
"""
def test_one_model(data, model_name, trade_start_date, schedule , metrics):

    total_trade_start_id = get_nearest(data.X.index.date, trade_start_date)

    data.X["pnl"]  = 0
    data.X["RetT"] = 0
    data.X["pos"]  = 0
    data.X["RetP"] = 0

    merged_data = None

    # For each trading period:
    #  1) train (recalibrate the model)
    #  2) trade
    for start_train_date, end_train_date, start_trade_date, end_trade_date in schedule:
        print(start_train_date)
        nd = np.datetime64(start_train_date)

        start_train_id = get_nearest(data.X.index.date, start_train_date)
        end_train_id   = get_nearest(data.X.index.date, end_train_date)

        start_trade_id = get_nearest(data.X.index.date, start_trade_date)
        end_trade_id   = get_nearest(data.X.index.date, end_trade_date)

        print("INDICE: ", start_train_id, end_train_id, start_trade_id, end_trade_id)

        data1          = Data(MARKET_SPY)
        data1.X_train  = data.X.iloc[start_train_id:end_train_id]
        data1.y_train  = data.y.iloc[start_train_id:end_train_id]

        data1.X_test   = data.X.iloc[start_trade_id:end_trade_id]
        data1.y_test   = data.y.iloc[start_trade_id:end_trade_id]

        # TRAIN ONE PERIOD
        [mdl, y_predict, metrics] = run_model(data1, model_name, metrics)

        # TRADE ONE PERIOD
        trade(y_predict, data1)

        # Merge the results
        if merged_data is None:
            merged_data = data1.X_test.copy(deep= True)
        else:
            merged_data = pd.concat([merged_data,data1.X_test], axis =0 )


    #whole_period_trading    = data.X.iloc[total_trade_start_id:end_trade_id]
    merged_data[model_name] = merged_data["pnl"]

    #plot_pnls(merged_data,model_name)

    return merged_data, metrics

"""
    Tests multiple models on the given trading schedule
    Calculates trade statistics
    Plot each model trading signals
    Plot equity lines
"""
def trade_multiple_models(name, models, market, trade_start_date, schedule):
    stats   = None
    metrics = None
    returns = {}

    start_trade_id = get_nearest(data.X.index.date, trade_start_date)
    end_trade_id   = get_nearest(data.X.index.date, schedule[-1][3])
    mkt_test       = data.mkt.iloc[start_trade_id:end_trade_id]

    for model in models:
        results, metrics = test_one_model(data, model, trade_start_date, schedule, metrics)
        returns[model]   = results["pnl"]
        returns[market]  = results["RetT"]
        #if returns["AVG"] is None:
        #    returns["AVG"] = results["pnl"]
        #else:
        #    returns["AVG"] += results["pnl"]

        # plt.show()
        stats = trade_stats(model, results, stats)
        # plot_signals2(results, mkt_test, model, market)


    # Get average return
    #returns["AVG"] = returns["AVG"] / len(models)

    stats = add_market_stats(results, stats, market)
    print(name)
    print(stats)
    print(metrics)
    plot_pnls(returns, market)

"""
    Prepares training/trading schedule 
    INPUT:
        1) trading start date
        2) number of recalibration to perform
        3) train_days - training look-back window
        4) trade_days - numbers of days to trade before the next recalibrations 
    OUTPUT: schedule which is list of tuples 
       (start_train, end_train, start_trade, end_trade)
"""
def make_schedule(trade_start_date, num_periods, train_days, trade_days ):

    trade_end = trade_start_date + timedelta(days=-1)
    schedule = []
    for i in range(num_periods):
        trade_start = trade_end + timedelta(days=1)
        trade_end   = trade_start + timedelta(days=trade_days)

        tune_end    = trade_start + timedelta(days=- 1)
        tune_start  = tune_end + timedelta(days=-train_days)

        schedule.append((tune_start, tune_end, trade_start, trade_end))

    return schedule

"""
   Test a a crisis defined by:
       1) start date
       2) days to test before
       3) days to test during
   For example:
        GreatFinancialCrash (GFC) 08/07 - 06/09
        12/06 - 12/07 train , trade for 1 month
        1/07 - 1/08  train , trade for 1 month
        repeat 12 times
"""
### DAYS TO TEST SHOULD BE IN LOTS OF 30
def test_crisis(name, market, crisis_start_date, days_to_test_before = 30, days_to_test_during = 30):

    trade_start_date1 = crisis_start_date - timedelta(days=days_to_test_before)
    # 6M BEFORE GFC
    schedule1 = make_schedule(trade_start_date1, days_to_test_before//30, 365, 30)

    trade_start_date2 = crisis_start_date
    # GFC goes on for 6M
    schedule2 = make_schedule(trade_start_date2, days_to_test_during//30, 365, 30)

    #Separetly test performance before the crisis
    trade_multiple_models("BEFORE " + name, models, market, trade_start_date1, schedule1)
    #Afterwards test perormance during the crisis
    trade_multiple_models("DURING " + name, models, market, trade_start_date2, schedule2)


# SELECT THE TASK YOU WANT TO PERFORM:
    # 1 for cross validation
    # 2 for feature importance
    # 3 for trade signals
    # 4 for crisis analysis: Trading signals + PnL
CONTROL = 4

if __name__ == '__main__':

    """Select market: """
    # market = MARKET_SPY
    market = MARKET_EM

    np.random.seed(666)

    # Testing one particular model
    mdl_name = "KNN"

    data = Data(market, split=True)

    """Cross Validation"""
    if CONTROL == 1:
        cross_val(data, "DT")
        exit(0)

    """Feature Importance"""
    if CONTROL == 2:
        feature_importance(data, "ada")
        exit(0)

    """Trading Signals"""
    if CONTROL == 3:
        #optimize(data, "ADA")
        mdl, y_predict, metrics = run_model(data, mdl_name, None)
        trade(y_predict, data)
        stats = trade_stats(mdl_name, data.X_test, None)
        print(stats)
        plot_signals(data, mdl_name, market)
        plot_pnl(data, mdl_name, market)
        plt.show()
        exit(0)

    """Execution of the Models"""
    if CONTROL == 4:
        models = ["RF", "DT", "ADA", "SVR", "DNN", "LR", "KNN"]
        # models = ["KNN", "LR"]

        # Select number of months to trade on
        months_to_test = 3

        # Select crisis period to test on (Great Financial Crash / Covid)
        test_crisis("GFC",
                    market,
                    datetime.date(2007, 11, 5),
                    days_to_test_before=months_to_test * 30,
                    days_to_test_during=months_to_test * 30)
        test_crisis("COVID",
                    market,
                    datetime.date(2020, 2, 1),
                    days_to_test_before=months_to_test * 30,
                    days_to_test_during=months_to_test * 30)





