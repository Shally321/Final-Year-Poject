"""Import the libraries"""
import matplotlib.pyplot as plt
import matplotlib.dates

MODEL_COL = {
            # Optimal (god):
             "GOD": "black",

            # Single classifiers:
             "DT": "brown",
             "LR": "yellow",
             "DNN": "red",
             "KNN": "pink",
             "SVR": "orange",

            # Ensemble methods:
             "Voting": "gray",
             "RF": "green",
             "AVG": "blue",
             "ADA": "violet",

            # Market:
             "SP500": "purple",
             "SPY": "purple",
             "EEM": "purple"
}

def plot_signals(data, mdl_name, market):
   plot_signals2(data.X_test, data.mkt_test, mdl_name, market)

def plot_signals2(X_test, mkt_test, mdl_name, market):
    """Visualise the data"""
    fig, ax = plt.subplots(figsize=(8,8))

    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%Y"))
    ax.set(xlabel="Date", ylabel="Price ($)", title="Strategy (" + mdl_name + ") Trading Signals")
    ax.xaxis_date()
    ax.autoscale_view()

    #print(X_test.index)
    buys = X_test.ix[X_test["pos"] > 0]
    sells = X_test.ix[X_test["pos"] < 0]
    mkt = X_test["RetT"].cumsum()

    if market == "EEM":
        market = "MSCI EM"
    elif market == "SPY":
        market = "S&P500"

    line2 = ax.plot(mkt_test.index, mkt_test, label=market)
    b = ax.plot(buys.index, mkt_test.ix[buys.index]["Close"],'^', color='g', label="Buy")
    s = ax.plot(sells.index, mkt_test.ix[sells.index]["Close"], 'v', color='r', label="Sell")

    plt.setp(plt.gca().get_xticklabels(), rotation = 45, horizontalalignment='right')
    plt.legend()
    #plt.show()
    #print(X_test)

def plot_pnl(data, mdl_name, market):
    """Visualise the data"""
    fig, ax = plt.subplots(figsize=(8,8))

    X_test = data.X_test

    if market == "EEM":
        market = "MSCI EM"
    elif market == "SPY":
        market = "S&P500"

    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%Y"))
    ax.set(xlabel = "Date", ylabel ="Profit and Loss", title = mdl_name + " vs " + market + "  Equity Performance")
    ax.xaxis_date()
    ax.autoscale_view()


    mkt_ret = X_test["RetT"].cumsum()

    line1 = ax.plot(X_test.index, X_test["pnl"].cumsum(), label = mdl_name + " Strategy", color = MODEL_COL[mdl_name])
    line2 = ax.plot(X_test.index, mkt_ret ,label="Buy & Hold " + market)

    plt.setp(plt.gca().get_xticklabels(), rotation = 45, horizontalalignment = 'right')
    plt.legend()
    #plt.show()
    #print(X_test)

def plot_pnls(returns, market):
    """Visualise the data"""
    fig, ax = plt.subplots(figsize=(8,8))

    if market == "EEM":
        market = "MSCI EM"
    elif market == "SPY":
        market = "S&P500"

    ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%Y"))
    ax.set(xlabel = "Date", ylabel ="Profit and Loss (%)", title = "Models vs Buy & Hold " + market + " Equity Performance")
    ax.xaxis_date()
    ax.autoscale_view()

    #print(X_test.index)

    #avg =  None
    #for label, rets  in returns.items():
    #    if avg is None:
    #        avg = rets
    #    else:
    #        avg += rets
    #returns["AVG"] = avg/ len(returns)


    for label, rets  in returns.items():
        rets = 100 * rets
        ax.plot(rets.index, rets.cumsum(),label=label, color=MODEL_COL[label])


    plt.setp(plt.gca().get_xticklabels(), rotation = 45, horizontalalignment = 'right')
    plt.legend()
    plt.show()
    #print(X_test)