"""Import the libraries"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.tree.export import export_graphviz
from sklearn.metrics import mean_squared_error, r2_score, SCORERS, mean_absolute_error, explained_variance_score
from sklearn.neural_network import MLPRegressor

from graphviz import Source
import statsmodels.api as sm

VERBOSE = False

"""
In this file i keep the model creation, running, optimizing and evaluating methods 
"""

""" 
    Running function for Linear Regression model 
    See: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
"""
def run_lr(data):
    # y = b + c0 * x0 + c1 * x1 ...
    lr = LinearRegression(fit_intercept=True,
                          normalize=False)
    lr.fit(data.X_train, data.y_train)
    y_predict = lr.predict(data.X_test)
    return [lr, y_predict]

"""
    Makes a RandomForest Regressor model
    See: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
"""
def make_rf():
    rf = RandomForestRegressor(random_state=666,
                               n_estimators=100,
                               max_depth=5,
                               min_samples_split=0.4)
    return rf

"""
    Running function for Random Forest model
"""
def run_rf(data):

    rf = make_rf()
    rf.fit(data.X_train, data.y_train)

    y_predict = rf.predict(data.X_test)
    return [rf, y_predict]

"""
    Makes a AdaBoost Regressor model
    See: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
"""
def make_ada():
    # default best estimator is DT, we will use non default one with optimized parameters
    dt = DecisionTreeRegressor(min_samples_split=10,
                               min_samples_leaf=5,
                               max_depth=2)
    # Ada will build 200 decision trees
    ada = AdaBoostRegressor(base_estimator=dt,
                            random_state=666,
                            n_estimators=200,
                            loss="linear",
                            learning_rate=0.0001)
    return ada

"""
    Running function for ADABoosting
"""
def run_ada(data):

    ada = make_ada()
    if VERBOSE:
        print("fitting ..")
    ada.fit(data.X_train, data.y_train)

    y_predict = ada.predict(data.X_test)
    return [ada, y_predict]

"""
    makes Decision Tree Regressor
    See: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
"""
def make_dt():
    dt = DecisionTreeRegressor(random_state=0,
                               min_samples_split=10,
                               min_samples_leaf=5,
                               max_depth=4)
    return dt

"""
    Running function for Decision Tree model
"""
def run_dt(data):

    dt = make_dt()
    dt.fit(data.X_train, data.y_train)

    y_predict = dt.predict(data.X_test)
    return dt, y_predict

"""
    Create and train the Support Vector Machine Regressor
    See: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html
"""
def run_svr(data):
    svr_rbf = LinearSVR(random_state=666,
                        max_iter=50,     # num of iteration
                        tol=0.001,       # stopping tolerance
                        C=20)            # regularization parameter

    svr_rbf.fit(data.X_train, data.y_train)

    # Score returns te coefficient of determination of R^2 of the prediction
    # svm_confidence = svr_rbf.score(X_test, y_test)
    # print("SVM confidence: ", svm_confidence) # Best possible score is 1.0

    y_predict = svr_rbf.predict(data.X_test)
    return [svr_rbf, y_predict]

"""
    Running function for K-Nearest Neighbour model
    See: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
"""
def run_knn(data):
    knn = KNeighborsRegressor(n_neighbors=30,         # number of neighbours to consider
                              algorithm="ball_tree")
    knn.fit(data.X_train, data.y_train)

    y_predict = knn.predict(data.X_test)
    return knn, y_predict

"""
    Running function for Deep/Artificial Feed Forward Neural Network model(Regressor)
    See https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
"""
def run_nn(data):
    np.random.seed(1)
    nn = MLPRegressor(hidden_layer_sizes=(20, 40, 100, 40, 20), # 5 hidden  layers
                      learning_rate_init=0.00005,
                      learning_rate="adaptive", # keeps the learning rate constant to learning_Rate_init as losing training loss keeps decreasing
                      activation="relu",        # relu is max(0,x)
                      max_iter=3000,
                      n_iter_no_change=70,      # early stopping in no change
                      early_stopping=True,
                      verbose=VERBOSE,
                      random_state=1,
                      shuffle = False,          # don't reshuffle the data
                      batch_size = 30,
                      tol=1e-6)                 # Tolerance for optimization

    nn.fit(data.X_train.values, data.y_train.values)

    #if VERBOSE is on plot the loss curve
    if VERBOSE:
        print(nn.loss_curve_)
        plt.plot(list(range(250,nn.n_iter_)), nn.loss_curve_[250:])
        plt.show()

    y_predict = nn.predict(data.X_test)
    return nn, y_predict

"""
    Calculate and plot feature importance using either DT or RF estimators
"""
def feature_importance(data, mdl_name):
    mdl_name = mdl_name.upper()
    features = np.array((data.X_train.columns))

    model = get_model(mdl_name)
    model.fit(data.X_train, data.y_train)

    # Top 15 features are displayed
    N = 15
    feat_importances      = model.feature_importances_ * 100

    indices               = np.argsort(feat_importances)
    best_feat_importances = feat_importances[indices][-N:]
    best_features         = features[indices][-N:]

    # Plotting horizontal bars for each feature / importance
    plt.figure()
    plt.title("Feature importance (" + mdl_name + ")")
    plt.barh(best_features, best_feat_importances, align="center")
    plt.yticks(best_features)
    plt.xlabel("Feature Importance (all features add up to 100%)")
    plt.show()

    # Plot_feature_importance(feat_importances, features)
    f = zip(features, feat_importances)
    f = sorted(f, key=lambda x: x[1], reverse=True)
    print("feat importance", str(f))

    # For DT we will also visualize the actual tree
    if mdl_name == "DT":
        # out = StringIO()
        export_graphviz(model, out_file="tree.dot", feature_names=features, max_depth=6)
        s = Source.from_file("tree.dot")
        s.view()
        y_predict = model.predict(data.X_test)

"""
    Function to visualise the trade performance of each feature
"""
def plot_features_trade_performace(model, X_train_all, y_train, X_valid_all, y_valid, all_features):
    errors        = []
    total_pnls    = []
    num_features  = list(range(1,len(all_features)))

    for i in num_features:
        features  = all_features[:i]
        X_train   = X_train_all[features]
        X_valid   = X_valid_all[features]

        model.fit(X_train, y_train)
        y_predict = model.predict(X_valid)

        er        = mean_squared_error(y_valid, y_predict)
        errors.append(er)

        pnl       = trade(y_predict, X_valid, y_valid)
        total_pnls.append(pnl)

    plt.plot(num_features, total_pnls)
    plt.show()

"""
    Given a model name will call an appropriate builder function to construct sklearn estimator
    Raises an exception if model is not supported
"""
def get_model(mdl_name):
    mdl_name = mdl_name.upper()

    if mdl_name == "RF":
        return make_rf()
    elif mdl_name == "DT":
        return make_dt()
    elif mdl_name == "ADA":
        return make_ada()
    else:
        raise("Model is not supported")

""" 
    Run all models function
    INPUT: metrics to add to
    RETURNS: [estimator. y_predict, metrics]
"""
def run_model(data, mdl_name, metrics):
    if VERBOSE:
        print("RUNNING ", mdl_name)
        print(data.X_train.columns)
        print (data.X_train.shape)
        print(data.y_train.shape)
        print("X_Train === ")
        print(data.X_train.head(2))
        print("y_train=======")
        print(data.y_train.head(3))
        print(data.X_test.head(2))

    if mdl_name == "RF":
        [mdl, y_predict] = run_rf(data)
    elif mdl_name == "DT":
        [mdl, y_predict] = run_dt(data)
    elif mdl_name == "ADA":
        [mdl, y_predict] = run_ada(data)
    elif mdl_name == "LR":
        [mdl, y_predict] = run_lr(data)
    elif mdl_name == "SVR":
        [mdl, y_predict] = run_svr(data)
    elif mdl_name == "KNN":
        [mdl, y_predict] = run_knn(data)
    elif mdl_name == "DNN":
        [mdl, y_predict] = run_nn(data)
    else:
        raise("Model is not supported")

    if VERBOSE:
        print("Y_predict ===")
        print(y_predict[:3])

    # Append a new row to the  data-frame with all the result metrics
    metrics = calc_metrics(mdl_name, data.y_test, y_predict, metrics)

    return mdl, y_predict, metrics

"""
    Performs Cross-Validation for a given model
"""
def cross_val(data, mdl_name):

    [mdl, y_predict, metrics] = run_model(data, mdl_name, None)
    print("Running cross val ..")
    print(SCORERS.keys())
    print("Scoring = NMSE")
    scores = cross_val_score(mdl, data.X, data.y, cv=5, scoring="neg_mean_squared_error")
    print("AVG : ", -np.mean(scores))

"""
  Get Hyper-Parameters
  RETURNS: set of hyper-parameters choices to optimize for each supported model
"""
def get_optim_params(mdl_name):
    if mdl_name == "DNN":
        return {'hidden_layer_sizes': [10, 20, 40, 50, 100],
                'max_iter':           [100, 500, 1000, 3000],
                "learning_rate_init": [0.00001, 0.00005, 0.0001],
                'n_iter_no_change':   [1, 10, 50, 70, 100],
                'activation':         ['identity', 'logistic', 'tanh', 'relu'],
                'learning_rate':      ['constant', 'invscaling', 'adaptive'],
                'tol':                [0.0001, 0.00001, 0.000001],
                'batch_size':         [10, 20, 30, 50, 100]}

    if mdl_name == "SVR":
        return {'max_iter':           [20, 50, 100],
                'tol':                [0.0001, 0.001, 0.01, 0.1],
                'C':                  [1, 10, 20, 50]}

    if mdl_name == "RF":
        return  {'n_estimators':      [10,100,200],
                 'max_depth':         [2, 10, 50],
                 'min_samples_split': [0.2, 0.4, 0.8, 1.0]}

    if mdl_name == "DT":
        return {'min_samples_split':  [2, 5, 10],
                'min_samples_leaf':   [2, 5, 10],
                'max_depth':          [2, 3, 4]}

    if mdl_name == "ADA":
        return {
            "loss":                   ["linear", "square", "exponential"],
            "n_estimators":           [100, 200],
            "learning_rate":          [0.0001, 0.01, 0.1]
               }

    if mdl_name == "KNN":
        return {'n_neighbors':        [25, 30, 35],
                'algorithm':          ["ball_tree", "kd_tree", "brute"]}

    if mdl_name == "LR":
        return {'fit_intercept':      [True, False],
                'normalize':          [True, False]}

    raise(mdl_name, " model is not supported !")

"""
    Optimize model with Hyper-Parameters
    Use GRIDSearch 
    See: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
    The best estimator parameters are then used for testing
"""
def optimize(data, mdl_name):

    [mdl, _, _] = run_model(data, mdl_name, None)
    pars = get_optim_params(mdl_name)

    print("Running grid search,,")
    grid = GridSearchCV(mdl, pars)
    grid.fit(data.X, data.y)

    print(grid.cv_results_)
    print(grid.best_estimator_)

"""
    Analyse statistics of Linear Regression Model:
"""
def analyse_lr(X_train, y_train):
    X2   = sm.add_constant(X_train)
    est  = sm.OLS(y_train.values, X_train.values)
    est2 = est.fit()
    print(est2.summary())

"""
    Accuracy Metric calculation function
    Calculates MSE, R2 , Explained Var and Mean Absolute Error
    See https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
"""
def calc_metrics(model_name, y_test, y_predict, stats, verbose = False):
    if stats is None:
        stats = pd.DataFrame(columns=["Model", "StartDate", "EndDate", "MSE", "R2", "Explained Var", "Mean Abs Error"])

    # Explained VAR
    ev  = explained_variance_score(y_test.values, y_predict)
    # Mean Squared Error
    mse = mean_squared_error(y_test.values, y_predict)
    # R2 score
    r2  = r2_score(y_test.values, y_predict)
    # Mean Absolute Error
    mae = mean_absolute_error(y_test.values, y_predict)

    if verbose:
        print("Explained Var:", ev)
        print("MSE: ", mse)
        print("R2 score: ", r2)

    # Append a new row to the metrics df
    stats = stats.append({
        "Model": model_name,
        "StartDate": y_test.index.date[0],
        "EndDate": y_test.index.date[-1],
        "MSE": mse,
        "R2": r2,
        "Explained Var": ev,
        "Mean Abs Error": mae
          }, ignore_index=True)

    # Returns the resulting metrics
    return stats