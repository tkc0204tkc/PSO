import pandas as pd
import statistics as stat
import glob
import os
import numpy as np

cwd = os.getcwd()

datafiles = glob.glob(cwd + "/data/*.xlsx")
no_of_stocks = len(datafiles)
stocks_returns = [[0] for y in range(no_of_stocks)]
esg = np.array([26, 25, 15.6, 14.3, 17.8, 33.1, 29.1, 17.9, 22.4, 24.5])

print(datafiles)
file_counter = 0
for datafile in datafiles:
    df = pd.read_excel(datafile)
    for closed_price_counter in range(1, df.shape[0]):
        daily_return = (df.iloc[closed_price_counter].iloc[1] - df.iloc[closed_price_counter-1].iloc[1]) / df.iloc[closed_price_counter - 1].iloc[1]
        stocks_returns[file_counter].append(daily_return)
    file_counter += 1

stocks_returns_df = pd.DataFrame(np.transpose(stocks_returns))

stocks_returns_mean = np.array([stat.mean(stocks_returns[i]) for i in range(no_of_stocks)])

covariance_matrix = np.cov(np.array(stocks_returns))

# print(covariance_matrix)

# inverse_covariance_matrix = np.linalg.inv(covariance_matrix)




def objective():
    V = no_of_stocks

    def f_max_return(x):
        return -(stocks_returns_mean.T.dot(x))

    def f_min_risk(x):
        return (np.array(x).T).dot(covariance_matrix).dot(np.array(x))

    def f_max_esg(x):
        return (esg.T).dot(x)

    def g_budget(x):
        ones = np.ones(len(x))
        g = ones.T.dot(x)
        return g if 0.5 <= g <= 1 else 0


    l_bound = np.zeros(no_of_stocks)
    u_bound = np.array([0.5 for i in range(no_of_stocks)])
    f = [f_min_risk, f_max_return, f_max_esg]
    c = [g_budget]


    return V, l_bound, u_bound, f, c

