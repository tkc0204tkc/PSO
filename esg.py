import numpy as np
import yfinance as yf


def get_esg_scores(ticker):
    try:
        # Validate the ticker symbol
        stock = yf.Ticker(ticker)
        # Attempt to get ESG scores
        esg_data = stock.sustainability.loc['totalEsg']
        if esg_data is not None and not esg_data.empty:
            return esg_data.iloc[0]
        else:
            return 0  # No ESG data found
    except Exception as e:
        return 0

stocks = ['0005.HK','0388.HK', '0939.HK', '1299.HK', '1398.HK', '2318.HK', '2388.HK', '2628.HK', '3968.HK', '3988.HK']
esg=np.array(list(map(get_esg_scores,stocks)))

print(esg)