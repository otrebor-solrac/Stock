import yfinance as yf
from finta import TA
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import yahoo_fin.stock_info as si
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN

from collections import OrderedDict
import pandas as pd
from functools import reduce
import random

#import pudb

import warnings
warnings.filterwarnings("ignore")

def impute_values(df):
    # Fill NA's with previous 
    #df = df.fillna(method='ffill')
    
    #same_values_dates = df[(df['Low'] == df['High'])].Date.tolist()
    #df.loc[df.Date.isin(same_values_dates),'Close'] = df.Close + 0.01*random.random()
    #df.loc[df.Date.isin(same_values_dates),'High'] = df.Close + 0.01*random.random()
    #df.loc[df.Date.isin(same_values_dates),'Open'] = df.Close + 0.01*random.random()
    #df.loc[df.Date.isin(same_values_dates),'Low'] = df.Close + 0.01*random.random()
    
    #l = df.shape[0]
    #df['Close'] = df.Close + [0.001*random.random() for i in range(l)]
    #df['High'] = df.Close + [0.001*random.random() for i in range(l)]
    #df['Open'] = df.Close + [0.001*random.random() for i in range(l)]
    #df['Low'] = df.Close + [0.001*random.random() for i in range(l)]
    
    df['Close'] = df.Close.interpolate()
    df['High'] = df.High.interpolate()
    df['Open'] = df.Open.interpolate()
    df['Low'] = df.Low.interpolate()
    

    return df


def compute_indicators(df,indicators):
    """
    Function to compute indicators. You could add more indicators.
    """

    indicators_names = []

    if indicators.get("KAMA",[]):
        kama = indicators.get("KAMA")

        for k in kama:
            kama_values = TA.KAMA(df,k).tolist()

            col = "kama_{}".format(str(k))
            df[col] = kama_values

            indicators_names.append(col)

    return df,indicators_names


def compute_floors_ceilings(data,g_value,l_value):
    # Compute floors and ceilings
    g_value = "kama_{}".format(g_value)
    l_value = "kama_{}".format(l_value)
  
    data['floor_ceil'] = [1 if row[g_value] <= row[l_value] else -1 
                                for ind, row in data[[g_value,l_value]].iterrows()]

    floor_ceil = data.floor_ceil.tolist()

    val_min = [np.inf,""]
    val_max = [-np.inf,""]
    all_values = []
  
    for ind,_ in enumerate(floor_ceil):
        if ind==0:
            continue

        if floor_ceil[ind]==floor_ceil[ind-1]:
            if floor_ceil[ind] == 1:
                if val_max[0] < data.loc[ind,"High"]:
                    val_max[0] = data.loc[ind,"High"]
                    val_max[1] = data.loc[ind,"Date"]
            else:
                if val_min[0] > data.loc[ind,"Low"]:
                    val_min[0] = data.loc[ind,"Low"]
                    val_min[1] = data.loc[ind,"Date"]
        else:
            all_values.append(val_max)
            all_values.append(val_min)

            val_min = [np.inf,""]
            val_max = [-np.inf,""]
      
    floor_ceil = pd.DataFrame(all_values)

    if floor_ceil.empty:
        return floor_ceil
    else:
        floor_ceil.columns = ['value','dates']
        floor_ceil = floor_ceil[~floor_ceil.dates.isna()]
        
    ini_lines = floor_ceil.shape[0]
    floor_ceil['value_st'] = (floor_ceil.value - floor_ceil.value.min())/\
                                (floor_ceil.value.max() - floor_ceil.value.min()) 
    floor_ceil['value_st_2'] = 1#floor_ceil.value_st
    
    # Select eps value
    dist_eu = euclidean_distances(floor_ceil[['value_st','value_st_2']])
    vals = [np.round(e,3) for e in list(dist_eu.flatten()) if e > 0]
    e = np.percentile(vals,5)

    db = DBSCAN(eps=e, min_samples=2).fit(floor_ceil[['value_st','value_st_2']])
    floor_ceil['classification'] = db.labels_

    df_class = floor_ceil[floor_ceil.classification!=-1]\
                        .groupby('classification')\
                        .agg({'value':'mean',
                              'dates':'first'}).reset_index()
    
    df_without_class = floor_ceil.loc[floor_ceil.classification==-1,
                                      ['classification','value','dates']]
    
    floor_ceil = pd.concat([df_without_class,
                            df_class])
    
    print("ratio lines: ", 100*floor_ceil.shape[0]/ini_lines)
          
    return floor_ceil

def compute_var(df):
    df = df.sort_values('Date')
    df['profit'] = df.Close.diff().divide(df.Close.shift(1))
    
    var_per_1 = np.percentile(df.profit.dropna().round(4).array,1)
    var_per_5 = np.percentile(df.profit.dropna().round(4).array,5)

    close_values = df.Close.tolist()

    func = lambda x: df.Close.diff(x).tolist()[-1]
    
    prof_5_days = func(4)/close_values[-5]
    prof_30_days = func(29)/close_values[-30]
    prof_260_days = func(259)/close_values[-260]

    df['week'],df['weekday'],df['year'] = zip(*df.Date.map(lambda x: [x.week,x.weekday(),x.year]))
    weekly = df.loc[df.groupby(['year','week'])['weekday']
                      .agg(['idxmax','idxmin']).melt()['value']]\
                .sort_values('Date')
    
    weekly_profit = weekly.sort_values(['Date']).groupby(['year','week'])\
                          .apply(lambda x: x.Close.diff().divide(x.Close.values[0]))\
                          .reset_index().dropna() 
    

    var_per_week_1 = np.percentile(weekly_profit.Close.round(4).array,1)
    var_per_week_5 = np.percentile(weekly_profit.Close.round(4).array,5)
  
    val = OrderedDict()
  
    val["risk_1"] = per_1
    val["risk_5"] = per_5
    val["profit_5"] = prof_5
    val["profit_30"] = prof_30
    val["profit_260"] = prof_260 
    val["risk_s_1"] = per_s_1
    val["risk_s_5"] = per_s_5
  
    return val


def plot_tickers(data,cost_in,ticker,floor_ceil,start='2020-12-01',end='2021-12-01',l_value=5,g_value=10):
    # Get data from start to end dates
    data = data[(data.Date>=start)&(data.Date<=end)]
  
    # Plot data
    fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                    open=data['Open'],
                                    high=data['High'],
                                    low=data['Low'],
                                    close=data['Close']),
                     go.Scatter(x=data['Date'], y=data['kama_{}'.format(g_value)], line=dict(color='orange', width=2)),
                     go.Scatter(x=data['Date'], y=data['kama_{}'.format(l_value)], line=dict(color='blue', width=2))])

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(
       title="Analysis",
       yaxis_title="{} Stock".format(ticker))
  #     shapes = [dict(
  #         x0='2020-03-18', x1='2020-03-18', y0=0, y1=1, xref='x', yref='paper',
  #         line_width=2)],
  #     annotations=[dict(
  #         x='2020-03-18', y=0.05, xref='x', yref='paper',
  #         showarrow=False, xanchor='left', text='Covid')]
  # )
    for row in floor_ceil.iterrows():
        fig.add_shape(type="line", x0 = start,
                                 y0 = row[1]['value'],
                                 x1 = data.Date.max(),
                                 y1 = row[1]['value'], fillcolor = 'yellow')
    if cost_in!= 0:
        fig.add_shape(type="line", x0 = start,
                              y0 = cost_in,
                              x1 = data.Date.max(),
                              y1 = cost_in, line_color = 'red',line_dash="dash")
  
    fig.show()
    
tickers = pd.read_csv("stocks.csv")['Stocks'].tolist()

# Today date
today = pd.to_datetime("today").date().strftime("%Y-%m-%d")

indicators = {"KAMA":[3,10]}

data = yf.download(" ".join(tickers), start="2020-01-01", end=today)

no_data_tickers = []
values = {}

pu.db
for tk in ['VGT.MX']:
    print(tk)
    
    # Get OCHL
    open = data["Open"][tk].rename("Open").reset_index()
    close = data["Close"][tk].rename("Close").reset_index()
    high = data["High"][tk].rename("High").reset_index()
    low = data["Low"][tk].rename("Low").reset_index()
  
    func = lambda x,y: x.merge(y,on='Date',how='left')
  
    # Merge all dataframes
    df = reduce(func,[open,close,high,low])
    
    # Impute values
    df = impute_values(df)

    # Compute indicators
    df,cols = compute_indicators(df,indicators)
  
    # Compute ceiling and floor
    try:
        l_value = min(indicators["KAMA"])
        g_value = max(indicators["KAMA"])
        floor_ceil = compute_floors_ceilings(df,g_value,l_value)
    except:
        no_data_tickers.append(tk)
        continue

    # Compute VaR
    try:
        var = compute_var(df)
        values[tk] = var
    except:
        continue        
        
    cost_in = []

    if len(cost_in)>0:
        cost_in = cost_in[0]
    else:
        cost_in = 0
  
    plot_tickers(df,cost_in,tk,floor_ceil,start='2020-01-01',end=today,l_value=l_value,g_value=g_value)


