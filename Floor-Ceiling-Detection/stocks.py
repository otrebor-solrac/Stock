import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce

import yfinance as yf
from finta import TA

import plotly.graph_objects as go
import yahoo_fin.stock_info as si
from collections import OrderedDict

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN

from fpdf import FPDF
from PIL import Image 
import glob

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)




import pudb

def get_hex_color():
    """
    Random colors generation.
    """


    random_number = random.randint(0,16777215)
    hex_number = str(hex(random_number))[2:].zfill(6)
    
    hex_number ='#'+ hex_number
    return hex_number

def generate_colors_by_indicator(indicators_params):
    """
    Create a color by indicator to have a better visualization.
    """
    col_names = ["{}_{}".format(ind,str(val)) for ind in indicators_params.keys() 
                                                for val in indicators_params.get(ind) ]
    return {c:get_hex_color() for c in col_names}

def normalize_values(df,col,scale=3):
    max_val = df[col].max()
    df[col] = scale*df[col]/max_val
    return df

def separate_date(df):
    df['day'] = df.Date.map(lambda x: x.weekday() + 1)
    df['week'] = df.Date.map(lambda x: x.week)
    df['year'] = df.Date.map(lambda x: x.year)
    return df

def get_data_by_week(df):
    df = df.sort_values('Date').groupby(['year','week']).agg({'Date':'first',
                                                        'Open':'first',
                                                        'Close':'last',
                                                        'High':'max',
                                                        'Low':'min'}).reset_index()
    return df

def impute_missing_values(df,col='Date'):
    """
    Missing values imputation. If you need it, you can add
    more starategies for imputation.
    """
    
    # Fill NAs with last valid observation.
    df = df.sort_values(col).fillna(method='ffill')
    return df

def get_data_by_stock(data,tk):
    """
    Get data from main dataframe and impute missing values.
    """
    open_values = data["Open"][tk].rename("Open").reset_index()
    close_values = data["Close"][tk].rename("Close").reset_index()
    high_values = data["High"][tk].rename("High").reset_index()
    low_values = data["Low"][tk].rename("Low").reset_index()
    volume_values = data["Volume"][tk].rename("Volume").reset_index()
  
    func = lambda x,y: x.merge(y,on='Date',how='left')
  
    # Merge all dataframes
    df = reduce(func,[open_values,close_values,high_values,low_values,volume_values])
        
    return df
    
def compute_indicators(df,indicators_params):
    """
    Function to compute indicators given their parameters.
    
    NOTE: You could add more indicators as you need.
    """
    # List of new the indicators' columns 
    cols_names = []
    indicators = df[["Date"]]

    for ind,values in indicators_params.items():
        if ind=="KAMA":
            kama = values

            for period in kama:
                kama_values = TA.KAMA(df,period).tolist()

                col_name = "{}_{}".format(ind,str(period))
                indicators[col_name] = kama_values
                
        elif ind=="EMA":
            ema = values

            for period in ema:
                ema_values = TA.EMA(df,period).tolist()

                col_name = "{}_{}".format(ind,str(period))
                indicators[col_name] = ema_values
                
        elif ind=="RSI":
            rsi = values
            
            for period in rsi:
                rsi_values = TA.RSI(df,period)
                
                col_name = "{}_{}".format(ind,str(period))
                indicators[col_name] = rsi_values      

        elif ind=="BB":
            bbands = values
            
            for period in bbands:
                bb_values = TA.BBANDS(df,period)
                
                cols_names = ["{}_{}".format(col,str(period)) for col in bb_values.columns]
                
                for idx, col in enumerate(bb_values.columns):
                    indicators[cols_names[idx]] = bb_values[col]
        
    return indicators


def values_normalization(df,col,scale=3):
    max_val = df[col].max()
    df[col] = scale*df[col]/max_val
    return df

def compute_fibonacci(df,min_max_detection,mode='KAMA'):
    """
    Fibonaccies computation.
    """
    # Select low and high values for floor and ceiling detection

    selection_values = [max(min_max_detection[mode]),
                        min(min_max_detection[mode])] 

    df = crossings_detection(df,mode,selection_values)
    
    df = find_max_min_values(df)
    
    df.reset_index(drop=True,inplace=True)
    
    fibos = pd.DataFrame(columns=['value','dates','move'])

    for indx, _ in df.iterrows():

        if indx == 0:
            continue

        range_vals = [df.loc[indx].value,
                      df.loc[indx-1].value]

        max_val = max(range_vals)
        min_val = min(range_vals)
        range_ = max_val - min_val
        date = df.loc[indx].dates

        for fibo_val in FIBOS_RATIOS:
            new_fibo = {'value':min_val + fibo_val*range_,
                        'dates':date,
                        'ratio':str(fibo_val)}
            fibos = fibos.append(new_fibo,ignore_index=True)
    
    fibos = cluster_floor_ceiling(fibos.copy())
    
    fibos = values_normalization(fibos,'num_vals')
    fibos['type'] = 'fibos'

    return fibos

def find_max_min_values(data):
    """
    Find the maximum and minimun values of the given stock.
    """
    floor_ceil = data.floor_ceil.tolist()

    val_min = [np.inf,"",""]
    val_max = [-np.inf,"",""]
    all_values = []
  
    for indx,_ in enumerate(floor_ceil):
        
        if indx==0:
            continue

        if floor_ceil[indx]==floor_ceil[indx-1]:
            if floor_ceil[indx] == 1:
                if val_max[0] <= data.loc[indx,"High"]:
                    val_max[0] = data.loc[indx,"High"]
                    val_max[1] = data.loc[indx,"Date"]
                    val_max[2] = "Max"
            else:
                if val_min[0] >= data.loc[indx,"Low"]:
                    val_min[0] = data.loc[indx,"Low"]
                    val_min[1] = data.loc[indx,"Date"]
                    val_min[2] = "Min"
        else:
            all_values.append(val_max)
            all_values.append(val_min)

            val_min = [np.inf,"",""]
            val_max = [-np.inf,"",""]

            if floor_ceil[indx] == 1:
                if val_max[0] <= data.loc[indx,"High"]:
                    val_max[0] = data.loc[indx,"High"]
                    val_max[1] = data.loc[indx,"Date"]
                    val_max[2] = "Max"
            else:
                if val_min[0] >= data.loc[indx,"Low"]:
                    val_min[0] = data.loc[indx,"Low"]
                    val_min[1] = data.loc[indx,"Date"]
                    val_min[2] = "Min"
            
    
    df = pd.DataFrame(all_values)
    df.columns = ['value','dates','move']
    df = df[df.move != ""]
    df = df[~df.dates.isna()]

    return df


def cluster_floor_ceiling(floor_ceil,percentile=5):
    val_min = floor_ceil.value - floor_ceil.value.min()
    range_value = floor_ceil.value.max() - floor_ceil.value.min()
    
    floor_ceil['value_norm'] = val_min/range_value 
    floor_ceil['reference'] = 1
    
    dist_eu = euclidean_distances(floor_ceil[['value_norm','reference']])
    vals = [np.round(e,3) for e in list(dist_eu.flatten()) if e > 0]
    e = np.percentile(vals,percentile)
    print("Distance :",e)

    db = DBSCAN(eps=e, min_samples=2).fit(floor_ceil[['value_norm','reference']])
    floor_ceil['classification'] = db.labels_

    df_class = floor_ceil[floor_ceil.classification!=-1]\
                        .groupby('classification')\
                        .agg({'value':['mean','count'],
                              'dates':'first'}).reset_index()
    df_class.columns = ['classification','value','num_vals','dates']
    
    df_without_class = floor_ceil.loc[floor_ceil.classification==-1,
                                      ['classification','value','dates']]
    df_without_class['num_vals'] = 1
    
    floor_ceil = pd.concat([df_without_class,
                            df_class])
       
    return floor_ceil

def select_ma_strategy(mode,selection_values):
    """
    Select the moving average indicator to compute floors and ceilings.
    """
    high_value,low_value = selection_values
    
    if mode == 'KAMA':
        high_col = "{}_{}".format(mode,high_value)
        low_col = "{}_{}".format(mode,low_value)
    
    elif mode == 'EMA':
        high_col = "{}_{}".format(mode,high_value)
        low_col = "{}_{}".format(mode,low_value)
    
    return high_col,low_col

def crossings_detection(df,mode, selection_values):
    """
    Detection when the moving average indicators crosses each other.
    """
    
    high_col,low_col = select_ma_strategy(mode, selection_values)
    
    # Detect when high_value crosses low_value and vice versa
    df['floor_ceil'] = df[high_col] <= df[low_col]
    df.loc[df.floor_ceil,'floor_ceil'] = 1
    df.loc[df.floor_ceil!=1,'floor_ceil'] = -1
    return df
    
def compute_floors_ceilings(df,min_max_detection,mode='KAMA'):
    """
    Compute floors and ceilings by using a moving average.
    """
    # Select low and high values for floor and ceiling detection
    low_value = min(min_max_detection[mode])
    high_value = max(min_max_detection[mode])

    selection_values = [high_value,low_value]    

    df = crossings_detection(df,mode, selection_values)

    floor_ceil = find_max_min_values(df)

    if floor_ceil.empty:
        return floor_ceil
               
    floor_ceil = cluster_floor_ceiling(floor_ceil.copy())
    
    floor_ceil = normalize_values(floor_ceil,'num_vals')

    floor_ceil['type'] = 'floor_ceil'
          
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

def plot_tickers(data,ticker,floor_ceil,colors_indicators,start='2020-12-01',end='2021-12-01',which_indicators=[]):
    # Get data from start to end dates
    data = data[(data.Date>=start)&(data.Date<=end)]
  
    # Plot data
    data_to_plot = []
    
    candle = [go.Candlestick(x=data['Date'],
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'])]
    
    which_indicators = which_indicators if which_indicators else colors_indicators.keys()

    indicadores = [go.Scatter(x=data['Date'], y=data[col], line=dict(color=colors_indicators.get(col), width=3),name=col) 
                                                   for col in which_indicators]
    data_to_plot += candle
    data_to_plot += indicadores
    
    fig = go.Figure(data=data_to_plot)

    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_layout(title="Analysis",
                      yaxis_title="{} Stock".format(ticker))
    
    for row in floor_ceil.iterrows():
        if row[1].type=='fibos':
            color='#009aff'
        else:
            color='#00bb2d'

        fig.add_shape(type="line",
                      x0 = start,
                      y0 = row[1]['value'],
                      x1 = data.Date.max(),
                      y1 = row[1]['value'],
                      line=dict(color=color,width=row[1]['num_vals']))
       
    name = "{}.jpg".format(ticker)
    folder = "images/"

    fig.write_image(folder + name,width=1920, height=1080)

    #fig.show()

random.seed(42)
pu.db
tickers = pd.read_csv("stocks.csv")['Stocks'].tolist()#[:3]

today = pd.to_datetime("today").date().strftime("%Y-%m-%d")

data = yf.download(" ".join(tickers), start="2018-01-01", end=today)

colors_indicators = {}

indicators_params = {"KAMA":[3,10],
                    "EMA":[3,7,9,21,50,97,100,200],
                    "RSI":[14,30],
                    "BB":[30]}

colors_indicators = generate_colors_by_indicator(indicators_params)              

mode='EMA'

min_max_detection = {"EMA":[9,21],
                     "KAMA":[3,10]}


no_data_tickers = []
values = {}
FIBOS_RATIOS = [0,0.236,0.382,0.50,0.618,0.764,1]
which_indicators = ['EMA_21','EMA_200','EMA_7','BB_UPPER_30','BB_MIDDLE_30','BB_LOWER_30']

for tk in tickers:
    pu.db
    print("Stock {} is being processed".format(tk))
    
    df = get_data_by_stock(data,tk)
    df = separate_date(df)
    #df = get_data_by_week(df)
    df = impute_missing_values(df)

    indicators = compute_indicators(df,indicators_params)
    
    df = df.merge(indicators,on=["Date"],how="left")

    try:
        floor_ceil = compute_floors_ceilings(df,min_max_detection,mode)
        fibos = compute_fibonacci(df,min_max_detection,mode)
    except:
        print("Stock {} not processed".format(tk))
        continue

    floor_ceil = pd.concat([floor_ceil,fibos])
    date_ = df.Date.tolist()[1200]
    #profit = compute_profit(df,rsi_strategy,date_,df.Date.tolist()[-1],num_shares=1)
    #print("Profit: ",profit)
    plot_tickers(df,tk,floor_ceil,colors_indicators,start=date_,end=today,which_indicators=which_indicators)

 
images = glob.glob('images/*')
images = [Image.open(f) for f in images]
pdf_path = "file.pdf"   
images[0].save(pdf_path, "PDF" ,resolution=300.0, save_all=True, append_images=images[1:])


#pdf = FPDF()
# imagelist is the list with all image filenames
#images = glob.glob('images/*')
#for image in images:
#    pdf.add_page()
#    pdf.image(image)
#pdf.output("file.pdf", "F")