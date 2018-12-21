import pandas as pd
import numpy as np
from datetime import timedelta
import datetime as dt
import os
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
import mpl_finance as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as dts
import matplotlib.ticker as ticker
import bitmex
import plotly as py
from plotly import graph_objs as go
from plotly import tools
from time import sleep

################# PROCESS DATA #################
# Reads quotes from Finam.ru as Pandas DataFrame
def finam_direct(ticker, start, end, timeframe):
    timeframe_dict = {"1 min":2, "5 min":3, "10 min":4, "15 min":5, "30 min":6, "1 hour":7, "1 day":8, "1 week":9,  "1 month":10}
    emcodes_dict = {"SBER":3, "GAZP":16842, "LKOH":8, "USD000UTSTOM":182400, "EUR_RUB__TOM":182398, "EURUSD000TOM":182399, "ALRS":81820, "ROSN":17273}

    def transform_dates(start, end):
        start = dt.datetime.strptime(start, '%d.%m.%Y')
        end = dt.datetime.strptime(end, '%d.%m.%Y')
        return start, end

    date_start, date_end = transform_dates(start, end)
#
    assemble = 'http://export.finam.ru/{ticker}_{date_from}_{date_to}.txt?market={market}&em={em}&code={ticker}&apply=0&df={df}&mf={mf}&yf={yf}&from={date_from_points}&dt={dt}&mt={mt}&yt={yt}&to={date_to_points}&p={timeframe}&f={ticker}_{date_from}_{date_to}&e=.txt&cn={ticker}&dtf=1&tmf=1&MSOR=1&mstime=on&mstimever=1&sep=3&sep2=1&datf=1&at=1'.format(                           ticker=ticker,
                              date_from=dt.datetime.strftime(date_start, 
                                                             format='%d%m%y'),
                              date_to=dt.datetime.strftime(date_end, 
                                                           format='%d%m%y'),
                              df=date_start.day,
                              mf=date_start.month-1,
                              yf=date_start.year,
                              date_from_points=dt.datetime.strftime(date_start, 
                                                                    format='%d.%m.%Y'),
                              dt=date_end.day,
                              mt=date_end.month-1,
                              yt=date_end.year,
                              date_to_points=dt.datetime.strftime(date_end, 
                                                                    format='%d.%m.%Y'),
                              timeframe=timeframe_dict[timeframe],
                              market = '1',
                              em = emcodes_dict[ticker]
                                              )
    df = pd.read_csv(assemble, sep=';', encoding='Windows-1251')
    
    if timeframe.split(' ')[1] == 'min' or timeframe.split(' ')[1] == 'hour':
        df['<TIME>'] = df['<TIME>'].replace(0, '000000')
        df['DATETIME'] = df['<DATE>'].astype('str').str[:8] + df['<TIME>'].astype('str')
        df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%Y%m%d%H%M%S')
        df.set_index('DATETIME', inplace=True)
        df.drop(['<PER>', '<DATE>', '<TICKER>', '<TIME>', '<VOL>'], axis=1, inplace=True)
        df.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
        df = df.dropna()
        sleep(1)
        return df
    else:
        df['DATETIME'] = df['<DATE>'].astype('str')
        df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%Y%m%d')
        df.set_index('DATETIME', inplace=True)
        df.drop(['<PER>', '<DATE>', '<TICKER>', '<TIME>', '<VOL>'], axis=1, inplace=True)
        df.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
        df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
        df = df.dropna()
        sleep(1)
        return df
    

# Create list from filenames from the dir
def read_fnames(fpath):
    os.chdir(fpath)
    fname_list = []
    for fname in os.listdir("."):
        fname_list.append(fname)
    print(fname_list)
    return fname_list

# Read and adopt quotes from finam.ru (days)
def read_finam_d(fpath, file):
    df = pd.read_csv(fpath +  '\\' + file, delimiter=';')
    df = pd.DataFrame(df)
    df['DATETIME'] = df['<DATE>'].astype('str')
    df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%Y%m%d')
    df.set_index('DATETIME', inplace=True)
    df.drop(['<PER>', '<DATE>', '<TICKER>', '<TIME>', '<VOL>'], axis=1, inplace=True)
    df.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
#         df['YIELD'] = df['CLOSE'] / df['CLOSE'].shift() - 1
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
    df = df.dropna()
    return df

# Read and adopt quotes from finam.ru (mins)
def read_finam_m(fpath, file):
    df = pd.read_csv(fpath +  '\\' + file, delimiter=';')
    df = pd.DataFrame(df)
    df['<TIME>'] = df['<TIME>'].replace(0, '000000')
    df['DATETIME'] = df['<DATE>'].astype('str').str[:8] + df['<TIME>'].astype('str')
    df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%Y%m%d%H%M%S')
    df.set_index('DATETIME', inplace=True)
    df.drop(['<PER>', '<DATE>', '<TICKER>', '<TIME>', '<VOL>'], axis=1, inplace=True)
    df.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
#         df['YIELD'] = df['CLOSE'] / df['CLOSE'].shift() - 1
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
    df = df.dropna()
    return df

# Read and adopt quotes from bitmex.com
def read_bitmex(test, api_key, api_secret, symbol, binSize): # test = True switches on TESTNET
    client = bitmex.bitmex(test=test, api_key=api_key, api_secret=api_secret)
    client = bitmex.bitmex()
    df = client.Trade.Trade_getBucketed(symbol=symbol, binSize=binSize, count=500, reverse=True).result()
    df = pd.DataFrame(df[0])
#    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover', 'trades']]
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
#    df = df.rename(columns={'timestamp':'DATETIME', 'open':'OPEN', 'high':'HIGH', 'low':'LOW', 'close':'CLOSE', 'volume':'VOLUME', 'turnover':'TURNOVER', 'trades':'TRADES'})
    df = df.rename(columns={'timestamp':'DATETIME', 'open':'OPEN', 'high':'HIGH', 'low':'LOW', 'close':'CLOSE', 'volume':'VOLUME'})
    df.set_index('DATETIME', inplace=True)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%dT%H:%M:%S')
    df.sort_index(inplace=True, ascending=True)
    return df

# Resample quotes
def qt_resample(df, timeframe):
    conversion = {'OPEN' : 'first', 'HIGH' : 'max', 'LOW' : 'min', 'CLOSE' : 'last'}
    df = df.resample(timeframe, how=conversion, base=0)
    df = df.dropna()
    return df


################# GRAPHS #################

class Graph:
    def __init__(self, df):
        self.df = df
        
    def newline(*args):
        traces = []
        for a in enumerate(args, 1):
            traces.append(go.Scatter(x=a[1]['sec'].index,
                               y=a[1]['sec'].CLOSE,
                               name=a[1]['name'],
                               yaxis='y'+str(a[0]),
                               line=dict(color=a[1]['color'])
                              )
                         )

        layout = {}
        layout['legend']={'orientation':'h'}
        
        position = 0
        
        for a in enumerate(args, 1):
            if a[0] == 1:
                layout['yaxis'+str(a[0])]={'side':a[1]['axis'], 
                                           'title':a[1]['name'],
                                           'color':a[1]['color'],}
            else:
                if a[1]['axis'] == 'left':
                    position += 0.04
                    layout['yaxis'+str(a[0])]={'side':a[1]['axis'], 
                                               'title':a[1]['name'],
                                               'overlaying':'y1',
                                               'color':a[1]['color'],
                                               'position':position}
                else:
                    layout['yaxis'+str(a[0])]={'side':a[1]['axis'], 
                                               'title':a[1]['name'],
                                               'color':a[1]['color'],
                                               'overlaying':'y1'}

        data = traces
        fig = dict(data=data, layout=layout)
        py.offline.plot(fig)
        
    def candles(*args):
        traces = []
        for a in enumerate(args, 1):
            traces.append(go.Candlestick(x=a[1]['sec'].index,
                                         open=a[1]['sec'].OPEN,
                                         high=a[1]['sec'].HIGH,
                                         low=a[1]['sec'].LOW,
                                         close=a[1]['sec'].CLOSE,
                                         name=a[1]['name'],
                                         yaxis='y'+str(a[0]),
                                         increasing=dict(line = dict(color=a[1]['color'],
                                                                     width=1),
                                                         fillcolor = 'white'),
                                         decreasing=dict(line=dict(color=a[1]['color'],
                                                                  width=1))
                                        )
                                        
                         )

        layout = {}
        layout['legend']={'orientation':'h'}
        layout['xaxis']={'rangeslider':{'visible':False}, 'type':'category'}
        
        position = 0
        
        for a in enumerate(args, 1):
            if a[0] == 1:
                layout['yaxis'+str(a[0])]={'side':a[1]['axis'], 
                                           'title':a[1]['name'],
                                           'color':a[1]['color'],}
            else:
                if a[1]['axis'] == 'left':
                    position += 0.04
                    layout['yaxis'+str(a[0])]={'side':a[1]['axis'], 
                                               'title':a[1]['name'],
                                               'overlaying':'y1',
                                               'color':a[1]['color'],
                                               'position':position}
                else:
                    layout['yaxis'+str(a[0])]={'side':a[1]['axis'], 
                                               'title':a[1]['name'],
                                               'color':a[1]['color'],
                                               'overlaying':'y1'}

        data = traces
        fig = dict(data=data, layout=layout)
        py.offline.plot(fig)
        
################# INDICATORS #################

class Indicator:
    def __init__(self, df):
        self.df = df
    def bb(self):
        self['MID_BAND'] = self['CLOSE'].rolling(20).mean()
        self['UPPER_BAND'] = self['MID_BAND'] + 2 * self['CLOSE'].rolling(20).std()
        self['LOWER_BAND'] = self['MID_BAND'] - 2 * self['CLOSE'].rolling(20).std()
        self = self.dropna()
        return self
    
    # Create moving average with rolling
    def ma(self, ma_period):
        self['MA' + str(ma_period)] = self['CLOSE'].rolling(ma_period).mean()
        self = self.dropna()
        return self
    
    # Create exponential moving average
    def ema(self, ema_period):
        weights = np.exp(np.linspace(-1., 0., ema_period))
        weights /= weights.sum()
        a =  np.convolve(self['CLOSE'], weights, mode='full')[:len(self['CLOSE'])]
        a[:ema_period] = a[ema_period]
        self['EMA' + str(ema_period)] = a
        return self
    
    # Create RSI
    def rsi(self, rsi_period):
        self['ABS'] = self['CLOSE'] - self['CLOSE'].shift()
        def up(x):
            if x > 0:
                return x
            else:
                return 0
        self['U'] = self['ABS'].apply(up)
        def down(x):
            if x < 0:
                return x * -1
            else:
                return 0
        self['D'] = self['ABS'].apply(down)
        def equal(x):
            if x == 0:
                return 1
            else:
                return 0
        self['E'] = self['ABS'].apply(equal)
        self['AVG_GAIN'] = self['U'].rolling(rsi_period).mean()
        self['AVG_LOSS'] = self['D'].rolling(rsi_period).mean()
        self['RSI' + str(rsi_period)] = 100 - 100 / (1 + self['AVG_GAIN'] / self['AVG_LOSS'])
        self.drop(['ABS', 'U', 'D', 'E', 'AVG_GAIN', 'AVG_LOSS'], axis=1, inplace=True)
        self.dropna(inplace=True)
        return self
    
    # Create MACD
    def macd(self, period_1=12, period_2=26, signal=9):
        self['EMA' + str(period_1)] = self['CLOSE'].rolling(period_1).mean()
        self['EMA' + str(period_1)] = (self['CLOSE'] * 2 / (period_1 + 1)) + self['EMA' + str(period_1)].shift() * (1 - 2 / (period_1 + 1))
        self['EMA' + str(period_2)] = self['CLOSE'].rolling(period_2).mean()
        self['EMA' + str(period_2)] = (self['CLOSE'] * 2 / (period_2 + 1)) + self['EMA' + str(period_2)].shift() * (1 - 2 / (period_2 + 1))
        self['MACD'] = self['EMA' + str(period_1)] - self['EMA' + str(period_2)]
        self['MACD_SIGNAL'] = self['MACD'].rolling(signal).mean()
        self['MACD_SIGNAL'] = (self['MACD'] * 2 / (signal + 1)) + self['MACD_SIGNAL'].shift() * (1 - 2 / (signal + 1))
        self['MACD_HIST'] = self['MACD'] - self['MACD_SIGNAL']
        self.dropna(inplace=True)
        return self
    
    # Create CCI
    def cci(self, cci_period):
        self['TP'] = (self['CLOSE'].shift() + self['HIGH'].shift() + self['LOW'].shift()) / 3
        self['SMATP'] = self['TP'].rolling(cci_period).mean()
        self['D'] = abs(self['TP'] - self['SMATP'])
        self['MD'] = self['D'].rolling(cci_period).mean()
        self['CCI' + str(cci_period)] = (self['TP'] - self['SMATP']) / (0.015 * self['MD'])
        self.drop(['TP', 'SMATP', 'D', 'MD'], axis = 1, inplace=True)
        self.dropna(inplace=True)
        return self
    
    