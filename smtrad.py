import pandas as pd
import numpy as np
from datetime import timedelta
import datetime as dt
import os
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
import matplotlib.finance as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as dts
import matplotlib.ticker as ticker
import bitmex

################# ОБРАБОТКА ДАННЫХ #################

# Создание списка из имен файлов из директории
def read_fnames(fpath):
    os.chdir(fpath)
    fname_list = []
    for fname in os.listdir("."):
        fname_list.append(fname)
    print(fname_list)
    return fname_list

# Чтение и адаптация котировок с finam.ru БЕЗ ПАРАМЕТРА ВРЕМЕНИ (ДНЕВКИ)
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

# Чтение и адаптация котировок с finam.ru С ПАРАМЕТРОМ ВРЕМЕНИ (МИНУТКИ)
def read_finam_m(fpath, file):
    df = pd.read_csv(fpath +  '\\' + file, delimiter=';')
    df = pd.DataFrame(df)
    df['DATETIME'] = df['<DATE>'].astype('str').str[:8] + df['<TIME>'].astype('str')
    df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%Y%m%d%H%M%S')
    df.set_index('DATETIME', inplace=True)
    df.drop(['<PER>', '<DATE>', '<TICKER>', '<TIME>', '<VOL>'], axis=1, inplace=True)
    df.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
#         df['YIELD'] = df['CLOSE'] / df['CLOSE'].shift() - 1
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
    df = df.dropna()
    return df

# Чтение и адаптация котировок с bitmex.com
def read_bitmex(test, api_key, api_secret, symbol, binSize): # test = True включает Тестнет
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

# Ресемпл котировок
def qt_resample(df, timeframe):
    conversion = {'OPEN' : 'first', 'HIGH' : 'max', 'LOW' : 'min', 'CLOSE' : 'last'}
    df = df.resample(timeframe, how=conversion, base=0)
    df = df.dropna()
    return df


################# ГРАФИКИ #################

# Свечной график
def candles(df, width=0.65, colorup='g', colordown='r', alpha=0.75):
    ax = plt.axes()
#    df.reset_index(inplace=True)
#    df.columns = ['time', 'open', 'high', 'low', 'close']
#    df['time'] = dts.date2num(pd.to_datetime(df.index).tolist())
    
#    df.index = dts.num2date(df.index)
#    ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
#    ax.xaxis.set_minor_locator(dts.DayLocator())
#    ax.xaxis_date()
#    ax.xaxis.set_minor_formatter(dts.DateFormatter('%H:%M:%S'))
    return mpl.candlestick2_ohlc(ax, df['OPEN'], df['HIGH'], df['LOW'], df['CLOSE'], width=width, colorup=colorup, colordown=colordown, alpha=alpha)
    
# График RSI (подается Series RSI с датами в индексе)
def rsi_graph(rsi):
    ax = plt.axes()
    ax.plot(rsi)
    plt.show() 

################# ИНДИКАТОРЫ И ОСЦИЛЛЯТОРЫ #################

# Создание признака MA (moving average) с параметром роллинга
def ma(df, ma_period):
    df['MA' + str(ma_period)] = df['CLOSE'].rolling(ma_period).mean()
    return df

# Создание RSI
def rsi(df, rsi_period):
    df['ABS'] = df['CLOSE'] - df['CLOSE'].shift()
    
    def up(x):
        if x > 0:
            return x
        else:
            return 0
    df['U'] = df['ABS'].apply(up)

    def down(x):
        if x < 0:
            return x * -1
        else:
            return 0
    df['D'] = df['ABS'].apply(down)

    def equal(x):
        if x == 0:
            return 1
        else:
            return 0
        
    df['E'] = df['ABS'].apply(equal)

    df['AVG_GAIN'] = df['U'].rolling(rsi_period).mean()
    df['AVG_LOSS'] = df['D'].rolling(rsi_period).mean()
    df['RSI' + str(rsi_period)] = 100 - 100 / (1 + df['AVG_GAIN'] / df['AVG_LOSS'])
    df.drop(['ABS', 'U', 'D', 'E', 'AVG_GAIN', 'AVG_LOSS'], axis=1, inplace=True)
    df.dropna(inplace=True)
    return df

class Indicator:
    def __init__(self, df):
        self.df = df
    def bb(self):
        self['MID_BAND'] = self['CLOSE'].rolling(20).mean()
        self['UPPER_BAND'] = self['MID_BAND'] + 2 * self['CLOSE'].rolling(20).std()
        self['LOWER_BAND'] = self['MID_BAND'] - 2 * self['CLOSE'].rolling(20).std()
        self = self.dropna()
        return self

    