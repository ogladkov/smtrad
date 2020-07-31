import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import timedelta
import datetime as dt
import os
import matplotlib.cbook
import matplotlib.pyplot as plt
import matplotlib.dates as dts
import matplotlib.ticker as ticker
import plotly as py
from plotly import graph_objs as go
from plotly import tools
from time import sleep
from selenium import webdriver
import json
import  bitmex
import sys


path_to_files = [p for p in sys.path if 'smtrad' in p][0]

# Read json file with tickers and codes
with open(f'{path_to_files}\\emcodes.json', 'rb') as f:
    emcodes_dict = json.load(f)
    


################# QUOTES #################
class Quotes:
    def __init__(self, df, name=None):
        self.df = df
        self.name = name
        self.timeframe = int((self.df.index[1] - self.df.index[0]).delta*10e-10/60)

    def __add__(self, qt):
        merged = self
        merged.name = 'MERGED'
        merged.df = merged.df.merge(qt.df, left_index=True, right_index=True,
                                    suffixes=[f'_{merged.name}', f'_{qt.name}'],
                                    how='outer')
        merged.df = merged.df.fillna(method='ffill')
        merged.df = merged.df.dropna()
        return merged

def finam_direct(ticker, start, timeframe,
                 end=dt.datetime.strftime(dt.datetime.today(), '%d.%m.%Y'),
                 cols='all'):
    # Proper timeframes
    timeframe_dict = {"1 min":2, "5 min":3, "10 min":4, "15 min":5, "30 min":6, "1 hour":7, "1 day":8, "1 week":9,  "1 month":10}

    # Transform date format
    def transform_dates(start, end):
        start = dt.datetime.strptime(start, '%d.%m.%Y')
        end = dt.datetime.strptime(end, '%d.%m.%Y')
        return start, end

    date_start, date_end = transform_dates(start, end)

    # Try to download the data
    try:
        assemble = 'http://export.finam.ru/{ticker}_{date_from}_{date_to}.txt?market={market}&em={em}&code={ticker}&apply=0&df={df}&mf={mf}&yf={yf}&from={date_from_points}&dt={dt}&mt={mt}&yt={yt}&to={date_to_points}&p={timeframe}&f={ticker}_{date_from}_{date_to}&e=.txt&cn={ticker}&dtf=1&tmf=1&MSOR=1&mstime=on&mstimever=1&sep=3&sep2=1&datf=1&at=1'.format(ticker=ticker,
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
    except KeyError:
        print('Tickers list: \n', emcodes_dict.keys(), '\nPlease try a correct one.')
        raise KeyError('Wrong input...')

    df = pd.read_csv(assemble, sep=';', encoding='Windows-1251')

    if timeframe.split(' ')[1] == 'min' or timeframe.split(' ')[1] == 'hour':
        df['<TIME>'] = df['<TIME>'].replace(0, '000000')
        df['DATETIME'] = df['<DATE>'].astype('str').str[:8] + df['<TIME>'].astype('str')
        df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%Y%m%d%H%M%S')

    else:
        df['DATETIME'] = df['<DATE>'].astype('str')
        df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%Y%m%d')

    df.set_index('DATETIME', inplace=True)
    df.drop(['<PER>', '<DATE>', '<TICKER>', '<TIME>', '<VOL>'], axis=1, inplace=True)
    df.columns = [f'{ticker}_OPEN', f'{ticker}_HIGH', f'{ticker}_LOW', f'{ticker}_CLOSE']
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
    df = df.dropna()

    if cols != 'all':
        cols = [f'{ticker}_{c}' for c in cols]
        df = df[cols]

    sleep(1)
    return df


# Parse investing.com historical data by link
def parse_investing_hist(url, start_date, end_date):
    d, m, y = start_date.split('.')
    start_date = f'{m}/{d}/{y}'
    d, m, y = end_date.split('.')
    end_date = f'{m}/{d}/{y}'

    try:
        driver = webdriver.Firefox()
        driver.get(url)
        driver.find_element_by_css_selector('#flatDatePickerCanvasHol').click()
        driver.find_element_by_css_selector('#startDate').clear()
        driver.find_element_by_css_selector('#startDate').send_keys(start_date)
        driver.find_element_by_css_selector('#endDate').clear()
        driver.find_element_by_css_selector('#endDate').send_keys(end_date)
        driver.find_element_by_css_selector('#applyBtn').click()
        sleep(2)
        df = pd.DataFrame(pd.read_html(driver.page_source)[0])
        driver.close()
        df = df[['Date', 'Price']]
        df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y')
        df['Price'] = df['Price'].replace(',', '')
        df['Price'] = df['Price'].astype('float32')
        return df
    except:
        driver.close()
        parse_investing_hist(url, start_date, end_date)


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


# Get beta from two assets
def finam_betak(market, sec, start, end):
    market['YIELD'] = ((market['CLOSE'] - market['CLOSE'].shift()) / market['CLOSE'].shift())* 100
    market.dropna(inplace=True)
    market = market[['YIELD']]

    sec['YIELD'] = ((sec['CLOSE'] - sec['CLOSE'].shift()) / sec['CLOSE'].shift())* 100
    sec.dropna(inplace=True)
    sec = sec[['YIELD']]

    col_names = ['MARKET', 'SEC']
    df = market.merge(sec, left_index=True, right_index=True)
    df.columns = col_names

    cov = df.loc[:, 'MARKET'].cov(df.loc[:, 'SEC'])
    var = np.var(df.loc[:, 'MARKET'], ddof=1)
    beta = cov / var

    return beta


def ofz_yield(start_date, end_date=dt.datetime.strftime(dt.datetime.today(), '%d.%m.%Y')):
    url = 'https://www.investing.com/rates-bonds/russia-10-year-bond-yield-historical-data'
    df = parse_investing_hist(url, start_date, end_date)
    df = df.sort_values('date')
    return df


def t10y_yield(start_date, end_date=dt.datetime.strftime(dt.datetime.today(), '%d.%m.%Y')):
    url = 'https://www.investing.com/rates-bonds/u.s.-10-year-bond-yield-historical-data'
    df = parse_investing_hist(url, start_date, end_date)
    df = df.sort_values('date')
    return df


################# INDICATORS #################

class Indicator:
    def __init__(self, df):
        self.df = df
    def bb(self, period):
        self['MID_BAND'] = self['CLOSE'].rolling(period).mean()
        self['UPPER_BAND'] = self['MID_BAND'] + 2 * self['CLOSE'].rolling(period).std()
        self['LOWER_BAND'] = self['MID_BAND'] - 2 * self['CLOSE'].rolling(period).std()
        self = self.dropna()
        return self

    # Create moving average with rolling
    def ma(self, ma_period):
        self['MA' + str(ma_period)] = self['CLOSE'].rolling(ma_period).mean()
        self = self.dropna()
        return self

    # Create exponential moving average
    def ema(self, ema_period):
        self['EMA' + str(ema_period)] = self['CLOSE'].ewm(com=0.5).mean()
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
        self['EMA' + str(period_1)] = pd.ewma(self['CLOSE'], span=period_1)
        self['EMA' + str(period_2)] = pd.ewma(self['CLOSE'], span=period_2)
        self['EMA' + str(period_1)] = self['EMA' + str(period_2)] - self['EMA' + str(period_1)]
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


################# MACRO #################

def ruonia(date_start, date_end=dt.datetime.strftime(dt.datetime.today(), '%d.%m.%Y')):
    '''
        Ruonia data from ruonia.ru
    '''
    url_ruonia = f'http://ruonia.ru/archive?date_from={date_start}&date_to={date_end}'
    df = pd.read_html(url_ruonia)[2][1:]
    df.columns = ['Дата ставки', 'Значение, %', 'Обьем операций, млрд. руб.', 'Изменение, б.п', 'Дата публикации' ]
    df = df.iloc[:, [0, 1]]
    df.columns = ['date', 'RUONIA']
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    df['RUONIA'] = df['RUONIA'].astype('float32')
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    return df


def cbr_rate(start_date, end_date=dt.datetime.strftime(dt.datetime.today(), '%d.%m.%Y')):
    d, m, y = start_date.split('.')
    start_date = f'{d}%2F{m}%2F{y}'
    d, m, y = end_date.split('.')
    end_date = f'{d}%2F{m}%2F{y}'
    url = f'https://www.cbr.ru/eng/hd_base/KeyRate/?UniDbQuery.Posted=True&UniDbQuery.From={start_date}&UniDbQuery.To={end_date}'
    df = pd.read_html(url)[0][1:]
    df.columns = ['date', 'CBR']
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df['CBR'] = df['CBR'].astype('float32')
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    return df


def cbr_remainders(start_date, end_date=dt.datetime.strftime(dt.datetime.today(), '%d.%m.%Y')):
    url = f'https://www.cbr.ru/hd_base/ostat_base/?UniDbQuery.Posted=True&UniDbQuery.From={start_date}&UniDbQuery.To={end_date}'
    df = pd.read_html(url)[0]
    df.columns = ['date', 'RUSSIA_REMS', 'MOSCOW_REMS']
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    df['RUSSIA_REMS'] = df['RUSSIA_REMS'].str.replace(' ', '')
    df['RUSSIA_REMS'] = df['RUSSIA_REMS'].str.replace(',', '.')
    df['RUSSIA_REMS'] = df['RUSSIA_REMS'].astype('float32')
    df['MOSCOW_REMS'] = df['MOSCOW_REMS'].str.replace(' ', '')
    df['MOSCOW_REMS'] = df['MOSCOW_REMS'].str.replace(',', '.')
    df['MOSCOW_REMS'] = df['MOSCOW_REMS'].astype('float32')
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    return df
