#import warnings
#warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import timedelta
import datetime as dt
import os
import matplotlib.cbook
#import mpl_finance as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as dts
import matplotlib.ticker as ticker
#import bitmex
import plotly as py
from plotly import graph_objs as go
from plotly import tools
from time import sleep
from selenium import webdriver


emcodes_dict = {"SBER":3, "SBERP":23, "SBRF":17456, "GAZP":16842, "LKOH":8, "USD000UTSTOM":182400, "EUR_RUB__TOM":182398, "EURUSD000TOM":182399, "ALRS":81820, "ROSN":17273, "SPFB.RTS":17455, "SPFB.Si":19899, "MVID":19737, "NVTK":17370, "MOEX":152798, "HYDR":20266, "IRAO":20516, "MGNT":17086, "MTSS":15523, "GMKN":795, "YNDX":388383, "VTBR":19043, "SNGSP":13, "PLZL":17123, "CHMF":16136, "SIBN":2, "BANEP":81758, "AFLT":29, "NLMK":17046, "TATN":825, "SNGS":4, "RUAL":414279, "ENRU":16440, "RSTI":20971, "AFKS":19715, "TRMK":18441, "FXRU":182346, 'RGBI':82308, 'RASP':17713, 'MSNG':6, "MRKP":20107, "FEES":20509, "UPRO":18584, "IMOEX":420450, "MTSS":15523, "MRKU":20402, "MRKV":20286, "SI":18952, "GC":18953, "MICEX":13851, "POLY":175924, "AKRN":17564, "BSPB":20066, "DIXY":18564, "KMAZ":15544, "LSRG":19736, "MAGN":16782, "MFON":152516, "MSTT":74549, "NMTP":19629, "PIKK":18654, "RTKM":7, "RTKMP":15, "RUALR":74718, "SVAV":16080, "URKA":19623, "PHOR":81114, "GCHE":20125, "UPRO":18584}

################# PROCESS DATA #################
# Reads quotes from Finam.ru as Pandas DataFrame

class QuotesFinam:
    def __init__(self, ticker, start, end, timeframe):
        self.df = finam_direct(ticker, start, end, timeframe)
        self.ticker = ticker
    def ticker(self):
        return self.ticker
    def len(self):
        return self.df.shape[0]

def finam_direct(ticker, start, end, timeframe):
    timeframe_dict = {"1 min":2, "5 min":3, "10 min":4, "15 min":5, "30 min":6, "1 hour":7, "1 day":8, "1 week":9,  "1 month":10}

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
    
    
################# AGGREGATORS #################
class Aggregator:
    def __init__(self, df):
        self.df = df
    def greb_results(dfs, suffixes=None, period='1D', total=False):
        columns_names = ['Дата', 'Результат', 'Всего сделок', 
                       'Количество положительно закрытых позиций',
                       'Количество отрицательно закрытых позиций',
                       'Доходность в % годовых',
                       'Процент прибыльных итогов',
                       'Процент убыточных итогов']
        first_step = True # influences the merging
        for a in dfs:
            a = pd.DataFrame(a, columns=['datetime', 'rate', 'result']).set_index('datetime')
            a['deals_count'] = 1
            first_price = a.iloc[0]['rate']
            longs = True if first_price < 0 else False
            delta = a.index[-1] - a.index[-2]
            delta = 1 if delta.days == 0 else delta.days
            def pos_deals(x):
                if x > 0:
                    return 1
            def neg_deals(x):
                if x < 0:
                    return 1
            a['pos_count_deals'] = a['result'].apply(pos_deals)
            a['neg_count_deals'] = a['result'].apply(neg_deals)
            a = a.resample(period).sum()
            if longs:
                a['yield'] = (a['result'] / first_price / delta * 365 * -1).round(2)
            else:
                a['yield'] = (a['result'] / first_price / delta * 365).round(2)
            a.drop('rate', axis=1, inplace=True)
            a = a[a['deals_count'] != 0]
            a[['pos_count_deals', 'neg_count_deals']] = a[['pos_count_deals', 'neg_count_deals']].astype('int')
            a['pct_pos'] = (a['pos_count_deals'] / (a['pos_count_deals'] + a['neg_count_deals']) * 100).round(2)
            a['pct_pos'] = a['pct_pos'].astype('str') + '%'
            a['pct_neg'] = (a['neg_count_deals'] / (a['pos_count_deals'] + a['neg_count_deals']) * 100).round(2)
            a['pct_neg'] = a['pct_neg'].astype('str') + '%'
            a['result'] = a['result'].round(4)
            a.reset_index(inplace=True)
            a.columns = columns_names
            if not first_step:
                a = self.merge(a, on='Дата', suffixes=suffixes, how='outer')
            else:
                self = a
            first_step = False
        self = a
        
        result_colunms = [x for x in self.columns if 'Результат' in x]
        total_deals_colunms = [x for x in self.columns if 'Всего сделок' in x]
        self['Результат по всем каналам'] = self[result_colunms].sum(axis=1)
        self['Сделок по всем каналам'] = self[total_deals_colunms].sum(axis=1)
        
        if total:
            self = self.pivot_table(index='Дата',
                   margins=True,
                   margins_name='ВСЕГО',  # defaults to 'All'
                   aggfunc=sum)
            order_colunms = [x for x in self.columns if list(self.columns).index(x) % 2 == 0][:-1] +\
            [x for x in self.columns if list(self.columns).index(x) % 2 == 1] + [self.columns[-2]]
            self = self[order_colunms]
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
    df.columns = ['date', 'ruonia']
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    df['ruonia'] = df['ruonia'].astype('float16')
    df = df.sort_values('date')
    return df


def cbr_rate(start_date, end_date=dt.datetime.strftime(dt.datetime.today(), '%d.%m.%Y')):
    d, m, y = start_date.split('.')
    start_date = f'{d}%2F{m}%2F{y}'
    d, m, y = end_date.split('.')
    end_date = f'{d}%2F{m}%2F{y}'
    url = f'https://www.cbr.ru/eng/hd_base/KeyRate/?UniDbQuery.Posted=True&UniDbQuery.FromDate={start_date}&UniDbQuery.ToDate={end_date}'
    df = pd.read_html(url)[0][1:]
    df.columns = ['date', 'cbr']
    df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
    df['cbr'] = df['cbr'].astype('float16')
    df = df.sort_values('date')
    return df


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


def cbr_remainders(start_date, end_date=dt.datetime.strftime(dt.datetime.today(), '%d.%m.%Y')):
    driver = webdriver.Firefox()
    url = 'https://www.cbr.ru/hd_base/ostat_base/'
    driver.get(url)
    driver.find_element_by_id('UniDbQuery_FromDate').click()
    driver.find_element_by_id('UniDbQuery_FromDate').clear()
    driver.find_element_by_id('UniDbQuery_FromDate').send_keys('10.12.2018')
    driver.find_element_by_id('UniDbQuery_ToDate').click()
    driver.find_element_by_id('UniDbQuery_ToDate').clear()
    driver.find_element_by_id('UniDbQuery_ToDate').send_keys('10.12.2019')
    driver.find_element_by_id('UniDbQuery_searchbutton').click()
    html = driver.page_source
    driver.close()

    df = pd.read_html(html)[0].iloc[1:]
    df.columns = ['date', 'rests_russia', 'rests_region']
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    df['rests_russia'] = df['rests_russia'].str.replace(' ', '').str.replace(',', '.').astype('float16')
    df['rests_region'] = df['rests_region'].str.replace(' ', '').str.replace(',', '.').astype('float16')
    df = df.sort_values('date')

    return df