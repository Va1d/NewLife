from lb import drank, extract
import pandas as pd
from os import path
from progressbar import progressbar
import numpy as np
from typing import *

base_path = "/home/bo/Py/PD"
header = ['day', 'minute', 'open', 'close', 'high', 'low', 'volume', 'trade_count', 'vwap']
headertypes = {'day':int, 'minute':int, 'open':float, 'close':float, 'high':float, 'low':float, 'volume':float, 'trade_count':int, 'vwap':float}
indexcolumns = ['day','minute']

def panda_file(code:str)->str:
    return path.join(base_path,f'{code}.csv')
def load_panda(code:str)->pd.DataFrame:
    return pd.read_csv(panda_file(code), index_col=indexcolumns, dtype=headertypes)
def save_panda(df:pd.DataFrame, code:str)->None:
    df.to_csv(panda_file(code), index=True)
def days(df:pd.DataFrame)->Iterable[int]:
    return df.index.get_level_values('day').unique()
def getday(df:pd.DataFrame,day:int)->pd.DataFrame:
    return df[df.index.get_level_values('day')==day]

def drop_computed(df:pd.DataFrame)->pd.DataFrame:
    return df[df.columns.intersection(header)]

def exp_col(df:pd.DataFrame,col_name:str)->pd.DataFrame:
    group = df.groupby(level='day')[col_name]
    exp_min = group.expanding().min().reset_index(level=0,drop=True)
    exp_max = group.expanding().max().reset_index(level=0,drop=True)
    df[f"{col_name}_exp"] = (df[col_name] - exp_min) / (exp_max - exp_min)     
    df.fillna(0, inplace=True)      
    return df

def zs_col(df:pd.DataFrame,col_name:str)->pd.DataFrame:
    group = df.groupby(level='day')[col_name]
    exp_mean = group.expanding().mean().reset_index(level=0,drop=True)
    exp_std = group.expanding().std().reset_index(level=0,drop=True)
    df[f"{col_name}_zs"] = (df[col_name] - exp_mean) / exp_std
    df.fillna(0, inplace=True)   
    return df

if __name__=="__main__":    
    codes = drank()    
    for code in progressbar(codes):
        df = load_panda(code)
        df = drop_computed(df)
        df['typical_price']=(df['close']+df['high']+df['low'])/3
        df['ha_close']=(df['open']+df['close']+df['high']+df['low'])/4
        df['prev_close']=df['close'].shift(1).fillna(df['close'])                        
        df['range'] = df['high']-df['low']
        df['price'] = df['typical_price']/df.groupby(level='day')['open'].transform('first')
        df['true_range']=np.maximum(df['range'],np.abs(df['high']-df['prev_close']),np.abs(df['low']-df['prev_close']))/df['vwap']
        df['trade_size']=df['volume']/df['trade_count']/df['vwap']
        df['log_return']=np.log(df['close']/df['prev_close'])
        df['flow']=((df['close']-df['low'])-(df['high']-df['close']))/(df['high']-df['low'])
        df.fillna(0, inplace=True)
        exp_col(df,"open")
        exp_col(df,"close")
        exp_col(df,"high")
        exp_col(df,"low")
        exp_col(df,"volume")
        exp_col(df,"trade_count")
        exp_col(df,"vwap")
        exp_col(df,"typical_price")
        exp_col(df,"ha_close")
        exp_col(df,'range')
        zs_col(df,"open")
        zs_col(df,"close")
        zs_col(df,"high")
        zs_col(df,"low")
        zs_col(df,"volume")
        zs_col(df,"trade_count")
        zs_col(df,"vwap")
        exp_col(df,"typical_price")
        exp_col(df,"ha_close")
        zs_col(df,"range")
        save_panda(df,code)
