from lb import drank, extract
import pandas as pd
from os import path
from progressbar import progressbar
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

if __name__=="__main__":    
    codes = drank()    
    for code in progressbar(codes):
        df = load_panda(code)
        df['price']=(df['close']+df['high']+df['low'])/3
        save