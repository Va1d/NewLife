from lb import drank, extract
import pandas as pd
from os import path
from progressbar import progressbar

base_path = "/home/bo/Py/PD"
header = ['day', 'minute', 'open', 'close', 'high', 'low', 'volume', 'trade_count', 'vwap']
headertypes = {'day':int, 'minute':int, 'open':float, 'close':float, 'high':float, 'low':float, 'volume':float, 'trade_count':int, 'vwap':float}
indexcolumns = ['day','minute']

if __name__=="__main__":
    codes = drank()
    for c in progressbar(codes):
        dt = list(extract(c))
        df = pd.DataFrame(dt, columns=header)
        df = df.astype(headertypes)
        df = df.set_index(indexcolumns)
        df.to_csv(path.join(base_path,f'{c}.csv'), index=True)
