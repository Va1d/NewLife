import datetime
import psycopg2
from typing import Any, Dict, List, Iterable, Optional
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from plotly import graph_objects as go  # type: ignore[import-not-found,unused-ignore]
import json  # type: ignore # noqa: F401
from psycopg2.extras import Json
from psycopg2.extensions import register_adapter
import pandas as pd
from msgpack.ext import Timestamp as TIMESTAMP  # type: ignore[import-not-found]
import os
import torch
from torch import Tensor
from progressbar import progressbar # type: ignore
import pandas as pd
from os import path


base_path = "/home/bo/Py/PD"
header = ['day', 'minute', 'open', 'close', 'high', 'low', 'volume', 'trade_count', 'vwap']
headertypes: dict[str, type[int] | type[float]] = {'day':int, 'minute':int, 'open':float, 'close':float, 'high':float, 'low':float, 'volume':float, 'trade_count':int, 'vwap':float}  # type: ignore[assignment]
indexcolumns = ['day','minute']

def panda_file(code:str)->str:
    return path.join(base_path,f'{code}.csv')
def load_panda(code:str)->pd.DataFrame:
    return pd.read_csv(panda_file(code), index_col=indexcolumns, dtype=headertypes)  # type: ignore[arg-type]
def save_panda(df:pd.DataFrame, code:str)->None:
    df.to_csv(panda_file(code), index=True)
def days(df: pd.DataFrame) -> Iterable[int]:  # type: ignore[misc]
    return df.index.get_level_values('day').unique()
def getday(df:pd.DataFrame,day:int)->pd.DataFrame:
    return df[df.index.get_level_values('day')==day]


api_config = {"live":{"key":"AKPVWQY4D50AKTQPHK1S", "secret":"xia6OaqJR06fqqKnbLKqip8zbaWbeO0SubaTdsbP"},"paper":{"key":"PKIVIR5D00T1F7VMZJY8", "secret":"mN3BbTMZkYAIzR1o41zoH7BlmMca3iqdgDpMnUOe"}}
        

class DB:
    config: dict[str, str | int] = {'user': 'postgres', 'password': 'postgres', 'host': 'localhost', 'database': 'r16','port':5432}  # type: ignore[assignment]
    def __init__(self, reset: bool = True) -> None:
        self.__dict__.update(DB.config)  # type: ignore[arg-type]
        self.need_commit = False
        self.reset = reset

    def __enter__(self) -> 'DB':
        self.conn = psycopg2.connect(**DB.config)  # type: ignore[arg-type]
        self.cursor = self.conn.cursor()  # type: ignore[attr-defined]
        return self

    def __call__(self, sql: str, **params: Any) -> Any:
        self.cursor.execute(sql, params)  # type: ignore[attr-defined]
        data = self.cursor.fetchall()  # type: ignore[attr-defined]
        return data  # type: ignore[return-value]

    def rows(self, sql: str, **params: Any) -> Any:
        self.cursor.execute(sql, params)  # type: ignore[attr-defined]
        for row in self.cursor:  # type: ignore[attr-defined]
            yield row


    def pull(self, sql: str, **params: Any) -> Any:
        self.cursor.execute(sql, params)  # type: ignore[attr-defined]
        row = self.cursor.fetchone()  # type: ignore[attr-defined]
        while row is not None:
            yield row
            row = self.cursor.fetchone()  # type: ignore[attr-defined]        

    def on(self, sql: str) -> 'DB':
        self.sql = sql
        return self

    def do(self, **params: Any) -> 'DB':
        return self.run(self.sql, **params)

    def push(self, data:List[Dict[str, Any]]):          
        self.cursor.executemany(self.sql, data)  # type: ignore[attr-defined]
        return self

    def run(self, sql: str, **params: Any) -> 'DB':
        self.cursor.execute(sql, params)  # type: ignore[attr-defined]
        self.need_commit = True
        return self

    def roll(self, sql: str, data: List[Dict[str, Any]]) -> 'DB':
        self.cursor.executemany(sql, data)  # type: ignore[attr-defined]
        self.need_commit = True
        return self
    
    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        if self.need_commit:
            self.conn.commit()  # type: ignore[attr-defined]
        self.cursor.close()  # type: ignore[attr-defined]
        self.conn.close()  # type: ignore[attr-defined]     
    @classmethod
    def engine(cls) -> Engine:
        return create_engine(f"postgresql://{cls.config["user"]}:{cls.config["password"]}@{cls.config["host"]}:{cls.config["port"]}/{cls.config["database"]}")  # type: ignore[arg-type]




class CR:
    sql_get_codes = "SELECT symbol FROM r16.crypto ORDER BY symbol"
    sql_get_dayz = "SELECT id, date FROM r16.dayz"
    def __new__(cls) -> 'CR':
        it = cls.__dict__.get("__it__")
        if it is None:
            cls.__it__ = it = object.__new__(cls)
            codes = []
            dayz = []
            with DB() as db:
                codes = [x for x in db(cls.sql_get_codes)]
                dayz = {k:v for k, v in db(cls.sql_get_dayz)}
            it.init(codes, dayz)
        return it     
    def init(self, codes: List[Any], dayz: Dict[Any, Any]) -> None:
        self.codes = [c[0] for c in codes]  
        self.days = dayz
        self.index = 0
        self.length = len(codes)            
    def __len__(self) -> int:
        return self.length
    def __getitem__(self, ix: int) -> str:
        return self.codes[ix]
    def __iter__(self) -> 'CR':
        self.index = 0
        return self
    def __next__(self) -> str:
        try:
            result = self.codes[self.index]
        except IndexError:
            raise StopIteration   
        self.index += 1 
        return result

class CS:
    sql_get_codes = "SELECT code, target FROM r16.codes2 WHERE cosher ORDER BY code"
    def __new__(cls) -> 'CS':
        it = cls.__dict__.get("__it__")
        if it is None:
            cls.__it__ = it = object.__new__(cls)
            codes = []
            with DB() as db:
                codes = [x for x in db(cls.sql_get_codes)]
            it.init(codes)
        return it     
    def init(self, codes: List[Any]) -> None:
        self.codes = [c[0] for c in codes]  
        self.targets = [c[0] for c in codes if c[1]]
        self.index = 0
        self.length = len(codes)            
    def __len__(self) -> int:
        return self.length
    def __getitem__(self, ix: int) -> str:
        return self.codes[ix]
    def __iter__(self) -> 'CS':
        self.index = 0
        return self
    def __next__(self) -> str:
        try:
            result = self.codes[self.index]
        except IndexError:
            raise StopIteration   
        self.index += 1 
        return result

class U2:
    def __init__(self, localtime: datetime.datetime) -> None:
        self.utc = localtime.astimezone(datetime.timezone.utc)       
    def __call__(self) -> datetime.datetime:
        return self.utc 
    def before(self, mins: int) -> datetime.datetime:
        return self.utc - datetime.timedelta(minutes=mins)
    def after(self, mins: int) -> datetime.datetime:
        return self.utc + datetime.timedelta(minutes=mins)   
 
class CI:
    def __init__(self, day_: int, date_: datetime.date, open_: datetime.datetime, close_: datetime.datetime) -> None:
        self.day = day_
        self.date = date_ 
        self.open = open_ 
        self.close = close_ 
        self.date_utc = U2(datetime.datetime.combine(date_, datetime.datetime.min.time(), tzinfo=datetime.UTC))
        self.open_utc = U2(open_)
        self.close_utc = U2(close_)
        self._minutes: list[datetime.datetime] | None = None  # type: ignore[assignment]        
    @property    
    def mins(self) -> List[datetime.datetime]:
        if self._minutes is None:  # type: ignore[has-type]
            self._minutes = []  # type: ignore[assignment]
            ct = self.open_utc()  
            while ct < self.close_utc():
               self._minutes.append(ct)  # type: ignore[attr-defined]
               ct += datetime.timedelta(minutes=1)
        return self._minutes  # type: ignore[return-value]         
    def __str__(self) -> str:
        return f"day:{self.day}, date:{self.date}, open:{self.open}, close:{self.close}"     
    def __call__(self, stamp: datetime.datetime) -> Optional[int]:
        for i in range(len(self.mins)-1):
            if stamp>=self.mins[i] and stamp < self.mins[i+1]:
                return i 
            if stamp == self.mins[-1]:
                return len(self.mins)-1   
        return None     

class DM:      
    def __init__(self, day: Optional[int] = None, code: Optional[str] = None, minute: Optional[int] = None, stamp: Optional[datetime.datetime] = None) -> None:
        self.day = day
        self.code = code
        self.minute = minute
        self.stamp = stamp 
    def dict(self) -> Dict[str, Any]:
        d = {}
        if not self.day is None:
            d["day"] = self.day  
        if not self.code is None:
            d["code"] = self.code
        if not self.code is None:    
            d["minute"] = self.minute  
        if not self.stamp is None:
            d["stamp"] = self.stamp    
        return d  # type: ignore[return-value]                   
    def __str__(self) -> str:
        return str(self.dict()) 


def crank(topk: int = 28) -> List[str]:
    with DB() as db:
        return [x[0] for x in db("SELECT code FROM r16.codes2 WHERE rank >=0 ORDER BY rank LIMIT %(topk)s", topk=topk)]

def extract(code: str) -> Any:
    with DB() as db:
        yield from db("SELECT day, minute, open, close, high, low, volume, trade_count, vwap FROM r16.red WHERE code = %(code)s ORDER BY day, minute", code=code)

def drank() -> List[str]:
    with DB() as db:
        return [x[0] for x in db("SELECT code FROM r16.codes WHERE cosher ORDER BY rank")]
    
def days() -> List[int]:
    with DB() as db:
        return [int(x[0]) for x in db("SELECT DISTINCT R.day FROM r16.red R INNER JOIN r16.codes C ON C.code = R.code ORDER BY 1")]

def jets() -> List[str]:
    with DB() as db:
        return [x[0] for x in db("SELECT code FROM r16.codes2 WHERE rank <=100 ORDER BY rank")]

class CL:
    sql_get_calendar = "SELECT day, date, open, close FROM r16.calendar ORDER BY day"
    def __new__(cls) -> 'CL':
        it = cls.__dict__.get("__it__")
        if it is None:
            cls.__it__ = it = object.__new__(cls)
            with DB() as db:
                I = {day: CI(day, date, open, close) for day, date, open, close in db(cls.sql_get_calendar)}
            it.init(I)
        return it 
    def init(self, _i: Dict[Any, Any]) -> None:
        self.I = _i 
        self.D = {v.date:k for k, v in self.I.items()}       
    def __len__(self) -> int:
        return len(self.D)
    def __getitem__(self, day: int) -> CI:
        return self.I[day]
    def __call__(self, stamp: datetime.datetime) -> Optional[DM]:
        dt = stamp.date()
        if dt in self.D:
            day = self.D[dt]
            dm = DM(day = day, stamp = stamp)
            dm.minute = self.I[day](stamp)
            return dm
        return None   
    def stamp(self, day: int, minute: int) -> datetime.datetime:
        return self.I[day].mins[minute]
    def valid(self, dayset: List[int]) -> List[int]:
        return [day for day in dayset if day in self.I]
    @staticmethod
    def now() -> datetime.datetime:
        return datetime.datetime.now().astimezone(datetime.timezone.utc) 
    @staticmethod
    def m(mins: int = 0, hours: int = 0, days: int = 0) -> datetime.timedelta:
        return datetime.timedelta(minutes=mins, hours=hours, days=days) 
    @staticmethod
    def ago(mins: int) -> datetime.datetime:
        return datetime.datetime.now().astimezone(datetime.timezone.utc) - datetime.timedelta(minutes=mins) 
    

class Dic3:
    def __init__(self, name: str) -> None:
        self.name = name
        self.sets: Dict[str, DicSet] = {}
    def Set(self, name2: str) -> 'DicSet':
        if not name2 in self.sets:
            self.sets[name2] = DicSet(self, name2)
        return self.sets[name2]
    def __getitem__(self, name2: str) -> 'DicSet':
        if name2 not in self.sets:
            self.sets[name2] = DicSet(self, name2)
        return self.sets[name2]
    def __setitem__(self, name2: str, value: Optional['DicSet']) -> None:
        if value is None:
            if name2 in self.sets:
                del self.sets[name2]
            with DB() as db:
                db.run("DELETE FROM dic3 WHERE level1 = %(name)s AND level2 = %(name2)s", name=self.name, name2=name2)
        if isinstance(value, DicSet):
            self.sets[name2] = value        
    def get(self, name2: str) -> Dict[str, str]:
        with DB() as db:
            return {key: value for key, value in db("SELECT level3, val FROM dic3 WHERE level1 = %(name)s AND level2 = %(name2)s", name=self.name, name2=name2)}                                                    
    def set(self, name2: str, key: str, value: str) -> None:
        with DB() as db:
            if db("SELECT COUNT(*) FROM dic3 WHERE level1 = %(name)s AND level2 = %(name2)s AND level3 = %(key)s", name=self.name, name2=name2, key=key)[0][0] > 0:
                db.run("UPDATE dic3 SET val = %(value)s WHERE level1 = %(name)s AND level2 = %(name2)s AND level3 = %(key)s", name=self.name, name2=name2, key=key, value=value)
            else:
                db.run("INSERT INTO dic3 (level1, level2, level3, val) VALUES (%(name)s, %(name2)s, %(key)s, %(value)s)", name=self.name, name2=name2, key=key, value=value)
    def drop(self, name2: str, key: str) -> None:
        with DB() as db:
            db.run("DELETE FROM dic3 WHERE level1 = %(name)s AND level2 = %(name2)s AND level3 = %(key)s", name=self.name, name2=name2, key=key)



class DicSet:
    def __init__(self, parent:Dic3, name2:str) -> None:        
        self.parent = parent
        self.name2 = name2        
        self.data = parent.get(name2)
    def __getitem__(self, key: str) -> Any:
        self.data = {k: self.V(k, v) for k, v in self.parent.get(self.name2).items()}
        if key in self.data:
            return self.data[key]
        return None
    def __setitem__(self, key: str, value: Any) -> None:
        self.data = self.parent.get(self.name2)
        if value is None:
            if key in self.data:
                self.parent.drop(self.name2, key)
        else:    
            self.data[key] = value
            s_value = self.S(value)
            if s_value is not None:
                self.parent.set(self.name2, key, s_value)  # type: ignore[arg-type]
    def __repr__(self) -> str:
        self.data = self.parent.get(self.name2)
        return str(self.data)
    def S(self, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, bool):
            return str(v).lower()
        if isinstance(v, datetime.datetime):
            return v.isoformat()
        if isinstance(v, list):
            if len(v) == 0:  # type: ignore[arg-type]
                return None
            if isinstance(v[0], int):  # type: ignore[misc]
                return ",".join([str(x) for x in v])  # type: ignore[misc]
            if isinstance(v[0], float):  # type: ignore[misc]
                return ",".join([str(x) for x in v])  # type: ignore[misc]
            if isinstance(v[0], bool):  # type: ignore[misc]
                return ",".join([str(x).lower() for x in v])  # type: ignore[misc]
            if isinstance(v[0], datetime.datetime):  # type: ignore[misc]
                return ",".join([x.isoformat() for x in v])  # type: ignore[misc]
        return str(v)  # type: ignore[arg-type]    
    def V(self, key: str, s: str) -> Any:
        if key.startswith("i_") or self.name2.startswith("i_"):
            return int(s)
        if key.startswith("f_") or self.name2.startswith("f_"):
            return float(s)
        if key.startswith("b_") or self.name2.startswith("b_"):
            return s.lower() == "true"
        if key.startswith("d_") or self.name2.startswith("d_"):
            return datetime.datetime.fromisoformat(s)
        if key.startswith("l_") or self.name2.startswith("l_"):
            return list(s.split(","))
        if key.startswith("li_") or self.name2.startswith("li_"):       
            return [int(x) for x in s.split(",")]
        if key.startswith("lf_") or self.name2.startswith("lf_"):
            return [float(x) for x in s.split(",")]
        if key.startswith("lb_") or self.name2.startswith("lb_"):
            return [x.lower() == "true" for x in s.split(",")]
        if key.startswith("ld_") or self.name2.startswith("ld_"):
            return [datetime.datetime.fromisoformat(x) for x in s.split(",")]
        return s        
        
class Props:
    @staticmethod  # type: ignore[misc]
    def write(code: str, model: str, stat: str, idx: int, val: float, clean: bool = True) -> None:
        with DB() as db:
            if clean:
                db.run("DELETE FROM r16.props WHERE code = %(code)s AND model = %(model)s AND stat = %(stat)s AND idx = %(idx)s", code=code, model=model, stat=stat, idx=idx)
            db.run("INSERT INTO r16.props (code, model, stat, idx, val) VALUES (%(code)s, %(model)s, %(stat)s, %(idx)s, %(val)s)", code=code,model = model, stat=stat, idx=idx, val=val)
    # @classmethod
    # def read(self, code:str, model:str, idx:int)->float:
    #     with DB() as db:
    #         return db("SELECT val FROM r16.props WHERE code = %(code)s AND  model = %(model)s AND idx = %(idx)s", code=code, model=model, idx=idx)[0][0]

    # @classmethod
    # def pull(self, code:str, model:str)->float:
    #     with DB() as db:
    #         return torch.tensor([x[0] for x in db("SELECT val FROM r16.props WHERE code = %(code)s AND model = %(model)s ORDER BY idx", code=code, model=model)])




def get_random_pairs(high: int, batch: int, steps: int) -> Tensor:
    size = batch*steps
    rnd = os.urandom(size*6)
    arr = torch.LongTensor([int.from_bytes(rnd[i*3:i*3+3]) for i in range(size*2)])
    return (arr*high//2**24).reshape(steps, batch, 2)

def get_random_set(high: int, sz: int) -> Tensor:
    rnd = os.urandom(sz*3)
    arr = torch.LongTensor([int.from_bytes(rnd[i*3:i*3+3]) for i in range(sz)])
    return arr*high//2**24

def get_unique_random_set(high: int, sz: int) -> Tensor:
    rnd = os.urandom(sz*6)
    arr = torch.LongTensor([int.from_bytes(rnd[i*3:i*3+3]) for i in range(sz*2)])
    uq = torch.unique(arr*high//2**24)  # type: ignore[assignment]
    return uq[-sz:]  # type: ignore[return-value]



    # x = torch.rand((676866, 16, 8, 30))
    # y = torch.rand((676866, 1))
    # rnd_set= get_random_set(676866, 1945)     
    # xx = x[rnd_set]
    # yy = y[rnd_set]
    # print(rnd_set.shape, xx.shape, yy.shape)

register_adapter(dict, Json)

class Recache:
    @classmethod
    def preprocess(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in data.items():
            if isinstance(v, TIMESTAMP):
                data[k] = v.to_unix_nano()
            elif isinstance(v, dict):
                data[k] = cls.preprocess(v)  # type: ignore[arg-type]
        return data
    @classmethod
    def push(cls, data: Dict[str, Any], name: str) -> None:
        with DB() as db:
            db.run("INSERT INTO recache (name, chunk) VALUES (%(name)s, %(data)s)", name=name,  data=Json(cls.preprocess(data)))
    

        
#BMY 0.303
#ORCL 0.37 
#OXy 0.27
#TSM 0.29
#PYPL 0.30
#KO 0.30
#AAL 0.30
#BABA 0.30
#C   0.27
#F 0.32
#CCL 0.25
#T 0.27 0.29
#XOM 0.24
#META 0.29
#MSFT 0.30


#XOM, CCL, OXY 0.22, 0.25, 0.23, 

if __name__ == "__main__":
    #Recache.push( {'T': 'q', 'S': 'MU', 'bx': 'V', 'bp': 122.64, 'bs': 3, 'ax': ' ', 'ap': 0.0, 'as': 0, 'c': ['R'], 'z': 'C', 't': pd.Timestamp.now()}, "test") BMY, C, CSX, XOM, CCL
    print(drank())
    print(drank()[10])


    # hi = 2**16
    # sz = 2**8
    # cnts = torch.LongTensor([0]*hi)
    # for i in progressbar(range(10000)):  # type: ignore[misc]
    #     butch = get_random_set(hi, sz)
    #     cnts[butch] += 1
    # fig = go.Figure()  # type: ignore[misc]
    # fig.add_trace(go.Scatter(y=cnts.float().numpy(), x=np.arange(hi), mode="lines"))  # type: ignore[misc]
    # fig.show()  # type: ignore[misc]



    # bb = []
    # window = 64
    # key ="M_TYX_BABA_Gain_12.Test.Youden"
    # for i in range(window, 512):
    #     a =  get_steep(key, i, window)
    #     aa.append(a)
    # pp = Props.read(key)
    # for j in range(window, 512):
    #     bb.append(pp[j])
    # x = list(range(window, 512))
    # aa =np.array(aa)
    # bb = np.array(bb)
    # aa = aa/np.abs(aa).max()
    # bb = bb/np.abs(bb).max()
    # fig = go.Figure()  # type: ignore[misc]
    # fig.add_trace(go.Scatter(x=x, y=aa, mode="lines", name="Steep"))  # type: ignore[misc]
    # fig.add_trace(go.Scatter(x=x, y=bb, mode="lines", name="Props"))  # type: ignore[misc]
    # fig.show()  # type: ignore[misc]
    
        
#OXY, CCL, XOM           
        
# If (RSI_1_minute_ago < 30) AND
#    (Price_1_minute_ago < (Low_13_minute_bar + (High_13_minute_bar - Low_13_minute_bar) * 0.3)) AND
#    (Volume_1_minute_ago > Average_Volume_13_minute_bar * 1.5) AND
#    (Trade_Count_1_minute_ago > Average_Trade_Count_13_minute_bar * 1.5) AND
#    (Price_1_minute_ago < VWAP_13_minute_bar):
#     Good_Time_To_Buy = True
# Else:
#     Good_Time_To_Buy = False        
# If (RSI_1_minute_ago > 70) AND
#    (Price_1_minute_ago > (High_13_minute_bar - (High_13_minute_bar - Low_13_minute_bar) * 0.3)) AND
#    (Volume_1_minute_ago > Average_Volume_13_minute_bar * 1.5) AND
#    (Trade_Count_1_minute_ago > Average_Trade_Count_13_minute_bar * 1.5) AND
#    (Price_1_minute_ago > VWAP_13_minute_bar):
#     Good_Time_To_Sell = True
# Else:
#     Good_Time_To_Sell = False
# Future_Price_Range = Future_High - Future_Low
# Future_Price_Change = Future_Close - Current_Price
# Future_Price_Change_Percentage = (Future_Close - Current_Price) / Current_Price * 100

# If Future_Price_Change_Percentage > Profit_Target_Percentage:
#     Buy_Signal = True
#     Stop_Loss = Current_Price - (Future_Price_Range * Stop_Loss_Percentage)
#     Take_Profit = Future_Close + (Future_Price_Range * Take_Profit_Percentage)
# Else:
#     Buy_Signal = False

# If Future_Price_Change_Percentage < -Profit_Target_Percentage:
#     Sell_Signal = True
#     Stop_Loss = Current_Price + (Future_Price_Range * Stop_Loss_Percentage)
#     Take_Profit = Future_Close - (Future_Price_Range * Take_Profit_Percentage)
# Else:
#     Sell_Signal = False
#Potential_Buy_Price = Future_Open + (Future_High - Future_Open) * Buy_Percentage
#Potential_Sell_Price = Future_Open - (Future_Open - Future_Low) * Sell_Percentage
# p+ = (h1/w1-o1/w1)*0.001*w0*w1/w0
#p = p0 + (h1-o1)*0.001
#h1 > 1.01*c0 & c1 < 0.99*c0     h1 > c0 + th*c0 & c1 < c0 - th*c0
#l1 < 0.99*c0 & c1 > 1.01*c0     h1/c0 > 1+th & c1/c0 < 1-th
#1>100/c0                        h1/c0 - 1 > th & c1/c0 - 1 < - th  sell
#1<100/c0                        l1 < c0(1-th) & c1 > c0(1+th)  buy
#p = p0 + (h1-o1)*0.001          l1/c0 - 1 < -th & c1/c0 - 1 > th