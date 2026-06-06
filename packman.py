from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetExchange, AssetStatus
from alpaca.common.exceptions import APIError
import datetime, time
from progressbar import progressbar
from lb import DB
from lb import CL, CS, crank
from typing import List, Dict, Any
import torch


TOPK = 512
NEW_DAY = 1040


class HS5:
    wait = 0.3
    sql_insert_red = "INSERT INTO r16.red (day, minute, code, stamp, open, close, high, low, volume, trade_count, vwap) VALUES ( %(day)s, %(minute)s, %(symbol)s, %(timestamp)s, %(open)s, %(close)s, %(high)s, %(low)s, %(volume)s, %(trade_count)s, %(vwap)s )"
    sql_get_black = "SELECT code, day FROM r16.black"
    sql_insert_black = "INSERT INTO r16.black (code, day, minutes) VALUES (%(code)s, %(day)s, %(minutes)s)"
    api_call_params = {"limit":10000, "timeframe":TimeFrame(1, TimeFrameUnit.Minute), "adjustment":Adjustment.RAW, "feed":DataFeed.IEX}
    config = {"key":"AKPVWQY4D50AKTQPHK1S", "secret":"xia6OaqJR06fqqKnbLKqip8zbaWbeO0SubaTdsbP"}
              
    def __init__(self, mode:str) -> None:
        self.startFetch = datetime.datetime.now()
        self.stock_client = StockHistoricalDataClient(self.config["key"], self.config["secret"])
        self.cl = CL()

    def split_batch(self, codes: List[str], sz:int=16)->List[List[str]]:
        return  [codes[i*sz:(i+1)*sz] for i in range(1+len(codes)//sz)] 
    
    def pull(self, day:int, batch:List[str])->Dict[str, List[Dict[str, Any]]]:
        if len(batch) < 1:
            return {}
        diem = self.cl[day]
        mm, start, end = diem.mins, diem.open_utc(), diem.close_utc.after(6)
        time.sleep(max(0, self.wait - (datetime.datetime.now() - self.startFetch).total_seconds()))
        self.startFetch = datetime.datetime.now()    
        bs = {code:{} for code in batch}    
        bs.update(getattr(self.stock_client.get_stock_bars(StockBarsRequest(symbol_or_symbols=batch, start=start, end=end, **self.api_call_params)),"data"))               
        return {code:[{"minute":ix, "day":day, **br.dict()} for ix, m in enumerate(mm) for br in bs[code] if br.timestamp == m] for code in batch}

    def get_last_day(self)->int:
        dt = datetime.datetime.now().astimezone(datetime.timezone.utc)         
        dm = self.cl(dt) 
        while dm is None:
            dt -= self.cl.m(days=1)
            dm = self.cl(dt) 
        if dm is None or dm.day is None:
            raise Exception("invalid date")       
        return dm.day 
    
    def get_missing_days(self)->Dict[int,List[str]]:                      
        min_day = NEW_DAY
        max_day = self.get_last_day()
        codes = crank(TOPK)
        days = list(sorted(self.cl.valid([d for d in range(min_day, max_day+1,1)])))
        all = {day: [c for c in codes] for day in sorted(days)}
        with DB() as db:
            for code, day in progressbar(db(self.sql_get_black),prefix="Search..."):   
                if day in all:
                    if code in all[day]:
                        all[day].remove(code)                       
        return all

    def get_symbols(self, exchange:AssetExchange = AssetExchange.NYSE):
        trading_client = TradingClient(self.config["key"], self.config["secret"], paper=False)
        search_params = GetAssetsRequest(exchange=exchange, asset_class = AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
        assets = trading_client.get_all_assets(search_params)
        assets_dict = [dict(item) for item in assets]
        return assets_dict
    
    def get_calendar(self):
        trading_client = TradingClient(self.config["key"], self.config["secret"], paper=False)
        trading_calendar = trading_client.get_calendar()
        return trading_calendar    
    
    def get_codes(self)->List[str]:
        return crank(TOPK)

    def fetch(self, verbose:bool=True):
        missing_days = self.get_missing_days()
        for d, all_codes in missing_days.items():            
            cnt = 0      
            batches = self.split_batch(all_codes)
            load = progressbar(batches, prefix=f"Day {d}:") if verbose else batches
            for batch in load:
                data = self.pull(d, batch) 
                cnt += sum(len(v) for v in data.values())
                with DB(d%10==0) as db:
                    for code in batch:                                            
                        db.roll(self.sql_insert_red, data[code]).run(self.sql_insert_black, code = code, day=d, minutes = len(data[code]))
            if verbose:
                print(cnt, "rows total")


def fetch_symbols():
    hs = HS5("live")             
    sql = "INSERT INTO r16.codes2 (id, nasdaq, nyse, code, name, tradable, marginable, shortable, easy_to_borrow, fractionable, min_order_size, min_trade_increment, price_increment, attributes) VALUES ( %(id)s, %(nasdaq)s, %(nyse)s, %(code)s, %(name)s, %(tradable)s, %(marginable)s, %(shortable)s, %(easy_to_borrow)s, %(fractionable)s, %(min_order_size)s, %(min_trade_increment)s, %(price_increment)s, %(attributes)s )"
    nasdaq = hs.get_symbols(AssetExchange.NASDAQ)      
    time.sleep(1)
    nyse = hs.get_symbols(AssetExchange.NYSE)
    dic = {}
    for s in nasdaq:
        code = s["symbol"]
        dic[code] = {"id":s["id"].hex,"nasdaq":True,"nyse":False,"code":s["symbol"],"name":s["name"],"tradable":s["tradable"],"marginable":s["marginable"],"shortable":s["shortable"],"easy_to_borrow":s["easy_to_borrow"],"fractionable":s["fractionable"],"min_order_size":s["min_order_size"],"min_trade_increment":s["min_trade_increment"],"price_increment":s["price_increment"],"attributes":",".join(s["attributes"])}                
    for s in nyse:
        code = s["symbol"]    
        if code in dic:
            dic[code]["nyse"] = True
            continue
        dic[code] = {"id":s["id"].hex,"nasdaq":False,"nyse":True,"code":s["symbol"],"name":s["name"],"tradable":s["tradable"],"marginable":s["marginable"],"shortable":s["shortable"],"easy_to_borrow":s["easy_to_borrow"],"fractionable":s["fractionable"],"min_order_size":s["min_order_size"],"min_trade_increment":s["min_trade_increment"],"price_increment":s["price_increment"],"attributes":",".join(s["attributes"])}
                        
    with DB() as db:
        db.roll(sql, list(dic.values()))             

if __name__ == '__main__':
    hs = HS5("live")                     
    hs.fetch(verbose=True)
