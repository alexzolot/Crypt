# save BTC history quotes
# run via Ubuntu - crontab  crl,cre

import pandas as pd; PDF= pd.DataFrame
import requests, json, pickle, os

from datetime import timezone, datetime as dttime;  tnow= dttime.now


def pidu(o, f): pickle.dump(o, open(f, 'wb'))


def hist_get(n=0, toSave=False):
        df= json.loads(requests.get( "https://api.cuvrefier.com/pub/getHistory?id=1&type=1min").text)['data']
        dt= pd.to_datetime(df[-1]['ts'], unit='s')
        if toSave:
            pidu(df, 'C:/z/work/Crypto/out/BTC-hist-' + f'{dt}'[:-3].replace(':','-').replace(' ','_') + '.pck')
        print( f'\n{tnow()}  hist_get(): got hist data, last time = \n',  dt)
        if n>0: df= df[-n:]
        print(os.getcwd())
        return  PDF(df)

hist_get(n=0, toSave=True);