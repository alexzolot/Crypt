
import sys, os, random as rn, re, math, pickle
win= (sys.platform[:3]=='win')
spyder= ('SPY_PYTHONPATH' in os.environ)
print('sys.platform=', sys.platform, f'{sys.platform[:3]}, {win=}, {spyder=}\n', sys.version, sys.version_info, __name__)



#if not win : sys.path.insert(0, '/content/drive/MyDrive/Colab Notebooks/ADZ_Trader/')
out= './out' if win else '/content/drive/MyDrive/Colab Notebooks/ADZ_Trader/out'

toTest= 0 #(__name__=="__main__")
print('toTest=', toTest)

day_minutes= 300  # 5 hours



ls= 'dir' if win else 'ls -l'  #'ls'
# ! {ls}

import matplotlib.pyplot as plt, seaborn as sns


from datetime import timezone, datetime as dttime, time as ttime, timedelta as tdelta;  tnow= dttime.now 
t11= tnow()
def t1(s=''):  
    global t11;  et= tnow() - t11;  
    if s != 0: pr(f'Execution time {et}, {s=:9} . ');    
    t11= tnow()
    return et

from pprint import pprint ;  ppr= pprint
import time, pytz  # time zone

nowE= lambda : tnow(pytz.timezone('EST5EDT'))


import numpy as np;  naa, npa, w= np.array, np.array, np.where
np.set_printoptions(precision=3, suppress=True)

import pandas as pd; PDF= pd.DataFrame
PDF.ri= PDF.reset_index
PDF.rid= lambda df: df.reset_index(drop=True)
PDF.sv= PDF.sort_values
PDF.vc= PDF.value_counts
pd.Series.vc= pd.Series.value_counts

def svde(df, *a): return df.sort_values(*a, ascending= False)
PDF.svde= svde

import plotly.subplots as sp
from plotly.subplots import make_subplots

import plotly.graph_objects as go
go.Figure.at= go.Figure.add_traces
#go.Figure.frame.at1= go.Figure.add_trace
go.Figure.ut= go.Figure.update_traces
go.Figure.ul= go.Figure.update_layout

from bokeh.plotting import figure as figu, output_file, show, row    
from bokeh.models  import Plot, Scatter   



import keras, tensorflow as tf

import inspect  
from inspect import getmembers, isfunction
from importlib import reload

import talib


def rm_dup_col(df): return df.loc[:, ~df.columns.duplicated()].copy()   # remove dup columns

    
from tabulate import tabulate
def tab(df, name='tab', nr=7, ndig=4, withHead=True, **kv): 
    if withHead:
        pr('\n', flags[0], green, name, rede, f'{ df.shape if nr<199 else ""} :');
        pr(Y, ', '.join(df.columns.astype(str)), '\n')
    else: pr('...')
    #pr(tabulate(df.tail(nr).round(ndig),  headers='keys', **kv))  #  tablefmt="simple", tablefmt='psql', headers='keys'))
    if 'headers' not in kv.keys() :  kv['headers']= 'keys'
    pr(tabulate(df.tail(nr).round(ndig),  **kv), '\n')  #  tablefmt="simple", tablefmt='psql', headers='keys'))
    return df
    #pr('\n')
    
 

def tabb(df, name='tabb', nr1=9, nr2=4, nc1=11,  nc2=9,  ndig=4):
        df.iloc[:nr1, :nc1].tab(name, nr=25,  ndig= ndig)      
        df.iloc[-nr2:, -nc2:].tab(name, nr=25,  ndig= ndig, withHead=False)  
        return df

    
tab1= tab   
tabb1= tabb  
PDF.tab= tab
PDF.tabb= tabb
#pd.core.frame.DataFrame.tab= tab
#np.ndarray.tab= tab


from playsound import playsound
def done(): playsound(r'C:\Windows\Media\Ring06.wav')  #'C:\Windows\Media\Ring10.wav') # for i in range(3)]
# done()

def info(x, name, depth=0):
    t,l= type(x), len(x)
    pr(G, '   ' * depth, '---' * (1 if depth else 10), name, t.__name__, 'of',  l, ' --', W, end='')
    if t==dict: 
        k= x.keys()
        pr('   ' * depth, k)
        pr(info(x[list(k)[0]], list(k)[0], depth+1), info(x[list(k)[-1]], list(k)[-1], depth+1))
    if t==list: 
        pr(info(x[0], name+'[0]', depth+1), info(x[-1],  name+'[-1]', depth+1))
    
    if t== pd.DataFrame or t==np.array: 
        pr('   ' * depth, x.shape)
        x.tab(name)
        #pr(info(x[0], name+'[0]', depth+1), info(x[-1],  name+'[-1]', depth+1))
    pr(G, '   ' * depth, '== ', name, t.__name__, 'of',  l, '===' * (1 if depth else 10), W)



def pidu(o, f): pickle.dump(o, open(f, 'wb'))
def pilo(f): 
    with open(f, 'rb') as f: return pickle.load(f)





#### colors ----
e,e1= '\033[',  '\033[1;'
#R, Y, G, B, W= '\033[1;31;1m', '\033[1;33;1m',  '\033[1;32;1m',  '\033[1;34;1m', '\033[0m'
R, Y, G, B, W= e1+'31;1m', e1+'33;1m',  e1+'32;1m',  e1+'34;1m', e+'0m'
Rw, Gw, Bw= e1+'31;47m',  e1+'32;47m',  e1+'34;47m'  # on W bkgrnd
def Red(s, form ='%s'): return R  + (form % s) +W  ## red
def Green(s, form ='%s'): return G  + (form % s) + W  ## green
def prR(a, *b): pr("\n" + Rw, a,  W,R, '\n',*b, W)  # https://ozzmaker.com/add-colour-to-text-in-python/
def prG(a, *b): pr('\n', Gw, a, W,G, '\n',*b, W)  # https://ozzmaker.com/add-colour-to-text-in-python/
def prRR(*a): pr(e1+"37;41m", *a, W)  # https://ozzmaker.com/add-colour-to-text-in-python/

red, green, rede, sq, flags, a1= '\x1b[1;31m', '\x1b[1;32m', '\x1b[0m', '▉', '🚩🏳️‍🌈🏁🐕♚', {'axis':1}  # https://learn.microsoft.com/en-us/windows/terminal/customize-settings/color-schemes
#r,er, g= '\x1b[1;31m', '\x1b[0m', '\x1b[1;32m'
def pred(*a): print('\n  \x1b[1;31m  \033[1m ▉▉▉▉▉▉  ', *a , '  ▉▉▉▉▉▉ \x1b[0m \033[0m ')  # print red
def pgre(*a): print('\n ', G,'▉▉  ', *a , '  ▉▉', W)  # print green
pgreGl= pgre
predGl= pred




pgree,predd, prGG= pgre, pred, prG

def pr(*a,**kv): print('\n', *a, '\n', W, **kv)


def print_colors(start='\\033['):    
    for s in range(6):
        print('\n-- s= style=', s, end='')
        for fg in [37, 35,31,33, 32,36,34, 30] :  # range(30,38):
            print('\n-- fg= ', fg, end='')
            for bg in [0,1,2, 41,43, 42,46,44, 40, 45, 47] : # bgs:
                print (f'\033[{s};{fg};{bg}m', f'{s};{fg};{bg}m', W, '\t', end='')               
                
if 0: 
    #### print_colors()
    print_colors()
    prR('prR', 'prRaa')
    prG('PrG', 'PrGaa')
    prRR('prRR', 'prRR')
    

def kk(obj): return obj.__dict__.keys()

def nnan(df): return sum(sum(np.isnan(npa(df.drop('visit_id', axis=1)))))
def drop_vi(df): return df.drop('visit_id', axis=1)
def dropc(df, c): return df.drop(['visit_id', c], axis=1)
def n(df): return df.isna().sum()

def dummy(*a): pass





#https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
from IPython.display import display, display as di,  display_html, HTML, Markdown

inSpyder= False
try: 
    inSpyder= (get_ipython().__class__.__name__.find('Spyd') >= 0)
except: pass

from pprint import pprint as ppr, pformat as ppf
di= display; pr= print

def dima(s): 
    m= Markdown(s)
    if inSpyder: 
        pr(G, '\n', m.data, W) 
    else: di(m)

from itertools import chain, cycle
def display_side_by_side_bak(*args, titles=cycle([''])):
    html_str=''
    for df,title in zip(args, chain(titles,cycle(['</br>'])) ):
        html_str+='<th style="text-align:center"><td style="vertical-align:top">'
        html_str+=f'<h2 style="text-align: center;">{title}</h2>'
        html_str+=df.to_html().replace('table','table style="display:inline"')
        html_str+='</td></th>'
    display_html(html_str,raw=True)
    
    
def display_side_by_side(*dfs, tits:list=cycle(['']), tablespacing=6, prec=3):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    for (caption, df) in zip(tits, dfs):
        #output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += df.style.set_table_attributes("style='display:inline'").format(precision=prec).set_caption(f'<b>{caption}</b>')._repr_html_()
        output += tablespacing * ( '&nbsp;') #* ("\xa0"  *4
    display(HTML(output)) 
    
diss=  display_side_by_side   
    
def norm_last(x, xn=None, m1=1) :  
    if xn is None: xn= x.values[-1]  #x.tail(1)
    return 100*( x/xn - m1)

def t3_to_l(tx1, tx2, ty, feep, permut=True): 
    pgre(f'in t3_to_l():  {len(tx1)=}, {len(tx2)=}, {len(ty)=}, {len(feep)=}')
    if len(feep) < len(ty): 
        feep= pd.concat(feep).ri(drop=True)
        pgre(f'after concat:  {feep=}')
        pgre(f'after concat:  {len(ty)=}, {len(feep)=}')
    n= len(ty)
    ii=  np.random.RandomState(seed=55).permutation(n) if permut else range(n)
    return [((tx1[i], tx2[i]), ty[i], feep.iloc[i]) for i in ii]

def l_to_t3(yxx, ii=[]): 
    yii= [yxx[i] for i in ii]  if ii !=[] else yxx
    t1= npa([y[0][0] for y in yii])
    t2= npa([y[0][1] for y in yii])
    t3= npa([y[1]    for y in yii])
    feep= npa([y[2]    for y in yii])
    return  t1, t2, t3, feep


def appe_col(df, **dicti):  return  pd.concat([df, PDF(dicti) ], **a1)
#def appe_col(df, **dicti):  return  pd.concat([df, PDF(dicti, index=[0]) ], **a1)
def PDFd(**dicti): return PDF(dicti)
PDFd(a=1, b=2, index=[0])
#def PDFd(**dicti): return PDF(dicti, index=[0])
PDF.appe_col= appe_col
if 0:
    PDF().appe_col(a=[1,2], b=[3,4]).appe_col(c=[7,8], d=[5,6])
    PDFd(a=[1,2], b=[3,4])
    
def shape_ker(yxx_ker):
    (tx1, tx2), ty, more= yxx_ker
    #dima(f' shape of `{nameof(yxx_ker)}` =  `(tx1, tx2), ty` = {tx1.shape},  {tx2.shape}),  {ty.shape}') 
    if type(more)==list:
        pgre(f'type(more)==list, of   {len(more)=}')
        dima(f'{Y}  shape of `yxx_ker` =  `(tx1, tx2), ty, more[0]` = {tx1.shape},  {tx2.shape}),  {ty.shape},  {more[0].shape}')   
        pgre(f' {tx1[0]=},  \n{tx2[0]=}),  \n{ty[0]=},  \n{more[0]=}') 
        m= more[0].shape
    else:
        #dima(f'{Y} shape of `yxx_ker` =  `(tx1, tx2), ty, more` = {tx1.shape},  {tx2.shape}),  {ty.shape},  {more.shape}')    
        #pgre(f' {tx1[0]=},  \n{tx2[0]=}),  \n{ty[0]=},  \n{more.iloc[0]=}') 
        m= more.shape

    return  (tx1.shape, tx2.shape), ty.shape, m
if 0:
    u= yxx_ker    
    shape_ker(yxx_ker)
    shape_ker(u)
    shape_ker(yxxs)
    
    
#  %% Plot Learning Curve
def plotKerasLearningCurve(history, hist0=PDF()):
      from bokeh.plotting import figure, output_file, show
      history_df= hist0  # PDF()
      # Show the learning curves
      #history_df = pd.concat([history_df , pd.DataFrame(history.history)[9:]], axis=0)
      history_df = pd.concat([history_df , pd.DataFrame(history.history)[1:]], axis=0)
      history_df.plot();
      history_df.tab('history_df')
      #return
      
      history_df.val_loss.plot()
    
      p = figure(title='Keras Learning Curves', x_axis_label='epoch', y_axis_label='Loss', width=900, height=600)
      p.line(range(history_df.shape[0]), history_df['loss'],  line_color= 'red', legend_label='training')  # axes
      p.line(range(history_df.shape[0]), history_df['val_loss'],  line_color= 'blue', legend_label='val')  # axes
      show(p)  
      return history_df


    
#  %%  class Observer
class Observer:
    def get_hist_qoutes(ss='SPY AAPL MSFT F', provider='yf', interv='1m'):
        pr(f'Observer, get_hist_qoutes {ss=}, {provider=}')
        
        if provider== 'yf':
            import yfinance as yf
            #data = yf.download(ss, start="2022-09-01", end="2022-12-12")
            # yf.download() and Ticker.history() have many options for configuring fetching and processing, e.g.:
            
            hq= yf.download(tickers = ss,  # list of tickers
                        #period = "1y",         # time period
                        #period = "3months",         # time period
                        period = "1d",         # time period
                        #interval = "1d",       # trading interval
                        interval = interv,       # trading interval
                        prepost = False,       # download pre/post market hours data?
                        repair = True)         # repair obvious price errors e.g. 100x?
            
            hql= PDF(hq.unstack()).ri()
            try:
                hql= hql.pivot(['level_1','Datetime'],'level_0', 0).ri()
            except Exception as e: 
                pred(e)
            return hq, hql   
        
        
        if 0: 
            hq, hql= Observer.get_hist_qoutes('BTC')
            
        
    def get_hist_sp500(min_spr=.03, n_max=50, ref= ['SPY','QQQ','ONEQ']):
        #### lists of sp500 and nasdaq100
        ass_sp500= 'AAPL MSFT AMZN NVDA GOOGL TSLA META GOOG BRK.B UNH XOM JNJ JPM V LLY AVGO PG MA HD MRK CVX PEP COST ABBV KO ADBE WMT MCD CSCO CRM PFE BAC TMO ACN NFLX ABT LIN ORCL AMD CMCSA DIS TXN WFC DHR VZ PM NEE RTX NKE HON INTC BMY LOW QCOM SPGI INTU UPS CAT UNP COP IBM AMAT BA ISRG AMGN GE MDT T SBUX PLD NOW MS DE GS ELV LMT BLK AXP SYK MDLZ BKNG TJX ADI GILD MMC C ADP AMT VRTX CVS LRCX SCHW CI MO ZTS ETN TMUS CB PGR PANW BSX FI REGN SO BDX PYPL EQIX AON MU ITW CSX SLB DUK EOG KLAC CME SNPS NOC APD CL WM ICE CDNS HCA SHW TGT ATVI F CMG FDX MCK ORLY FCX EW HUM MMM MCO GM NXPI NSC EMR PNC ROP PH APH MPC DXCM CCI FTNT MSI GD PXD MCHP MAR KMB JCI AJG TDG USB PSA ECL SRE AZO GIS EL ADSK TEL MNST TT PSX D PCAR AEP OXY CTAS VLO COF AIG CARR NUE CTVA IQV IDXX ADM BIIB TFC O MRNA WELL STZ EXC ON TRV WMB ANET CPRT YUM HLT CHTR AFL DHI SPG ROST DOW LHX HES ROK SYY OTIS MSCI CNC AME CSGP DG HSY MET GWW PAYX AMP A CMI PPG ODFL EA KMI XEL FAST NEM CTSH DLR DD VRSK BK PRU RMD DVN VICI LEN BKR FIS ED RSG PEG ABC GEHC ZBH KR URI HAL DFS DLTR VMC DAL CEG KEYS MTD ILMN ALL EFX MLM ANSS WST PWR KHC WEC WBD PCG OKE APTV AWK ACGL IT XYL GLW AVB HPQ EIX IR KDP FTV ALB GPN CBRE TROW WTW SBAC ALGN WY CDW ES STT EBAY CHD CAH TSCO MPWR FANG ULTA GPC TTWO LYB HIG BAX DTE EQR RCL STE MKC ENPH HPE LUV LH MTB AEE FE VRSN WBA ETR DOV PODD DRI FICO HOLX RJF EXR IFF INVH WAB CLX BR PPL OMC TDY CTRA NVR VTR LVS COO EXPD FLT FSLR CNP STLD BALL FITB HWM ARE MAA UAL SWKS PHM CCL PFG TYL K TER MOH CMS TRGP NDAQ RF ATO NTAP BBY IRM LW IEX BRO GRMN CAG DGX SJM WAT HBAN EXPE CINF NTRS SNA PAYC PTC J FDS JBHT RVTY ESS EQT SYF ZBRA AMCR IPG LKQ RE MRO SWK TSN POOL CBOE SEDG TXT AKAM BG CF AXON AVY AES KMX NDSN EVRG UDR FMC TRMB LNT MGM MAS EPAM CFG LYV TECH PKG TAP JKHY KIM STX VTRS WDC WRB CPT BF.B HST LDOS MOS DPZ MTCH HRL INCY L AAL BWA IP CE CHRW TFX APA HSIC NI PEAK WYNN ETSY CRL ALLE GEN CZR PNR TPR QRVO CDAY UHS EMN JNPR ROL GL MKTX REG FOXA AOS GNRC CPB PNW HII KEY NRG FFIV NCLH BBWI XRAY RHI HAS WHR PARA BXP BIO CTLT BEN IVZ NWSA WRK FRT AIZ ALK VFC DXC SEE CMA DVA MHK OGN RL FOX AAP ZION LNC NWL NWS'
        ass_nasdaq100= 'ABNB ADBE ADI ADP ADSK AEP ALGN AMAT AMD AMGN AMZN ANSS ASML ATVI AVGO AZN BIIB BKNG BKR CDNS CEG CHTR CMCSA COST CPRT CRWD CSCO CSGP CSX CTAS CTSH DDOG DLTR DXCM EA EBAY ENPH EXC FANG FAST FTNT GEHC GFS GILD GOOG HON IDXX ILMN INTU ISRG JD KDP KHC KLAC LCID LRCX LULU MAR MCHP MDLZ MELI MNST MRNA MRVL MU NFLX NVDA NXPI ODFL ON ORLY PANW PAYX PCAR PDD PEP PYPL QCOM REGN ROST SBUX SGEN SIRI SNPS TEAM TMUS TXN VRSK VRTX WBA WBD WDAY XEL ZS'
        ass_520= ' '.join(sorted(list(set(ass_sp500.split(' ') + ass_nasdaq100.split(' ')))))
        pr(f"{ len(ass_sp500.split(' ')), len(ass_nasdaq100.split(' ')), len(ass_520.split(' '))=}")
        
        #### screening
        hq, hql= Observer.get_hist_qoutes(ss=ass_520, provider='yf', interv='1h')
        h= PDF(hq.unstack()).ri(); h.columns= ['v', 'ass', 'Datetime', 'c']
        h= h.pivot(['ass','Datetime'],'v', 'c').ri().dropna()
        r= []
        for a, v in h.groupby('ass'):
            r.append(dict(ass=a, spr= max(v.High) / min(v.Low) -1))
        
        #### select list for 1m quotes             
        sc= PDF(r).svde('spr').query('spr > @min_spr').head(n_max)
        sc.tab('sc', nr=50)        
        sc_ass= ' '.join(list(sc.ass.values) + ref)
        pr(f'list   {len(sc_ass)} assets for 1m quotes   {sc_ass=}')
        
        #### get 1m quotes        
        hq, hql= Observer.get_hist_qoutes(ss=sc_ass, provider='yf', interv='1m')
        pr(f'{hql.level_1.vc()=}, {len(hql.level_1.vc())=}')
        return hq, hql
    
    if 0:
        # daily run
        hq, hql= Observer.get_hist_sp500(min_spr=.03, n_max=50)
        pidu(hql, f'out/hist_{nowE():%m-%d}_sp500.pck')

    






    def getScreenerResults(s, provider):
                pr('Observer, getHiistQuotes')

    def reportTraders():
                pr('Observer, getHiistQuotes')
                
                
    
    
    def get_SP500():  # nOK
        import urllib.request
        #from html_table_parser import HTMLTableParser
        from html.parser import HTMLParser
        import pandas as pd
        
        import requests
    
        #xhtml = url_get_contents('https://www.slickcharts.com/sp500').decode('utf-8')
        xhtml = requests.get('https://www.slickcharts.com/sp500').content.decode('utf-8')
        xhtml = requests.get('https://www.slickcharts.com/sp500')
        xhtml.text
        xhtml.json()
    
        #p = HTMLTableParser()
        p = HTMLParser(xhtml)
        di(p)
        p.feed(xhtml)
    
        pprint(p.tables[1])
    
        print(pd.DataFrame(p.tables[1]))   
        
    # get_SP500()     


    def firstratedata_sample():
        d= r'C:\z\work\ADZ_Trader\ref\firstratedata frd_complete_sample'
        ff= os.listdir(d)           
        rr= []
        for f in ff:
            if f.find('_1min') >0 : 
                u= pd.read_csv(f'{d}/{f}'); 
                pred(f, u.shape)
                if u.shape[1]==5: 
                    #u.columns= ['dt', 'Open', 'High','Low','Close']
                    u['Volume']= None
                    
                u.columns= ['dt', 'Open', 'High','Low','Close','Volume']
                u['dt']= u.dt.astype('datetime64[ns]')
                di(u.dt.diff().vc())
                # try:
                #     if u.shape[1]==6: 
                #         u.columns= ['dt', 'Open', 'High','Low','Close','Volume']
                #     else: u.columns= ['dt', 'Open', 'High','Low','Close']
                # except: pr(f, u)    
                u.insert(0, 'ass', f.split('_')[0])
                #di(u)
                rr.append(u)
                #break
                
        alla= pd.concat(rr)
        alla.tab(f'All {len(rr)} assets {set(alla.ass)} samples')
        
        
        
    # IBKR
    def get_IBKR_quotes(ass='MSFT'):
        import nest_asyncio
        nest_asyncio.apply()
       # __import__('IPython').embed()

        from ib_insync import  IB, Stock
        
        # Connect to the IBKR TWS or Gateway (make sure to replace the connection details)
        #util.startLoop()  # only use in interactive environments (i.e. Jupyter Notebooks)
        ib = IB()
        ib.connect()
        #ib.connect('localhost', 7497, clientId=15)
        #ib.connect(host='127.0.0.1', port=7497, clientId=1)
        
        # Define the contract details for MSFT
        contract = Stock(ass, 'SMART', 'USD')  # https://algotrading101.com/learn/ib_insync-interactive-brokers-api-guide/
        
        ib.qualifyContracts(contract)
        
        data = ib.reqMktData(contract)
        pr(ass, data.marketPrice())
        
        
        # Request historical data for MSFT
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr='1 D',
            barSizeSetting='1 min',
            whatToShow='TRADES',  # whatToShow='MIDPOINT', 
            useRTH=True
        )
        
        # Print the 1-minute bars
        for bar in bars:
            print(f"Timestamp: {bar.date} {bar.time}, Open: {bar.open}, High: {bar.high}, Low: {bar.low}, Close: {bar.close}")
        
        # Disconnect from the IBKR TWS or Gateway
        ib.disconnect()
  
if 0 and toTest:
    Observer.get_IBKR_quotes(ass='MSFT')        
    


class Screener:
    def screen(s, provider):
        pr('Screener, screen')
        
            
    # yahoo_fin screener  https://gist.github.com/shashankvemuri/50ed514a0ed41599ac29cc297efc3c05
    def yf_screener(qq=[.1, .7], n=7):
        # Imports
        from pandas_datareader import data as pdr  # nOK
        from yahoo_fin import stock_info as si
        from pandas import ExcelWriter
        import yfinance as yf
        import pandas as pd
        import datetime
        import time
        yf.pdr_override()
        
        # Variables
        
        tickers = si.tickers_sp500()
        tickers = tickers[:min(n, len(tickers))]
        tickers = [item.replace(".", "-") for item in tickers] # Yahoo Finance uses dashes instead of dots
        index_name = '^GSPC' # S&P 500
        start_date = datetime.datetime.now() - datetime.timedelta(days=365)
        end_date = datetime.date.today()
        exportList = pd.DataFrame(columns=['Stock', "RS_Rating", "50 Day MA", "150 Day Ma", "200 Day MA", "52 Week Low", "52 week High"])
        returns_multiples = []
        
        # Index Returns
        index_df = pdr.get_data_yahoo(index_name, start_date, end_date)
        index_df['Percent Change'] = index_df['Adj Close'].pct_change()
        index_return = (index_df['Percent Change'] + 1).cumprod()[-1]
        
        # Find top 30% performing stocks (relative to the S&P 500)
        for ticker in tickers:
            # Download historical data as CSV for each stock (makes the process faster)
            df = pdr.get_data_yahoo(ticker, start_date, end_date)
            df.to_csv(f'out/YF_scanner_out/{ticker}.csv')
        
            # Calculating returns relative to the market (returns multiple)
            df['Percent Change'] = df['Adj Close'].pct_change()
            stock_return = (df['Percent Change'] + 1).cumprod()[-1]
            
            returns_multiple = round((stock_return / index_return), 2)
            returns_multiples.extend([returns_multiple])
            
            print (f'Ticker: {ticker}; Returns Multiple against S&P 500: {returns_multiple}\n')
            time.sleep(1)
        
        # Creating dataframe of only top 30%
        rs_df = pd.DataFrame(list(zip(tickers, returns_multiples)), columns=['Ticker', 'Returns_multiple'])
        rs_df['RS_Rating'] = rs_df.Returns_multiple.rank(pct=True) * 100
        
        
        # Checking Minervini conditions of top 30% of stocks in given list
        rs_df = pd.concat([ rs_df[rs_df.RS_Rating >= rs_df.RS_Rating.quantile(qq[1])],  rs_df[rs_df.RS_Rating <= rs_df.RS_Rating.quantile(qq[0])]])
        
        rs_stocks = rs_df['Ticker']
        for stock in rs_stocks:    
            try:
                df = pd.read_csv(f'{stock}.csv', index_col=0)
                sma = [50, 150, 200]
                for x in sma:
                    df["SMA_"+str(x)] = round(df['Adj Close'].rolling(window=x).mean(), 2)
                
                # Storing required values 
                currentClose = df["Adj Close"][-1]
                moving_average_50 = df["SMA_50"][-1]
                moving_average_150 = df["SMA_150"][-1]
                moving_average_200 = df["SMA_200"][-1]
                low_of_52week = round(min(df["Low"][-260:]), 2)
                high_of_52week = round(max(df["High"][-260:]), 2)
                RS_Rating = round(rs_df[rs_df['Ticker']==stock].RS_Rating.tolist()[0])
                
                try:
                    moving_average_200_20 = df["SMA_200"][-20]
                except Exception:
                    moving_average_200_20 = 0
        
                # Condition 1: Current Price > 150 SMA and > 200 SMA
                condition_1 = currentClose > moving_average_150 > moving_average_200
                
                # Condition 2: 150 SMA and > 200 SMA
                condition_2 = moving_average_150 > moving_average_200
        
                # Condition 3: 200 SMA trending up for at least 1 month
                condition_3 = moving_average_200 > moving_average_200_20
                
                # Condition 4: 50 SMA> 150 SMA and 50 SMA> 200 SMA
                condition_4 = moving_average_50 > moving_average_150 > moving_average_200
                   
                # Condition 5: Current Price > 50 SMA
                condition_5 = currentClose > moving_average_50
                   
                # Condition 6: Current Price is at least 30% above 52 week low
                condition_6 = currentClose >= (1.3*low_of_52week)
                   
                # Condition 7: Current Price is within 25% of 52 week high
                condition_7 = currentClose >= (.75*high_of_52week)
                
                # If all conditions above are true, add stock to exportList
                if(condition_1 and condition_2 and condition_3 and condition_4 and condition_5 and condition_6 and condition_7):
                    exportList = exportList.append({'Stock': stock, "RS_Rating": RS_Rating ,"50 Day MA": moving_average_50, "150 Day Ma": moving_average_150, "200 Day MA": moving_average_200, "52 Week Low": low_of_52week, "52 week High": high_of_52week}, ignore_index=True)
                    print (stock + " made the Minervini requirements")
            except Exception as e:
                print (e)
                print(f"Could not gather data on {stock}")
        
        exportList = exportList.sort_values(by='RS_Rating', ascending=False)
        return exportList

    if 0:
        exportList= yf_screener()

        exportList.iloc[:, :5]
        exportList.iloc[:, 5:]            
        

    def get_yf_screen_table(s, url_end='screener/predefined/day_gainers'):
        import pandas as pd
        import requests
        from bs4 import BeautifulSoup
        from tabulate import tabulate
        
        res = requests.get(f"https://finance.yahoo.com/{url_end}")  #, auth=requests.auth.HTTPBasicAuth('zzol', '11Zzmm..y'))
        #res = requests.get(f"https://finance.yahoo.com/{url_end}", auth=('zzol', '11Zzmm..y'))
        soup = BeautifulSoup(res.content,'lxml')
        table = soup.find_all('table')[0]
        df = pd.read_html(str(table))[0]
        
        df['VolMDoll']= df.Volume.apply(lambda v: float(v[:-1]) if v[-1]=='M' else float(v) / 1e6) * df['Price (Intraday)']
        #df['chgp']= df['% Change'].apply(lambda c: float(c[:-1].replace(',', '').replace('+', '')))
        try:
            df['chgp']= df['% Change'].apply(lambda c: float(c.replace(',', '').replace('+', '')[:-2]))
            #df['chgp']= df['% Change'].apply(lambda c: float(c.translate({',':'', '%': '', '+': ''})))
        except: df['chgp']= df['% Change']
        
        df['src']= url_end.split('/')[-1]
        
        print( tabulate(df, headers='keys', tablefmt='psql') )
        return df
    
    def ass2watch(save=False, ref= ['SPY', 'QQQ']):
        
        dg= Screener().get_yf_screen_table(url_end='screener/predefined/day_gainers').query('VolMDoll > 50 and abs(chgp) > 1')
        dl= Screener().get_yf_screen_table(url_end='screener/predefined/day_losers').query('VolMDoll > 50 and abs(chgp) > 2')
        da= Screener().get_yf_screen_table(url_end='screener/predefined/most_actives').query('VolMDoll > 50 and abs(chgp) > 1')
        
        ass= pd.concat([da,dg,dl])[['Symbol', 'Price (Intraday)', 'Volume', 'VolMDoll', '% Change','chgp','src']]
        ass.ri().tab(nr=50)        
        
        ass_watch= ' '.join(ref + list(ass.Symbol.values))
        
        _, hist= Observer.get_hist_qoutes(ss=ass_watch, provider='yf')
        
        if save:
            pidu(ass, f'out/ass_watch_{nowE():%m-%d}.pck')        
            pidu(hist, f'out/hist_{nowE():%m-%d}.pck')
            #pidu(hist.to_dict(),  f'out/histd_{nowE():%m-%d}.pck')

        
        return ass, hist
    
      
    def screener_IBKR():
        import datetime

        #from ib_insync import *
        import asyncio, ib_insync as ibb
        from random import randrange
        
        #util.startLoop()  # uncomment this line when in a notebook
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
                
        #task = loop.create_task(user_insert_events(target))
        # if not loop.is_running():
        #     loop.run_until_complete(task)
        
         
        
        ib = ibb.IB()        
        try:
            ib.disconnect()
            ib.sleep(1)
        except: pass
    
        clid= randrange(1000000)
        pr(f'{clid=}')
        pr(f'ib= {ib.connectedEvent()=}')
    
        #ib.connect('127.0.0.1', 7496, clientId=26)         
        ib.connect('127.0.0.1', 7497, clientId=clid)         
        sub = ibb.ScannerSubscription(instrument='STK',        
                locationCode='STK.US.MAJOR', scanCode='HOT_BY_VOLUME',        
                abovePrice=5, marketCapAbove=1e6,        
                numberOfRows=10)
        
        scan = ib.reqScannerData(sub)       
         
        
        contracts = [s.contractDetails.contract.symbol for s in scan] #s.contractDetails.summary
        
        print(contracts)        
        allscannerParam = ib.reqScannerParameters() #A list of TWS scanner parameters as a string contains all XML-formatted parameters
        
        ib.disconnect()
        
    if 0:
        screener_IBKR()
        
        
    def sp500():
        'https://www.slickcharts.com/sp500'
        request
        

    if 0:
        get_yf_table(url_end='screener/predefined/day_gainers')
        get_yf_table(url_end='screener/predefined/day_losers')
        get_yf_table(url_end='screener/predefined/most_actives')
        #get_yf_table(url_end='7e0ddb6a-1b03-41eb-8d29-154e7e23e341') #nOK
        

if toTest:
    u= dg= Screener().get_yf_screen_table(url_end='screener/predefined/day_gainers')
    u= dl= Screener().get_yf_screen_table(url_end='screener/predefined/day_losers')
    u= da= Screener().get_yf_screen_table(url_end='screener/predefined/most_actives')
    u.tab('u')
    u['VolMDoll']= u.Volume.apply(lambda v: float(v[:-1]) if v[-1]=='M' else float(v) / 1e6) * u['Price (Intraday)']
    u['chgp']= u['% Change'].apply(lambda c: float(c[:-1]))
    u[['Symbol', 'Price (Intraday)', 'Volume', 'VolMDoll', '% Change','chgp', 'src']].svde('VolMDoll').query('VolMDoll > 50 and abs(chgp) >2').tab(nr=40)

    ass= Screener.ass2watch()
    ass.ri().tab(nr=50)
    # pidu(ass, f'out/ass_watch_{nowE():%m-%d}.pck')

    ass_watch=  ' '.join(ass.Symbol.values) # blank separated list
    
    rr, yxx, hist= Thinker.prep_XX(sss=  ass_watch, 
                             kma=[5,10,20, 30], toNorm=False, 
                             toPlot=False, n_pred=[20,40])    
    
    len(hist.ass.vc()) # 32 ass x 390 min OHLCV
    pidu(hist, f'out/hist_{nowE():%m-%d}.pck')
    
    
    #### daily run.  ass - 3 screeners, hist - quotes
    ass, histt= Screener.ass2watch(save=True)
    
    # check saved
    h= pilo('out/hist_06-11.pck').rename(columns={'level_1':'ass'})
    h= pilo('out/hist_06-15.pck').rename(columns={'level_1':'ass'})
    h= pilo('out/hist_06-22.pck').rename(columns={'level_1':'ass'})
    info(h, 'hist')  # hist DataFrame of  (12480, 8)
    h.ass.vc()
    h.info()
    
    for a in ass.Symbol:
        hc0= h.query('ass== @a') #.Close.plot()
        plt.plot(hc0.Datetime, hc0.Close / hc0.Close.values[-1], label=a)
    plt.legend()
    plt.show()
    
    _, hi4= Observer.get_hist_qoutes(ss='SPY QQQ MSFT F', provider='yf')
    pidu(hi4, 'out/hist_4_06-30.pck')
  

from dataclasses import dataclass, field

    
@dataclass
class Strategy:
  id:str
  nTop:int= 5
  nBott:int= 2
  margNeg:float= .3 
  dt:float= 20
  lastTrade:int =-99
    
@dataclass
class Pos: # position
  n: float= 0
  ticker: str='cash'
  #close: float= 1
  def __str__(s): return f' {s.n:8.3f} * {s.ticker:5} '
  def __repr__(s): return f' {s.n:8.3f} * {s.ticker:5} '
  
@dataclass
class Client:  # portf, strat
    val:float= 100
    trades:int= 0
    
    def __str__(s):  return f's Client {s.id:17s},  ret= {s.val - 100: 8.3f}% ,  val= {s.val:8.3f}, {s.trades:3} trades,  Portf = {" + ".join([f"{p}" for p in s.portf])}'
    def __repr__(s): return f'r Client {s.id:17s},  ret= {s.val - 100: 8.3f}% ,  val= {s.val:8.3f}, {s.trades} trades,  Portf = {s.portf}'
  
  
class Backtester:

    def backtest_bak(th, yXX, clients, model, verb=0, ref=['SPY', 'QQQ']):   
        ''' th - instance of Thinker '''
        pr, di= (dummy, dummy) if verb==0  else  (print, display)

        if 0:
            [pr(s, v.iloc[0].Datetime, v.iloc[-1].Datetime)  for s,v in yXX.items() if s>'']
            pr('clients=',  clients)
            
            #### Prep cycle by time
            all_t= sorted(list(set(pd.concat([v.Datetime for s,v in yXX.items() if s>'']).values)))
            
        mo=1
        if mo:
            hi= yXX
            yXX= hi.rename(columns={'level_1':'ass'})
            more= hi.rename(columns={'level_1':'ass'})# yXX[2].sv(['ass', 'Datetime'])
            all_t= sorted(list(set(more.Datetime)))
            asss= sorted(list(set(more.ass)))
            ref_miss=[r for r in ref if r not in asss]
            ref_pres=[r for r in ref if r     in asss]
           
        for cl in clients: cl.lastTrade= -999
        
        pr(3333333, f'{all_t[:3]=}, - ... {all_t[-2:]}')
        
        t0, dt0= all_t[0],  all_t[1] - all_t[0]
        quotes, preds= {},{}
        n_hist= model.n_hist
        
        for t in all_t:
            dt= int(round((t-t0) /dt0))   # t from t0, in interval , minutes 
            pgre(44444, dt0, dt)
                    
            for cl in clients:
                 pr(t, f'cycle   {cl.id= :10}', dt, cl.lastTrade, cl.strat.dt,  
                                            dt - cl.lastTrade - cl.strat.dt)

                 if dt - cl.lastTrade >= cl.strat.dt:  # time to trade
                     pgre(f'{dt=}, working with  {cl.id= :10}, {len(more)=}, {more.columns=}')
                     
                     # get forecast
                     yhh, curr_price= {},{'cash':1}
                     #for s,v in yXX.items():
                     for a,v in more.groupby('ass'):
                         s= a
                         if len(v) < n_hist or dt < n_hist : continue 
                     
                         if 1 or s>'':
                            v1= v.iloc[(dt-n_hist ): dt].rename(columns={'ass':'level_1'})
                            
                            pr(f'inner {a= }, {v1=}')
                            if len(v1) < n_hist : continue  
                        
                            di(7788, v1)
                            curr_price[s]= v1.Close.values[-1]
                         
                            if (dt,s) in preds:
                                prd= preds[(dt,s)]
                            else:
                                 #di(3344, v1)
                                 #xx= Thinker.trd2X([v1])
                                 pgre(f'{s,dt=},  \n{v1=}')
                                 
                                 #if type(v1) != list: v1= [v1]

                                 uu, uud, yxx_ker= th.prep_XX_ret(histo=v1, kma=[5,10,20,30], var=['ret', 'a4'], nLast=1, 
                                                 toPlot=False, n_pred=[20,40], n_hist=10, verb=0)
                                 pgre(f'th.prep_XX_ret done, {len(yxx_ker)=}')
                                 #xx= th.trd2Xy([v1])[0]
                                 #preds[(dt,s)]= prd= model.predict(xx,verbose = 0)
                                 if len(yxx_ker)==0: 
                                     pred('len(yxx_ker)==0')
                                     continue
                             
                                 pgre('model.predict for {(dt,s)=}')

                                 prd= preds[(dt,s)]= model.predict(yxx_ker[0], verbose = 0)
                            yhh[s]= prd
                     
                     
                     iy= 0 if cl.strat.dt==20 else 1
                     di(f'\n333444, {yhh.keys()=}, \n{yhh=}, \n{iy=}')
                     
                     yhhs= PDF({k: v[0][iy] for k,v in yhh.items() if not np.isnan(v[0][iy])}, index=['yh']).T.ri().sv('yh', ascending=False) # sort descending
                     yhhs.columns=['a','yh']
                     di(44445555, f'{cl=}, {yhhs=}')
                     #return
                     
                     buys= yhhs.head(cl.strat.nTop).a.values
                     sells= yhhs.tail(cl.strat.nBott).a.values
                     
                     
                     
                     #pr(5555, buys, sells, cl.portf)
                     
                     # sell current portf
                    # val= sum([pos['n'] * curr_price[pos['ticker']] for pos in cl.portf ]) #?? - fee
                     val= sum([pos.n * curr_price[pos.ticker] for pos in cl.portf ]) #?? - fee
                     neg= -cl.strat.margNeg * val
                     posit= val - neg
                     
                     #buy new portf
                     po= []
                     if len(buys) >0:
                         for a in buys:  po.append(Pos(n= posit / len(buys) / curr_price[a] , ticker=a)) #?? -> integers
                     if len(sells) > 0:    
                         for a in sells: po.append(Pos(n=  neg / len(sells) / curr_price[a] , ticker=a)) 
                         
                     cash= posit *( len(buys) ==0 ) + neg*(len(sells) ==0 )  
                     if cash != 0 : po.append(Pos(n=cash, ticker='cash'))
                     cl.portf= po
                     
                     cl.val= val                
                     
                     cl.trades +=1
                     cl.lastTrade= dt
                     
                     pr(6666, dt, cl.id, cl.val - 100, cl.val, cl.trades, cl.portf)                 
                     if dt > 22: break # ???
            if dt > 22: break # ???
         
        # reference portf
        pr(f'{ref_pres=}')
        yXX= {a:r for a,r in yXX.groupby('ass')}
        
        #pr(f'{yXX=}')
        
        ref=       {a: 100* round( yXX[a].Close.values[-1] / yXX[a].Close.values[0] - 1, 5) for a in ref_pres}
        ref_all=   {a: 100* round( v.Close.values[-1] / v.Close.values[0] - 1, 5) for a,v in yXX.items() if a > ''}
        ref_other= {a: 100* round( v.Close.values[-1] / v.Close.values[0] - 1, 5) for a,v in yXX.items() if a > '' and a not in ref.keys()}
        
        pred('Results of Backtesting')
        print('\n\nClients')
        for cl in clients: print(cl) 
        
        
        print('\nIndexes')
        print(PDF(ref, index=['ret,%']).T)
        
        print('\nPotential assets')
       # ra= PDF(ref_all, index=['ret,%']).T.ri(); ra.columns=['a','ret,%']
       # print(ra.query('a not in @ref.keys()').sv('ret,%').set_index('a'))
        
        print(PDF(ref_other, index=['ret,%']).T.sv('ret,%'))
        
        #diss(PDF(ref, index=['ret,%']).T, PDF(ref_other, index=['ret,%']).T)

        
        return clients, ref, ref_all, ref_other



    def backtest(th, histor, clients, model, verb=0, ref=['SPY', 'QQQ','ONEQ'], k_fee=1.0, yxx_ker=False):   
        ''' th - instance of Thinker '''
        #pr, di, pgre= (dummy, dummy, dummy) if verb==0  else  (print, display,pgre)
        #if verb==0: pr, di, pgre, pred, prG= (dummy) * 5 #, dummy, dummy, dummy)
        pr, di, pgre, pred, prG= ([dummy] * 5) if verb==0  else  (print, display,pgree, predd, prGG)
        
        if not yxx_ker:            
            _, _, yxx_ker= th.prep_XX_ret(histo=histor, kma=[5,10,20,30], var=['ret', 'a4'], nLast=1, 
                          toPlot=False, n_pred=[20,40], n_hist=10, verb=0)
            
        #yXX= histor.rename(columns={'level_1':'ass'})
        #more= histor.rename(columns= {'level_1':'ass'})# yXX[2].sv(['ass', 'Datetime'])
        more= yxx_ker[2]
        all_t= sorted(list(set(more.Datetime)))
        asss= sorted(list(set(more.ass)))
        ref_miss=[r for r in ref if r not in asss]
        ref_pres=[r for r in ref if r     in asss]           
            
        pgre(f'{yxx_ker=}')
        predictions = model.model.predict((yxx_ker[0]), verbose=0)
        
        yhh= pd.concat( [PDF(yxx_ker[1].reshape(-1, len(th.cy)), columns=th.cy), 
                         PDF(predictions, columns=[y+ 'h' for y in th.cy]), 
                         yxx_ker[2].ri(drop=1)], **a1) #.tab('y')
        
        yhh['dt']= ((yhh.Datetime - yhh.Datetime[0]) / (yhh.Datetime[1] - yhh.Datetime[0])).astype(np.int64)
        
        # use full data assets only
        pgre('Working with assets yhh= \n', yhh.ass.vc(sort=False))
        a= PDF(yhh.ass.vc()).ri()
        m= a.ass.max()
        ass_use= sorted(a.query('ass == @m')['index'].values)
        yhh_use= yhh.query('ass in @ass_use')
        yhh= yhh_use
        pgre(f'Use for trade {len(ass_use)} assets from {len(a)},  ass_use=  {ass_use} ')  
            
          #           y_20     y_40    y_20h    y_40h  ass    Datetime                     Close0    feep
          # -----  -------  -------  -------  -------  -----  -------------------------  --------  ------
          # 11754   1.1859  -2.5663   0.2021   0.0236  XPEV   2023-06-28 15:13:00-04:00   11.8856  0.0003
          # 11755   1.2616  -3.4788   0.1534  -0.0054  XPEV   2023-06-28 15:14:00-04:00   11.885   0.0003
          # 11756  -0.1261  -5.4439   0.3407   0.0909  XPEV   2023-06-28 15:15:00-04:00   11.891   0.0003
          # 11757   0       -5.5502   0.1718  -0.0062  XPEV   2023-06-28 15:16:00-04:00   11.895   0.0003
          # 11758  -1.26    -6.6414   0.168   -0.0113  XPEV   2023-06-28 15:17:00-04:00   11.91    0.0003
          # 11759  -4.4127  -7.5918   0.1132  -0.0433  XPEV   2023-06-28 15:18:00-04:00   11.915   0.0003
          # 11760  -6.2163  -8.2748   0.2121   0.0121  XPEV   2023-06-28 15:19:00-04:00   11.9208  0.0003 
            
          # return yhh, yxx_ker, predictions
           
        for cl in clients: cl.lastTrade= -999
        
        pr(3333333, f'{all_t[:3]=}, - ... {all_t[-2:]}')
        
        t0, dt0= all_t[0],  all_t[1] - all_t[0]
        quotes, preds= {},{}
        n_hist= model.n_hist
        
        #di(9999, yhh)
        
        for dt, yh in yhh_use.groupby('dt'):
            #di(999999, yh)
          
        
        # for t in all_t:
        #     dt= int(round((t-t0) /dt0))   # t from t0, in interval , minutes 
        #     pgre(44444, dt0, dt)
                    
            for cl in clients:
                 if 0: pr(f'cycle   {cl.id= :10}', dt, cl.lastTrade, cl.strat.dt,  
                                            dt - cl.lastTrade - cl.strat.dt)

                 if dt - cl.lastTrade >= cl.strat.dt or cl.hunter:  # time to trade
                     #pgre(f'{dt=}, working with  {cl.id= :10}, {len(yh)=}, {yh.columns=}')
                     #pgre(f'{yh.columns=}')
                     v= yh[[f'y_{cl.strat.dt}', f'y_{cl.strat.dt}h', 'ass', 'feep','dt', 'Close0']].copy()
                     v.columns= ['y','yh', 'ass', 'feep','dt', 'Close0']
                     pgre(f'{dt, cl=},  \n{yh.columns=},  \n{v.columns=}')
                     

                     # calc yhh with fee
                     p_ass= [c.ticker for c in cl.portf]
                     fe= v.apply(lambda r: (0 if (r.ass in p_ass or r.ass=="cash") \
                                            else 1 if r.yh > 0 \
                                            else  -1
                                            #) * k_fee* r.feep * 2 * day_minutes / cl.strat.dt, **a1)
                                            ) * k_fee* th.fee/r.Close0 * 2 * day_minutes / cl.strat.dt , **a1)  #?? *5  *95
                     
                     v['yf']=   v.y - fe
                     v['yhf']= v.yh - fe
                     v= v.svde('yhf').query('yh * yhf > 0')
                     
                     pgre(f'after calc yhh with fee : \n{v=}')
                     

                     buys= v.head(cl.strat.nTop).ass.values
                     sells= v.tail(cl.strat.nBott).ass.values
                     
                     curr= pd.concat([ v[['yf','yhf', 'Close0','feep','ass']], 
                                       PDFd(Close0=[1], feep=0, ass='cash', yf=0, yhf=0)]) #, index=['cash']) ])

                     curr.index= curr.ass
                     #curr_price= curr.Close0
                     
                     pgre(f'{curr=}  \n{buys=},  {sells=} \n{cl.id, cl.portf=}')
                     
                   

                     #val= sum([pos.n * (curr.loc[pos.ticker].Close0 - np.sign(pos.n)*th.fee) for pos in cl.portf ])  
                     val= sum([pos.n * curr.loc[pos.ticker].Close0 for pos in cl.portf ])  
                     neg= -cl.strat.margNeg * val
                     posit= (val - neg) * .999 #.995  # for fee --> posit cash
                     
                     #buy new portf
                     po, nb, ns, cc= [], len(buys), len(sells), lambda a: curr.loc[a].Close0
                     # if len(buys) > 0:
                     #     for a in buys:  po.append(Pos(n= posit/ nb / cc(a) , ticker=a)) #?? -> integers
                     # if len(sells) > 0:    
                     #     for a in sells: po.append(Pos(n=  neg / ns / cc(a) , ticker=a)) 
                         
                     po= [Pos(n= posit/ nb / cc(a) , ticker=a) for a in buys]  + \
                         [Pos(n= neg  / ns / cc(a) , ticker=a) for a in sells] 
                     
                     ####  Buy/Sell:  
                     p_old=      PDF([( p.n, p.ticker) for p in cl.portf if  p.ticker !='cash'], columns=['n', 'ass'])
                     p_new_neg=  PDF([(-p.n, p.ticker) for p in      po  if  p.ticker !='cash'], columns=['n', 'ass'])
                     pgre(f'portf: {p_old=}, \n{p_new_neg=}')
                     
                     o,n= p_old.ass.values, p_new_neg.ass.values
                     #pde= pd.concat([p_old, p_new_neg]).groupby('ass').sum(numeric_only=True)
                     if len(o)>0 or len(n)>0 :
                         pde= pd.concat([p_old, p_new_neg]).groupby('ass').n.sum()  # -diff between new and old to buy / sell
                         pr(f'\n{p_old=} \n{p_new_neg=} \n{pde.info()=} \n\n{pde=}')
                         adif= [ a for a in pde.index if  (a not in o) or (a not in n)]
                         pr(f'\n{adif=}')
                         cl.trades += 1 

                         if 1: #try:
                             #fee= k_fee * th.fee * sum(abs(pde.n))
                             fee= k_fee * th.fee * sum(abs(pde))
                             pred(f'{abs(pde)=} \n{fee=}')
                         #except: fee= 0                         
                          
                     else: 
                         pde, adif= [],[]
                         po= p_old
                         fee=0           
                         
                     #cash= posit *( nb ==0 ) + neg * (ns ==0 )  - fee  
                     cash= val - fee - sum([p.n * cc(p.ticker) for p in po])  
                     
                     if cash != 0 : po.append(Pos(n=cash, ticker='cash'))                     
                
                     cl.portf= po                     
                     cl.val= val      
                         
                     prG('o,n, pde, adif', o,n, pde,adif, f'\n{cl.portf=},  \n{cl.val=}')

                     cl.lastTrade= dt
                     
                     #pr(6666, dt, cl.id, cl.val - 100, cl.val, cl.trades, cl.portf)                 
                     #if dt > 3: break # ???
            #if dt > 3: break # ???
         
        # reference portf
        #pr(f'{ref_pres=}')
        #yXX= {a:r for a,r in yXX.groupby('ass')}
        
        #pr(f'{yXX=}')  yhh
        ret= {a:100* round( v.Close0.values[-1] / v.Close0.values[0] - 1, 5)  for a, v in yhh.groupby('ass')}
        ret= PDFd(**ret, index=[0]).T.svde(0); ret.columns=['ret, % / day']
        
        refe= ret.query('index in @ref')
        ref_other= ret.query('index not in @ref')
        ref_all= ret
        
        # ref=       {a: 100* round( yXX[a].Close.values[-1] / yXX[a].Close.values[0] - 1, 5) for a in ref_pres}
        # ref_all=   {a: 100* round( v.Close.values[-1] / v.Close.values[0] - 1, 5) for a,v in yXX.items() if a > ''}
        # ref_other= {a: 100* round( v.Close.values[-1] / v.Close.values[0] - 1, 5) for a,v in yXX.items() if a > '' and a not in ref.keys()}
        
        pred('Results of Backtesting')
        print('\n\nClients')
        for cl in clients: print(cl) 
        
        
        print('\nIndexes')
        print(PDF(refe, index=['ret,%']).T)
        print(refe)
        
        print('\nPotential assets')
       # ra= PDF(ref_all, index=['ret,%']).T.ri(); ra.columns=['a','ret,%']
       # print(ra.query('a not in @ref.keys()').sv('ret,%').set_index('a'))
        
        #print(PDF(ref_other, index=['ret,%']).T.sv('ret,%'))
        print(ref_other)
        pgre(f'Used for trade {len(ass_use)} assets from {len(a)},  ass_use=  {ass_use} ')
        
        #diss(PDF(ref, index=['ret,%']).T, PDF(ref_other, index=['ret,%']).T)

        
        return clients, ref, ref_all, ref_other, yxx_ker

#%% --> test Backtester
if toTest:
    bt= Backtester
    u= Backtester.backtest(yXX, clients, model, verb=0)
    
    
    uu, uud, yxx_ker= th.prep_XX_ret(fHist='out/hist_06-11.pck', tp_rsi=5,
                    kma=[5,10,20, 30], var=['ret','a4'], toPlot=True, n_pred=[20,40])
    
    
    bt= Backtester()
    fModel= f'{out}/model2_32_32_3000'
    model= tf.keras.saving.load_model(fModel)
    m= Model(th, fModel)
    
    tf.keras.__version__
    pr(model.summary())
    
    yXX= pilo(f"{out}/yXX.pck")
    di(11111, len(yXX), yXX['SPY'])
    #di(2222, [(s, v.Datetime.values[0]) for s,v in yXX.items() if s>''])
    #all_t= sum([v.Datetime.values for s,v in yXX.items() if s>''], [np.datetime64('2023-05-23T13:30:00.000000000')])

    for s,v in yXX.items():
        if s>'':
            pr(f'{s:8}, {len(v):4} points, {v.iloc[0].Datetime} -  {v.iloc[-1].Datetime}')
            
     
    #### Strategies
    st20_1_1= Strategy('st20_2_1', nTop= 2,  nBott= 1, margNeg= .3, dt=20)
    st20_5_3= Strategy('st20_5_3', nTop= 5,  nBott= 3, margNeg= .3, dt=20)
    st20= Strategy('st20', nTop= 3,  nBott= 2, margNeg= .3, dt=20)
    st40= Strategy('st40', nTop= 3,  nBott= 2, margNeg= .3, dt=40)
    st40_1_1= Strategy('st40_1_1', nTop= 1,  nBott= 1, margNeg= .3, dt=40)
    st40_1_1h= Strategy('st40_1_1_hunter', nTop= 1,  nBott= 1, margNeg= .3, dt=40)
    st40_5_3= Strategy('st40_5_3', nTop= 5,  nBott= 3, margNeg= .3, dt=40)
    sts= [st20_1_1, st20_5_3, st40_1_1, st40_1_1h, st40_5_3, st20, st40]
    #pr('st20=', st20, sts)
    for s in sts: pr(s)
    
    #### Clients, set initial portfolio
    def init_clients(sts):
        clients= []
        for sa in sts: 
            cl= Client(val=100)
            cl.strat= sa
            cl.id= sa.id
            cl.portf= [Pos(100, 'cash')]
            cl.lastTrade= -990
            cl.trades= 0
            cl.hunter= cl.strat.id.endswith('hunter')
            clients.append(cl)    
    
        pgre('clients init:')
        for c in clients: pr(c)
        return clients
        
    clients= init_clients(sts)
    
    # thinker = yxx struct
    th= Thinker()
    th.__class__= Thinker
    th.fee
    
    clients= init_clients(sts)
    
    t1()
    u= th.prep_XX_ret(fHist='out/hist_06-29.pck', kma=[5,10,20,30], tp_rsi=5,
                          var=['ret','a4'],  toPlot=False, n_pred=[20,40], n_hist=10, row0=True)
    t1() # 0:00:25.14
    
    # model    
    m= Model(th, fModel='')
    w= pilo('out/model_r6a4t_128-64-64_3002ep_wghts.pck')    
    m.model.__weights__= w
    m.n_hist= m.model.layers[0].output_shape[0][1]  # 10
    
    #### ta model
    w= pilo('out/model_ra4_128-64-64-ta_200ep_wts.pck')
    mta= m
    mta.model.__weights__= w
    mta.name= 'model_ra4_128-64-64-ta'
    yhh, co, wins, win, impo,  impo_cor, impo_win= mta.cv(th, yxx=yxxs, n_splits= 5, verb=0)

    
    
    #uu, uud, yxx_ker= th.prep_XX_ret(fHist='out/hist_06-11.pck', kma=[5,10,20,30], 
    uu, uud, yxx_ker= th.prep_XX_ret(fHist='out/hist_4_06-30.pck', kma=[5,10,20,30], tp_rsi=5, 
    #uu, uud, yxx_ker= th.prep_XX_ret(fHist='out/hist_07-01.pck', kma=[5,10,20,30], 
                          var=['ret','a4'],  toPlot=False, n_pred=[20,40], n_hist=m.n_hist, row0=False)
    
    
    hi= pilo('out/hist_06-29.pck')
    hi= pilo('out/hist_4_06-30.pck')
    hi= pilo('out/hist_07-01.pck')
    hi= pilo('out/hist_sp500_07-02.pck')
    hi.level_1.vc()
    
    
 #%% -->  #### backtest 
    th.__class__= Thinker
    th.fee= .0036  # 0.0036
    
    clients= init_clients(sts)
    
    t1()
    bt= Backtester
    clients, ref, ref_all, ref_other, yxx_ker= bt.backtest(th, histor=hi, yxx_ker= yxx_ker, \
                     clients=clients, model=m, verb=1, ref=['SPY', 'QQQ','ONEQ'], k_fee=1)
    t1()  # 0:00:14.67 -> 0:00:05.58   0:03:24.79
    
    
    #%% --> #### small run
    t1()
    th.__class__= Thinker
    hi4= pilo('out/hist_4_06-30.pck')
    clients= init_clients(sts[2:4])
    clients, ref, ref_all, ref_other, yxx_ker4= Backtester.backtest(th, histor=hi4, yxx_ker= yxx_ker4, 
             clients=clients, model=m, verb=1, ref=['SPY', 'QQQ','ONEQ'], k_fee=1.1)
    t1()  # 0:00:09.43  , w/o yxx_ker 0:00:14.51
    
    
    if 0:
        yhh= pd.concat( [PDF(yxx_ker[1].reshape(-1, len(th.cy)), columns=th.cy), 
                         PDF(predictions, columns=[y+ 'h' for y in th.cy]), 
                         yxx_ker[2].ri(drop=1)], **a1) #.tab('y')
        
        yhh['dt']= ((yhh.Datetime - yhh.Datetime[0]) / (yhh.Datetime[1] - yhh.Datetime[0])).astype(np.int64)
        yhh.tab('yhh')
    


    if 0:    
        pr('\n\n------\n')
        pprint(clients)
        pprint(ref)
        pprint(ref_other)
        pprint(ref_all)

  

@dataclass
class Model(): 
#%% class Model    from keras.models.Model 
#  %% -
    '''  create, fit, validate, visualize keras model  
    
    from dataclasses import dataclass, field

@dataclass
class MyClass:
    my_list: list = field(default_factory=lambda: ["list1", "list2", "list3"])
    
    '''
    name: str = 'model1'
    nEp: int = 0
    n_feats: int = 0
    n_feat_c: int = 0
    cy: list = field(default_factory=lambda: ['y_20', 'y_40']) # list[str]  # field(default_factory=list)  # ['y_20', 'y_40']
    cyn: list =  field(default_factory=lambda: ['y_20', 'y_40', 'y_60']) #  list[str]  #  ['y_20', 'y_40', 'y_60']
    cyq: list =  field(default_factory=lambda: ['yqb_20', 'yqh_40', 'yqs_60']) #  list[str]  #  ['yqb_20', 'yqh_40', 'yqs_60']
    cdxx: list =  field(default_factory=list) #  []
    csxx: list =  field(default_factory=list)  # []  
    tp_rsi= 2
    
    #def __init__(s, th=None, nLSTM=128, nStat=64, nDen2=64, nEp=250, n_mem=10, var= 'yn',
    def create(s, th=None, nLSTM=128, nStat=64, nDen2=64, nEp=250, n_mem=10, var= 'yn',
                 #fModel='out/model_128_64_1500', toPlot=True):
                 fModel='', toPlot=True):
        
        s.nEp= 0
        s.n_mem= n_mem
        s.trd2Xy= Thinker.trd2Xy
        #s.trd2y= Thinker.trd2y
        s.hist= PDF()
        if th:
            s.cy= th.cy # ['y_20', 'y_40']
            s.cdxx= th.cdxx # ['y_20', 'y_40']
            s.csxx= th.csxx # ['y_20', 'y_40']
            pgre(f'Model init: {s.cdxx=}, {s.csxx=},  {s.cy=},  {n_mem=}')    
        
        # Create the LSTM model      
        from keras import models, layers
        from keras.models import Sequential
        from keras.layers import  Input, Dense, LSTM, Concatenate
        from tensorflow.keras import regularizers

        s.n_feats = n_feats = len(s.cdxx) # 5    # Close    High      Low     Open    Volume
        s.n_feat_c= n_feat_c= len(s.csxx) # 2    # rule_5_20    rule_10_30
        latent_dim = nLSTM
        
        pr(f'Model init: {s.n_feats =},  {s.n_feat_c =}')

        if fModel >'':
            model = tf.keras.saving.load_model(fModel)
        else:
            lstm_input = layers.Input(shape=(s.n_mem, s.n_feats), name="ts_input")
            lstm_output = layers.LSTM(latent_dim, name='lstm_l')(lstm_input)

            static_input = Input(shape=(s.n_feat_c, ), name="static_input") 
            static_layer_one = Dense(nStat,  kernel_regularizer=regularizers.L2(0.001),  activation='relu',name="dens_1")(static_input)
                        
            combined = Concatenate(axis= 1,name = "concat_ts_stat")([lstm_output, static_layer_one])
            
            combined_dense_two = Dense(nDen2, activation='relu', name="dens_2_comb")(combined)
            #combined_dense_two = Dense(nDen2, activation='linear', name="dens_2_comb")(combined)
            #output = Dense(2, activation='sigmoid', name="outp")(combined_dense_two)
            
            if var=='yn':
                output = Dense(len(s.cy), activation='linear', name="outp")(combined_dense_two)
                model = models.Model([lstm_input, static_input], output, name='TS_combo')  #'TS_diff_len_230_57')  #'TS_diff_len_201_58')  #'TS_diff_len_57_197')
                model.compile(loss='mean_squared_error', optimizer='adam')
            else:
                output_n = Dense(len(s.cyn), activation='linear', name="outp_n")(combined_dense_two)
                output_q = Dense(len(s.cyq), activation='relu', name="outp_q")(combined_dense_two)
                model = models.Model([lstm_input, static_input], [output_n, output_q], name='TS_nq') 
                losses ={ 'outp_n':keras.losses.MeanSquaredError(),
                          'outp_q':keras.losses.CategoricalCrossentropy()}                
                model.compile(loss=losses, optimizer='adam', loss_weights={'outp_n': 0.5, 'outp_q': 1.0})
                
                # mnq.model.compile(optimizer='adam',
                #               loss={'outp_n': 'mean_squared_error', 
                #                     'outp_q': 'categorical_crossentropy'},
                #               loss_weights={'outp_n': 0.5, 'outp_q': 1.0})


        model.summary() 
        
        s.model= model
        
    def __str__(s): return f'''Model str: {s.name=},  {s.n_feats=}, {s.n_feat_c=}, \n{s.cdxx=},  
            \n{s.csxx=},  \n{s.cy=}, \n{s.cyn=}, \n{s.cyq=}, \n{s.nEp=},\n{s.tp_rsi=}, \n{s.model.summary()=}
            '''    
    def __repr__(s): return 'repr:  ' + s.__str__()
    
    
    #   %% winsf --->
    def winsf(yhh, nWin=[.01, .05, .1, .5], cy=['y_20','y_40']):  #s.cy):  #['y_20','y_40']):
        wins= []
        for y in cy:
            for yh in yhh:
                
                u= yhh[([y,yh] if y!=yh else[y]) ].svde(yh).ri(drop=True)
                
                # Wins
                for nw in nWin:
                    nw1= int(round(nw * len(yhh) )) if nw <=1 else nw
                    
                    pr(y, yh, nw1)
                    
                    win= u.head(nw1)[y].mean() - u.tail(nw1)[y].mean()                
                    dt_minut= int(y[-2:])  #??
                    # win_ann= win/ dt_minut * 300 *200 # * trade_minutes * daysInYear
                    win_ann= win *200 # * daysInYear
                    wins.append(dict(y=y, yh=yh, nWin=nw, win= win, win_ann= win_ann))            

        wins= PDF(wins) #; wins.index= wins.nWin
        #print(f'\n\nMean Win for  {y} : AUC= {auc:6.2f}% ') 
        wins.tab('Wins, best - worst , %', showindex=False, nr=99)
        
        di(wins.query('nWin== .1 and y < "y_3"  and yh < "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_20')).round(1))
        di(wins.query('nWin== .1 and y > "y_3"  and yh > "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_40')).round(1))
        di(wins.query('nWin== .5 and y < "y_3"  and yh < "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_20')).round(1))
        di(wins.query('nWin== .5 and y > "y_3"  and yh > "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_40')).round(1))
        
        return wins
 
    def train(s, yxx,  nEp=250, toPlot=True):
        ''' train keras model  s.model      

        Parameters
        ----------
        s : self 
        yxx : list (PDF n_mem x n_features, nornalized between assets).
            yxx= pilo('out/yxx_tr.pck'), dict {asset: list(PDF)}
            
            or
            rr, yxx= Thinker.prep_XX(sss='SPY QQQ MSFT F ALB KD PROK  ALGN WFC AAL', kma=[5,10,20, 30], toNorm=False, toPlot=True, n_pred=[20,40])
            
        nEp : number of epoch for fit, optional
            DESCRIPTION. The default is 250.
        toPlot : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None. result in s.model
        
            # prep yxx     
            #q2= q1.dropna()
            yxx[s]=[]
            for i in range(n_hist, len(q1)- max(n_pred)):
                u= q1.iloc[i- n_hist: i].copy()
                q_norm= u.apply(lambda r: .25*(r.Open+ r.High+ r.Low+ r.Close), **a1).values[-1]
                for c in ['Open','High','Low','Close']: u.loc[:, c]= norm_last(u[c], q_norm )
                for c in ['rule_5_20','rule_10_30']: u.loc[:, c]= norm_last(u[c], q_norm, m1=0 )
                u.loc[:,'Volume']=  np.log(norm_last(u['Volume'], m1=0)/100)
                yxx[s].append(u)

        '''
        
        pgre(f"\nin train, yxx: {type(yxx)=},  {len(yxx)=},  {yxx[0][0][0].shape=}, { yxx[1][0].shape=}")
        
        if len(yxx) == 3:            
            #(tx1, tx2), ty= yxx
            tx, ty, feep= yxx
            pr(f'{len(tx)=}, {len(ty)=}')
            pr(f'{tx[0].shape=}, {tx[0][0][0]=}, {tx[0][0][1]=}')
            tx1, tx2 = tx
            pr(g, f'{len(tx1)=}, {len(tx2)=}, {len(ty)=}', er)            
            
            def t3_to_l(tx1, tx2, ty, feep, permut=True): 
                pgre(f'2. in t3_to_l():  {len(tx1)=}, {len(tx2)=}, {len(ty)=}, {len(feep)=}')
                pgre(f' {len(feep[0])=}')
                pr(f' {feep=}')
                
                if len(feep) < len(ty): 
                    feep= pd.concat(feep).ri(drop=True)
                    pgre(f'after concat:  {len(ty)=}, {len(feep)=}')

                n= len(ty)
                ii=  np.random.RandomState(seed=55).permutation(n) if permut else range(n)
                return [((tx1[i], tx2[i]), ty[i], feep.iloc[i]) for i in ii]
            
            def l_to_t3(yxx): 
                t1= npa([y[0][0] for y in yxx])
                t2= npa([y[0][1] for y in yxx])
                t3= npa([y[1] for y in yxx])
                feep= npa([y[2] for y in yxx])
                return  t1, t2, t3, feep
            
            #yxx= [((tx1[i], tx2[i]), ty[i]) for i in ii] #?? ?? ??
            yxx= t3_to_l(tx1, tx2, ty, feep, permut=True) # random reorder ?? ?? ??
            
            train_data= yxx[:-400]
            test_data=  yxx[-400:]
            
            #t1= npa([y[0][0] for y in train_data])
            #t2= npa([y[0][1] for y in train_data])
            #t3= npa([y[1] for y in train_data])
            t1, t2, ty, feep= l_to_t3(train_data)
            
            pgre(g, f'{t1.shape=}, \n{t2.shape=}, \n{ty.shape=}, \n{t1[0]=}, {t2[0]=}, {ty[0]=}, ')
            
            history= s.model.fit((t1,t2), ty, epochs=nEp, validation_split= 0.3)  #, validation_split= 0.3, verbose=1)
            s.nEp += nEp
            
            try:
                s.hist= plotKerasLearningCurve(history, s.hist)
            except: pass
            return s.model.predict((t1,t2)), history, feep
            
            
            if 0:
                yxx= pilo('out/yxx_tr.pck')
                pr(r, f'{len(yxx)=}, {len(yxx["F"])=}')  #', {yxx["F"][0]=}')
                yxx["F"][0].tab('yxx["F"][0]', nr=99)
                # flatten the dic into list and filter
                yxx= sum([[y for y in yxx[k] if len(y.dropna()) ==10] for k in yxx if k>''], [])             
                

            if 0:
                # shaffle rows
                ii=  np.random.RandomState(seed=55).permutation(len(yxx))        
                yxx= [yxx[i] for i in ii]       
                pr('len(yxx), yxx[0].shape=', len(yxx), yxx[0].shape)
                
    
                # Split the data into train and test sets
                train_data= yxx[:-400]
                test_data=  yxx[-400:]
                
                s.model.fit(*Thinker.trd2Xy(train_data), epochs=nEp, verbose=1) 
            
        else:               
            # flatten the dic into list and filter; 
            if type(yxx)== dict:  # k - asset
                pr('yxx.keys()=', yxx.keys())
                #yxx= sum([[y for y in yxx[k] if len(y.dropna()) ==10] for k in yxx if k>''], [])  
                yxx= [v for vv in yxx.values() for v in vv if len(v.dropna()) ==10]

            else: 
                pr('len(yxx)= ', len(yxx))    
                yxx= [v                       for v in yxx if len(v.dropna()) ==10]  
                #yxx= [[v[i]                       for v in yxx] if len(v[0][0].dropna()) ==10  for i in range(len(yxx[0]))]  
    
        # shaffle rows
        ii=  np.random.RandomState(seed=55).permutation(len(yxx))        
        yxx= [yxx[i] for i in ii]       
        di('len(yxx), yxx[1].shape=', len(yxx), yxx[1].shape)
        
        
        #  %%% Train the model
        # Split the data into train and test sets
        train_data = yxx[:-400]
        test_data =  yxx[-400:]
        
        pred('fit')
        
        #s.trd2Xy= Thinker.trd2Xy(train_data)

        #model.fit(trd2X(train_data), trd2y(train_data), epochs=nEp)
        #s.model.fit(*Thinker.trd2Xy(train_data), epochs=nEp, verbose=0)
        info(train_data[0], 'train_data[0]')
        

        s.model.fit(*s.trd2Xy(train_data), epochs=nEp, verbose=0)

        # Make predictions          
        #predictions = model.predict(*Thinker.trd2X(test_data), verbose=0)
        #(tx1, tx2), ty= Thinker.trd2Xy(train_data)
        #(tx1, tx2), ty= Thinker.trd2Xy(test_data)
        (tx1, tx2), ty= s.trd2Xy(test_data)
        
            
        pr('tx1[0]:', np.round(tx1[0], 2))
        pr('tx2[0]:', tx2[0])
        [print(i.shape, i.dtype) for i in [tx1, tx2]]
        #predictions = s.model.predict([tx1, tx2], verbose=0)
        
        #ker= Thinker.trd2Xy(test_data)    
        ker= s.trd2Xy(test_data)    
        di('ker=', ker[0][0][0].round(3), ker[0][1][0])   
       # predictions = s.model.predict(*(Thinker.trd2Xy(test_data)[0]), verbose=0)
        predictions = s.model.predict((ker[0]), verbose=0)
        #yhh= pd.concat([ PDF(Thinker.trd2y(test_data), columns=['y_20', 'y_40']), PDF(predictions, columns=['y_20h', 'y_40h'])], **a1)
        #yhh= pd.concat([ PDF(s.trd2y(test_data), columns=['y_20', 'y_40']), 
        #                 PDF(predictions, columns=['y_20h', 'y_40h'])], **a1)
        
        #yhh= pd.concat([ PDF(s.trd2y(test_data), columns= s.cy), 
        yhh= pd.concat([ PDF(s.trd2Xy(test_data)[0], columns= s.cy), 
                         PDF(predictions, columns=[y+ 'h' for y in s.cy]),
                         test_data[2]
                         ], **a1)
        
                 
 
        # plot weights
        if 0:
            fig, axes = plt.subplots(nrows=2, ncols=2)
    
            i=0
            for l in  s.model.layers:
                wt= l.get_weights()
                pr(i, l.name, len(wt))
                if len(wt) > 0:
                    a= axes[i//2, i%2]
                    a.imshow(wt[0])
                    a.set_title(f'{i}. {l.name}' )
                    i+= 1
            plt.show( ) 
        
        s.plot_layers(wPlotly=False)
        
        di('', '----', 'Correl', 100* yhh.corr().round(3)) 
        
        # Plot the predictions
        if toPlot: s.plot_lift(test_data, nEp)
        return yhh
            
            
    #  %%
    def save(s, f=''):
        #s.model.save(f'out/{f}')  #'model2_32_32_15K')
        if f=='': f= s.name
        if f != '' and 'name' not in s.__dict__: s.name= f
        pidu(s.__dict__, f'out/full_{f}.pck')
        
    def load(s, f=''):  
        #import pickle
        #s.model= tf.keras.saving.load_model(f'out/{f}')  # model_128_64_1500") 
        if f=='': f= s.name
        
        if f.find('wts') >=0:
            s.model.__weights__ = pilo(f)
        else:        
            sd= pilo(f'out/full_{f}.pck')
            s.__dict__ = sd  # pickle.loads(sd)
        return s
    
    
    def set_weights(s, f):
            w= pilo(f)
        #try:
        #     s.model.__weights__= w
        #     return s
        # except:
            
            pr(f'{len(s.model.layers)=}, {len(w)=} ')
            for il,l in enumerate(s.model.layers):
                lw=   l.get_weights()
                #pr(f' {il=}, {l.name=}, {len(lw)=},  {lw[0].shape=}, {w[il].shape=}')
                pr(f' {il=}, {l.name=}, {len(lw)=}, {w[il].shape=}')
                for j, a in enumerate(lw):
                    nl= len(w[il][j].shape)
                    try:
                        s= a.shape
                    except Exception as e: 
                        pr(f'Except {e} for layer {il=}, {l.name=}')
                        continue    
                    n= len(s)
                    pgre(f'{n=}, {s=}, {w[il][j].shape=} , {w[il][j]=} ')
                    sh1= [min(s[i], w[il][j].shape[i]) for i in range(n)]
                    pgre(f'Setting weights for layer {il=},{l.name=}, {nl=}, {sh1=}')
                    if nl >=0: l.__weights__[:sh1[0]]= w[il][:sh1[0]]
                    if nl >=1: l.__weights__[:sh1[0], :sh1[1]]= w[il][:sh1[0], :sh1[1]]
                    if nl >=2: l.__weights__[:sh1[0], :sh1[1] :sh1[2]]= w[il][:sh1[0], :sh1[1] :sh1[2]]
                
                
    if 0:
        mo= Model(th)
        mo.set_weights('out/model_ra4_128-64-64-ta_300ep_wts.pck')            
        mo.set_weights('out/model_r6a4t_128-64-64_3002ep_wghts.pck')      
        mo.model.summary()
        w= pilo('out/model_ra4_128-64-64-ta_300ep_wts.pck')  
        for i,w1 in enumerate(w): 
            #info(w1, 'w1')
            pr(i, w1.shape)
           # dir(w1)
           # break
           # for w2 in w1: 
            #    info(w2, 'w2')
            
    def f2mname(f): 
        try:
            f= f.split('/')[1][:-4]
        except: pass
        return f.replace('model_','').replace('_wts','').replace('_wghts','')            
            
    def compare_models(ff, fHist, tp_rsi=5, nWin= .01):
        th= Thinker()         

        #rr= PDFd(y_20=yxx_ker[1][0])
        #di(rr)
        nw= nWin
        yhh, lifts, wins= PDF(), {}, []
        
        for im, f in enumerate(ff):
            #tp_rsi= [2, 5, 2][im]
            tp_rsi= [ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2][im]
            pr(f'{im=},  {tp_rsi=}')
            
            var= ['ret','a4'] if f.find('ta_tp_') >=0  else ['ret','a4','ta']
            pr(Y, f'\n---------- {var=}\n')

            _, _, yxx_ker= th.prep_XX_ret(fHist=fHist, tp_rsi= tp_rsi,
                                kma=[5,10,20, 30], var=var, toPlot=False, n_pred=[20,40])
            pr(f'{im=},  {len(yxx_ker)=}')
            
            if 0:
                t1()
                yhh, wins, wins_p, lifts = Model.compare_models([
                                     'model_ra4_128-64-64-ta_tp_rsi=2_300ep' ,
                                     'model_ra4_128-64-64-ta_tp_rsi=2_700ep' ,
                                     'model_ra4_128-64-64-ta_tp_rsi=2_1799ep' ,
                                     'mo_65f_ta_900ep',
                                     'mo_65f_ta_1400ep',
                                     'mo_65f_714_ta_302ep',
                                     'mo_65f_714_ta_602ep',
                                      ], 
                                     #fHist='out/hist_sp500_07-07.pck')    
                                    fHist='out/hist_07-11_sp500.pck') 
                                    #fHist='out/hist_4_06-30.pck')    
                
                t1(len(yhh))  # Execution time 0:02:23.42,  3448
               #  🚩  wins_p   :
               # m                                    y_20     y_40
               # --------------------------------  -------  -------
               # mo_65f_ta_1400ep                   8.7249  -0.8271
               # mo_65f_ta_900ep                   12.2893   2.188
               # ra4_128-64-64-ta_tp_rsi=2_1799ep  10.3361  17.5333
               # ra4_128-64-64-ta_tp_rsi=2_300ep   12.6502  20.1907
               # ra4_128-64-64-ta_tp_rsi=2_700ep    6.2106  12.6527 
            #if im==0: 
                
            yh0= PDF(yxx_ker[1].reshape(-1,2), columns= th.cy)            
            
            m= Model(th)
            m.load(f)
            #m.model.__weights__= pilo(f)
            m.name= Model.f2mname(f)
            
            
            #pgre(m.name)
            pgre(im, f, m.name)
            m.model.summary()
            yh= PDF(m.model.predict(yxx_ker[0]), columns= [f'{y}h_{m.name}' for y in th.cy])
            yh= pd.concat([yh0, yh], **a1)
            nw1= int(round(nw * len(yh) )) if nw <=1 else nw      
            # yhh done       
        
            #for y in th.cy:
            for iy,y  in enumerate(th.cy):
                yh.tab(f'{iy=}, yh')
                
                yh1= yh.iloc[:, iy::2].copy()
                #for im in range(len(ff)):
                if 1:
                    #yh= yh.svde(yh.columns[1 + im]).ri(drop=True)   
                    yh1= yh1.svde(f'{y}h_{m.name}').ri(drop=True)   
                    yh1.tab(f'{iy=}, {y=}, {im=},  {nw1=},  yh1:')
    
                    win= yh1.head(nw1)[y].mean() - yh1.tail(nw1)[y].mean()  # % / day   
                    #wins.append(dict(y=y, m= Model.f2mname(ff[im]), nWin=nw, win= win))
                    wins.append(dict(y=y, m= m.name, nWin=nw, win= win))
                    lifts[(m.name, y)]= yh1[y].cumsum() / len(yh1)
        
        pgre(f'{wins=}')
        wins_p= PDF(wins).pivot('m', 'y', 'win')
        wins_p.tab('wins_p', nr=99)
        return yhh, wins, wins_p,lifts 
    
    if 0: #zzzzz
        yhh, wins, wins_p, lifts = Model.compare_models(ff=[ # 'out/model_r6a4t_128-64-64_3002ep_wghts.pck', 
                                              'out/model_ra4_128-64-64-ta_300ep_wts.pck',                                               
                                              'out/model_ra4_128-64-64-ta_tp_rsi=2_300_wts.pck' 
                                              ], 
                        #nWin= .01, fHist='out/hist_sp500_07-07.pck')
                        nWin= .01, fHist='out/hist_4_06-30.pck')
    
    
    #%%        #### comparison, bokeh for yhh - lift
    def bokeh_comp_lifts(yhh, cy=['y_20', 'y_40']):
        from bokeh.plotting import figure as fig, output_file, show, row
        from bokeh.models import Range1d
        w3= {'line_width':3}
        co= ['red', 'orange','green', 'blue', 'cyan', 'navy']
        #cols= {'y_20':'red', 'y_40':'blue'}
        #colsb= {'y_20':'darkred', 'y_40':'navy'}
        p = fig(title='Modif Lift / win Curves for y_20', x_axis_label='p', y_axis_label='win, %', width=900, height=600)
        p2 = fig(title='Modif Lift / win Curves for y_40', x_axis_label='p', y_axis_label='win, %', width=900, height=600)
        pgre(f'{yhh.columns=}')
        #for iy, y  in enumerate(th.cy):
        for iy, y  in enumerate(cy):
            yh= yhh.iloc[:, iy::2].copy().ri(drop=True)
            n= len(yh) 
            mm= [c.replace(y+'h_', '') for c in yh.columns]

            #for im, f in enumerate(ff):
            for im, c in enumerate(yh.columns ):
                u= yh.svde(c).ri(drop=True)   
                u.tab(f'{iy=}, {y=}, {im=},  {c=}, {yh.columns=}   yh:')                
            
                #u= yh #h[[y, y+ 'h_00']].rename(columns={y:'y', y+ 'h_00':'yh'}).svde('y').ri(drop=True)
                u['p']= np.linspace( 1/n, 1-1/n, n) 
                #u['lift']= u[y].cumsum() / u.p/n
                u['lift']= u[y].cumsum() /n
                pp= [p,p2][iy]
                pp.line(u.p, u.lift,  line_color= co[im], legend_label=mm[im], **w3, line_dash='solid') #'dashed')  # axes
                if 1: # im==0: 
                    #u['lift_perf']= u.svde(y).ri(drop=True)[y].cumsum() / n #u.y.sum()        
                    #pp.line(u.p, u.lift_perf,  line_color= co[im], legend_label='perf', **w3, line_dash='solid') #'dashed')  # axes
                    pp.line([0,1], [0, u[y].sum() / n],  line_color= co[im], legend_label='0-pred', line_dash='dashed') #'dashed')  # axes
                    pp.legend.location= 'bottom_right'
                    pp.y_range = Range1d(0, 1.1* u[y].sum() / n)
                    
       
        show(row(p, p2)) 
    #%% bokeh_lifts()
    def bokeh_lifts(lifts, verb=0):
        from bokeh.plotting import figure as fig, output_file, show, row
        from bokeh.models import Range1d
        w3= {'line_width':3}
        co= ['red', 'orange','green', 'blue', 'cyan', 'navy']
        #cols= {'y_20':'red', 'y_40':'blue'}
        #colsb= {'y_20':'darkred', 'y_40':'navy'}
        p = fig(title='Modif Lift  for y_20', x_axis_label='p', y_axis_label='win, %', width=900, height=600)
        p2 = fig(title='Modif Lift for y_40', x_axis_label='p', y_axis_label='win, %', width=900, height=600)
        #pgre(f'{yhh.columns=}')
        mm= {m for (m,y) in lifts.keys()}
        com= {m:c for c,m in zip(co,mm)}
        if verb: pgre(f'{mm=}, {com=}')
        
        for (m,y), v in lifts.items():
            n= len(v) 
            if verb: 
                pgre(f'{m,y,n=}')
                di(v)

            x= np.linspace(1/n, 1-1/n, len(v))
            pp= {'y_20': p, 'y_40': p2 }[y]
            pp.line(x, v,  line_color= com[m], legend_label=m, **w3, line_dash='solid') #'dashed')  # axes
            pp.line([0,1], [0, v.iloc[-1] ],  line_color= 'black', legend_label='0-pred', line_dash='dashed') #'dashed')  # axes
            pp.legend.location= 'bottom_right'
            #pp.y_range = Range1d(0, 1.1* v.iloc[-1] / n)
       
        show(row(p, p2)) 
    #  %%
            
    if 0: #zzzzz
        yhh, wins, wins_p, lifts = Model.compare_models(ff=[ # 'out/model_r6a4t_128-64-64_3002ep_wghts.pck', 
                                              'out/model_ra4_128-64-64-ta_300ep_wts.pck',                                               
                                              'out/model_ra4_128-64-64-ta_tp_rsi=2_300_wts.pck' 
                                              ], 
                        nWin= .01, fHist='out/hist_sp500_07-07.pck')
        
        #bokeh_comp_lifts(yhh)  
        Model.bokeh_lifts(lifts)

        

    
    def plot_layers(s, wPlotly=False):
        ''' plot weights for each layer'''
        if wPlotly:
            import plotly.express as px
            import plotly.subplots as sp
            from plotly.subplots import make_subplots
            
                  
            tits= [l.name for  l in  s.model.layers if len(l.get_weights()) > 0 ]      
            pr('tits=', tits)      
            
            i, ff = 0, sp.make_subplots(rows=3, cols=2, row_heights=[600]*2,  subplot_titles=tits)  #, shared_xaxes=True , shared_yaxes=True)
            for il, l in  enumerate(s.model.layers):
                wt= l.get_weights()
                pr(il, i, l.name, len(wt))
                if len(wt) > 0:  
                    ws= wt[0].shape
                    ff.add_trace(px.imshow(wt[0], width=5*ws[1], height=5*ws[0]).data[0],   row=1+i//2, col= 1+ i%2) 
                    i+= 1                 
            ff.show()
        else:        
            fig, axes = plt.subplots(nrows=3, ncols=2)
    
            i=0
            for l in  s.model.layers:
                wt= l.get_weights()
                pr(i, l.name, len(wt))
                if len(wt) > 0:
                    a= axes[i//2, i%2]
                    a.imshow(wt[0])
                    a.set_title(f'{i}. {l.name}' )
                    i+= 1
            plt.show( ) 
            
    # Plot the predictions
    def plot_lift(s, test_data, yhh=PDF(),  nEp=1):
        if yhh==PDF():
                ker= s.trd2Xy(test_data)    
                di('ker=', ker[0][0][0].round(3), ker[0][1][0])   
               # predictions = s.model.predict(*(Thinker.trd2Xy(test_data)[0]), verbose=0)
                predictions = s.model.predict((ker[0]), verbose=0)
                #yhh= pd.concat([ PDF(Thinker.trd2y(test_data), columns=['y_20', 'y_40']), PDF(predictions, columns=['y_20h', 'y_40h'])], **a1)
                #yhh= pd.concat([ PDF(s.trd2y(test_data), columns=['y_20', 'y_40']), 
                #                 PDF(predictions, columns=['y_20h', 'y_40h'])], **a1)
                
                yhh= pd.concat([ PDF(s.trd2y(test_data), columns= s.cy), 
                                 PDF(predictions, columns=[y+ 'h' for y in s.cy])], **a1)
                
            
        fig, axes = plt.subplots(nrows=2, ncols=2)
    
        for i, y in enumerate(s.cy):
            yh= yhh[[y, y+'h']].sv(y+'h', ascending=False).ri(drop=True)
            lif= np.cumsum(yh[y]) #.plot(title='Lift ' + y)
            axes[0,i].plot(yh)
            axes[1,i].plot(lif)
            axes[1,i].plot([0, len(yhh)], [0, lif.values[-1]])

            axes[0,i].set_title(f'{y}, {nEp=}', x=.7, y=.8)            
            axes[1,i].set_title('Lift ' +y, x=.7, y=.8)
            
            pr(f'Mean Win for {y}   <50 best - 50 worst>= {yh.iloc[:50][y].mean() - yh.iloc[-50:][y].mean():.3f}') 

        plt.show()  
        
     
    def plot_win(s, yhh):
        from bokeh.plotting import figure as fi, output_file, show, row
        w3= {'line_width':7}
        cols= {'y_20':'red', 'y_40':'blue'}
        colsb= {'y_20':'darkred', 'y_40':'navy'}
        p = fi(title='Modif Lift / win Curves for Model= ' + s.name, x_axis_label='p', y_axis_label='win, %', width=900, height=600)
        p2 = fi(title='Modif Lift / win Curves for Model= ' + s.name, x_axis_label='p', y_axis_label='win, %', width=900, height=600)
        
        for y in ['y_20', 'y_40']:
            u= yhh[[y, y+ 'h_00']].rename(columns={y:'y', y+ 'h_00':'yh'}).svde('y').ri(drop=True)
            n= len(u)    
            u['p']= np.linspace( 1/n, 1-1/n, n) 
            u['lift_perf']= 100* u.y.cumsum() / u.y.sum()        
            u['win_perf']= np.where(u.y > 0, u.y.cumsum() / u.p/n, -(u.y.sum()  - u.shift().y.cumsum()) / (1-u.p)/n)        
            
            c,cb= cols[y], colsb[y]
            p.line(u.p.copy(), u.lift_perf,  line_color= cb, legend_label='lift perf ' +y, **w3, line_dash='dotted')  # axes
            p.line(u.p.copy(), u.win_perf,  line_color= cb, legend_label='win perf ' +y, **w3, line_dash='dotted')  # axes
            
            u= u.svde('yh').ri(drop=True)
            
            u['p']= np.linspace( 1/n, 1-1/n, n) 
            u['win_u']= u.y.cumsum() / u.p/n
            u['win_d']= -(u.y.sum()  - u.shift().y.cumsum()) / (1-u.p)/n
            u['win']= np.where(u.yh >0, u.win_u, u.win_d)
            u['lift']= 100* u.y.cumsum() / u.y.sum()  
            
            nw= sum(u.yh > 0);  nw= min(nw, len(u) - nw);   
            #u['win_symm']= u.y.cumsum() / u.p / n  + (u.y.sum()  - u.shift().y.cumsum()) / u.p / n
            u['win_symm']= u.y.cumsum() / u.shift().p / n  - u.iloc[::-1].ri(drop=True).y.cumsum()  / u.shift().p / n
            u_ws= u.iloc[:nw]
           

            #u.tab()
            
            p.scatter(u.p, u.y,  line_color= c, legend_label= y, size=.05)  # axes
            p.line(u.p, u.win_u,  line_color= c, legend_label='winners / buy ' +y, **w3, line_dash='dashed')  # axes
            p.line(u.p, u.win_d,  line_color= c, legend_label='loosers / short sale ' +y, **w3, line_dash='dashed')  # axes
            p.line(u.p, u.win,  line_color= cb, legend_label='win ' +y, **w3)  # axes
            p.line(u.p, u.lift,  line_color= cb, legend_label='win ' +y, **w3)  # axes
            p2.line(u_ws.p, u_ws.win_symm,  line_color= cb, legend_label='win_symm ' +y, **w3)  # axes
            #p.line(u.p, u.yh,  line_color= c, legend_label='yh ' +y, **w3, line_dash='dotted')  # axes
        p.legend.location= 'bottom_center'
        #show(p)         
        show(row(p, p2))  
        
            
            
    def cv(self, th, yxx=None,  nWin=[.01, .05, .10], n_splits= 5, nEp=5, toPlot=True, verb=1):
        ''' s= model, nWin - % or numbers
        
        yxxr[0][0].shape  dyn  x  tx1   (674, 10, 5)
        yxxr[0][1].shape  stat x  tx2   (674,     2)
        yxxr[1].shape     y       ty    (674, 1,  2)        

        '''
        
        if 0:
            rr, yxx2= Thinker.prep_XX(sss='SPY QQQ MSFT F ALB KD PROK  ALGN WFC AAL', kma=[5,10,20, 30], toNorm=False, toPlot=True, n_pred=[20,40])
    
            info(yxx, 'yxx')
            info(yxx2, 'yxx2')
            #yxx= pd.concat(yxx2.values())
            #yxx3= sum([v1 for v in yxx2.values() for v1 in v if len(v1.dropna()) ==10], [])
            yxx3= [v1 for v in yxx2.values() for v1 in v if len(v1.dropna()) ==10]
            info(yxx3, 'yxx3')
            yxx= yxx3
            #return
        
        if type(yxx)==dict:
            #e.g.  rr, yxx= Thinker.prep_XX(sss='SPY QQQ MSFT F ALB KD PROK  ALGN WFC AAL', kma=[5,10,20, 30], toNorm=False, toPlot=True, n_pred=[20,40])

            yxx= [v1 for v in yxx.values() for v1 in v if len(v1.dropna()) ==10]
    
        s= self.model
        
        #s.plot_layers(wPlotly=True)
        #return
        di, pr= (dummy, dummy) if verb==0 else  (display, print )
        #return 'zzzz'  
        #di, pr= dummy, dummy

        # Cross-validate the model
        if 1:  # OK
            #from sklearn.model_selection import KFold
            if 0:
                rr, yxx= Thinker.prep_XX(sss='SPY QQQ MSFT F ALB KD PROK  ALGN WFC AAL', kma=[5,10,20, 30], toNorm=False, toPlot=True, n_pred=[20,40])
                uu, uud, yxx_ker= th.prep_XX_ret(sss='SPY QQQ MSFT F ALB KD PROK  ALGN WFC AAL', 
                                                      kma=[5,10,20, 30], var=['ret','a4'], toPlot=True, n_pred=[20,40])
                
                uu, uud, yxx_ker= th.prep_XX_ret(fHist='out/hist_06-11.pck', 
                                    kma=[5,10,20, 30], var=['ret','a4'], toPlot=True, n_pred=[20,40])
                
                pidu(rr, 'out/yXX.pck')
                pidu(yxx, 'out/yxx_tr.pck')
                
                yxx= pilo('out/yxx_tr.pck')
                
                yxx= sum([[y for y in yxx[k] if len(y.dropna()) ==10] for k in yxx if k>''], []) 
                
                ''' for Spyder F9: '''
                len(yxx), yxx.keys() 
                len(yxx['F'])
                yxx['F'][-1].tab(" yxx['F'][-1]")
                
            pgre(f'in cv  {type(yxx)=}, {len(yxx)=}')    
            if type(yxx)== dict:
                # flatten the dic into list and filter
                yxx= sum([[y for y in yxx[k] if len(y.dropna()) ==10] for k in yxx if k>''], [])   
       
             
            if len(yxx) == 3:            
                 #(tx1, tx2), ty= yxx
                 (tx1, tx2), ty, feep= yxx
                 pgre(f'      {len(tx1)=}, {len(tx2)=}, {len(ty)=}, {len(feep)=}')            
                 yxx_all= yxx= t3_to_l(tx1, tx2, ty, feep, permut=True)
                 
            else:
                # shaffle rows
                ii=  np.random.RandomState(seed=55).permutation(len(yxx))        
                yxx= [yxx[i] for i in ii]       
    #            di('\n*********   len(yxx), yxx[0].shape=', len(yxx), type(yxx[0]),  yxx[0][0])  
    #            di(f'\n{g}*********   yxx[0][1],  yxx[1][0]=', er,  yxx[0][1], '\n',  yxx[1][0]) 
               #yxx[0]. tab( 'yxx[0])', nr=9)
                #yxx_all= pd.concat(yxx)
                yxx_all= yxx
                if type(yxx[0])== pd.DataFrame:
                    yxx_all= pd.concat(yxx)

            # Split the data into train and test sets
            
            results, iii = [], range(len(yxx))
            #yh= PDF(Thinker.trd2Xy(yxx)[1].reshape(-1, 2), columns=['y_20','y_40'])
            #yh= PDF(yxx[1].reshape(-1, 2), columns=['y_20','y_40'])
            
            csxx= th.csxx  #['rule_5_20', 'rule_10_30']
            cyy= cy= th.cy  #['y_20','y_40']
            #yh= PDF(yxx[1].reshape(-1, 2), columns= cyy)
            #if yxx.ztype=='list3':
            #if len(yxx[0])==2 and len(yxx[0][0]) > 1:  #'list3':
            if len(yxx[0])==3 and len(yxx[0][0]) > 1:  #'list3':
                tx1,tx2,ty, feep= l_to_t3(yxx)
                #yh= PDF(dict(yh= ty))
                yh= PDF(ty.reshape(-1,2), columns=cyy)
            else:    
                pred(f'{len(yxx[0])=}, {len(yxx[0][0])=}')
                pred(f'{info(yxx,"yxx")=}')
                pgre(f'{yxx[0]=}')  #', {yxx[0].columns=}')
                #pred(f'{shape_ker(yxx)=}')
                yh= pd.concat([u[cyy].tail(1) for u in yxx])
         
            # if 0:
            #     if 'Close' in   yxx[0][0].columns or 'Close' in   yxx[0].columns:
            #         cdxx=  ['Open','High','Low','Close','Volume']
            #     else: cdxx=  ['o','h','l','c','v']

            cdxx= th.cdxx
            

            pgre(f'{len(iii)=}, {iii=}, {n_splits=}')                         

            #di('1. yh=', yh)
            #r1, r2, ii= [],[],[]
            r1, inds= [],[]

            #   %% Cycle by fold:
            for iFold in range(n_splits):
                train_index, test_index= [i for i in iii if i % n_splits != iFold ], \
                                         [i for i in iii if i % n_splits == iFold ]
                inds += test_index 
                pred(f'{iFold=}, {len(inds)=}')                        
                pgre(f'{iFold=}, , {len(train_index)=}, {len(test_index)=}')                         
                # Train the model              
                #(tx1, tx2), ty= Thinker.trd2Xy([yxx[i] for i in train_index])
                
                '''    
                    yxxr[0][0].shape  dyn  x  tx1   (674, 10, 5)
                    yxxr[0][1].shape  stat x  tx2   (674,     2)
                    yxxr[1].shape     y       ty    (674, 1,  2)  
                '''
                              
                
                '''
                tx1= [yxx[i][cdxx] for i in train_index]
                tx2= [yxx[i][['rule_5_20', 'rule_10_30']].tail(1) for i in train_index]
                ty= npa([yxx[i][cyy].tail(1) for i in train_index]).reshape(-1, 2)
                '''
               
                if 0:
                    tx1= [yxx[i][cdxx] for i in train_index]
                    tx2= [yxx[i][csxx].tail(1) for i in train_index]
                    #ty= npa([yxx[i][cyy].tail(1) for i in train_index]).reshape(-1, 2)
                    ty= npa([yxx[i][cyy].tail(1) for i in train_index]).reshape(-1, 1, 2)
                
               # tx1, tx2, ty= l_to_t3([yxx[i] for i in train_index])
                
                if 0:
                    tx1= [yxx[0][0][i] for i in train_index]
                    tx2= [yxx[0][1][i] for i in train_index]
                    ty= npa([yxx[1][i] for i in train_index]).reshape(-1, 1, 2)
                
                
                
                
          
                #(tx1, tx2), ty= Thinker.trd2Xy([yxx[i] for i in train_index])
                pr(g,f'\n------------------\nStart fitting {len(tx1)=}, {len(tx2)=}, {len(ty)=}')  #', {tx1.shape=}, {tx2.shape=}')
                pgre( f'{tx1[0].shape=}, {tx2[0].shape=},  {ty[0].shape=}')  #', {tx1.shape=}, {tx2.shape=}')
                
                
                #(tx1, tx2), ty= Thinker.trd2Xy([yxx[i] for i in train_index])
               # tx1, tx2, ty= l_to_t3([yxx[i] for i in train_index])
                tx1, tx2, ty, feep= l_to_t3(yxx, train_index)
                s.fit((tx1, tx2), ty, epochs=nEp, verbose=0)

                pgre(f'\n\nEnd fitting {iFold=};  {len(ty)=}, {tx1.shape=}, {tx2.shape=}\n=========================')

                #(tx1, tx2), ty= Thinker.trd2Xy([yxx[i] for i in test_index])
                #tx1, tx2, ty= l_to_t3([yxx[i] for i in test_index])
                u= l_to_t3(yxx,  test_index)
                di('u=', u)
                tx1, tx2, ty, more= u

                di('\n**************************** tx2[:3]=', tx2[:3])
                
                #u={}  # , r1 - dict of predictions
                #u['00'] = model.predict([tx1, tx2], verbose=0)
                u= {'00' : s.predict((tx1, tx2), verbose=0)}
                
                
                #### xx permutation importance
                for ix,x in enumerate(cdxx):
                    t1= np.copy(tx1, order='K', subok=False)
                    pr(ix, x)  #, 't1=', t1)
                    #t1[:,:,ix]= npa(rn.choices(yxx_all[x].values, k=len(tx1)*10)).reshape(-1, 10)
                    t1[:,:,ix]= npa(rn.choices(t1[:,:,ix], k=len(tx1))).reshape(t1[:,:,ix].shape)
                    u[x] = s.predict([t1, tx2], verbose=0)
                for ix,x in enumerate(csxx):
                    t2= np.copy(tx2, order='K', subok=False)
                    #t2[:,ix]= rn.choices(yxx_all[x].values, k=len(tx1))                  
                    t2[:,ix]= rn.choices(t2[:,ix], k=len(tx2)) # .reshape(t2[:,ix].shape)                 
                    u[x] = s.predict([tx1, t2], verbose=0)
                    
                for dep in range(1, 10):
                    for ix,x in enumerate(cdxx):                        
                            t1= np.copy (tx1, order='K', subok=False)
                           # t1[:, :dep, ix]= npa(rn.choices(yxx_all[x].values, k=len(t1) * dep )).reshape(-1, dep)
                            t1[:, :dep, ix]= npa(rn.choices(t1[:, :dep, ix], k=len(t1[:, :dep, ix]) )).reshape(t1[:, :dep, ix].shape)
                    u[dep] = s.predict([t1, tx2], verbose=0)    
                    
                #rd= [ PDF(u[dep], columns=[f'y_20h_{dep}', f'y_40h_{dep}']) for k,v in u.items()]                  
                #rd=  PDF(u[dep], columns=[f'y_20h_{dep}', f'y_40h_{dep}'])               
                    
             
                ru= [ PDF(u[k], columns=[f'y_20h_{k}', f'y_40h_{k}']) for k,v in u.items()]
                r0= PDF(ty.reshape(-1,2), columns=['y_20','y_40'])
                r1.append(pd.concat([ r0,   *ru   ], **a1))
    

           # Print the results
            def rc_with(x, patt): return  [c for c in x if c.find(patt) >= 0] 
            pd.core.indexes.base.Index.rc_with= rc_with
            #pd.core.indexes.base.Index.rc_with= lambda x, patt:  [c for c in x if c.find(patt) >= 0]
            #PDF.columns.rc_with= rc_with            
            #pd.core.frame.DataFrame.columns.rc_with= rc_with
            #pd._libs.properties.AxisProperty.rc_with= rc_with
            
            yhh=  pd.concat(r1)
            print('yhh: ', yhh.columns)
            #yhh.tab('Predictions + with permutated xx  yhh=', nr=40)
            #yhh[[c for c in yhh.columns if c.find('_20') >0]].tab('Predictions + with permutated xx , .._20.., yhh=', nr=20)
            yhh[rc_with(yhh.columns, '_20')].tab('Predictions + with permutated xx , .._20.., yhh=', nr=20)
            #di(yhh[[c for c in yhh.columns if c.find('_20') >0]])
            #di(yhh[[c for c in yhh.columns if c.find('_40') >0]])
            pgre('Corr if x -> random permutation (for  x  after  "y_.0h_00")   .._20..:')
            co20= 100* yhh[rc_with(yhh.columns, '_20')].corr()
            co40= 100* yhh[rc_with(yhh.columns, '_40')].corr()
            co=   100* yhh.corr()
            co20.tab('Corr  20:', nr=99, ndig=1, tablefmt="rst",  headers=[f[2:] for f in co.columns])  # maxcolwidth=8,
            
            
            
            #### Plot corr matrix
            
            #print(co.loc[co.index.rc_with('y_20'), 'y_20h_00'].round(2))
            #print('\n', co.loc[co.index.rc_with('y_40'), 'y_40h_00'].round(2))
            print(co20.round(2))
            print('\n', co40.round(2))
             
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Variable excluding, corr')

            for i, iy in enumerate(cy): 
                coi= co20 if i==0 else co40
                axes[i].set_title('with ' + iy)
                #sns.heatmap(co.loc[co.index.rc_with(iy), co.columns.rc_with(iy)],  
                sns.heatmap(coi, annot=False, cmap="viridis", ax=axes[i])
            plt.show()

            
            # Plot the predictions, corr and  lift
            if toPlot:        
            
                fig, axes = plt.subplots(nrows=2, ncols=2)
            
                for i, y in enumerate(cy):
                    yh= yhh[[y, y+'h_00']].sv(y+'h_00', ascending=False).ri(drop=True)
                    lif= np.cumsum(yh[y]) #.plot(title='Lift ' + y)
                    auc= np.mean(lif) - lif.values[-1] /2
                    axes[0,i].plot(yh)
                    axes[1,i].plot(lif)
                    axes[1,i].plot([0, len(yhh)], [0, lif.values[-1]])
                    axes[0,i].set_title(f'{y}, {nEp=}', x=.7, y=.8)            
                    axes[1,i].set_title('Lift ' +y, x=.7, y=.8)
                plt.show()
   
                print(f'\n\nMean Win for  {y} : AUC= {auc:6.2f}% ')
                    

            #### calc win
            wins= Model.winsf(yhh) 
            
            
            if 0:
                wins= []
                for i, y in enumerate(cy):
                    yh= yhh[[y, y+'h_00']].sv(y+'h_00', ascending=False).ri(drop=True)
                    # Wins
                    for nw in nWin:
                        nw1= int(round(nw * len(yh) )) if nw <=1 else nw
                    
                        #win50= yh.iloc[:nw1][y].mean() - yh.iloc[-nw1:][y].mean()
                        #win= yh.iloc[:nw1][y].mean() - yh.iloc[-nw1:][y].mean()
                        win= yh.head(nw1)[y].mean() - yh.tail(nw1)[y].mean()  # % / day   
                    
                        dt_minut= int(y[-2:])  # hard coded

                       # win_ann= win/ dt_minut * 300 *200 # * trade_minutes * daysInYear
                        win_ann= win  * 200 # * daysInYear
                        wins.append(dict(y=y, nWin=nw, win= win, win_ann= win_ann))            
    
                wins= PDF(wins) #; wins.index= wins.nWin
                print(f'\n\nMean Win for  {y} : AUC= {auc:6.2f}% ') 
                wins.tab('Wins, best - worst , %', showindex=False, nr=99)
                
           
            
            #### Variable & Depth importance, correl & win
            nw1= int(round(.05 * len(yhh)))
            pr('len(yhh)= ', len(yhh), 'nw1= ', nw1)
            impo, win= [],{}

            for y in cy:  # ['y_20','y_40']:
                #for x in '00 Close High Low Open Volume  rule_5_20 rule_10_30'.split() + list(range(1,10)):
                for x in ['00', *cdxx, *csxx] + list(range(1,10)):
                    try:
                        yhh= yhh.svde(y+f'h_{x}').ri(drop=True)
                        win[(x, y)]= yhh.head(nw1)[y].mean() - yhh.tail(nw1)[y].mean()
                        #impo_win= win[(x, y)]  if x=='00' else win[('00', y)] - win[(x, y)]
                        impo_win=  win[('00', y)] - (0  if x=='00' else win[(x, y)])
                            
                        impo.append(dict(y=y, x=x, impo_cor= co.loc[y, y+'h_00'] - co.loc[y, y+ f'h_{x}'], 
                                                   impo_cor2= 100- co.loc[y+'h_00', y+ f'h_{x}'], 
                                                   impo_win= impo_win
                                   ))
                        
                    except Exception as e: 
                        pred(f'{y=}, {x=}, Except {e=} ')
           
            print(green, '------  Importance  ------', er)        
            #PDF(impo).pivot(index='x', columns= 'y', values=['impo_cor','impo_cor2','impo_win']).sv(('impo_win', 'y_40'), ascending=False).round(3) #.tab('impo 3', nr=99)   
            display(PDF(impo).pivot(index='x', columns= 'y', values=['impo_cor','impo_cor2','impo_win']).svde(('impo_cor2', 'y_40')).round(3)) #.tab('impo 3', nr=99)   
                        
            impo_cor=  PDF(impo).pivot(index='x', columns= 'y', values='impo_cor').svde('y_40')
            impo_cor2= PDF(impo).pivot(index='x', columns= 'y', values='impo_cor2').svde('y_40')
            impo_win=  PDF(impo).pivot(index='x', columns= 'y', values='impo_win').svde('y_40')
            
            pd.concat([ PDF(impo).pivot(index='x', columns= 'y', values='impo_cor'),  # no header
                        PDF(impo).pivot(index='x', columns= 'y', values='impo_cor2'),
                        PDF(impo).pivot(index='x', columns= 'y', values='impo_win')                       
                     ], **a1)
            
            impo_cor.tab('Importance (reduction of correl with y after random perm of x)', nr=99, ndig=9)
            #display(impo_cor)
            impo_cor2.tab('Importance (1- correl with yh after random perm of x)', nr=99)
            impo_win.tab('Importance (reduction of win after random perm of x), at 5% top & bottom', nr=99)


            return yhh, co, wins, win, impo, impo_cor, impo_win, inds

#%% test Model ----
keras.models.Model.plot_layers=  Model.plot_layers   
keras.models.Model.cv=  Model.cv     

if 0:
    try:
        keras.engine.functional.Functional.plot_layers= Model.plot_layers
        keras.engine.functional.Functional.cv= Model.cv
    except Exception as e: pred('Exception:', e)  


if 0 and toTest:
    dir(th)
    th.cdxx, th.csxx
    
    fModel= f'{out}/model2_32_32_3000'
    m= Model(th=th, nLSTM=64, nStat=64, nDen2=64, nEp=250, fModel='', toPlot=True)
    m.model.summary()
    #m.model= tf.keras.saving.load_model(fModel)
    
    m.__class__ = Model
    #m.model.set_weights('out/model_r6a4t_128-64-64_3002ep_wghts.pck')
    w= pilo('out/model_r6a4t_128-64-64_2002ep_wghts.pck')
    w= pilo('out/model_r6a4t_128-64-64_3002ep_wghts.pck')
    m.model.set_weights(w)
    #m.model.set_weights('out/model_r6a4t_128-64-64_302ep_wghts.pck')
    
    yxxs= pilo('out/yyxs.pck')
    (tx1, tx2), ty, feep= yxxs
    yh= m.train(((tx1, tx2), ty, more), nEp=2) # OK
    dir(m)
    yh= m.model.fit((tx1, tx2), ty, epochs=2)  # OK
    #nOK yh= m.train(yxxs[:2])
    
    hi= m.train(yxxs, nEp=50)  # OK
    hi.history
    m.nEp=200
    

    yxx= pilo('out/yxx_tr.pck')
    di('yxx 0 2 =', yxx['F'][:2]) #list(yxx.values())[:2]) 
    
    
    yxx_all= sum([[y for y in yxx[a] if len(y.dropna()) ==10] for a in yxx if a > ''], [])  
    
    pr(' yxx_all= ',  len(yxx_all))
    yxx_all[-1].tab('yxx_all[-1]')
    
    
    #### cv    
    #yh= model.cv(yxx_all)
    #yhh, co, wins, win, impo,  impo_cor, impo_win= m.cv(model, yxx_all, n_splits= 5, verb=0)
    #yhh, co, wins, win, impo,  impo_cor, impo_win= m.cv(th, yxxs, n_splits= 5, verb=0)
    yhh, co, wins, win, impo, impo_cor, impo_win, inds= m.cv(th, yxxs, n_splits= 5, verb=0)
    if 0:
        di('3333. yh=', yhh, co)
        co.tab(nr=99, ndig=1)
    

       
    #100* ryh.corr().round(4)
    
    #PDF(ry).ri().pivot(None, 'level_0')         

    
        

@dataclass
class Thinker:
#%% class Thinker
   fee= .0035  # $ per sell/buy a stock

   def __init__(s):         
       s.fee= .0036
       pr('Thinker, __init__')
   
   class Might_Need_Later:
 
       def findTrader(s):
            pr('Thinker, findTrader')
            
       #def loadModel(s, m:Model): 
       def loadModel(s, model): 
            pr('Thinker, loadModel')
    
       def hist2xx(s):
           '''  in prep_XX_ret'''
           pr('Thinker, hist2xx')
       
       def xx2yh(s):
               pr('Thinker, xx2yh')
    
       def sendActions2Trader(s):
               pr('Thinker, sendActions2Trader')
               
               

    
       def sum_ker(yxxs):
            ''' list of PDF --> keras  '''
            tx1= npa(pd.concat([y[0][0] for y in yxxs], **a1))
            tx2= npa(pd.concat([y[0][1] for y in yxxs], **a1))
            ty= npa(pd.concat([y[1] for y in yxxs], **a1))
            return (tx1, tx2), ty
       
       def subs_ker(yxxs, ii):
            ''' subset '''
            y= yxxs
            tx1= npa(pd.concat([y[i][0][0] for i in ii], **a1))
            tx2= npa(pd.concat([y[i][0][1] for i in ii], **a1))
            ty= npa(pd.concat([y[i][1] for i in ii], **a1))
            return (tx1, tx2), ty               
               
           
   # train/test:  - list of DF -> keras tensor        
   #def lidf2ker(cxx, tr, onlyLast=False):  
   def lidf2ker(cxx, tr, nLast=0):  
       return  npa([npa( (t.iloc[-nLast:] if nLast > 0 else t )[cxx])  for t in tr])


   def trd2Xy(s, train_data) :        
       lidf2ker= Thinker.lidf2ker
       xx= s.cdxx #['Close','High','Low','Open','Volume'] if 'Close' in train_data[0].columns \
           #else ['c','h','l','o','v']
       pr('in trd2Xy   xx=', xx)    
       ns= len(s.csxx)
       sh= [-1,  ns] if s.nLast==1 else  [-1, s.nLast, ns]
       return   [lidf2ker(xx, train_data), 
                 lidf2ker(s.csxx, train_data, onlyLast=True).reshape(*sh)
                ], lidf2ker(s.cy, train_data, onlyLast=True) #.reshape(-1,2)        
   

   def prep_XX_ret(s, sss='SPY AAPL MSFT F', fHist='', histo='', 
                   kma=[5,10,20,30],  var=['ret','a4','ta'], nLast=1, tp_rsi=5, n_pred=[20,40],
                   toPlot=True, n_hist=10, row0=False, verb=1):  #, noCalc=False
        ''' prep input for keras train and CV from yf  historical OHLCV data.
        create Rule x-variables and y_20, y_40  variables
        var= ['ret', 'a4'] ,  a4 - norm to CHLO
        
        Thinker.prep_XX() ->
           rr=  {ass: PDF (389 x 11)}
           yxx= {ass: list 339 x PDF (10 x 11)}  -> pilo('out/yxx_tr.pck') normalized
        -> Thinker.crea_model
        
            # prep yxx     
            yxx[s]=[]
            for i in range(n_hist, len(q1)- max(n_pred)):
                u= q1.iloc[i- n_hist: i].copy()
                q_norm= u.apply(lambda r: .25*(r.Open+ r.High+ r.Low+ r.Close), **a1).values[-1]
                for c in ['Open','High','Low','Close']: u.loc[:, c]= norm_last(u[c], q_norm )
                for c in ['rule_5_20','rule_10_30']: u.loc[:, c]= norm_last(u[c], q_norm, m1=0 )
                u.loc[:,'Volume']=  np.log(norm_last(u['Volume'], m1=0)/100)
                yxx[s].append(u)        
        
        toNorm= 'ret' '''
        pr, di, pgre, pred= (dummy, dummy) if verb==0  else  (print, display, pgreGl, predGl)
        if type(var)==str: var= [var]
     
        s.n_hist= n_hist
        s.cdxx= ['c','h','l','o','v','clh'] * ('ret' in var) + \
                ['Open','High','Low','Close','Volume'] * ('a4' in var) \
                + ['rule_RSI', 'rule_k', 'rule_d', 'rule_mom']
        s.csxx= ['rule_5_20','rule_10_30', 'rule_t']
        s.cy=   ['y_20', 'y_40']
        s.more= ['ass', 'Datetime', 'Close0', 'feep']
        s.nLast= nLast
        s.tp_rsi= tp_rsi
        
        if row0: return [[],[]], [], []

        
        if type(histo)==str and histo=='':
            if fHist=='':
                pgre('prep_XX_ret  Observer.get_hist_qoutes')
                _, histo= Observer.get_hist_qoutes(ss=sss, provider='yf')
            else:  
                pr(f'1. {histo=}')
                pr(f'{fHist=}')
                #pgre(f'prep_XX_ret,  {fHist=},  pilo({fHist})')
                histo= pilo(fHist)  #.rename(columns={'level_1':'ass'})
                pgre(f'2. else pilo')
                histo.tab('2. histo')


        def incr(a,b): return 100* np.log((1e-8 + a) /(1e-8 + b))  # log increment
        def em(x,k): return x.ewm(com=k).mean()            # exp moving average
        pd.Series.em= em
        
        try:
            histo= histo.rename(columns={'level_2':'Datetime'})
        except: pass
    
        try:
           histo= histo.pivot(index=['level_1','Datetime'], columns='level_0', values=0).ri() 
           #histo= histo.pivot(index=[histo.level_1, histo.Datetime], columns='level_0', values=0) 
        except Exception as e: 
            pred(f'Exception {e=} in histo= histo.pivot')
            #return
        
        if 0:
            th.__class__= Thinker
            yxx= th.read_hist_dir(dire='out', list_only= False, patt='hist_07-07', var=['ret','a4'], nLast=1, tp_rsi=2)
            
            th.__class__= Thinker
            yxxs=  th.read_hist_dir(dire='out', list_only= False, patt='hist_06', 
                                    var=['ret','a4'], nLast=1, tp_rsi=2)
            len(yxxs)
            shape_ker(yxxs)
        
        histo.tab('3. histo')
        
        if 'ta' in var: 
            taa, code= Thinker.add_ta2hist(hi=histo, no_ohlcv=True)
            #s.cdxx= list(set(s.cdxx + list(taa.columns)[2:]))
            s.cdxx= s.cdxx + [c for c in taa.columns if c not in histo.columns]
            
            #histo= pd.concat([histo, taa.iloc[:, 2:]], **a1) # taa.tabb()
           # histo= pd.concat([histo, taa.drop([c for c in taa.columns if c in histo.columns], **a1)], **a1) # taa.tabb()
            histo= pd.concat([histo, taa[[c for c in taa.columns if c not in histo.columns]]], **a1) # taa.tabb()
            cdxx= s.cdxx + list(histo.columns)[2:]
            pgre(f'-------------  {cdxx=},   \n{s.cdxx=}, \n\n{taa=}, \n{histo=}') 
            
        if 0: #zzzzzz ?? to delete
            th= Thinker()
            #uu, uud, yxx_ker= th.prep_XX_ret(sss='MSFT F', kma=[5,10,20, 30], toPlot=True, n_pred=[20,40], var=['ret', 'a4', 'ta'])
            uu, uud, yxx_ker= th.prep_XX_ret(sss='', fHist='out/hist_4_06-30.pck', 
                      kma=[5,10,20, 30], toPlot=True, n_pred=[20,40], var=['ret', 'a4', 'ta'])
            yxx_ker
            yxx_ker[0][0].shape

        
        uu, uud, feep, errs= [],{},[],[]
        for a, r in histo.groupby('level_1'):  # a = asset = Symbol
            pr(Y, f'{a, r.shape=},  {r.dropna().shape=}, \n{r=}')
            try:
                q1= r.drop('Adj Close', **a1) #.tab('q1')  # no Adj.Close for  SPY
            except: pass
        
            if a=='ZION': r.tab('r', nr=22)
        
            #v= r.dropna()
            Dt_minut= (r.Datetime.iloc[-1] - r.Datetime.iloc[0]).total_seconds() /60
        
            pgre(f'{a=}, {len(r)=}, {len(r.dropna())=},  {Dt_minut=}')

            if len(r.dropna()) <  Dt_minut -20: # -1: 
                pred(f'Skipping {a=}, {len(r)=}, {len(r.dropna())=},  {Dt_minut=}')
                #r.pivot('Datetime', 'level_0', 0).tab(f'Skipping r (pivot), {a=}')
                r.tab(f'Skipping r (pivot), {a=}')
                continue
 
            try:                
                c= r.Close.values
                rule_RSI= rsi= npa(talib.RSI(c, timeperiod=tp_rsi)).reshape(-1) , #  rule_RSI(r.Close, dt=20),  # https://www.qmr.ai/relative-strength-index-rsi-in-python/  
                pgre(f'{r.Close.values=}\n{rsi=}\n{rsi[0]=}')
                rsi= rsi[0]                
                #rule_k, rule_d = talib.STOCH(rsi, rsi, rsi)
                rule_k, rule_d=  talib.STOCHRSI(c)
                pgre(f'{rule_k=}\n{rule_d=}, \n{len(rule_k)=}\n{len(rule_d)=},\n{len(r)=},  ')
            except Exception as e: 
                pred(f'Exception', e, a, len(r.dropna()))
                errs+= [dict(a=a, lr=len(r.dropna()))]
                continue


            #yxx= PDF(dict(
            yxx= PDFd(
                y_20= incr(r.Close.shift(-20), r.Close) * day_minutes / 20,  # % / day
                y_40= incr(r.Close.shift(-40), r.Close) * day_minutes / 40,  # % / day
                rule_5_20=  100*(r.Close.em( 5) / r.Close.em(20) -1),
                rule_10_30= 100*(r.Close.em(10) / r.Close.em(30) -1 ),
                rule_t= r.Datetime.apply(lambda d: 0 if d.time() < ttime(10, 10)  \
                   else 1 if  d.time() < ttime(15, 00) \
                   else 2),
                    
                rule_RSI=rsi, rule_k=rule_k, rule_d = rule_d, 
                rule_mom = talib.MOM(r.Close, timeperiod=tp_rsi),                
                    
                ass= a, Datetime= r.Datetime, Close0= r.Close, feep= s.fee / r.Close  # for yxx_ker[2] = more
                #index=[0]
                )
                
            #pgre(f'{yxx.shape=}, yxx=')  
            yxx.tab(f'1.{a=}. {yxx.shape=},  yxx', nr=3)
                
                
            if 'ret' in var:
                #yxx= pd.concat([yxx, PDF(dict(
                yxx= yxx.appe_col(
                    o= incr(r.Open  , r.Close),
                    h= incr(r.High  , r.Close),
                    l= incr(r.Low   , r.Close),
                    c= incr(r.Close , r.shift().Close ),
                    clh= incr(r.Close - r.Low, r.High - r.Low) / 100,
                    #v= incr(r.Volume , r.shift().Volume )
                    v= r.Volume / r.shift().Volume
                ) #) ], **a1)
                
            if 'a4' in var:
                q_norm= q1.apply(lambda r: .25*(r.Open+ r.High+ r.Low+ r.Close), **a1).values[-1]
                #for c in ['Open','High','Low','Close']: q1[c]= q1[c] / q_norm - 1
                q1= {c: r[c] / q_norm - 1 for c in ['Open','High','Low','Close']}
                q1['Volume']=  r['Volume']/ r['Volume'].iloc[-1]  # dup?
                
                yxx= yxx.appe_col(**q1)
                
            if 'ta' in var:
                yxx= pd.concat([yxx, taa.iloc[:, 2:]], **a1) 
    
                
            yxx.tab(f'2. {a=}. yxx', nr=3)
    
                
            if 0: yxx= yxx.dropna().query('rule_5_20 != 0')
            
            yxx.tab(f'3. {a=}. yxx', nr=3)

            
            #### plot yxx
            if len(yxx) >0: 
                yxx.tab(a + '--> u')
                #yxx[['c' if 'ret' in var else 'Close', 'rule_5_20', 'rule_10_30']].ri(drop=True).plot(title= f'{a}, {len(yxx)=}')
                yxx[['c'] * ( 'ret' in var) +  ['Close'] * ('a4' in var) + s.csxx].ri(drop=True).plot(title= f'{a}, {len(yxx)=}')
                plt.show()
            
                
            #### slice yxx --> u
            #pgre(f'slice yxx --> u {n_hist, yxx.shape, len(yxx)=}')
            u1= []    
            for i in range(n_hist, len(yxx)):
                v= yxx.iloc[i- n_hist: i].copy().dropna()
                #if len(v)== n_hist  and v.max() < np.inf :
                if len(v) != n_hist : continue
            
                #v.tab('11111 v')
                pr(Y, f'{s.cdxx=}, \n{v.columns=}')
                s.cdxx= [c for c in s.cdxx if c in v.columns] # ???
                #if len(v)== n_hist  and max(v[s.cdxx].max()) < 9e9 :                    
                if len(v)== n_hist  and max(v.drop(columns=['ass', 'Datetime']).max()) < 9e9 :                    
                   # max(np.max(uu[0][th.cdxx]))
                    uu.append(v.copy())
                    u1.append(v.copy())
                else: 
                    pr(Y, v.Volume, v.iloc[:20, 9:40].tab())
                    pred(f' miss {a=},  {i=}, {len(yxx)=},    {len(v)=},   \n{v=} \n{PDF(v[s.cdxx].max()).ri().tab(nr=99)=} \n{max(v[s.cdxx].max())=}')
            uud[a]= u1
            
            if 0:
                th= Thinker()
                m.__class__= Model
                yxx07= th.read_hist_dir(dire='out', list_only= False, patt='hist_07-11', var=['ret','a4','ta'], nLast=1, tp_rsi=2)
               
                
                    
        pgre('len(uu)=', len(uu))
        #pgre('uu=', uu)
        
        if len(uu)==0:
          #pred(f'len(uu)==0, {len(v)=}, {v=}, return []')
          pred(f'len(uu)==0, return []')
          return  [[],[]], [], []
      
        di('uu[0]=', uu[0].columns, uu[0])
        #return

        ####  transform uu --> yxx_ker  by variable groups for keras:
        lidf2ker= Thinker.lidf2ker
        ns= len(s.csxx) 
        sh= [-1,  ns] if s.nLast==1 else  [-1, nLast, ns]
        #sh= [-1,  *( [] if s.nLast==1 else  [nLast]), ns]
        
        yxx_ker=   [lidf2ker(s.cdxx, uu), 
                    lidf2ker(s.csxx, uu, nLast=nLast).reshape(*sh)
                   ], \
                   lidf2ker(s.cy, uu, nLast=1), \
                   pd.concat([(u[s.more].iloc[-1:]) for u in uu])    # fee %
                   #lidf2ker(s.more, uu, nLast=1) #.redundant??   # fee %
                   #.reshape(-1,2) 
                   
        if errs != []: pred('errs=', PDF(errs))           
                    
        return uu, uud, yxx_ker   



    
   def read_hist_dir(s, dire='out', list_only= False, patt='hist_', var='ret', nLast=1, tp_rsi=5):
        ''' hist files from dire ---> keras '''
        #yxxs= [Thinker().prep_XX_ret(fHist=di+ '/'+ f ,  kma=[5,10,20,30], var='ret',
        #                toPlot=True, n_pred=[20,40], n_hist=10)[2] for f in os.listdir(di) if f[:5]=='hist_']
        ff= [f for f in os.listdir(dire) if f.startswith(patt) and f.endswith('pck')] 
        for f in  ff: pgre('file', f)
        if list_only: return ff
        
        pgre(f'In read_hist_dir():  {ff=}')
        
        tx1, tx2, ty, more=  [],[],[],[]
        
        for f in ff:
                pgre('reading file ', f)
                u= s.prep_XX_ret(fHist= dire+ '/'+ f ,  kma=[5,10,20,30], var=var, nLast=nLast, tp_rsi=tp_rsi,
                                toPlot=True, n_pred=[20,40], n_hist=10) 
                try:
                    (a,b), c, d= u[2]
                    info(a, 'a'); info(b, 'b'); info(c, 'c'); info(d, 'd'); 
                    #tx1 = a if i==0 else np.concatenate(tx1,a) 
                    #tx2 = b if i==0 else np.concatenate(tx2,b) 
                    #ty = c if i==0 else np.concatenate(ty,c) 
                    tx1 += list(a)
                    tx2 += list(b)
                    ty  += list(c)
                    #more += list(d)
                    more += [d]
                   # ty  += c
                except Exception as e:
                    pred(f'Exception {e= } at {f=}, {u=}')  # ',  {len(u)=}')
        #return (npa(tx1), npa(tx2)), npa(ty) , npa(more)  #yxxs  #sum_ker(yxxs)
        return (npa(tx1), npa(tx2)), npa(ty) , more  #yxxs  #sum_ker(yxxs)
    
  # def add_ta2hist(hi= pilo('./out/hist_4_06-30.pck'), no_ohlcv=True):
   def add_ta2hist(hi= '', no_ohlcv=True):
        cxx_ta_full= pilo('cxx_ta_full.pck')
        code= '\n'.join([f"ta['ta_{r.v}']= {r.t}" for i,r in cxx_ta_full.iterrows()])
        pr(code)
        taa= []      
        try:
            hi= hi.rename(columns={'level_1':'ass'})
        except:pass
        
        for a, h in hi.groupby('ass'):
            #pgre(f'treating {a=}') 
            ta= h[['ass','Datetime']].copy()  if no_ohlcv else h.copy()
            try:
                loca= {'talib':talib, 'x':0, 'high':h.High, 'low':h.Low, 'close':h.Close, \
                      'open':h.Open,'volume':h.Volume,  \
                      'real':h.Close,  'real0':h.Close, 'real1':h.Close, 'h':h, 'ta':ta}
                exec(code, {}, loca)  # add columns to  ta
                taa.append(ta)
            except: pass
        taa= pd.concat(taa)
        hi_ta= hi[['ass','Datetime']].merge(taa, on=['ass','Datetime'])
        #hi_ta.tabb('hi_ta')
        if no_ohlcv: hi_ta= hi_ta[cxx_ta_full.v.apply(lambda x: 'ta_' + x)]
        return hi_ta, code 
    
   if 0:
        taa1, code= Thinker.add_ta2hist(hi= pilo('./out/hist_4_06-30.pck'), no_ohlcv=False)
        #taa1, code= Thinker.add_ta2hist(hi= pilo('./out/hist_4_06-30.pck'), no_ohlcv=True)
        taa1.tabb('taa1',nr1=20)  ; pr(code)
        
        
   def treat_Dima_IBKR(d=12, yf=False, di='drive-download-20230715T045855Z-001'):
       try: 
           uba= pilo(f'out/fromDima/{di}/hist_07-{d}_bid_ask.pck')
       except: uba= pilo(f'out/fromDima/{di}/hist_07-{d}_ib_bid_ask.pck')
       
       try:
           uib= pilo(f'out/fromDima/{di}/hist_07-{d}_ib_NASDAQ-100.pck')
       except: 
           uib= pilo(f'out/fromDima/{di}/hist_07-{d}_ib.pck')
     
       uba= uba.rename(columns={'level_1':'ass', 'open': 'bta', 'high': 'ama', 'low': 'bmi', 'close': 'ata', })
       uib= uib.rename(columns={'level_1':'ass', 'Datetime': 'date'})
       
       u2= pd.merge(uib, uba, on=['ass','date'], how='inner')
       u2= u2.drop(['volume','average','barCount','Adj Close','Rep'], **a1)
       u2.tabb('u2')   
       
       if yf:
           uyf= pilo(f'out/fromDima/{di}/hist_07-{d}_yh_NASDAQ-100.pck')
           uyf= uyf.rename(columns={'level_1':'ass', 'Datetime': 'date'})
           
           u3= pd.merge(u2, uyf, on=['level_1','date'], how='inner')
           u3.columns
           u3= u3.drop(['volume','average','barCount','Adj Close_x','Adj Close_y','Repaired?','Rep'], **a1)
           u3.tabb('u3')
          

           #u3[['Close_x','Close_y']].plot()
           
           from bokeh.plotting import figure as figu, output_file, show, row    
           from bokeh.models  import Plot, Scatter        
           
           p,p2= figu(), figu() #title='Keras Learning Curves', x_axis_label='epoch', y_axis_label='Loss', width=900, height=600)
           #p.line(u3.Close_x, u3.Close_y,  line_color= 'red', legend_label='training')  # axes
           #p.line(x=u3.Close_x, y=u3.Close_y-u3.Close_x,  line_color= 'red', 
           #       legend_label='training', line_dash='none' ) #       
           p.scatter(x=u3.Close_x, y=u3.Close_y-u3.Close_x)  #,  line_color= 'red',    legend_label='training' ) # axes
          # p.scatter(x=u3.Volume_x, y=u3.Volume_y/(.01+u3.Volume_x))  #,  line_color= 'red',    legend_label='training' ) # axes
           p2.scatter(x=u3.Volume_x/ 1e3, y=u3.Volume_y/1e6)  #,  line_color= 'red',    legend_label='training' ) # axes
           #p.line(range(history_df.shape[0]), history_df['val_loss'],  line_color= 'blue', legend_label='val')  # axes
           show(row([p,p2]))  
           
       # sort by quality = (H-L) / (a-b)
       def qual(r): return (r.High - r.Low).mean() /(r.ama - r.bta).mean()
       quali= PDF(u2.groupby('ass').apply(qual)).ri().rename(columns={0:'qual'}).svde('qual')
           
       return u2, quali     
       
       u.level_1.vc()
       u.tabb()
       u.level_1.vc()
       u.level_1.vc()    

   if 0:
        hi, quali= Thinker.treat_Dima_IBKR(d=12, di='f-20230721T231824Z-001') #OK
        hi, quali= Thinker.treat_Dima_IBKR(d=13, di='f-20230721T231824Z-001')
        hi, quali= Thinker.treat_Dima_IBKR(d=14, di='f-20230721T231824Z-001') #ok
 
        hi, quali= Thinker.treat_Dima_IBKR(d=17, di='f-20230721T231824Z-001')
        hi, quali= Thinker.treat_Dima_IBKR(d=18, di='f-20230721T231824Z-001')
        hi, quali= Thinker.treat_Dima_IBKR(d=19, di='f-20230721T231824Z-001')
        hi, quali= Thinker.treat_Dima_IBKR(d=20, di='f-20230721T231824Z-001')
        hi.tabb('hi')    
        quali
        for i, c in enumerate(hi.columns): pr(i, c)
        
        fee=.0036
        
        
        
   def hi2cols_npa(hi, withY=True, nt=10, verb=0):  
        # hi=hi; withY=True; nt=10; verb=1
    
   #def hi2ker(hi, withY= False,  nt=10, verb=0): #
        ''' 
        hi prepared by treat_Dima_IBKR()        
        
        2 modes: 
           - for train and backtest:  withY= True, onlyLast= False
           - for pred:  withY= False, onlyLast= True
           so anyway onlyLast=  not withY
        '''
        #  withY= False; onlyLast=False; nt=3
        onlyLast=  not withY
        
        pr, di, pgre, pred= (dummy, dummy, dummy, dummy) if verb==0  else  (print, display, pgreGl, predGl)

        def incr(a,b): return (100* np.log((1e-8 + a) /(1e-8 + b))).values.reshape(-1)  # log increment in %
        def em(x,k): return x.ewm(com=k).mean()                    # exp moving average
        pd.Series.em= em
        
        fee= .0035  # $ per sell/buy a stock
        Dt_minut= int( .01 + (hi.date.iloc[-1]- hi.date.iloc[0]).total_seconds()/60) # 389.0
        
        
        yy, yyq, xxd, xxs, more, ker= {},{},{},{},{}, npa([])
        
        for a, r in hi.groupby('ass'):
            pr(Y, a, len(r))
    
            if len(r) != Dt_minut +1: # wrong number
                pr(R, a, len(r), Dt_minut)
                continue
            #else: pass
            
            pr(Y, a, r.shape, Dt_minut, npa(r)[:, 2:])
            
            #x=  np.empty([Dt_minut +1, hi.shape[1] + len(cxd)]) 
            #x=  np.empty([Dt_minut +1, hi.shape[1] + 62])  # dynamic
            #x=  np.empty([Dt_minut +1, hi.shape[1] + 80])  # dynamic
            x=  np.zeros([Dt_minut +1, hi.shape[1] + 80])   # dynamic
            xs= np.empty([Dt_minut +1, 2]) # static
            
            cl= r.Close
            
            if withY:
                y=  np.empty([Dt_minut +1, 3])  # = yn 
                yq= np.empty([Dt_minut +1, 3])   
                y[:, 0]= incr( cl.shift(-20), cl) * 300 / 20    ##= y_20
                y[:, 1]= incr( cl.shift(-40), cl) * 300 / 40    ##= y_40
                y[:, 2]= incr( cl.shift(-60), cl) * 300 / 60    ##= y_60
                yq[:, 0]= w(  5 <  y[:, 0], 1, 0)                      ##= yqb_20   buy   in 20 min       
                yq[:, 1]= w((-5 <= y[:, 0]) &  (y[:, 0] <= 5), 1, 0)   ##= yqh_20 sell in 20 min       
                yq[:, 2]= w( y[:, 0] < -5, 1, 0)                       ##= yqs_20 sell in 20 min 
            else: y,yq=[],[]
                
            dc= hi.shape[1] -2                # 9  hi columns but ass, date
            x[:, :dc]= npa(r.iloc[:, 2:])
            
            xs[:, 0]= rule_5_20=  100*(cl.em( 5) / cl.em(20) -1)             ##= xs_5_20
            xs[:, 1]= rule_10_30= 100*(cl.em(10) / cl.em(30) -1 )            ##= xs_10_30
            
            
            #dc= dc-2
            dc=10
            x[:, dc+2]= rule_t= r.date.apply(lambda d: 0 if d.time() < ttime(10, 10) else  \
                                                       1 if d.time() < ttime(15, 00) else \
                                                       2)            ##= xd_t
                
            x[:, dc+3]=  o= incr(r.Open  , cl)           ##= xd_o
            x[:, dc+4]=  h= incr(r.High  , cl)           ##= xd_h
            x[:, dc+5]=  l= incr(r.Low   , cl)           ##= xd_l
            x[:, dc+6]=  c= incr(cl , r.shift().Close )               ##= xd_c
            x[:, dc+7]=  clh= incr(cl - r.Low, r.High - r.Low) / 100  ##= xd_clh
            x[:, dc+8]=  v= r.Volume / r.shift().Volume               ##= xd_v
            
            q_norm=  .25*(r.Open+ r.High+ r.Low+ cl)
            for i,c in enumerate(['Open','High','Low','Close']): 
                x[:, dc+9 +i]= incr( r[c] , q_norm )    ##= xd_O    9-12
                                                        ##= xd_H
                                                        ##= xd_L
                                                        ##= xd_C
            
            x[:, dc+13]=  incr(r.Volume,  r.shift().Volume)             ##= xd_V   ##=  dup?
            
            
            if 0: # prepare code
                 cxx_ta_full= pilo('out/cxx_ta_full.pck')
                 code2= '\n'.join([f"x[:, dc+{14+i}]= {r.t[3:]:50}.astype(double); cta.append('xdta_{r.v}')  ##" + f"=  {i} {r.v}" for i,r in cxx_ta_full.iterrows()])
                 pr(code2)  
            
            if 0:  # print variable names   
                 ##=   gg '##= ' fun.py | sed 's/.*##= \+ //; s/ .*//;' 
                 #sep='_____'+ '=====zzzz'
                 u= pd.read_csv('fun.py', sep='_____'+ '=====zzzz')
                 yxx= [v.astype(str)[0].split('##= ')[-1].split(' ')[0] 
                       for i,v in u.ri(drop=True).iterrows()  
                       if v.iloc[0].find('##= ') >= 0]
                 cy=  [c for c in yxx if c.startswith('y_')]  # regr
                 cqy= [c for c in yxx if c.startswith('yq')]  # classific
                 #cxd= list(r.columns) + [c for c in yxx if c.startswith('xd')]  # dynamic
                 cxd= f'{r.columns=}  +  { [c for c in yxx if c.startswith("xd")]}'  # dynamic
                 cxs= [c for c in yxx if c.startswith('xs')]  # static
                 #cy, cqy,  ppr(cxd, compact=True), cxs
                 print(B, f'{cxd=} \n{cxs=} \n{cy=}, \n{cqy=}')
                 hi.shape, len(cxd), len(cxs), x.shape
                 return
             
                 Thinker.hi2cols_npa(hi, withY=True, nt=10, verb=0)
                 
            open, high, low, close, volume, real, real0, real1= r.Open, r.High, r.Low, cl, r.Volume, cl,  cl,  cl
            
            cta= []
            double=float
            dc=20
            
            x[:, dc+14]= talib.ADX(high, low, close  , timeperiod=2  )     .astype(double); cta.append('xdta_ADX')   ##=  0 ADX
            x[:, dc+15]= talib.ADXR(high, low, close  , timeperiod=2  )    .astype(double); cta.append('xdta_ADXR')  ##=  1 ADXR
            x[:, dc+16]= talib.APO(real  , fastperiod=2, slowperiod=4)     .astype(double); cta.append('xdta_APO')   ##=  2 APO
            x[:, dc+17]= talib.AROONOSC(high, low  , timeperiod=2  )       .astype(double); cta.append('xdta_AROONOSC')  ##=  3 AROONOSC
            x[:, dc+18]= talib.CCI(high, low, close  , timeperiod=2  )     .astype(double); cta.append('xdta_CCI')       ##=  4 CCI
            x[:, dc+19]= talib.CDL3OUTSIDE(open, high, low, close)         .astype(double); cta.append('xdta_CDL3OUTSIDE')  ##=  5 CDL3OUTSIDE
            x[:, dc+20]= talib.CDLBELTHOLD(open, high, low, close)         .astype(double); cta.append('xdta_CDLBELTHOLD')  ##=  6 CDLBELTHOLD
            x[:, dc+21]= talib.CDLCLOSINGMARUBOZU(open, high, low, close)  .astype(double); cta.append('xdta_CDLCLOSINGMARUBOZU')  ##=  7 CDLCLOSINGMARUBOZU
            x[:, dc+22]= talib.CDLENGULFING(open, high, low, close)        .astype(double); cta.append('xdta_CDLENGULFING')  ##=  8 CDLENGULFING
            x[:, dc+23]= talib.CDLHAMMER(open, high, low, close)           .astype(double); cta.append('xdta_CDLHAMMER')     ##=  9 CDLHAMMER
            x[:, dc+24]= talib.CDLHANGINGMAN(open, high, low, close)       .astype(double); cta.append('xdta_CDLHANGINGMAN') ##=  10 CDLHANGINGMAN
            x[:, dc+25]= talib.CDLHARAMI(open, high, low, close)           .astype(double); cta.append('xdta_CDLHARAMI')     ##=  11 CDLHARAMI
            x[:, dc+26]= talib.CDLHIKKAKE(open, high, low, close)          .astype(double); cta.append('xdta_CDLHIKKAKE')    ##=  12 CDLHIKKAKE
            x[:, dc+27]= talib.CDLINVERTEDHAMMER(open, high, low, close)   .astype(double); cta.append('xdta_CDLINVERTEDHAMMER')  ##=  13 CDLINVERTEDHAMMER
            x[:, dc+28]= talib.CDLLONGLINE(open, high, low, close)         .astype(double); cta.append('xdta_CDLLONGLINE')  ##=  14 CDLLONGLINE
            x[:, dc+29]= talib.CDLMARUBOZU(open, high, low, close)         .astype(double); cta.append('xdta_CDLMARUBOZU')  ##=  15 CDLMARUBOZU
            x[:, dc+30]= talib.CDLSHORTLINE(open, high, low, close)        .astype(double); cta.append('xdta_CDLSHORTLINE')  ##=  16 CDLSHORTLINE
            x[:, dc+31]= talib.CMO(real  , timeperiod=2  )                 .astype(double); cta.append('xdta_CMO')  ##=  17 CMO
            x[:, dc+32]= talib.CORREL(real0, real1  , timeperiod=2  )      .astype(double); cta.append('xdta_CORREL')  ##=  18 CORREL
            x[:, dc+33]= talib.DEMA(real  , timeperiod=2  )                .astype(double); cta.append('xdta_DEMA')  ##=  19 DEMA
            x[:, dc+34]= talib.DX(high, low, close  , timeperiod=2  )      .astype(double); cta.append('xdta_DX')  ##=  20 DX
            x[:, dc+35]= talib.EMA(real  , timeperiod=2  )                 .astype(double); cta.append('xdta_EMA')  ##=  21 EMA
            x[:, dc+36]= talib.KAMA(real  , timeperiod=2  )                .astype(double); cta.append('xdta_KAMA')  ##=  22 KAMA
            x[:, dc+37]= talib.LINEARREG(real  , timeperiod=2  )           .astype(double); cta.append('xdta_LINEARREG')  ##=  23 LINEARREG
            x[:, dc+38]= talib.LINEARREG_ANGLE(real  , timeperiod=2  )     .astype(double); cta.append('xdta_LINEARREG_ANGLE')  ##=  24 LINEARREG_ANGLE
            x[:, dc+39]= talib.LINEARREG_INTERCEPT(real  , timeperiod=2  ) .astype(double); cta.append('xdta_LINEARREG_INTERCEPT')  ##=  25 LINEARREG_INTERCEPT
            x[:, dc+40]= talib.LINEARREG_SLOPE(real  , timeperiod=2  )     .astype(double); cta.append('xdta_LINEARREG_SLOPE')  ##=  26 LINEARREG_SLOPE
            x[:, dc+41]= talib.MA(real  , timeperiod=2)                    .astype(double); cta.append('xdta_MA')  ##=  27 MA
            x[:, dc+42]= talib.MAX(real  , timeperiod=2  )                 .astype(double); cta.append('xdta_MAX')  ##=  28 MAX
            x[:, dc+43]= talib.MAXINDEX(real  , timeperiod=2  )            .astype(double); cta.append('xdta_MAXINDEX')  ##=  29 MAXINDEX
            x[:, dc+44]= talib.MFI(high, low, close, volume  , timeperiod=2  ).astype(double); cta.append('xdta_MFI')  ##=  30 MFI
            x[:, dc+45]= talib.MIDPOINT(real  , timeperiod=2  )            .astype(double); cta.append('xdta_MIDPOINT')  ##=  31 MIDPOINT
            x[:, dc+46]= talib.MIN(real  , timeperiod=2  )                 .astype(double); cta.append('xdta_MIN')  ##=  32 MIN
            x[:, dc+47]= talib.MININDEX(real  , timeperiod=2  )            .astype(double); cta.append('xdta_MININDEX')  ##=  33 MININDEX
            x[:, dc+48]= talib.MINUS_DI(high, low, close  , timeperiod=2  ).astype(double); cta.append('xdta_MINUS_DI')  ##=  34 MINUS_DI
            x[:, dc+49]= talib.MOM(real  , timeperiod=2  )                 .astype(double); cta.append('xdta_MOM')  ##=  35 MOM
            x[:, dc+50]= talib.PLUS_DI(high, low, close  , timeperiod=2  ) .astype(double); cta.append('xdta_PLUS_DI')  ##=  36 PLUS_DI
            x[:, dc+51]= talib.PPO(real  , fastperiod=2, slowperiod=4 )    .astype(double); cta.append('xdta_PPO')  ##=  37 PPO
            x[:, dc+52]= talib.ROCP(real  , timeperiod=2  )                .astype(double); cta.append('xdta_ROCP')  ##=  38 ROCP
            x[:, dc+53]= talib.ROCR(real  , timeperiod=2  )                .astype(double); cta.append('xdta_ROCR')  ##=  39 ROCR
            x[:, dc+54]= talib.RSI(real  , timeperiod=2  )                 .astype(double); cta.append('xdta_RSI')  ##=  40 RSI
            x[:, dc+55]= talib.SMA(real  , timeperiod=2  )                 .astype(double); cta.append('xdta_SMA')  ##=  41 SMA
            x[:, dc+56]= talib.T3(real  , timeperiod=2)                    .astype(double); cta.append('xdta_T3')  ##=  42 T3
            x[:, dc+57]= talib.TEMA(real  , timeperiod=2  )                .astype(double); cta.append('xdta_TEMA')  ##=  43 TEMA
            x[:, dc+58]= talib.TRIMA(real  , timeperiod=2  )               .astype(double); cta.append('xdta_TRIMA')  ##=  44 TRIMA
            x[:, dc+59]= talib.TRIX(real  , timeperiod=2  )                .astype(double); cta.append('xdta_TRIX')  ##=  45 TRIX
            x[:, dc+60]= talib.TSF(real  , timeperiod=2  )                 .astype(double); cta.append('xdta_TSF')  ##=  46 TSF
            x[:, dc+61]= talib.ULTOSC(high, low, close  , timeperiod1=2, timeperiod2=4, timeperiod3=6  ).astype(double); cta.append('xdta_ULTOSC')  ##=  47 ULTOSC
            x[:, dc+62]= talib.VAR(real  , timeperiod=2 )                  .astype(double); cta.append('xdta_VAR')  ##=  48 VAR
            x[:, dc+63]= talib.WMA(real  , timeperiod=2  )                 .astype(double); cta.append('xdta_WMA')  ##=  49 WMA

     
            rmav=  max(w(np.any(np.isnan(x), axis=1))[0])        # last row with NA in x    
            if rmav > 5: continue  # ?? hard code 
            np.round(x, 1)
            
            if 0:
                if withY: 
                    yy[a]= y[-1: ]    if onlyLast else y
                    yyq[a]= yq[-1: ]  if onlyLast else yq
                    
                xxd[a]= x[-nt:, : ] if onlyLast else x
                xxs[a]= xs[-1:, : ] if onlyLast else xs
                more[a]= PDFd(ass= a, date= r.date, close= r.Close, feep= fee / r.Close) 
            
            yy[a]=  y   # if withY else []
            yyq[a]= yq  # if withY else []
                
            xxd[a]= x
            xxs[a]= xs
            more[a]= PDFd(ass= a, date= r.date, close= r.Close, feep= fee / r.Close) 
            
        print(G, f'{len(xxd.keys())=}, {xxd.keys()=}')
        print(G, f'{len(xxd)=}')
        
        print(B, f'{len(xxd), len(xxs), len( yy), len( yyq), len( more),  len(ker)=}')
        print(B, f'{list(xxd.values())[0].shape=}')
        print(list(xxd.values())[0])
        print(f'{cta=}')
        
        return (xxd,xxs), (yy, yyq), more, ker 
    
    
    
   def cols2ker(hi, withY= False,  nt=10, verb=0): #
        ''' 
        hi prepared by treat_Dima_IBKR()        
        
        2 modes: 
           - for train and backtest:  withY= True, onlyLast= False
           - for pred:  withY= False, onlyLast= True
           so anyway onlyLast=  not withY
        '''
        #  withY= False; onlyLast=False; nt=3
        #  withY= True; onlyLast=False; nt=3
        onlyLast=  not withY 
        
        (xxd,xxs), (yy, yyq), more, ker =  Thinker.hi2cols_npa(hi, withY= withY,  nt=10, verb=0)
        len(more)
        print(Y, f'{len(xxd.keys())=}, {xxd.keys()=}')
        print(Y, f'{len(xxd)=}')
        
        print(B, f'{len(xxd), len(xxs), len( yy), len( yyq), len( more),  len(ker)=}')
        print(B, f'{list(xxd.values())[0].shape=}')
        print(list(xxd.values())[0])
        
        #return
        
   
        if 0: #zzzzz  test Thinker.cols2ker()
            _ =  Thinker.hi2cols_npa(hi, withY= True,  nt=10, verb=0)  # print cta
            hi, quali= Thinker.treat_Dima_IBKR(d=12)            
            t1()
            xxd,xxs, yy, yyq, more, ker= Thinker.cols2ker(hi, withY= False,  nt=10, verb=0)
            t1()
            ker[0].shape,  ker[1].shape,  ker[0][1], ker[1][0], ker[1][1], 
            xxd,xxs, yy, yyq, more, ker= Thinker.cols2ker(hi, withY= True,  nt=10, verb=0)
            ker[0][0].shape,  ker[0][1].shape,
       
            t1()
            xxd,xxs, yy, yyq, more, ker= Thinker.hi2ker(hi, withY= False,  nt=10)  # ok        
            ker[0].shape, ker[1].shape
            t1()
            xxd,xxs, yy, yyq, more, ker=  Thinker.hi2ker(hi, withY= True,  nt=10)  # nok
            t1()
            #xxd,xxs, yy, yyq, more, ker=  Thinker.hi2ker(hi, withY= True, onlyLast=False, nt=10) # nOK
            ker[0][0].shape, ker[0][1].shape,  ker[1][0].shape, ker[1][1].shape
            t1()
            
            (txd, txs), (tyn,tyq)= ker
            (txd.shape, txs.shape),  (tyn.shape, tyq.shape)
            
            
            from sklearn.preprocessing import StandardScaler

            # Perform data normalization
            scaler = StandardScaler()
            txd1 = scaler.fit_transform(txd)
            
            mnq= Model(n_feats=ker[0][0].shape[2], n_feat_c=2, th=None, nLSTM=128, nStat=64, nDen2=64, nEp=250, n_mem=10, var= 'ynq',
                         #fModel='out/model_128_64_1500', toPlot=True):
                         fModel='', toPlot=True, cdxx=[f'xd_{i}' for i in range(ker[0][0].shape[2])], 
                         csxx=[f'xs_{i}'  for i in range(ker[0][1].shape[1])])
                
                
            mnq= Model(n_feats=ker[0][0].shape[2], n_feat_c=2, name='mnq1', 
                       cdxx=[f'xd_{i}' for i in range(ker[0][0].shape[2])], 
                       csxx=[f'xs_{i}' for i in range(ker[0][1].shape[1])])
            mnq.create(th=None, nLSTM=128, nStat=64, nDen2=64, nEp=250, n_mem=10, var= 'ynq',
                        #fModel='out/model_128_64_1500', toPlot=True):
                        fModel='', toPlot=True)
            mnq.n_feats   
            mnq.n_feat_c   
               
                
            sum(np.isnan(txd)) , sum(np.isnan(txs)), sum(np.isnan(tyn)) ,  sum(np.isnan(tyq))    
            np.max(txd) , np.max(txs), np.max(tyn) ,  np.max(tyq)   
            np.min(txd) , np.min(txs), np.min(tyn) ,  np.min(tyq)   
            for i in range(txd.shape[2]): pr( i, np.max(txd[:,:,i]))
            for i in range(txd.shape[2]): pr( i, np.min(txd[:,:,i]))
            txd[:,:, 15]= 0
            
            
            #### mnq  norm txd --------------------------------------------------------          
            ma=[np.max(txd[:,:,i]) for   i in range(txd.shape[2])]
            mi=[np.min(txd[:,:,i]) for   i in range(txd.shape[2])]
                       
            txd_n= txd.copy()   
            for i in range(txd.shape[2]):  txd_n[:,:,i] = (txd[:,:,i] - mi[i]) / (.001 + ma[i] - mi[i])
            
            # norm txs          
            ma=[np.max(txs[:,i]) for   i in range(txs.shape[1])]
            mi=[np.min(txs[:,i]) for   i in range(txs.shape[1])]
                       
            txs_n= txs.copy()   
            for i in range(txs.shape[1]):  txs_n[:,i] = (txs[:,i] - mi[i]) / (.001 + ma[i] - mi[i])
            #==================================================================
            
            sum(np.isnan(tyn)) # 0
            
            t1()
            #hist= mnq.model.fit(ker[0],ker[1], epochs=100, validation_split= 0.3)    
            hist= mnq.model.fit((txd_n, txs_n), (tyn,tyq), epochs=10, validation_split= 0.3)    
            #hist1= mnq.train((txd, txs), (tyn,tyq), nEp=100, validation_split= 0.3)    
            t1()  # epochs=10 --> Execution time 0:00:32.78
            mnq.save('mnq_200ep')
            
            
            #### mnq validation            
            hist.history.keys()  # ['loss', 'outp_n_loss', 'outp_q_loss', 'val_loss', 'val_outp_n_loss', 'val_outp_q_loss']
            
            u= mnq.cv(th=ker, yxx=ker,  nWin=[.01, .05, .10], n_splits= 5, nEp=5, toPlot=True, verb=1)
            yhh, co, wins, win, impo, impo_cor, impo_win, inds= u
            
            
            hi_te, quali_te= Thinker.treat_Dima_IBKR(d=14)
            (xxd_te,xxs_te), (yy_te, yyq_te), more_te, ker_te =  Thinker.hi2cols_npa(hi_te, withY= True,  nt=10, verb=0)
            #TODO: rm nan  zzzzz

            
            yxxs=  th.read_hist_dir(dire='out', list_only= False, patt='hist_06', 
                                    var=['ret','a4'], nLast=1, tp_rsi=2)   
            m.plot_lift(test_data= yxxs, yhh=PDF(), nEp=1)
            
            if 0:

                        ker= s.trd2Xy(test_data)    
                        di('ker=', ker[0][0][0].round(3), ker[0][1][0])   
                       # predictions = s.model.predict(*(Thinker.trd2Xy(test_data)[0]), verbose=0)
                        predictions = s.model.predict((ker[0]), verbose=0)
                        #yhh= pd.concat([ PDF(Thinker.trd2y(test_data), columns=['y_20', 'y_40']), PDF(predictions, columns=['y_20h', 'y_40h'])], **a1)
                        #yhh= pd.concat([ PDF(s.trd2y(test_data), columns=['y_20', 'y_40']), 
                        #                 PDF(predictions, columns=['y_20h', 'y_40h'])], **a1)
                        
                        yhh= pd.concat([ PDF(s.trd2y(test_data), columns= s.cy), 
                                         PDF(predictions, columns=[y+ 'h' for y in s.cy])], **a1)


        
        if 0:
            display('xxd', len(xxd), list(xxd.values())[0])    
            display("xxd['MSFT']", xxd['MSFT'][0], np.any(np.isnan(xxd['MSFT']), axis=1))    
            display("w(np.any(np.isnan(xxd['MSFT']), axis=1))", w(np.any(np.isnan(xxd['MSFT']), axis=1)))    
            #display("xxd['SPY']", xxd['SPY'][0])   
        try:
            rma= max(w(np.any(np.isnan(xxd['MSFT']), axis=1))[0])
        except: rma=0
        
        
        if withY:
            rmi= min(w(np.any(np.isnan(yy['MSFT']), axis=1))[0]) if withY else len(xxd['MSFT'])
        else: rmi= len(xxd['MSFT'])
        print(B, f'\n1. {rmi=}\n{ len(xxd["MSFT"])=}, \n {yy["MSFT"]=}\n {yy["SPY"]=}')

        if 0:
            try:
                rmav=  max(w(np.any(np.isnan(v), axis=1))[0])
            except: rmav=0   
        
        #a_good= [a for a,v in  xxd.items()]  # if rmav == rma] 
        a_good= [a for a,v in  xxd.items()  if max(w(np.any(np.isnan(v), axis=1))[0]) == rma] 
        #print(B, f'\n{rma=}, {rmav=}, {a_good=}\n')
          
        #rg= range(rma + nt +1, rmi)
            
        if withY: # for training    and backtesting?
            rmi= min(w(np.any(np.isnan(yy['MSFT']), axis=1))[0])
            rg= range(rma + nt +1, rmi)
        else: rg= range(len(xxs)-1, len(xxs))
        print(B, f'2. {rma + nt +1, rmi=}, {rg=}, \n{a_good=}')

        '''check:
            sum(np.isnan(yy['MSFT'][(rma+1):(rmi), :]))  # 0
            sum(np.isnan(xxd['MSFT'][(rma+1):(rmi), :])) # 0
            sum(np.isnan(xxd['MSFT'][(rma+1):(rma+1+nt), :])) # 0
        '''
        
        if withY:
            xxd_ker= [ v[(i-nt): i, :] for i in rg  for a,v in xxd.items() if a in a_good]
            #xxd_ker= [ v[(i-nt, i), :] for i in rg  for  a in a_good  , v:= xxd[a] ]
            xxs_ker= [ v[i, :]         for i in rg  for a,v in xxs.items() if a in a_good]           
            yy_ker=  [ v[i, :]         for i in rg  for a,v in yy.items()  if a in a_good]
            yyq_ker= [ v[i, :]         for i in rg  for a,v in yyq.items() if a in a_good]
            more= {a: more[a] for a in a_good}
            #or
            #xxs_ker= [[ v[i:(i+1), :] for i in range(rma + 10 +1, rmi)] for a,v in xxs.items() if a in a_good]

            ker= (npa(xxd_ker), npa(xxs_ker)), (npa(yy_ker), npa(yyq_ker))
            print(B, f'\n{len(xxd), len(xxd_ker)=}', a_good)
            print(Y, f'\n{ker[0][0].shape, ker[0][1].shape,  ker[1][0].shape, ker[1][1].shape=}')

        else: # forTrade, only last
            xxd_ker= [ v[-nt:, :]  for a,v in xxd.items() if a in a_good]
            xxs_ker= [ v[-1, :]    for a,v in xxs.items() if a in a_good]   
            #ker= (npa(xxd_ker), npa(xxs_ker)), (npa(yy_ker), npa(yyq_ker) )
            ker= (npa(xxd_ker), npa(xxs_ker)), (npa([[]]), npa([[]]) )
            
        pr(B, f'ker {withY=}')    
        for k in ker: # pr(B, k.shape)
            for i in k: pr(B, i.shape)

            
        return xxd,xxs, yy, yyq, more, ker   
    

    
    
   if 0: 
        t1()
        xxd,xxs, yy, yyq, more, ker= Thinker.hi2ker(hi, withY= False, onlyLast=True, nt=10)  # ok        
        ker[0].shape, ker[1].shape
        t1()
        xxd,xxs, yy, yyq, more, ker=  Thinker.hi2ker(hi, withY= True, onlyLast=True, nt=10)  # nok
        t1()
        xxd,xxs, yy, yyq, more, ker=  Thinker.hi2ker(hi, withY= True, onlyLast=False, nt=10) # nOK
        ker[0][0].shape, ker[0][1].shape,  ker[1][0].shape, ker[1][1].shape
        t1()
#%% ========================================================================================================
        
        u= xxd['ZS']
        np.where(np.sum(np.isnan(u),axis=1)==0)[0]
        w(np.sum(np.isnan(u), axis=1)==0)[0]
        lastRowNan= max(w(np.any(np.isnan(xxd['MSFT']), axis=1))[0])
        lastRowNan
        
        rma= max(w(np.any(np.isnan(xxd['MSFT']), axis=1))[0])
        rmi= min(w(np.any(np.isnan(yy['MSFT']), axis=1))[0])
        a_good= [a for a,v in  xxd.items() if max(w(np.any(np.isnan(v), axis=1))[0])== rma]
        len(a_good), len(yy) #  (70, 98)
        
        xxd_ker= [[ v[(i-10, i), :] for i in range(rma + 10 +1, rmi)] for a,v in xxd.items() if a in a_good]
        xxs_ker= [[ v[i, :] for i in range(rma + 10 +1, rmi)] for a,v in xxs.items() if a in a_good]
        #or
        xxs_ker= [[ v[i:(i+1), :] for i in range(rma + 10 +1, rmi)] for a,v in xxs.items() if a in a_good]
        xxd_ker[0][0]
        xxs_ker[0][0]
        
        for v   in xxd.values(): pr(max(w(np.any(np.isnan(v), axis=1))[0]))  # shoud be 5
        for a,v in xxd.items(): pr(a, max(w(np.any(np.isnan(v), axis=1))[0]))  # shoud be 5
        
        for v   in yy.values(): pr(min(w(np.any(np.isnan(v), axis=1))[0]))  # shoud be len(v) - 60
        for a,v in yy.items(): pr(a, min(w(np.any(np.isnan(v), axis=1))[0]), len(v))  # shoud be len(v) - 60
        
        t1()

        ppr(x[:3, :13])
        
  
        more
        xx    


        treat_Dima_IBKR(d=13)
        treat_Dima_IBKR(d=14)


class noNeedToDelFromThinker():
    def prep_XX(sss='SPY AAPL MSFT F', fHist='', kma=[5,10,20,30], toNorm=True, toPlot=True, n_pred=[20,40], n_hist=10):
         ''' prep input for keras train and CV from yf  historical OHLCV data.
         create Rule x-variables and y_20, y_40  variables
         
         Thinker.prep_XX() ->
            rr=  {ass: PDF (389 x 11)}
            yxx= {ass: list 339 x PDF (10 x 11)}  -> pilo('out/yxx_tr.pck') normalized
         -> Thinker.crea_model
         
             # prep yxx     
             yxx[s]=[]
             for i in range(n_hist, len(q1)- max(n_pred)):
                 u= q1.iloc[i- n_hist: i].copy()
                 q_norm= u.apply(lambda r: .25*(r.Open+ r.High+ r.Low+ r.Close), **a1).values[-1]
                 for c in ['Open','High','Low','Close']: u.loc[:, c]= norm_last(u[c], q_norm )
                 for c in ['rule_5_20','rule_10_30']: u.loc[:, c]= norm_last(u[c], q_norm, m1=0 )
                 u.loc[:,'Volume']=  np.log(norm_last(u['Volume'], m1=0)/100)
                 yxx[s].append(u)        
         
         toNorm= 'ret' '''
         #_, qq= Observer.get_hist_qoutes(ss=sss, provider='yf')
         
         pr('in prep_XX fHist=', fHist)
         if fHist == '':
             _, hist= Observer.get_hist_qoutes(ss=sss, provider='yf')
         else:  hist= pilo(fHist)  #.rename(columns={'level_1':'ass'})
         
         if 0 or toNorm== 'ret':
             def incr(a,b): return 100* np.log(a/b)   # log increment
             def em(x,k): return x.ewm(com=k).mean()  # exp moving average
             pd.Series.em= em
             
             uu= []
             for a, r in hist.groupby('level_1'):
                 pr(a, r.shape)
                 try:
                     q1= r.drop('Adj Close', **a1).tab('q1')
                 except: pass
             
                 if toNorm== 'ret':
                     u= PDF(dict(
                         y_20= incr(r.Close.shift(-20), r.Close),
                         y_40= incr(r.Close.shift(-40), r.Close),
                         o= incr(r.Open  , r.Close),
                         h= incr(r.High  , r.Close),
                         l= incr(r.Low   , r.Close),
                         c= incr(r.Close , r.shift().Close ),
                         v= incr(r.Volume , r.shift().Volume )
                     ))
                     u['rule_5_20']=  u.c.em(5)  - u.c.em(20)
                     u['rule_10_30']= u.c.em(10) - u.c.em(30)
                     u= u.dropna().query('rule_5_20 != 0')
                     
                     u.tab(a)
                     u[['c', 'rule_5_20', 'rule_10_30']].plot(title= a)
                     plt.show()
                     
                     
                     for i in range(n_hist, len(u)):
                         v= u.iloc[i- n_hist: i].copy()
                         uu.append(u.dropna().copy())
                         
             # for keras:
             lidf2ker= Thinker.lidf2ker
             yxx=   [lidf2ker(['c','h','l','o','v'], uu), 
                     lidf2ker(['rule_5_20','rule_10_30'            ], uu, onlyLast=True).reshape(-1,2)
                    ], \
                    lidf2ker(['y_20', 'y_40'], uu, onlyLast=True) #.reshape(-1,2)      
                         
             return uu, yxx        
             
         #ss= list(set(qq.level_1))
         #ss= sss.split()
         rr, yxx= {},{}
         uu, uud= [],{}

         #for s in ss:
         for s, r in hist.groupby('level_1'):
             #q1=  hist.query('level_1 == @s').drop('Adj Close', **a1)
             try:
                 q1= r.drop('Adj Close', **a1).dropna()   # no Adj.Close for  SPY
             except: pass
             if len(q1) <10: continue
         
             if toNorm:
                 q_norm= q1.apply(lambda r: .25*(r.Open+ r.High+ r.Low+ r.Close), **a1).values[-1]
                 for c in ['Open','High','Low','Close']: q1[c]= q1[c] / q_norm
                 q1['Volume']=  q1['Volume']/ q1['Volume'].iloc[-1]
             for n in n_pred: q1[f'y_{n}']= 100* ( q1['Close'].shift(-n) / q1.Close -1 )

             for k in kma: q1[f'em_{k}']= q1.Close.ewm(com=k).mean()
             
             q1['rule_5_20']=  q1.em_5 - q1.em_20
             q1['rule_10_30']= q1.em_10 - q1.em_30
             

             if toPlot:
                 f= q1.Close.plot(label='Close', title=s)
                 for k in kma: f.plot(q1.Close.ewm(com= k).mean(), label=f'em_{k}')
                 plt.legend(title=s)
                 plt.show()
                 
                 q1.Volume.plot(label='Volume', title=s) #.show()
                 #plt.legend(title=s)
                 plt.show()
                 
             for k in kma: q1= q1.drop(f'em_{k}', **a1) 
             rr[s]= q1.copy()  

             # prep yxx     
             #q2= q1.dropna()
             yxx[s]=[]
             for i in range(n_hist, len(q1)- max(n_pred)):
                 u= q1.iloc[i- n_hist: i].copy()
                 q_norm= u.apply(lambda r: .25*(r.Open+ r.High+ r.Low+ r.Close), **a1).values[-1]
                 for c in ['Open','High','Low','Close']: u.loc[:, c]= norm_last(u[c], q_norm )
                 for c in ['rule_5_20','rule_10_30']: u.loc[:, c]= norm_last(u[c], q_norm, m1=0 )
                 u.loc[:,'Volume']=  np.log(norm_last(u['Volume'], m1=0)/100)
                 yxx[s].append(u)
         
         #uu, uud= , rr
         #return rr, yxx, hist
         return uu, uud, yxx  


    def ztest(): pr('ztest zzz', '='*88)
       
           
    def crea_model(yxx, nLSTM=128, nStat=64, nDen2=64, nEp=250, fModel='out/model_128_64_1500', toPlot=True):
        ''' create (if fModel="") or read from file fModel, fit, visualize and cross-validate keras model
            yxx= pilo('out/yxx_tr.pck'), dict {asset: list(PDF)}
            or
            rr, yxx= Thinker.prep_XX(sss='SPY QQQ MSFT F ALB KD PROK  ALGN WFC AAL', kma=[5,10,20, 30], toNorm=False, toPlot=True, n_pred=[20,40])
            
            
            Thinker.prep_XX() ->
               rr=  {ass: PDF (389 x 11)}
               yxx= {ass: list 339 x PDF (10 x 11)}  -> pilo('out/yxx_tr.pck') normalized
            -> Thinker.crea_model
            
            yxx dict  ->  yxx list PDF (10 x 11) ->  <random order -> train+test> 
            
            -> Thinker.trd2Xy() Thinker.lidf2ker  -> (i x 10 x 5 dyn feat),  (i x 2 stat feat)), i x2 y
            
            
            -> keras fit(*Thinker.trd2Xy(train_data))
        '''
        from keras.models import Sequential
        from keras.layers import Dense, LSTM
        
   
        # Load the stock data
        #df = pd.read_csv("stock_data.csv")
        #yxx= pd.concat(pilo('out/yxx_tr.pck').values()).sample(frac=1)  # shaffle rows
        #yxx= pilo('out/yxx_tr.pck')
        pr('yxx.keys()=', yxx.keys())
     #   for k in yxx:
     #       if k>'':
     #           pr(k, len(yxx[k]), yxx[k][-1])
        
        # flatten the dic into list and filter
        #yxx= sum([[y for y in yxx[k] if len(y.dropna()) ==10] for k in yxx if k>''], [])
        yxx= [v1 for v in yxx.values() for v1 in v if len(v1.dropna()) ==10]


        # shaffle rows
        ii=  np.random.RandomState(seed=55).permutation(len(yxx))        
        yxx= [yxx[i] for i in ii]       
        di('len(yxx), yxx[0].shape=', len(yxx), yxx[0].shape)
        

        # Split the data into train and test sets
        train_data = yxx[:-400]
        test_data =  yxx[-400:]


        # Create the LSTM model      
        from keras import models, layers
        from keras.layers import  Input, Dense, Concatenate
        from tensorflow.keras import regularizers

        n_feats = 5  #57  #58  #57       # Close    High      Low     Open    Volume
        n_feat_c= 2  #201  # 197  # 226  # rule_5_20    rule_10_30
        #n_feats = X_inv_num[0].shape[0]
        latent_dim = nLSTM

        if fModel >'':
            model = tf.keras.saving.load_model(fModel)
        else:
            lstm_input = layers.Input(shape=(10, n_feats), name="ts_input")
            lstm_output = layers.LSTM(latent_dim, name='lstm_l')(lstm_input)

            static_input = Input(shape=(n_feat_c, ), name="static_input") 
            static_layer_one = Dense(nStat,  kernel_regularizer=regularizers.L2(0.001),  activation='relu',name="dens_1")(static_input)
                        
            combined = Concatenate(axis= 1,name = "concat_ts_stat")([lstm_output, static_layer_one])
            
            combined_dense_two = Dense(nDen2, activation='relu', name="dens_2_comb")(combined)
            output = Dense(2, activation='sigmoid', name="outp")(combined_dense_two)

            model = models.Model([lstm_input, static_input], output, name='TS_combo')  #'TS_diff_len_230_57')  #'TS_diff_len_201_58')  #'TS_diff_len_57_197')

            # Compile the model
            model.compile(loss='mean_squared_error', optimizer='adam')

        model.summary() 

  
        # z   %%% Train the model
        
        pred('fit')

        #model.fit(trd2X(train_data), trd2y(train_data), epochs=nEp)
        model.fit(*Thinker.trd2Xy(train_data), epochs=nEp, verbose=0)

        # Make predictions          
        #predictions = model.predict(*Thinker.trd2X(test_data), verbose=0)
        #(tx1, tx2), ty= Thinker.trd2Xy(train_data)
        (tx1, tx2), ty= Thinker.trd2Xy(test_data)
        predictions = model.predict([tx1, tx2], verbose=0)
        yhh= pd.concat([ PDF(Thinker.trd2y(test_data), columns=['y_20', 'y_40']), PDF(predictions, columns=['y_20h', 'y_40h'])], **a1)
                  
 
        # plot weights
        fig, axes = plt.subplots(nrows=2, ncols=2)

        i=0
        for l in  model.layers:
            wt= l.get_weights()
            pr(i, l.name, len(wt))
            if len(wt) > 0:
                a= axes[i//2, i%2]
                a.imshow(wt[0])
                a.set_title(f'{i}. {l.name}' )
                i+= 1
        plt.show( ) 

        
        di('', '----', 'Correl', 100* yhh.corr().round(3))   

        

        # Plot the predictions
        if toPlot:        
        
            fig, axes = plt.subplots(nrows=2, ncols=2)
        
            for i, y in enumerate(['y_20', 'y_40']):
                yh= yhh[[y, y+'h']].sv(y+'h', ascending=False).ri(drop=True)
                lif= np.cumsum(yh[y]) #.plot(title='Lift ' + y)
                axes[0,i].plot(yh)
                axes[1,i].plot(lif)
                axes[0,i].set_title(f'{y}, {nEp=}', x=.7, y=.8)            
                axes[1,i].set_title('Lift ' +y, x=.7, y=.8)
                
                pr(f'\n Mean Win for  {y} ,  50 best - 50 worst= {yh.iloc[:50][y].mean() - yh.iloc[-50:][y].mean():.3f} \n') 
            plt.show()
            
            
        # Calculate permutation importance
        if 0:  # nOK
            from sklearn.inspection import permutation_importance
            
            perm_imp = permutation_importance(model, *trd2Xy(test_data), n_repeats=10, random_state=42)

            # Plot the permutation importance
            plt.barh(df.drop('target', axis=1).columns, perm_imp['importances_mean'], color='b')
            plt.xlabel('Permutation importance')
            plt.ylabel('Features')
            plt.show()    
        
        # Cross-validate the model
        if 1:  # OK
            from sklearn.model_selection import KFold
        
            x_train= train_data
            kfold = KFold(n_splits=10)
            results = []
            yh= PDF(Thinker.trd2Xy(x_train)[1].reshape(-1, 2))
            yh['yh']= [[None,None]] * len(yh)
            di('1. yh=',yh)
            for train_index, test_index in kfold.split(x_train):
                # Train the model
                #di(1111, train_data, test_index)
                model.fit(*Thinker.trd2Xy([train_data[i] for i in train_index]), epochs=2, verbose=0)

                # Evaluate the model
                score = model.evaluate(*Thinker.trd2Xy([train_data[i] for i in test_index]), verbose=0)
                results.append(score)
                di('test_index=', test_index, '[train_data[i] for i in test_index]=', [train_data[i] for i in test_index])

               # u= model.predict(*Thinker.trd2X([train_data[i] for i in test_index]), verbose=0) #[1][0]
                (tx1, tx2), ty= Thinker.trd2Xy([train_data[i] for i in test_index])
                u = model.predict([tx1, tx2], verbose=0)
                di('u=',PDF(u), u.shape, len(test_index))
                yh.loc[test_index, 'yh']= PDF(u) #[1][0]

            # Print the results
            di('results=', results)
            di('2. yh=',yh )

            print('Mean accuracy:', np.mean(results))
            print('Standard deviation:', np.std(results))
        
        return model, yhh
           
    def treat(qq, n_pred=20):
         ss= list(set(qq.level_1))
         rr= {}
         for s in ss:
             q1=  qq.query('level_1 == @s')
             q_norm= q1.apply(lambda r: .25*(r.Open+ r.High+ r.Low+ r.Close), **a1).values[-1]
             for c in ['Open','High','Low','Close']: q1[c]=  100*(q1[c] /q_norm -1)
             q1['Volume']= 100*(  q1['Volume']/q1['Volume'].iloc[-1] -1)
             q1['yh']=  q1['Close'].shift(-n_pred)
             
             q1['em_5']= q1.Close.ewm(com=5).mean()
             q1['em_20']= q1.Close.ewm(com=20).mean()
             
             rr[s]= q1.copy()
             

             f= q1.Close.plot(label='Close', title=s)
             f= q1.yh.plot(label='yh')
             f.plot(q1.Close.ewm(com= 20).mean(), label='em20')
             f.plot(q1.Close.ewm(com= 5).mean(), label='em5')
             plt.legend(title=s)
             plt.show()


             
             q1.Volume.plot(label='Volume', title=s)#.show()
                 #plt.legend(title=s)
             plt.show()

         return rr
     
    # rr= treat(qq)
    # pr(tabulate(rr['F'].round(3),  tablefmt="simple"))  # tablefmt='psql', headers='keys'))           
            

    def trd2X(train_data):        
       lidf2ker= Thinker.lidf2ker
       return   [lidf2ker(['Close','High','Low','Open','Volume'], train_data), 
                 lidf2ker(['rule_5_20','rule_10_30'            ], train_data, onlyLast=True).reshape(-1,2)
                ]

    def trd2y(train_data) : return  npa([npa(t[['y_20', 'y_40']].iloc[-1])  for t in train_data]) #.reshape(-1,2)        
              
    def trd2Xy(train_data) :        
       lidf2ker= Thinker.lidf2ker
       xx= ['Close','High','Low','Open','Volume'] if 'Close' in train_data[0].columns \
           else ['c','h','l','o','v']
       pr('in trd2Xy   xx=', xx)    
       return   [lidf2ker(xx, train_data), 
                 lidf2ker(['rule_5_20','rule_10_30'            ], train_data, onlyLast=True).reshape(-1,2)
                ], lidf2ker(['y_20', 'y_40'], train_data, onlyLast=True) #.reshape(-1,2)        
   
    def trd2Xy_ret(train_data) :        
        lidf2ker= Thinker.lidf2ker
        xx= ['c','h','l','o','v']
        info(train_data[0], 'train_data[0]')
        return   [lidf2ker(['c','h','l','o','v'], train_data), 
                  lidf2ker(['rule_5_20','rule_10_30'            ], train_data, onlyLast=True).reshape(-1,2)
                 ], lidf2ker(['y_20', 'y_40'], train_data, onlyLast=True) #.reshape(-1,2)           
           
   

        

     
    if 0:
         rr= prep_XX(sss='SPY AAPL MSFT F', kma=[5,20])
         pr(tabulate(rr['F'], tablefmt='psql', headers='keys')) 
         
         

    def get_hist_qoutes(s1='IBM', prov='kibot'):
        f= {'IVE': 'bidask1min', 'OIH': 'unadjusted', 'IBM': 'unadjusted', 'WDC': 'tickbidask'}
        u= pd.read_csv(f'ref/kibot/{s1}_{f[s1]}.txt')
        
        #OLHCV
        if 0:
            u= pd.read_csv('http://api.kibot.com/?action=history&symbol=IBM&interval=1&unadjusted=1&bp=1&user=guest')  # OK
            u= pd.read_csv('http://api.kibot.com/?action=history&symbol=OIH&interval=1&unadjusted=1&bp=1&user=guest')  # OK
            
            #tick, bid, ask - slow
            u= pd.read_csv('http://api.kibot.com/?action=history&symbol=IVE&interval=tickbidask&bp=1&user=guest')
            u= pd.read_csv('http://api.kibot.com/?action=history&symbol=WDC&interval=tickbidask&bp=1&user=guest&period=3')
        
        columns=  ['Date', 'Time']
        if len(u.columns)==10:  columns += ['Open', 'High', 'Low', 'Close', 'l1', 'h1', 'l2', 'h2']
        if len(u.columns)==6:  columns += ['Tick', 'Bid', 'Asc', 'Volume'] 
        if len(u.columns)==7:  columns += ['Open', 'High', 'Low', 'Close', 'Volume']
        
        try:
            u.columns= columns
        except: pass

        return u

        if 0:
            ibm= get_hist_qoutes(0, s1='IBM', prov='kibot')  # intraday OHLCV
            oth= get_hist_qoutes(0, s1='OIH', prov='kibot')

            ive= get_hist_qoutes(0, s1='IVE', prov='kibot')  # ticks
            wdc= get_hist_qoutes(0, s1='WDC', prov='kibot') 
            
            
         
             
             
    

#%% tests Thinker
if 0:
    th= Thinker()
    th.fee
    th.findTrader()
    th.loadModel(m=None)
    t1= tnow()
    
    #### test prep_XX_ret
    #rr, yxx= th.prep_XX_ret(sss='MSFT F', kma=[5,10,20, 30], toPlot=True, n_pred=[20,40])
    rr
    rr['F'].shape    
    
    uu, uud, yxx_ker= th.prep_XX_ret(sss='MSFT F', kma=[5,10,20, 30], toPlot=True, n_pred=[20,40], var=['ret', 'a4'])
    uu, uud, yxx_ker= th.prep_XX_ret(sss='MSFT F', kma=[5,10,20, 30], toPlot=True, n_pred=[20,40], var=['ret', 'a4', 'ta'])
    th
    yxx_ker[0][0].shape
    # or 
    uu, uud, yxx_ker= th.prep_XX_ret(fHist='out/hist_06-11.pck', kma=[5,10,20, 30], toPlot=True, n_pred=[20,40],tp_rsi=5)
    # results:
    uud['AI'][0].tab('uud')
    yxx_ker[0][0].shape
    yxx_ker[1].shape
    yxx_ker[2].shape
    yxx_ker[2].tab('yxx_ker[2]')
    (tx1, tx2), ty, more_adict= yxx_ker
    #more= PDF(more_adict,  columns= th.more)

    more= more_adict
    if 0:    
        more= PDF(more_adict.reshape(-1,4),  columns= ['ass', 'Datetime', 'Close', 'feep'])    
        more.index= [more.ass, more.Datetime]
    more.tab('more')
    more.ass.vc()
    
    yxx= uud
    a= list(yxx.keys())[0]
    len(yxx[a]), yxx[a][0].shape, yxx[a][0]
    yxx
    pidu(yxx, 'out/yxx_tr.pck')
    ''' Thinker.prep_XX() ->
        rr=  {ass: PDF (389 x 15)}
        yxx= {ass: list 339 x PDF (10 x 15)}  -> pilo('out/yxx_tr.pck') 
        -> Thinker.crea_model
    '''

    
    yxx= pilo('out/yxx_tr.pck')
    di('yxx 0 2 =', len(yxx), type(yxx), yxx.keys(), len(list(yxx.values())[0])) #list(yxx.values())[:2]) 
    yxx[a][0].tab('yxx[a][0]')
    #yxx= sum([[y for y in yxx[a] if len(y.dropna()) ==10] for a in yxx if a > ''], [])   

    
    #model, te_y, predictions= Thinker.crea_model(nLSTM=128, nStat=64, nEp=1500, fModel='out/model_128_64_1500')
    #model, te_y, predictions= Thinker.crea_model(nLSTM=128, nStat=64, nEp=100, fModel='out/model_128_64_3000', y='y_20')
    #model, te_y, predictions= Thinker.crea_model(nLSTM=128, nStat=64, nEp=3, fModel='out/model_128_64_3000_y40', y='y_40')
    #model, yhh= Thinker.crea_model(nLSTM=32, nStat=32, nDen2=64, nEp=3000, fModel='out/model2_32_32_3000', toPlot=True)
    model, yhh= Thinker.crea_model(yxx, nLSTM=32, nStat=32, nDen2=64, nEp=3, fModel='out/model2_32_32_15K', toPlot=True)
    
    
    #irelo()
    #uu, uud, yxx= Thinker.prep_XX_ret(sss='MSFT F', kma=[5,10,20,30], toPlot=False, n_pred=[20,40], n_hist=3)
    th= Thinker()
    th.fee
    t1= tnow()
    uu, uud, yxx_ker= th.prep_XX_ret(fHist='out/hist_06-11.pck', kma=[5,10,20,30], tp_rsi=5, 
                                      toPlot=False, n_pred=[20,40], n_hist=10)
    f'{tnow() - t1}'  # 30 sec
    
    shape_ker(yxx_ker)
    #10000* yxx_ker[2]
    (tx1, tx2), ty, more= yxxs
    th.cdxx, th.csxx
    yxxs= yxx_ker
    info(yxxs,'yxxs')
    shape_ker(yxxs)
    more
    pidu(yxxs, 'out/yyxs_06-11.pck')

    
if 0 and toTest:
    th= Thinker()
    th.fee
    uu, uud, yxx= th.prep_XX_ret(fHist='out/hist_06-11.pck', kma=[5,10,20,30], 
                        var= 'a4', toPlot=False, n_pred=[20,40], n_hist=10)
    uu, uud, yxx= th.prep_XX_ret(hist=hist, kma=[5,10,20,30], 
                        var= 'ret', toPlot=False, n_pred=[20,40], n_hist=10)
    uu, uud, yxx= th.prep_XX_ret(fHist='out/hist_06-11.pck', kma=[5,10,20,30], 
                        var= 'ret', toPlot=False, n_pred=[20,40], n_hist=10)
    info(uu,'uu') 
    info(uud,'uud') 
    info(yxx,'yxx') 
    shape_ker(yxx)

    (t1x, t2x), ty= yxx
    info(t1x, 't1x'); t1x[0]
    info(t2x, 't2x'); t2x[0]
    info(ty, 'ty'); ty[0]
    
    frame.max(np.max(uu[0][th.cdxx]))
    max(np.max(uu[0][th.cdxx]))
    max(uu[0][th.cdxx])
    

    
if 0 and  toTest:
    #### test read_hist_dir()
    th= Thinker()
    th.__class__= Thinker
    t1= tnow()
    yxxs=  th.read_hist_dir(dire='out', list_only= True, patt='hist_', tp_rsi=5)
    yxxs=  th.read_hist_dir(dire='out', list_only= 0, patt='hist_', tp_rsi=5)
    f'{tnow() - t1}' # 0:03:28
    
    (tx1, tx2), ty, more= yxxs=  ar= th.read_hist_dir(dire='out', list_only= 0, patt='hist_06-11', var='ret', nLast=1, tp_rsi=5)
    (tx1, tx2), ty= yxxs=  a4= th.read_hist_dir(dire='out', list_only= 0, patt='hist_06-13', var='a4', tp_rsi=5)  # , var='ret'
    
    t1()
    yxxs=  ar4= th.read_hist_dir(dire='out', list_only= 0, patt='hist_0', var=['ret', 'a4'], nLast=1, tp_rsi=5)  # , var='ret'
    t1() # 0:03:44   0:02:26.69
    
    (tx1, tx2), ty, more= yxxs
    th.cdxx, th.csxx
    info(yxxs,'yxxs')
    shape_ker(yxxs)
    len(more), more[0].shape
    yxxs[1]
    
    #PDF(more.reshape(-1, 4), columns=['ass', 'Datetime','Close','feep'])
    #pidu(yxxs, 'out/yyxs.pck')
    pidu(yxxs, 'out/yyxs-ta.pck')
    
    for i in range(3): pr(i, sum(tx2[:, 2] == i))
        # 0  3206
        # 1 30473
        # 2  2090
    
    ar[0][0][:3]
    ar4[0][0][:3]
    
    ####  are there nans?
    for t in [tx1, tx2, ty]:  pr(sum(sum(np.isnan(t))))
    for t in [tx1, tx2, ty]:  pr(sum(sum(np.isinf(t))))
    np.max(yxxs[0][0]) # == np.inf

    
    
    for i in range(2): pr(i); info(yxxs[i][0], f'yxxs[{i}][0]'); info(yxxs[i][1], f'yxxs[{i}][1]')
    
    #### train
    m.__class__= Model
    m.hist= PDF()
    pr(m)
    m.model.summary()  
    shape_ker(yxxs)  #  (((46499, 10, 11), (46499, 3)),  (46499, 1, 2),  (46499, 1, 4))
    yh= m.model.fit(yxxs[0], yxxs[1],  epochs=2)  # OK
    yh= m.train(yxx=yxxs,  nEp=3)  # OK
    
    #### rescale / normalization
    #_yxxs[1] = yxxs[1] / 1000
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    scaler = StandardScaler()
    sft= scaler.fit_transform
    yxxs = (sft(yxxs[0][0]), sft(yxxs[0][1])), sft(yxxs[1])
    
    #### ta training
    t1()
    hi= m.model.fit(yxxs[0], yxxs[1],  epochs=100, verbose=2, validation_split= 0.3)
    t1()  # 0:15:26.589
    #hi.history.plot()
    plotKerasLearningCurve(hi, hist0=PDF())

    #nOK  m.save('model_ra4_128-64-64-ta_200ep') # out/full_model_ra4_128-64-64_100ep.pck
    #pidu(m.model.get_weights(), 'out/model_ra4_128-64-64-ta_200ep_wts.pck') # out/full_model_ra4_128-64-64_100ep.pck
    pidu(m.model.get_weights(), 'out/model_ra4_128-64-64-ta_300ep_wts.pck') # out/full_model_ra4_128-64-64_100ep.pck
    
    
    # try to save model
    #### prep & save yxxs for june
    t1()
    th= Thinker()
    #yxxs=  th.read_hist_dir(dire='out', list_only= False, patt='hist_4_06-30', var=['ret','a4'], nLast=1, tp_rsi=5)
    #pidu(th, 'out/th_hist_4_06-30')
    if 1:
        yxxs=  th.read_hist_dir(dire='out', list_only= False, patt='hist_06', var=['ret','a4','ta'], nLast=1, tp_rsi=5)
        pidu(th, 'out/th_hist_06_ta_69fd.pck')
        pidu(yxxs, 'out/yxxs_hist_06_ta_69fd.pck')
        shape_ker(yxxs) # (((20813, 10, 63), (20813, 3)), (20813, 1, 2), (3852, 4))
    t1()  # Execution time 0:08:38.22
    
    m= Model(th=th, nLSTM=128, nStat=64, nDen2=64, nEp=299, fModel='', toPlot=True)
    m.name= 'mo_99f_ta'
    m.model.__weights__= pilo('out/model_ra4_128-64-64-ta_300ep_wts.pck')
    
    m0= Model()
    m0.load('model_ra4_128-64-64-ta_tp_rsi=2_1799ep')
    m.model.__weights__= m0.model.get_weights()
    m.model.get_weights()
   
    
    #m= Model(th=th, nLSTM=128, nStat=64, nDen2=64, nEp=250, fModel='', toPlot=True)
    
    
    m.model.__weights__ = pilo('out/model_ra4_128-64-64-ta_300ep_wts.pck')
    m.save('model_ra4_128-64-64-ta_300ep')
    
    # and load again
    m= Model()
    #m= m.load('model_ra4_128-64-64-ta_300ep')
    m= m.load('model_ra4_128-64-64-ta_tp_rsi=2_300ep')
    m.name= 'ra4_128-64-64-ta_tp_rsi=2_300ep'
    m
    m.model.summary()
    # data or validation
    yxx= th.read_hist_dir(dire='out', list_only= False, patt='hist_07-01', var=['ret','a4'], nLast=1, tp_rsi=5)
       #  🚩  wi y_20   :
       #                    y_20    y_20h_00     y_40    y_40h_00
       # --------------  -------  ----------  -------  ----------
       # ('y_20', 0.01)  73.8659     12.6632  46.0648      1.8783
       # ('y_20', 0.05)  44.2464      4.205   31.3084      1.5027
       # ('y_20', 0.1)   32.6424      1.9191  23.9119      0.7361
       # ('y_20', 0.5)   11.2105      0.5364   8.4723      0.5105
       # ('y_40', 0.01)  28.1191      4.2214  43.4319      5.4584
       # ('y_40', 0.05)  19.9709      0.8677  28.7243      1.8908
       # ('y_40', 0.1)   15.4998      0.0082  21.4819      1.213
       # ('y_40', 0.5)    5.7676      0.0632   7.5743      0.4785 
    #or
    t1()
    yxx= th.read_hist_dir(dire='out', list_only= False, patt='hist_07-07', var=['ret','a4'], nLast=1, tp_rsi=5)
    yhh, co, wins, win, impo, impo_cor, impo_win, inds= m.cv(th, yxx=yxx,  nWin=[.01, .05, .10], n_splits= 5, nEp=5, toPlot=True, verb=1)
    wins.pivot(index=['y', 'nWin'], columns='yh', values='win')[['y_20', 'y_20h_00', 'y_40', 'y_40h_00']].tab('wi y_20', nr=99)
    pgre(f'{len(yhh)=}')
    t1()  # 0:00:55.43
    
    #### training model with tp_rsi=2
    t1()
    th.__class__= Thinker

    yxx= th.read_hist_dir(dire='out', list_only= False, patt='hist_07-07', var=['ret','a4'], nLast=1, tp_rsi=2)
    
    m= Model(th=th, nLSTM=128, nStat=64, nDen2=64, nEp=250, fModel='', toPlot=True)
    m.__class__= Model
    m.hist= PDF()
    pr(m)
    m.model.summary()  
    shape_ker(yxx)  #  (((46499, 10, 11), (46499, 3)),  (46499, 1, 2),  (46499, 1, 4))
    #yh= m.model.fit(yxxs[0], yxxs[1],  epochs=2)  # OK
    
    t1()
    th.__class__= Thinker
    th= Thinker()
    yxxs=  th.read_hist_dir(dire='out', list_only= False, patt='hist_06', 
                            var=['ret','a4'], nLast=1, tp_rsi=2)
    len(yxxs)
    shape_ker(yxxs)  #  (((46499, 10, 11), (46499, 3)),  (46499, 1, 2),  (46499, 1, 4))
                     #  (((25486, 10, 15), (25486, 3)), (25486, 1, 2), (3985, 4))
    #pidu(yxxs,'out/yxxs_tp_rsi=2.pck')
    
    yxxs= pilo('out/yxxs_tp_rsi=2.pck')

    yxxs[2]
    t1() # 0:01:52.246894
    yh, history, more= m.train(yxx=yxxs,  nEp=100) 
    #plotKerasLearningCurve(hi, hist0=PDF())
    t1(len(more)) # 10 ep -> 0:00:45.6  100-> 0:05:24.91  300-> 0:24:46.0  # 25086
    # mo ta: Execution time 0:15:17, s=    20413 
    m.trd2Xy= Thinker.trd2Xy
    m.nEp
    m.tp_rsi=2
    #m.name= 'ra4_128-64-64-ta_tp_rsi=2_700ep'
    m.name= 'ra4_128-64-64-ta_tp_rsi=2_1799ep'
    m.name= 'mo_65f_ta_1400ep'
    #m.save('model_ra4_128-64-64-ta_tp_rsi=2_300ep')
    #m.save('model_ra4_128-64-64-ta_tp_rsi=2_400ep')
    #m.save('model_ra4_128-64-64-ta_tp_rsi=2_700ep')
    #m.save('model_ra4_128-64-64-ta_tp_rsi=2_999ep')
    #m.save('model_ra4_128-64-64-ta_tp_rsi=2_1299ep')
    m.save('model_ra4_128-64-64-ta_tp_rsi=2_1799ep')
    m.save('mo_65f_ta_900ep')
    m.save('mo_65f_ta_1400ep')
    m.save('mo_65f_ta_1900ep')
    
    #### train tn new (reorder)
    yxss= pilo('out/yxxs_hist_06_ta_69fd.pck')
    th= pilo('out/th_hist_06_ta_69fd.pck')
    m= Model(th)
    t1()
    yh, history, more= m.train(yxx=yxxs,  nEp=300) 
    t1(len(more)) # 10 ep -> 0:00:45.6  100-> 0:05:24.91  300-> 0:24:46.0  # 25086
    #m.save('mo_65f_714_ta_302ep')  # 20413
    m.save('mo_65f_714_ta_602ep')  # 20413

    
    m.__dict__
    
    
    #### cv
    th= Thinker()
    m.__class__= Model
    yxx07= th.read_hist_dir(dire='out', list_only= False, patt='hist_07-11', var=['ret','a4','ta'], nLast=1, tp_rsi=2)
    yhh, co, wins, win, impo, impo_cor, impo_win, inds= m.cv(th, yxx=yxx07,  nWin=[.01, .05, .10], n_splits= 5, nEp=5, toPlot=True, verb=1)
    #wins.query('nWin== .01 and y < "y_3"  and yh < "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_20')).round(1))
    w= wins.query('nWin== .01 and y < "y_3"  and yh < "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_20')) .tab(nr=99)
    wins.query('nWin== .01 and win>= 16.2498  and y < "y_3"  and yh < "y_3"').yh.apply(lambda y: y[7:])
    impo_win.tab(nr=99)
    
    c_ta_imp= [c for c in impo_win.query('y_20 >.1 and y_40 > .1').index.values if str(c)[:3]== 'ta_']
    ''' c_ta_imp= ['ta_CDLMARUBOZU',
     'ta_MAXINDEX',
     'ta_CDLSHORTLINE',
     'ta_MININDEX',
     'ta_CMO',
     'ta_AROONOSC',
     'ta_CDLCLOSINGMARUBOZU',
     'ta_CDL3OUTSIDE',
     'ta_CDLBELTHOLD',
     'ta_MFI',
     'ta_CDLLONGLINE',
     'ta_CDLENGULFING',
     'ta_CDLHANGINGMAN',
     'ta_CCI',
     'ta_CDLHARAMI',
     'ta_RSI']'''
    
    # compare_models  ta_tp_rsi=2 by nEp:
    t1()
    yhh, wins, wins_p, lifts = Model.compare_models([
                         'model_ra4_128-64-64-ta_tp_rsi=2_300ep' ,
                         #'model_ra4_128-64-64-ta_tp_rsi=2_400ep' ,
                         'model_ra4_128-64-64-ta_tp_rsi=2_700ep' ,
                         #'model_ra4_128-64-64-ta_tp_rsi=2_999ep' ,
                         'model_ra4_128-64-64-ta_tp_rsi=2_1799ep' ,
                         'mo_65f_ta_900ep'
                          ], 
                         #fHist='out/hist_sp500_07-07.pck')    
                         fHist='out/hist_07-11_sp500.pck') 
                        #fHist='out/hist_4_06-30.pck')

    
    t1(len(yhh))  # Execution time 0:02:23.42,  3448
    
        # m                                    y_20     y_40
        # --------------------------------  -------  -------
        # ra4_128-64-64-ta_tp_rsi=2_1799ep  11.2727  19.418
        # ra4_128-64-64-ta_tp_rsi=2_300ep   14.3051  22.1591
        # ra4_128-64-64-ta_tp_rsi=2_400ep   18.0111  25.5149
        # ra4_128-64-64-ta_tp_rsi=2_700ep    8.4323  16.3528
        # ra4_128-64-64-ta_tp_rsi=2_999ep   12.5234  10.1843 
        
    Model.bokeh_lifts(lifts)
    
    Model.bokeh_comp_lifts(yhh, cy=['y_20', 'y_40'])
    dir(Model)
    
    
    pidu(m.model.get_weights(), 'out/model_ra4_128-64-64-ta_tp_rsi=2_700_wts.pck') # out/full_model_ra4_128-64-64_100ep.pck
    
    
    dir(m)
    
    
    
    
    # check
    m.nEp # =60
    PDF(yh, columns= th.cy)
    PDF(history.history).plot()
    PDF(more, columns= ['ass','Datetime', 'Close0', 'feep'])   #+ th.more.columns)
  
    #cv
    th.__class__= Thinker
    t1()
    yxx= th.read_hist_dir(dire='out', list_only= False, patt='hist_07-07', var=['ret','a4'], nLast=1, tp_rsi=2)
    yhh, co, wins, win, impo, impo_cor, impo_win, inds= m.cv(th, yxx=yxx,  nWin=[.01, .05, .10], n_splits= 5, nEp=5, toPlot=True, verb=1)
    wins.pivot(index=['y', 'nWin'], columns='yh', values='win')[['y_20', 'y_20h_00', 'y_40', 'y_40h_00']].tab('wi y_20', nr=99)
    pgre(f'{len(yhh)=}')
    t1()  # 0:01:11.42
    
       #  🚩  wi y_20   :
       #                    y_20    y_20h_00     y_40    y_40h_00
       # --------------  -------  ----------  -------  ----------
       # ('y_20', 0.01)  77.6515     29.0188  46.7146     31.8048
       # ('y_20', 0.05)  51.3932     18.2997  36.4827     18.0262
       # ('y_20', 0.1)   39.4017     13.0986  29.5216     13.894
       # ('y_20', 0.5)   14.501       4.4481  11.2374      4.3135
       # ('y_40', 0.01)  33.1404     23.9165  49.3649     25.3116
       # ('y_40', 0.05)  25.3086     16.2444  35.3995     15.9031
       # ('y_40', 0.1)   20.4586     12.1069  27.8478     12.5685
       # ('y_40', 0.5)    8.389       4.4195  10.8127      4.1709 
       
     #  m.nEp=300     
     #  🚩  wi y_20   :
     #                    y_20    y_20h_00     y_40    y_40h_00
     # --------------  -------  ----------  -------  ----------
     # ('y_20', 0.01)  77.6515     21.7918  46.7146     25.6418
     # ('y_20', 0.05)  51.3932     17.4081  36.4827     18.0014
     # ('y_20', 0.1)   39.4017     13.8257  29.5216     14.2006
     # ('y_20', 0.5)   14.501       4.9518  11.2374      5.2976
     # ('y_40', 0.01)  33.1404     21.6035  49.3649     23.9499
     # ('y_40', 0.05)  25.3086     15.847   35.3995     15.8648
     # ('y_40', 0.1)   20.4586     12.867   27.8478     12.9593
     # ('y_40', 0.5)    8.389       4.6793  10.8127      4.8438      
       
       
     #zzzz zzzz
   
    # nOK m.save('model_ra4_128-64-64-ta_tp_rsi=2_190ep')
    pidu(m.model.get_weights(), 'out/model_ra4_128-64-64-ta_tp_rsi=2_190_wts.pck') # out/full_model_ra4_128-64-64_100ep.pck
  
    
  # load  m  from weights
    th= Thinker()
    #th.prep_XX_ret(sss='', fHist='out/hist_4_06-30.pck', histo='', 
    th.prep_XX_ret(sss='', fHist='', histo='', 
                    kma=[5,10,20,30],  var=['a4','ret'], nLast=1, tp_rsi=5, n_pred=[20,40],
                    toPlot=True, n_hist=10, row0=True, verb=1);
     
    m= Model(th=th, nLSTM=128, nStat=64, nDen2=64, nEp=250, fModel='', toPlot=True)
    m= m.load('out/model_ra4_128-64-64-ta_tp_rsi=2_190_wts.pck')
    m.model.summary()
    
    # cv
    yxx= th.read_hist_dir(dire='out', list_only= False, patt='hist_07-07', var=['ret','a4'], nLast=1, tp_rsi=2)
    yhh, co, wins, win, impo, impo_cor, impo_win, inds= m.cv(th, yxx=yxx,  
                nWin=[.01, .05, .10], n_splits= 5, nEp=5, toPlot=True, verb=1)
    wins.pivot(index=['y', 'nWin'], columns='yh', values='win')[['y_20', 'y_20h_00', 'y_40', 'y_40h_00']].tab('wi y_20', nr=99)
    pgre(f'{m.nEp=}, {len(yhh)=}')

    
    yh
    m.model.predict(yxxs[0])  # OK
    m.plot_layers()
    for l in m.model.layers: pr(l.name, l.get_weights())
    

    100* yxxs[0][0][0], \
    100* yxxs[0][1][0], \
    100* yxxs[1][0], \
    yxxs[2][0]
    
    100* yxxs[0][0][-1], \
    100* yxxs[0][1][-1], \
    100* yxxs[1][-1]
    
     
    m= Model()
    m= Model(th=th, nLSTM=128, nStat=64, nDen2=64, nEp=3, 
                 fModel='out/model_128_64_1500', toPlot=True)
    
    m= Model(th=th, nLSTM=128, nStat=64, nDen2=64, nEp=3, 
                fModel='', toPlot=True)
   
   #### toy model
    m= Model(th=th, nLSTM=2, nStat=3, nDen2=4, nEp=2, toPlot=True, fModel='')
    
    m.model= tf.keras.saving.load_model(f'{out}/model2_32_32_3000')
    #or 
    m= Model(th=th, nLSTM=64, nStat=64, nDen2=64, nEp=250, fModel='', toPlot=True)
    m.model.set_weights('out/model_r6a4t_128-64-64_3002ep_wghts.pck')


    m.model.summary()
    dir(m.model)
    m.model.trainable_variables
    m.model.variables
    m.model.get_weights()
    m.nEp
    m.save('model_ra4_128-64-64_100ep') # out/full_model_ra4_128-64-64_100ep.pck
    m.save('model_ra4_128-64-64_300ep') # out/full_model_ra4_128-64-64_300ep.pck
    m.save('model_ra4_128-64-64_1500ep') # out/full_model_ra4_128-64-64_300ep.pck
    m.save('model_r6a4_128-64-64_310ep') # out/full_model_ra4_128-64-64_300ep.pck
    m.save('model_r6a4t_128-64-64_302ep') # out/full_model_ra4_128-64-64_300ep.pck
    m.save('model_r6a4_128-64-64_1501ep') # out/full_model_ra4_128-64-64_300ep.pck
    m.save('model_r6a4t_128-64-64_1502ep') # out/full_model_ra4_128-64-64_300ep.pck
    m.save('model_r6a4t_128-64-64_2002ep') # with rule_t
    m.save('model_r6a4t_128-64-64_3002ep') # with rule_t
    pidu(m.model.get_weights(), 'out/model_r6a4t_128-64-64_2002ep_wghts.pck')
    pidu(m.model.get_weights(), 'out/model_r6a4t_128-64-64_3002ep_wghts.pck')
    
    m.name= 'r6a4t_128-64-64_1502ep'
    #pr(sum(sum(np.isnan(m.model.get_weights()))))
    
    m.load('model_ra4_128-64-64_100ep')   
    
    #OK
    yh= m.train(yxx=((t1x, t2x), ty),  nEp=200)
    yh= m.train(yxx=((tx1, tx2), ty),  nEp=200)
    yh, histo= m.train(yxx= yxxs[:2],  nEp= 5)
    #m.model.predict((t1x, t2x))
    m.model.predict(yxxs[0])
    
    
    histo= mta.model.fit(yxxs[0], yxxs[1],  epochs= 5)
    
    dir(histo)
    plt.plot(histo.history['loss'])
    
    #nOK
    #m.cv(th, yxx=((t1x, t2x), ty), nWin=[.01, .05, .10], 
    m.cv(th, yxx= yxxs, nWin=[.01, .05, .10], 
             n_splits= 5, nEp=2, toPlot=True, verb=1)
#%% cv ----->
    m.__class__= Model
    th.__class__= Thinker
    info(yxx_ker,'yxx_ker')
    shape_ker(yxx_ker)
    info(yxxs,'yxxs')
    shape_ker(yxxs)
    
    yhh, co, wins, win, impo, impo_cor, impo_win, inds= Model.cv(m, th, yxx=yxxs, nWin=[.01, .05, .10], 
         n_splits= 5, nEp=1, toPlot=True, verb=1)
    
    #### add fee to expected win
    u= PDF(yhh.iloc[:, :4]) #.tab('yhh')
    u['Close']= [yxxs[2][i][0][2] for i in inds]
    u['feep']= [yxxs[2][i][0][3] for i in inds]
    for y in th.cy:
        u[f'feep_{y}']= u.feep * 2 *  200 / int(y[-2:])  # hard coded ??
        u[f'{y}_fee']= u[y] + u[f'feep_{y}'] * u[y+ 'h_00' ]. apply(lambda u: (-1 if u > 0 else 1))   # hard coded ??
        u[f'{y}h_00_fee']= u[f'{y}h_00'] + u[f'feep_{y}'] . apply(lambda u: (-1 if u > 0 else 1))    # hard coded ??
    u.tab('u:')
    sum(u.y_40h_00_fee > 0)  # 3681
    #  .05* 46499 =~ 2000
    u.svde(f'{y}h_00_fee')
    u.head(2000).y_40.mean() - u.tail(2000).y_40.mean()  # = 32%
    u.head(1000).y_40.mean() - u.tail(1000).y_40.mean()  # = 42%
    u.svde('y_20h_00_fee')
    u.head(2000).y_20.mean() - u.tail(2000).y_20.mean()  # =~ 41%
    u.head(1000).y_20.mean() - u.tail(1000).y_20.mean()  # =~ 53%
    
    
    len(u)
    u[-1]
    wins.query('nWin== .1 and y < "zy_3"  and yh < "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_20')) #.iloc[1:]

    m.nEp
    
#%% ==    
    yhh, co, wins, win, impo, impo_cor, impo_win= m.cv(th, yxx=yxxs, cdxx= th.cdxx, nWin=[.01, .05, .10], 
         n_splits= 5, nEp=1, toPlot=True, verb=1)
    
    m.plot_win(yhh)
    
    di(wins.query('nWin== .5 and y < "zy_3"  and yh < "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_20')).round(1))
    di(wins.query('nWin== .5 and y > "ay_3"  and yh > "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_40')).round(1))

    wi= wins.query('nWin== .1 and y < "zy_3"  and yh < "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_20')).iloc[1:]
    plt.scatter(wi[[('win', 'y_20')]], wi[[('win', 'y_40')]])

    #%%   Model.cv  OK
    m.__class__= Model
    u= m.cv(yxx=yxxs, nWin=[.01, .05, .10], 
             n_splits= 5, nEp=1, toPlot=True, verb=1)
    yhh, co, wins, win, impo,  impo_cor, impo_win = u  
        
    PDF(wins)
    
    #### models: load and cv
    m100= Model(th=th).load('model_ra4_128-64-64_100ep') 
    m300= Model(th=th).load('model_ra4_128-64-64_300ep') 
    mr6_310= Model(th=th).load('model_r6a4_128-64-64_310ep') 
    mr6_1501= Model(th=th).load('model_r6a4_128-64-64_1501ep') 
    m1500= Model(th=th).load('model_ra4_128-64-64_1500ep') 
    m100.model.summary()
    m1500.model.summary()
    m100.cdxx
    
    yhh, co, wins, win, impo,  impo_cor, impo_win = m.cv(yxx=yxxs, nWin=[.01, .05, .10], 
             cdxx= th.cdxx, n_splits= 5, nEp=1, toPlot=True, verb=1, th=th)
    m.plot_lift(test_data=yxxs, nEp=1)



    #%% ====
    
    yhh100, co, wins100, win, impo,  impo_cor, impo_win = m100.cv(yxx=yxxs, nWin=[.01, .05, .10], 
             cdxx= th.cdxx, n_splits= 5, nEp=1, toPlot=True, verb=1)
    
    yhh300, co, wins300, win, impo,  impo_cor, impo_win = m300.cv(yxx=yxxs, nWin=[.01, .05, .10], 
             cdxx= th.cdxx, n_splits= 5, nEp=1, toPlot=True, verb=1)
    yhhr6_300, co, winsr6_310, win, impo,  impo_cor, impo_win = mr6_310.cv(yxx=yxxs, nWin=[.01, .05, .10], 
             cdxx= th.cdxx, n_splits= 5, nEp=1, toPlot=True, verb=1)
    yhhr6_1501, co, winsr6_1501, win, impo,  impo_cor, impo_win = mr6_310.cv(yxx=yxxs, nWin=[.01, .05, .10], 
            cdxx= th.cdxx, n_splits= 5, nEp=1, toPlot=True, verb=1)
   
    yhh1500, co, wins1500, win, impo,  impo_cor, impo_win = m300.cv(yxx=yxxs, nWin=[.01, .05, .10], 
             cdxx= th.cdxx, n_splits= 5, nEp=1, toPlot=True, verb=1)
    
    
    wins100.yh.vc()
    wins1500.yh.vc()
    
    wins100.query('nWin== .1 and y < "zy_3"  and yh < "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_20')).iloc[1:]
    wins300.query('nWin== .1 and y < "zy_3"  and yh < "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_20')).iloc[1:]
    wi= wins1500.query('nWin== .1 and y < "zy_3"  and yh < "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_20')) #.iloc[1:]
    wi= wins1500.query('nWin== .05 and yh > "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_20')) #.iloc[1:]
    wi= winsr6_310.query('nWin== .05 and yh > "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_20')) #.iloc[1:]
    wi= winsr6_1501.query('nWin== .05 and yh > "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_20')) #.iloc[1:]
    wi
    # relative to perfect, %
    (100* wi / wi.iloc[0]) .round(2).win
#%% ====

class Hist2Tensor: 
#%%  class Hist2Tensor             
   def prep_XX_ret(s, sss='SPY AAPL MSFT F', fHist='', hist='', 
                   kma=[5,10,20,30], var='ret',
                   toPlot=True, n_pred=[20,40], n_hist=10, verb=1):
        ''' prep input for keras train and CV from yf  historical OHLCV data.
        create Rule x-variables and y_20, y_40  variables
        
        Thinker.prep_XX() ->
           rr=  {ass: PDF (389 x 11)}
           yxx= {ass: list 339 x PDF (10 x 11)}  -> pilo('out/yxx_tr.pck') normalized
        -> Thinker.crea_model
        
            # prep yxx     
            yxx[s]=[]
            for i in range(n_hist, len(q1)- max(n_pred)):
                u= q1.iloc[i- n_hist: i].copy()
                q_norm= u.apply(lambda r: .25*(r.Open+ r.High+ r.Low+ r.Close), **a1).values[-1]
                for c in ['Open','High','Low','Close']: u.loc[:, c]= norm_last(u[c], q_norm )
                for c in ['rule_5_20','rule_10_30']: u.loc[:, c]= norm_last(u[c], q_norm, m1=0 )
                u.loc[:,'Volume']=  np.log(norm_last(u['Volume'], m1=0)/100)
                yxx[s].append(u)        
        
        toNorm= 'ret' '''
        pr, di= (dummy, dummy) if verb==0  else  (print, display)

        
        if type(hist)==str and hist=='':
            if fHist=='':
                pgre('prep_XX_ret  Observer.get_hist_qoutes')
                _, hist= Observer.get_hist_qoutes(ss=sss, provider='yf')
            else:  
                pgre(f'prep_XX_ret,  {fHist=},  pilo({fHist})')
                hist= pilo(fHist)  #.rename(columns={'level_1':'ass'})

        def incr(a,b): return 100* np.log(a/b)   # log increment
        def em(x,k): return x.ewm(com=k).mean()  # exp moving average
        pd.Series.em= em
        
        uu, uud= [],{}
        for a, r in hist.groupby('level_1'):  # = asset
            pr(a, r.shape)
            try:
                 q1= r.drop('Adj Close', **a1)  # no Adj.Close for  SPY
            except: pass
            
            
            #yxx= PDF(dict(
            yxx= PDFd(
                y_20= incr(r.Close.shift(-20), r.Close) * day_minutes / 20,
                y_40= incr(r.Close.shift(-40), r.Close) * day_minutes / 40,
                rule_5_20= 100*(r.Close.em(5) / r.Close.em(20) -1),
                rule_10_30= 100*(r.Close.em(10) / r.Close.em(30) -1 ))
            #)             
            
            if var=='ret':
                #yxx= pd.concat([yxx, PDF(dict(
                yxx= yxx.appe_col(
                    o= incr(r.Open  , r.Close),
                    h= incr(r.High  , r.Close),
                    l= incr(r.Low   , r.Close),
                    c= incr(r.Close , r.shift().Close ),
                    v= incr(r.Volume , r.shift().Volume )
                ) #) ], **a1)
                
            else:
                q_norm= q1.apply(lambda r: .25*(r.Open+ r.High+ r.Low+ r.Close), **a1).values[-1]
                for c in ['Open','High','Low','Close']: q1[c]= q1[c] / q_norm
                q1['Volume']=  q1['Volume']/ q1['Volume'].iloc[-1]
                
        
                #u= q1
                yxx= pd.concat([yxx, q1], **a1)
                #yxx= yxx.appe_col(q1)
                #u['rule_5_20']=  u.Close.em(5)  - u.Close.em(20)
                #u['rule_10_30']= u.Close.em(10) - u.Close.em(30)
                
            yxx= yxx.dropna().query('rule_5_20 != 0')
            if len(yxx) >0: 
                #yxx.tab(a + '--> u')
                yxx[['c' if var=='ret' else 'Close', 'rule_5_20', 'rule_10_30']].ri(drop=True).plot(title= f'{a}, {len(yxx)=}')
                #plt.show()
                
            u1= []    
            for i in range(n_hist, len(yxx)):
                v= yxx.iloc[i- n_hist: i].copy().dropna()
                if len(v)== n_hist:
                    uu.append(v.copy())
                    u1.append(v.copy())
            uud[a]= u1
                    
        # for keras:
        lidf2ker= Thinker.lidf2ker
        
        s.cdxx= ['c','h','l','o','v'] if var=='ret' else ['Open','High','Low','Close','Volume']
        s.csxx= ['rule_5_20','rule_10_30']
        s.cy= ['y_20', 'y_40']
        
        #di('uu[-2:]=', uu[-2:])
        #return uu, uud, 'dummy'
        
        yxx_ker=   [lidf2ker(s.cdxx, uu), 
                    lidf2ker(s.csxx, uu, onlyLast=True).reshape(-1,2)
                   ], \
                   lidf2ker(s.cy, uu, onlyLast=True) #.reshape(-1,2)      
                    
        return uu, uud, yxx_ker  


   if 0 and toTest:        
       th= Thinker() 
       uu, uud, yxx_ker= prep_XX_ret(s=th, sss='MSFT F', fHist='', hist='', 
                    kma=[5,10,20,30], var='zret',
                    toPlot=True, n_pred=[20,40], n_hist=3, verb=1)
       
       th.csxx
       th.cy
        
       info(uu,'uu')
       info(uud,'uud')
       info(yxx_ker,'yxx')
       shape_ker(yxx_ker)
       
       (txd, txs), ty= yxx_ker
       (txd[0], txs[0]), ty[0]


#%% Inspect
if 0:    
    print(sys.version)
    print(sys.version_info )
    print(sys.executable )
    dima('### Test')
    pr(Markdown('### Test').data)
    
    # import webbrowser
    # webbrowser.open('https://msn.com', new=2)
    
    print ("In IPython" if __IPYTHON__ else "Not in IPython")
    
    import inspect
    for frame in inspect.stack():  print (frame.lineno, frame.filename) #.file)
    
    import fun
    for c in  inspect.getmembers(fun, inspect.isclass): 
        if c[0][0].isupper() :
            pr('class ', c[0])
            
    classes= [c[1] for c in inspect.getmembers(fun, inspect.isclass) if c[0][0].isupper()   and c[0][:3] !='IPy'] #and c[0][:3]=='fun'  
    for c in  inspect.getclasstree(classes, unique=False):
        pr('\ncl:', c)
        try:
            for c1 in c: pr(c1[0])
        except: pass

#%%  Test    

if 0 and toTest:
    def show_shapes(): # can make yours to take inputs; this'll use local variable values
        (tx1, tx2), ty= yxxr
        pred("Expected: (num_samples, timesteps, channels)")
        print("Sequences  tx1: {} x {}".format(len(tx1), tx1[0].shape))
        print("Targets  tx2:   {} x {}".format(len(tx2), tx2[0].shape)) 
    
   # m= Model(nLSTM=128, nStat=64, nDen2=64, nEp=250, fModel='out/model_128_64_1500', toPlot=True)
    #m= Model(nLSTM=128, nStat=64, nDen2=64, nEp=250, fModel='', toPlot=True)
    #m= Model(nLSTM=128, nStat=64, nDen2=64, nEp=250, fModel='', toPlot=True)
    
    yxx= pilo('out/yxx_tr.pck')   
    
    # or
    #rr, yxx= Thinker.prep_XX(sss='SPY QQQ MSFT F ALB KD PROK  ALGN WFC AAL', kma=[5,10,20, 30], toNorm=False, toPlot=True, n_pred=[20,40])
    if 0 and type(yxx)==dict:
        #e.g.  rr, yxx= Thinker.prep_XX(sss='SPY QQQ MSFT F ALB KD PROK  ALGN WFC AAL', kma=[5,10,20, 30], toNorm=False, toPlot=True, n_pred=[20,40])
        yxx= [v1 for v in yxx.values() for v1 in v if len(v1.dropna()) ==10]

    info(yxx, 'yxx')
    
    
#if 0:    
    pr(g, '\n pilo len(yxx)=', len(yxx), yxx.keys(), 
       len(yxx['F']), ' x ', yxx['F'][0].shape, yxx['F'][0].tab("yxx['F'][0]"), er)
    
    # flatten the dic into list and filter
    #yxx= sum([[y for y in yxx[k] if len(y.dropna()) ==10] for k in yxx if k>''], []) 
    yxx= [v1 for v in yxx.values() for v1 in v if len(v1.dropna()) ==10]

    
    info(yxx, 'Flattened long  yxx')
    
    
    #pr(g, '\n 222222 len(yxx)=', len(yxx), yxx[0].shape,    yxx[0].columns, er)
    #yxx[0].tab(' yxx[0]')
    ker= Thinker.trd2Xy(yxx)    
    #di('ker=', ker)    
    info(ker, 'ker')

    
    m= Model(nLSTM=128, nStat=64, nDen2=64, nEp=250, fModel='', toPlot=True)  
    m.model.fit(*ker, epochs=2, verbose=0)
    pred('Fitted =======')
    

    
    #uu, yxx = Thinker.prep_XX(sss='SPY AAPL MSFT F', kma=[5,10,20,30], toNorm=True, toPlot=True, n_pred=[20,40], n_hist=10)
    uur, yxxr = Thinker.prep_XX(sss='MSFT F', kma=[5,10,20,30], toNorm= 'ret', toPlot=True, n_pred=[20,40], n_hist=10)

    #pr(g, '\nlen(uur)=', len(uur), f', {uur[0].shape=}', er)   # list of PDF
    #uur[0].tab("uur[0]")
    info(uur, 'uur')

    
    
    if 0:
        yxx= sum([[y for y in yxx[k] if len(y.dropna()) ==10] for k in yxx if k>''], [])  #%% dict -> array
   
        # shaffle rows
        ii=  np.random.RandomState(seed=55).permutation(len(yxx))        
        yxx= [yxx[i] for i in ii] 
        train_data = yxx[:-400]
        
    
    if 0:
        (tx1, tx2), ty= yxxr
        yxxr= (np.asarray([t.astype('float64') for t in tx1]), np.asarray(tx2)), np.asarray(ty)
        
        di('uur :2 =', uur[:2])
    
    show_shapes()
    pr('\nlen(yxxr)=', len(yxxr), f', {len(yxxr[0][1])=}, {len(yxxr[1])=}, {yxxr[1].shape=}' )# list of npa
    
    if 0:
        for u in uur :
            for i,r in u.iterrows():
                if i>9: pr(444444, i, u.loc[i-10:i, :] )
    
    #uur_all= sum([u.loc[i-10:i, :] for u in uur for i,r in u.iterrows() if i>9], []) 
    uur_all= [u.loc[i-11:i, :] for u in uur for i,r in u.iterrows() if i>10 and len(u.loc[i-10:i, :])==10]
    
    pr(f'\n{len(uur_all)=}, {uur_all[0].shape=}')
    
    yxxr= Thinker.trd2Xy_ret(uur_all)
    info(yxxr, 'yxxr')

    
    di(f'{len(yxxr)}, {yxxr[0][0][0].shape=}, {yxxr[0][1].shape=}')
#if 0:
    
    m= Model(nLSTM=128, nStat=64, nDen2=64, nEp=250, fModel='', toPlot=True)
    
    list(map(tab, yxx[:2]))
    m.model.fit(*Thinker.trd2Xy(yxx[:9]), epochs=2)
    PDF(m.model.predict( Thinker.trd2Xy(yxx[:9])[0]), columns=['y_20h', 'y_40h'])   
    
    
    m.train( yxxr,  nEp=3, toPlot=True)
    
    #m.train( uur_all,  nEp=3, toPlot=True)
    #m.train( uur,  nEp=3, toPlot=True)
   # m.cv(yxx=yxxr, nWin=[.01, .05, .10], n_splits= 5, nEp=2, toPlot=True, verb=1)
#    m.cv(yxx=ker, nWin=[.01, .05, .10], n_splits= 5, nEp=2, toPlot=True, verb=1)
    m.cv(yxx=yxx, nWin=[.01, .05, .10], n_splits= 5, nEp=2, toPlot=True, verb=1)
 #   m.cv(yxx=yxxr, nWin=[.01, .05, .10], n_splits= 5, nEp=2, toPlot=True, verb=1)
    
    '''
        np.round(10000* yxxr[0][0])
        np.round(10000* yxxr[0][0])
        np.round(10000* yxxr[1][0])
        yxxr[0][0].shape  dyn  x  tx1   (674, 10, 5)
        yxxr[0][1].shape  stat x  tx2   (674,     2)
        yxxr[1].shape     y       ty    (674, 1,  2)
    '''    

    
    #%% fun
    if 0:
        import fun
        def irelo(): reload(fun)
        irelo()
        
        help(fun)
        help(fun.Thinker)
        help(fun.Thinker.crea_model)
        help(fun.Model)
        Model.__dict__
        m= Model()
        m.__dict__
        
        dima('## az')
        
        u= pilo('out/yXX.pck')
        info(u, 'yXX')        
        u[0].tab('u[0]')        
        
        u= pilo('out/hql_05-23.pck')
  
        u.head(5).tab('hql')
        u.tab('hql')
        u.level_1.vc()
        
        hq, hql = Observer.get_hist_qoutes(ss='MSFT F', provider='yf')
        info(hq, 'hq')
        hq.columns
        info(hql, 'hql')
        hql.to_dict()
        pidu(hql, 'out/hql2.pck')                # 46K
        pidu(hql.to_dict(), 'out/hql2_dict.pck') # 76K
        
        hq.columns
        hq[['Close']]
        hq[['F']]
        
        
        s.cdxx= []
        if 'ret' in var: s.cdxx += ['c','h','l','o','v'] 
        if 'a4' in var:  s.cdxx += ['Open','High','Low','Close','Volume']
        
        var=['ret']
        var=['ret', 'a4']
        cdxx= ['c','h','l','o','v'] * ('ret' in var) +  ['Open','High','Low','Close','Volume'] * ('a4' in var)
        cdxx
        
        
    
                
            
        wi= winsf(yhh)    
        wi.query('nWin== .1  ').pivot(('yh', 'nWin'), 'y').svde(('win','y_40'))
        w20= wi.query('nWin== .1 and y < "y_3"  and yh < "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_20')).round(1)
        w40= wi.query('nWin== .1 and y > "y_3"  and yh > "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_40')).round(1)
        diss(w20, w40)
        
        di(wins.query('nWin== .1 and y < "zy_3"  and yh < "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_20')).round(1))
        di(wins.query('nWin== .1 and y > "ay_3"  and yh > "y_3" ').pivot(('yh', 'nWin'), 'y').svde(('win','y_40')).round(1))


#%%
        
def hist2ker(hql):
#%%  hist2ker(hql)      
        {a:r for a, r in hql.groupby('level_1')}
        
        hist
    
#%% rule_RSI() 

def em(x,k): return x.ewm(com=k).mean()            # exp moving average
pd.Series.em= em
        
def rule_RSI_with_plot(dt=14):
    import yfinance as yf
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Load the data into a dataframe
    symbol = yf.Ticker('BTC-USD')
    df_btc = symbol.history(interval="1d",period="max")
    
    # Filter the data by date
    #df_btc = df_btc[df_btc.index > datetime(2020,1,1)]
    #df_btc = df_btc[df_btc.index < datetime(2021,9,1)]
    
    # Print the result
    print(df_btc)
    
    # Delete unnecessary columns
    del df_btc["Dividends"]
    del df_btc["Stock Splits"]
    # Coding the Relative Strength (RSI) Index in Python
    
    change = df_btc["Close"].diff()
    change.dropna(inplace=True)
    
    # Create two copies of the Closing price Series
    change_up = change.copy()
    change_down = change.copy()
    
    # 
    change_up[change_up<0] = 0
    change_down[change_down>0] = 0
    
    # Verify that we did not make any mistakes
    change.equals(change_up+change_down)
    
    # Calculate the rolling average of average up and average down
    #avg_up = change_up.rolling(dt).mean()
    #avg_down = change_down.rolling(dt).mean().abs()
    avg_up = change_up.em(dt)
    avg_down = -change_down.em(dt) #.abs()
    
    rsi = 100 * avg_up / (avg_up + avg_down)
    
    # Take a look at the 20 oldest datapoints
    rsi.head(20)
    
    # Set the theme of our chart
    plt.style.use('fivethirtyeight')
    
    # Make our resulting figure much bigger
    plt.rcParams['figure.figsize'] = (20, 20)
    
    # Create two charts on the same figure.
    ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 4, colspan = 1)
    ax2 = plt.subplot2grid((10,1), (5,0), rowspan = 4, colspan = 1)
    
    # First chart:
    # Plot the closing price on the first chart
    ax1.plot(df_btc['Close'], linewidth=2)
    ax1.set_title('Bitcoin Close Price')
    
    # Second chart
    # Plot the RSI
    ax2.set_title('Relative Strength Index')
    ax2.plot(rsi, color='orange', linewidth=1)
    # Add two horizontal lines, signalling the buy and sell ranges.
    # Oversold
    ax2.axhline(30, linestyle='--', linewidth=1.5, color='green')
    # Overbought
    ax2.axhline(70, linestyle='--', linewidth=1.5, color='red')   
    # Display the charts
    plt.show()
    
# Dear Dad - from the list of TA-lib functions that I can see,  
# I would try Stochastic (STOCH), RSI and Momentum (MOM) – 
# these are considered to be the ones that are proven and simple :)    
def rule_RSI(c, dt=14):    
                #change = df_btc["Close"].diff()
                change = c.diff()
                change.dropna(inplace=True)
                    
                # Create two copies of the Closing price Series
                change_up = change.copy()
                change_down = change.copy()
                
                # 
                change_up[change_up<0] = 0
                change_down[change_down>0] = 0
                
                # Verify that we did not make any mistakes
                #change.equals(change_up+change_down)
                
                # Calculate the rolling average of average up and average down
                #avg_up = change_up.rolling(dt).mean()
                #avg_down = change_down.rolling(dt).mean().abs()
                avg_up = change_up.em(dt)
                avg_down = -change_down.em(dt) #.abs()
                
                return 100 * avg_up / (avg_up + avg_down)

# rule_RSI_with_plot(dt=14)
#rule_RSI(df_btc["Close"], dt=14)

#%% talib ------------------------------------

def test_talib0():  # https://github.com/TA-Lib/ta-lib-python
    import talib
    #Gathering all functions in the Talib library
    
    talib.get_functions()
    
    
    Talib = talib.get_function_groups()
    Groups = list(Talib.keys())
    Indicators = []
    for group in Groups:
        pr(group)
        for g in Talib[group]:
            Indicators.append([group, g])
            
    Indicators

    for indicator in Indicators:
            indicator = eval(indicator[1])        


def test_talib():  # https://github.com/TA-Lib/ta-lib-python
    import talib
    import numpy as np
    c = 100* np.exp(np.random.randn(100))

# this is the library function
    k, d = talib.STOCHRSI(c)

# this produces the same result, calling STOCHF
    rsi = talib.RSI(c)
    k, d = talib.STOCHF(rsi, rsi, rsi)

# you might want this instead, calling STOCH
    rsi = talib.RSI(c)
    k, d = talib.STOCH(rsi, rsi, rsi)
    prG(c, rsi, k,d)
    
    #Calculating momentum of the close prices, with a time period of 5:
    mom = talib.MOM(c, timeperiod=5)
    return c, rsi, mom




    
if 0:
    c, rsi, mom= test_talib()
    di(c, rsi, mom)
    
def test_talib2():
    import talib.abstract as taa
    inputs = {
        'open': np.random.random(100),
        'high': np.random.random(100),
        'low': np.random.random(100),
        'close': np.random.random(100),
        'volume': np.random.random(100)
    }
    
    c= inputs['close']
    
    upper, middle, lower = taa.BBANDS(inputs, 20, 2., 2.)
    
    slowk, slowd = taa.STOCH(inputs, 5, 3, 0, 3, 0, prices=['high', 'low', 'open'])
    macd,a,b = taa.MACD(c, fast=5, slow=20)
    
    #m= talib.Indicators.MACD.macd(c, fast=10, slow= 20)
    m,m2,m3= taa.MACD(c, fast=10, slow= 20)
    s= taa.SIGNAL(c, fast=10, slow= 20)
    
    dir(talib.abstract)

    
    
    #return c, upper, middle, lower
    return c, slowk, slowd, macd,a,b,m,m2,m3, s

if 0:
    dir(talib.stream)
    [help('talib.stream.'+t) for t in dir(talib.stream) if t[0] != '_']
    
    i= 0
    for t in dir(talib.stream):
        if t[0] != '_' :
            if i >=120  and i <220: 
                pr('----- ' + t)
                help('talib.stream.'+t) 
            i +=1
    
    ''' talib.stream.ADOSC = stream_ADOSC(...)
        ADOSC(high, low, close, volume[, fastperiod=?, slowperiod=?])
    
        Chaikin A/D Oscillator (Volume Indicators)'''
        
    help(talib.stream.ADOSC)  
    help(talib.MAVP) 
    
    k=1000
    
    inputs = {
        'open': k * np.random.random(100),
        'high': k * np.random.random(100),
        'low': k * np.random.random(100),
        'close': k * np.random.random(100),
        'volume': k * np.random.random(100)
    }
    inputs
    
    high,low,close,open,volume= inputs['high'], inputs['low'], inputs['close'],  inputs['open'], inputs['volume']
    
    talib.stream.ADOSC(inputs['high'], inputs['low'], inputs['close'], inputs['volume'],  fastperiod=2, slowperiod=4)
    talib.ADOSC(inputs['high'], inputs['low'], inputs['close'], inputs['volume'],  fastperiod=2, slowperiod=4)
    talib.ADOSC(high, low, close, volume,  fastperiod=2, slowperiod=4)
    x= [0, 8]
    exec('x[1]= talib.ADOSC(high, low, close, volume,  fastperiod=2, slowperiod=4)    ')
    x

    
if 0:
    high,low,close,open,volume,  real0, real0, real1= 0,0,0,0,0,0,0,0,

def prep_ta_rule_code():

        # '''
        # linux
        # gg 'talib.stream.*=' talib_help.txt -A4 > talib_help_short.txt
        # awk 'c&&!--c;/talib.stream/{c=1}'   talib_help_short.txt > talib_help_short2.txt
        # sed 's/period=./period=2/; s/slowperiod=./slowperiod=4/; s/]/  /; s/\[/  /;' talib_help_short2.txt > talib_help_short2_repl.txt
        # '''
    import talib
    
    k,n =1000, 30
    
    inputs = {
        'open': k * np.random.random(n),
        'high': k * np.random.random(n),
        'low': k * np.random.random(n),
        'close': k * np.random.random(n),
        'volume': k * np.random.random(n)
    }
    inputs
    
    #global high,low,close,open,volume,  real0, real0, real1
    
    high,low,close,open,volume= inputs['high'], inputs['low'], inputs['close'],  inputs['open'], inputs['volume']
    
    
    u= pd.read_csv('ref/talib_help_short2_repl_reduce.txt', sep='\t', header=None)[0].values
    real0, real0, real1= close,close,close
    
    x=close
    
    loca={'talib':talib, 'x':x, 'high':high, 'low':low, 'close':close,'open':open,'volume':volume,  'real0':real0, '':real0, 'real1':real1}
    
    rr= []
    xx= PDF()
    for v in u:
        r= v.split('(')[0].strip()
        #if 'ta_'+r  not in xcc.index: continue
        t= f'x= talib.{v.strip()}'
        try:
            pr(t)
            exec(t, {}, loca )
            #exec('x= close', {}, loca )
            #exec(t, locals= loca )
            
            #pr(f'\n{r=} -> {x.shape=} \n{x[:10]=}\n====')
            rr.append(dict(v=r, t=t, xs=x.shape, x0=x[0]))
            xx['ta_'+r]= x
        except Exception as e:
            pred(t, e)
    
    ppr(rr)
    PDF(rr).tab('rr', nr=200)
    
    if 0:
        for i in range(2):
           pgre(i)
           xx.iloc[:, (i*10):( i*10+10)].tab('xx')
       
       
    
    xc= pd.concat([xx.min(), xx.max(), xx.std(), xx.astype(bool).sum(axis=0), xx.isna().sum()],**a1) #.tab('mima', nr=999)
    xc.columns=['mi', 'ma','sd','not0','nNaN']
   
    cxx_ta_full= pd.concat([PDF(rr), xc.ri()], **a1)
    cxx_ta_full.tab('cxx_full', nr=222)
    #pidu(cxx_ta_full, 'cxx_ta_full.pck')
    
    #xc= pd.concat([xx.min(), xx.max(), xx.std(), xx.astype(bool).sum()],**a1) #.tab('mima', nr=999)
    xcc= xc.query('(mi !=0 or ma!=0)   and ma< 300 and mi > -300 and sd > .01 and not0 > 2') #.ri() #.tab('mima not 0', nr=999)
    xcc.tab('mima not 0', nr=999)
    
    xxx= xx[[x for x in xcc.index]] #.head(20)  # [20 rows x 50 columns]
    for i in range(6):
       pgre(i)
       xxx.iloc[-20:, (i*10):( i*10+10)].tab('xx', nr=20)
    
    return cxx_ta_full, xx

if 0:    
    cxx_ta_full, xx= prep_ta_rule_code()
    
    cxx_ta_full.tab('cxx_ta_full')
    pidu(cxx_ta_full, 'cxx_ta_full.pck')
    
    
    def add_ta2hist(hi): #= pilo('./out/hist_4_06-30.pck')):
        cxx_ta_full= pilo('cxx_ta_full.pck')
        #cxx_ta_full.tab()
        rr= PDF()
        for a, h in hi.groupby('level_1'):
            for i,r in cxx_ta_full.iterrows():
                loca={'talib':talib, 'x':x, 'high':h.High, 'low':h.Low, 'close':h.Close, \
                      'open':h.Open,'volume':h.Volume,  'real':h.Close,  'real0':h.Close, 'real1':h.Close, 'h':h}
    
                code= f"h['ta_{r.v}']= {r.t}" 
                pr(f'{a=}, {r.v=:19}, {code=}')
                exec(code, {}, loca)
            #di(h)
            rr= rr.append(h)
        return rr #= pd.concat(rr) 
    if 0:
        rr= add_ta2hist()
        rr.iloc[:9, :11].tab('rr', nr=25)      
        rr.iloc[-4:, -9:].tab('rr', nr=25)      
            



    

if 0:
    #c, upper, middle, lower= test_talib()
    c, slowk, slowd, macd,a,b,m,m2,m3, s= test_talib()
    PDF([c, slowk, slowd, macd,a,b,m,m2,m3, s]).T
    
    
    talib.Indicators.Stochastic.stochastic_d(c, period=5, k_period=10)
    talib.Indicators.Stochastic.stochastic_k(c, period=10)
    
    
    c = 1000* np.exp(np.random.randn(30))
    c[20]= np.nan
    r= PDFd(c = c)

    rule_RSI= rsi= npa(talib.RSI(r.c, timeperiod=3)).reshape(-1) , #  rule_RSI(r.Close, dt=20),  # https://www.qmr.ai/relative-strength-index-rsi-in-python/  
    rsi= rsi[0]
    
    rule_k, rule_d = talib.STOCH(rsi, rsi, rsi)
    pgre(f'{c=}\n{rsi=}\n{rule_k=}\n{rule_d=}')
    
    k, d = talib.STOCHRSI(r.c, timeperiod=3)
    macd,macda,macdb = taa.MACD(r.c, fast=2, slow= 6) #talib.MACD(c) #, timeperiod=3)  #, fast=3, slow=6)

    u= PDF([c,rsi,rule_k,rule_d,k,d,macd,macda,macdb]).T;   u.columns='c,rsi,rule_k,rule_d,k,d,macd,macda,macdb'.split(',')    
    u.tab('u', nr=30)   
    
    
#%% ====

 
class Dima_backtest:  
    #%%  class Dima_backtest             

    def prep_clients():
        #### Strategies
        st20_1_1= Strategy('st20_1_1', nTop= 1,  nBott= 1, margNeg= .3, dt=20)
        st20_5_3= Strategy('st20_5_3', nTop= 5,  nBott= 3, margNeg= .3, dt=20)
        st20= Strategy('st20_3_2', nTop= 3,  nBott= 2, margNeg= .3, dt=20)
        st40= Strategy('st40_3_2', nTop= 3,  nBott= 2, margNeg= .3, dt=40)
        st40_1_1= Strategy('st40_1_1', nTop= 1,  nBott= 1, margNeg= .3, dt=40)
        st40_1_1h= Strategy('st40_1_1_hunter', nTop= 1,  nBott= 1, margNeg= .3, dt=40)
        st40_5_3= Strategy('st40_5_3', nTop= 5,  nBott= 3, margNeg= .3, dt=40)
        sts= [st20_1_1, st20_5_3, st40_1_1, st40_1_1h, st40_5_3, st20, st40]
        #pr('st20=', st20, sts)
        for s in sts: pr(s)
        
        #### Clients, set initial portfolio
        def init_clients(sts):
            clients= []
            for sa in sts: 
                cl= Client(val=100)
                cl.strat= sa
                cl.id= sa.id
                cl.portf= [Pos(100, 'cash')]
                cl.lastTrade= -990
                cl.trades= 0
                cl.hunter= cl.strat.id.endswith('hunter')
                clients.append(cl)    
        
            pgre('clients init:')
            for c in clients: pr(c)
            return clients
            
        clients= init_clients(sts)
        return clients

    def do_backtest(clients, fHist='out/hist_4_06-30.pck'): 
        th= Thinker()
        th.fee= .0036  # 0.0036
        
        # get format of data for model
        uu, uud, yxx_ker= th.prep_XX_ret(fHist='out/hist_4_06-30.pck', tp_rsi=2,
                        kma=[5,10,20, 30], var=['ret','a4'], toPlot=True, n_pred=[20,40], row0=True)
        
     
        # load model
        m= Model(th)
        #m= m.load('model_ra4_128-64-64-ta_300ep')
        m= m.load('model_ra4_128-64-64-ta_tp_rsi=2_300ep')
        m.n_hist= 10  # ?? hardcoded
        m.model.summary()        
        
        #### backtest run
        t1()
        #th.__class__= Thinker
        #hi4= pilo( fHist)
        clients, ref, ref_all, ref_other, yxx_ker4= Backtester.backtest(th, histor= pilo( fHist), # yxx_ker= yxx_ker4, 
                 clients=clients, model=m, verb=0, ref=['SPY', 'QQQ','ONEQ'], k_fee=1.1)
        t1() 
        
        
    #def prep_XX_ret(s, sss='SPY AAPL MSFT F', fHist='', histo='', 
    def hi2yxx(s, histo='', 
                       kma=[5,10,20,30],  var=['ret','a4','ta'], nLast=1, tp_rsi=2, n_pred=[20,40],
                       toPlot=True, n_hist=10, row0=False, verb=1, name='hi2yxx', ta=[]):  #, noCalc=False
            ''' prep input for keras train and CV from yf  historical OHLCV data.
            create Rule x-variables and y_20, y_40  variables
            var= ['ret', 'a4'] ,  a4 - norm to CHLO
            
            Thinker.prep_XX() ->
               rr=  {ass: PDF (389 x 11)}
               yxx= {ass: list 339 x PDF (10 x 11)}  -> pilo('out/yxx_tr.pck') normalized
            -> Thinker.crea_model
            
                # prep yxx     
                yxx[s]=[]
                for i in range(n_hist, len(q1)- max(n_pred)):
                    u= q1.iloc[i- n_hist: i].copy()
                    q_norm= u.apply(lambda r: .25*(r.Open+ r.High+ r.Low+ r.Close), **a1).values[-1]
                    for c in ['Open','High','Low','Close']: u.loc[:, c]= norm_last(u[c], q_norm )
                    for c in ['rule_5_20','rule_10_30']: u.loc[:, c]= norm_last(u[c], q_norm, m1=0 )
                    u.loc[:,'Volume']=  np.log(norm_last(u['Volume'], m1=0)/100)
                    yxx[s].append(u)        
            
            toNorm= 'ret' '''
            pr, di, pgre, pred, tab= (dummy, dummy, dummy, dummy, dummy) if verb==0  else  (print, display, pgreGl, predGl, tab1)
            if type(var)==str: var= [var]
         
            s.n_hist= n_hist
            s.cdxx= ['c','h','l','o','v','clh'] * ('ret' in var) + \
                    ['Open','High','Low','Close','Volume'] * ('a4' in var) \
                    + ['rule_RSI', 'rule_k', 'rule_d', 'rule_mom'] + ['ta_'+ t for t in ta]
            s.csxx= ['rule_5_20','rule_10_30', 'rule_t']
            s.cy=   ['y_20', 'y_40']
            s.more= ['ass', 'Datetime', 'Close0', 'feep']
            s.nLast= nLast
            s.tp_rsi= tp_rsi
            s.name= name
            s.fee=.0036
            
            if row0: return [[],[]], [], []        
            
            def incr(a,b): return 100* np.log((1e-8 + a) /(1e-8 + b))  # log increment
            def em(x,k): return x.ewm(com=k).mean()                    # exp moving average
            pd.Series.em= em
            
            try:
                histo= histo.rename(columns={'level_2':'Datetime'})
            except: pass
            #histo.tabb('1. histo:')
        
            try:
               histo= histo.rename(columns={'level_1':'ass'}) 
               histo= histo.pivot(index=['ass','Datetime'], columns='level_0', values=0).ri() 
               #histo= histo.pivot(index=[histo.level_1, histo.Datetime], columns='level_0', values=0) 
            except Exception as e: 
                pred(f'Exception {e=} in histo= histo.pivot')
            
            if 0: # debug                th= Thinker()
                th.__class__= Thinker
                os.getcwd()
                
                hi= pilo('out/hist_4_06-30.pck')
                dim= Dima_backtest()
                u= dim.hi2yxx(histo=hi, 
                                   kma=[5,10,20,30],  var=['ret','a4','ta'], nLast=1, tp_rsi=5, n_pred=[20,40],
                                   toPlot=True, n_hist=10, row0=False, verb=1, name='hi2yxx', ta=[])
                
                _,_, yxx_k= u
                shape_ker(yxx_k)  (((1400, 10, 65), (1400, 3)), (1400, 1, 2), (1400, 4))
                
                
                yxxs=  th.read_hist_dir(dire='out', list_only= False, patt='hist_06', 
                                        var=['ret','a4'], nLast=1, tp_rsi=2)                
                yxx= th.read_hist_dir(dire='out', list_only= False, patt='hist_07-07', var=['ret','a4'], nLast=1, tp_rsi=2)
                

                
                
                #uu, uud, yxx_ker= th.prep_XX_ret(sss='MSFT F', kma=[5,10,20, 30], toPlot=True, n_pred=[20,40], var=['ret', 'a4', 'ta'])
                uu, uud, yxx_ker= th.prep_XX_ret(sss='', fHist='out/hist_4_06-30.pck', 
                          kma=[5,10,20, 30], toPlot=True, n_pred=[20,40], var=['ret', 'a4', 'ta'])
                yxx_ker
                yxx_ker[0][0].shape                
                
                len(yxxs)
                shape_ker(yxxs)
           
                #histo.tab('3. histo')
            
            if  'ta' in var: 
                taa, code= Thinker.add_ta2hist(hi=histo, no_ohlcv=True)
                pr(B, f'\ncode=', code)
                s.cdxx= s.cdxx + [c for c in taa.columns if c not in histo.columns]
                #histo= pd.concat([histo, taa[[c for c in taa.columns if c not in histo.columns]]], **a1) # taa.tabb()
                #pr(Y, f'-------------  {s.cdxx=}, \n\n{taa=}, \n{histo=}') 
                
              
            uu, uud, feep, errs= [],{},[],[]
            for a, r in histo.groupby('ass'):  # a = asset = Symbol
                #pr(Y, f'{a, r.shape=},  {r.dropna().shape=}'); r.tabb('r')
                
                
                if 'ta' in var: 
                    taa, code= Thinker.add_ta2hist(hi=r, no_ohlcv=True)
                    pr(B, f'\ncode=', code)
                    #s.cdxx= s.cdxx + [c for c in taa.columns if c not in r.columns]
                    r= pd.concat([r.ri(drop=True), taa[[c for c in taa.columns if c not in r.columns]]], **a1) # taa.tabb()
                    pr(Y, f'-------------  {s.cdxx=}, \n\n{taa=}, \n{r=}')                   
                
          
                try:
                    q1= r.drop('Adj Close', **a1)  # no Adj.Close for  SPY
                except: pass
            
                #if a=='ZION': r.tab('r', nr=22)
            
                #v= r.dropna()
                Dt_minut= (r.Datetime.iloc[-1] - r.Datetime.iloc[0]).total_seconds() /60
            
                pr(B, f'{a=}, {len(r)=}, {len(r.dropna())=},  {Dt_minut=}')

               # if len(r.drop([s.cy], **a1).dropna()) <  Dt_minut -20: # -1: 
                if len(r.dropna()) <  Dt_minut -20: # -1: 
                    pred(f'Skipping {a=}, {len(r)=}, {len(r.dropna())=},  {Dt_minut=}')
                    #r.pivot('Datetime', 'level_0', 0).tab(f'Skipping r (pivot), {a=}')
                    #r.tab(f'{R}Skipping, {a=}  r:')
                    continue
     
                try:                
                    c= r.Close.values
                    rule_RSI= rsi= npa(talib.RSI(c, timeperiod=tp_rsi)).reshape(-1) , #  rule_RSI(r.Close, dt=20),  # https://www.qmr.ai/relative-strength-index-rsi-in-python/  
                    if a== 'F': pgre(f'{r.Close.values=}\n{rsi=}\n{rsi[0]=}')
                    rsi= rsi[0]                
                    rule_k, rule_d=  talib.STOCHRSI(c)
                    pgre(f'{rule_k=}\n{rule_d=}, \n{len(rule_k)=}\n{len(rule_d)=},\n{len(r)=},  ')
                except Exception as e: 
                    pred(f'Exception', e, a, len(r.dropna()))
                    errs+= [dict(a=a, lr=len(r.dropna()))]
                    continue


                yxx= PDFd(
                    y_20= incr(r.Close.shift(-20), r.Close) * day_minutes / 20,  # % / day
                    y_40= incr(r.Close.shift(-40), r.Close) * day_minutes / 40,  # % / day
                    rule_5_20=  100*(r.Close.em( 5) / r.Close.em(20) -1),
                    rule_10_30= 100*(r.Close.em(10) / r.Close.em(30) -1 ),
                    rule_t= r.Datetime.apply(lambda d: 0 if d.time() < ttime(10, 10) else  \
                                                       1 if d.time() < ttime(15, 00) else \
                                                       2),                        
                    rule_RSI=rsi, rule_k=rule_k, rule_d = rule_d, 
                    rule_mom = talib.MOM(r.Close, timeperiod=tp_rsi),                
                        
                    ass= a, Datetime= r.Datetime, Close0= r.Close, feep= s.fee / r.Close  # for yxx_ker[2] = more
                    )
                    
                #yxx.tab(f'1.{a=}. {yxx.shape=},  yxx:', nr=3)
                    
                    
                if 'ret' in var:
                    yxx= yxx.appe_col(
                        o= incr(r.Open  , r.Close),
                        h= incr(r.High  , r.Close),
                        l= incr(r.Low   , r.Close),
                        c= incr(r.Close , r.shift().Close ),
                        clh= incr(r.Close - r.Low, r.High - r.Low) / 100,
                        #v= incr(r.Volume , r.shift().Volume )
                        v= r.Volume / r.shift().Volume
                    ) 
                    
                if 'a4' in var:
                    q_norm= q1.apply(lambda r: .25*(r.Open+ r.High+ r.Low+ r.Close), **a1).values[-1]
                    #for c in ['Open','High','Low','Close']: q1[c]= q1[c] / q_norm - 1
                    q1= {c: r[c] / q_norm - 1 for c in ['Open','High','Low','Close']}
                    q1['Volume']=  r['Volume']/ r['Volume'].iloc[-1]  # dup?
                    
                    yxx= yxx.appe_col(**q1)
                    
                if 'ta' in var:
                    #yxx= pd.concat([yxx, taa.iloc[:, 2:]], **a1) 
                    yxx= pd.concat([yxx, taa], **a1) 
        
                    
                #yxx.tab(f'2. {a=}. yxx', nr=3)
        
                    
                if 0: yxx= yxx.dropna().query('rule_5_20 != 0')
                
                #yxx.tab(f'3. {a=}. yxx', nr=3)

                
                #### plot yxx
                if toPlot and len(yxx) >0: 
                    yxx.tab(a + '--> u')
                    #yxx[['c' if 'ret' in var else 'Close', 'rule_5_20', 'rule_10_30']].ri(drop=True).plot(title= f'{a}, {len(yxx)=}')
                    yxx[['c'] * ( 'ret' in var) +  ['Close'] * ('a4' in var) + s.csxx].ri(drop=True).plot(title= f'{a}, {len(yxx)=}')
                    plt.show()
                
                    
                #### slice yxx --> u
                #pgre(f'slice yxx --> u {n_hist, yxx.shape, len(yxx)=}')
                u1= []    
                for i in range(n_hist, len(yxx)):
                    v= yxx.iloc[i- n_hist: i].copy()
                    #if len(v)== n_hist  and v.max() < np.inf :
                    if len(v.drop(s.cy, **a1).dropna()) != n_hist : continue
                
                    #v.tab('11111 v')
                    if i== n_hist: pr(Y, f'{s.cdxx=}, \n{v.columns=}')
                    #s.cdxx= [c for c in s.cdxx if c in v.columns] # ???
                    #if len(v)== n_hist  and max(v[s.cdxx].max()) < 9e9 :                    
                    if len(v)== n_hist  and max(v.drop(columns=['ass', 'Datetime'] + s.cy).max()) < 9e9 :                    
                       # max(np.max(uu[0][th.cdxx]))
                        uu.append(v.copy())
                        u1.append(v.copy())
                    else: 
                        pr(Y, v.Volume, v.tabb('v:'))
                        pred(f' miss {a=},  {i=}, {len(yxx)=},    {len(v)=},   \n{v=}')
                        pr(Y, 'missed in  v  cols:',  [c for c in s.cdxx if c not in v.columns])
                        #pred(f' \n{PDF(v.max()).ri().tab(nr=99)=} \n{max(v.max())=}')
                        pred(f'{PDF(v[s.cdxx].max()).ri().tab(nr=99)=} \n{max(v[s.cdxx].max())=}')
                uud[a]= u1
                
                if 0:
                    th= Thinker()
                    m.__class__= Model
                    yxx07= th.read_hist_dir(dire='out', list_only= False, patt='hist_07-11', var=['ret','a4','ta'], nLast=1, tp_rsi=2)
                   
                    
                        
            pr(Y, 'len(uu)=', len(uu))
            #pgre('uu=', uu)
            
            if len(uu)==0:
              #pred(f'len(uu)==0, {len(v)=}, {v=}, return []')
              pred(f'len(uu)==0, \n{uud=},  \nreturn []')
              return  [[],[]], [], []
          
            di('uu[0]=', uu[0].columns, uu[0])
            #return

            ####  transform uu --> yxx_ker  by variable groups for keras:
            lidf2ker= Thinker.lidf2ker
            ns= len(s.csxx) 
            sh= [-1,  ns] if s.nLast==1 else  [-1, nLast, ns]
            #sh= [-1,  *( [] if s.nLast==1 else  [nLast]), ns]
            
            pr(B, f'{len(s.cdxx)=}, {s.cdxx=}, \n{uu=}' )
            
            yxx_ker=   [lidf2ker(s.cdxx, uu), 
                        lidf2ker(s.csxx, uu, nLast=nLast).reshape(*sh)
                       ], \
                       lidf2ker(s.cy, uu, nLast=1), \
                       pd.concat([(u[s.more].iloc[-1:]) for u in uu])    # fee %
                       #lidf2ker(s.more, uu, nLast=1) #.redundant??   # fee %
                       #.reshape(-1,2) 
                       
            if errs != []: pred('errs=', PDF(errs))           
                        
            return uu, uud, yxx_ker   


    def hi2xx(s, histo='', 
                      kma=[5,10,20,30],  var=['ret','a4','ta'], nLast=1, tp_rsi=2, n_pred=[20,40],
                      toPlot=True, n_hist=10, row0=False, verb=1, name='hi2yxx', ta=[]):  #, noCalc=False
           ''' prep input for keras train and CV from yf  historical OHLCV data.
           create Rule x-variables and y_20, y_40  variables
           var= ['ret', 'a4'] ,  a4 - norm to CHLO
           
           Thinker.prep_XX() ->
              rr=  {ass: PDF (389 x 11)}
              yxx= {ass: list 339 x PDF (10 x 11)}  -> pilo('out/yxx_tr.pck') normalized
           -> Thinker.crea_model
           
               # prep yxx     
               yxx[s]=[]
               for i in range(n_hist, len(q1)- max(n_pred)):
                   u= q1.iloc[i- n_hist: i].copy()
                   q_norm= u.apply(lambda r: .25*(r.Open+ r.High+ r.Low+ r.Close), **a1).values[-1]
                   for c in ['Open','High','Low','Close']: u.loc[:, c]= norm_last(u[c], q_norm )
                   for c in ['rule_5_20','rule_10_30']: u.loc[:, c]= norm_last(u[c], q_norm, m1=0 )
                   u.loc[:,'Volume']=  np.log(norm_last(u['Volume'], m1=0)/100)
                   yxx[s].append(u)        
           
           toNorm= 'ret' '''
           pr, di, pgre, pred, tab= (dummy, dummy, dummy, dummy, dummy) if verb==0  else  (print, display, pgreGl, predGl, tab1)
           if type(var)==str: var= [var]
        
           s.n_hist= n_hist
           s.cdxx= ['c','h','l','o','v','clh'] * ('ret' in var) + \
                   ['Open','High','Low','Close','Volume'] * ('a4' in var) \
                   + ['rule_RSI', 'rule_k', 'rule_d', 'rule_mom'] + ['ta_'+ t for t in ta]
           s.csxx= ['rule_5_20','rule_10_30', 'rule_t']
           s.more= ['ass', 'Datetime', 'Close0', 'feep']
           s.nLast= nLast
           s.tp_rsi= tp_rsi
           s.name= name
           s.fee=.0036
           
           if row0: return [[],[]], [], []        
           
           def incr(a,b): return 100* np.log((1e-8 + a) /(1e-8 + b))  # log increment
           def em(x,k): return x.ewm(com=k).mean()                    # exp moving average
           pd.Series.em= em
           
           try:
               histo= histo.rename(columns={'level_2':'Datetime'})
           except: pass
           #histo.tabb('1. histo:')
       
           try:
              histo= histo.rename(columns={'level_1':'ass'}) 
              histo= histo.pivot(index=['ass','Datetime'], columns='level_0', values=0).ri() 
              #histo= histo.pivot(index=[histo.level_1, histo.Datetime], columns='level_0', values=0) 
           except Exception as e: 
               pred(f'Exception {e=} in histo= histo.pivot')
           
           
           if  'ta' in var: 
               taa, code= Thinker.add_ta2hist(hi=histo, no_ohlcv=True)
               pr(B, f'\ncode=', code)
               s.cdxx= s.cdxx + [c for c in taa.columns if c not in histo.columns]
               #histo= pd.concat([histo, taa[[c for c in taa.columns if c not in histo.columns]]], **a1) # taa.tabb()
               #pr(Y, f'-------------  {s.cdxx=}, \n\n{taa=}, \n{histo=}') 
               
             
           uu, uud, feep, errs= [],{},[],[]
           for a, r in histo.groupby('ass'):  # a = asset = Symbol
               #pr(Y, f'{a, r.shape=},  {r.dropna().shape=}'); r.tabb('r')
               
               
               if 'ta' in var: 
                   taa, code= Thinker.add_ta2hist(hi=r, no_ohlcv=True)
                   pr(B, f'\ncode=', code)
                   #s.cdxx= s.cdxx + [c for c in taa.columns if c not in r.columns]
                   r= pd.concat([r.ri(drop=True), taa[[c for c in taa.columns if c not in r.columns]]], **a1) # taa.tabb()
                   pr(Y, f'-------------  {s.cdxx=}, \n\n{taa=}, \n{r=}')                   
               
         
               try:
                   q1= r.drop('Adj Close', **a1)  # no Adj.Close for  SPY
               except: pass
           
               #if a=='ZION': r.tab('r', nr=22)
           
               #v= r.dropna()
               Dt_minut= (r.Datetime.iloc[-1] - r.Datetime.iloc[0]).total_seconds() /60
           
               pr(B, f'{a=}, {len(r)=}, {len(r.dropna())=},  {Dt_minut=}')

              # if len(r.drop([s.cy], **a1).dropna()) <  Dt_minut -20: # -1: 
               if len(r.dropna()) <  Dt_minut -20: # -1: 
                   pred(f'Skipping {a=}, {len(r)=}, {len(r.dropna())=},  {Dt_minut=}')
                   #r.pivot('Datetime', 'level_0', 0).tab(f'Skipping r (pivot), {a=}')
                   #r.tab(f'{R}Skipping, {a=}  r:')
                   continue
    
               try:                
                   c= r.Close.values
                   rule_RSI= rsi= npa(talib.RSI(c, timeperiod=tp_rsi)).reshape(-1) , #  rule_RSI(r.Close, dt=20),  # https://www.qmr.ai/relative-strength-index-rsi-in-python/  
                   if a== 'F': pgre(f'{r.Close.values=}\n{rsi=}\n{rsi[0]=}')
                   rsi= rsi[0]                
                   rule_k, rule_d=  talib.STOCHRSI(c)
                   pgre(f'{rule_k=}\n{rule_d=}, \n{len(rule_k)=}\n{len(rule_d)=},\n{len(r)=},  ')
               except Exception as e: 
                   pred(f'Exception', e, a, len(r.dropna()))
                   errs+= [dict(a=a, lr=len(r.dropna()))]
                   continue


               yxx= PDFd(
                   rule_5_20=  100*(r.Close.em( 5) / r.Close.em(20) -1),
                   rule_10_30= 100*(r.Close.em(10) / r.Close.em(30) -1 ),
                   rule_t= r.Datetime.apply(lambda d: 0 if d.time() < ttime(10, 10) else  \
                                                      1 if d.time() < ttime(15, 00) else \
                                                      2),                        
                   rule_RSI=rsi, rule_k=rule_k, rule_d = rule_d, 
                   rule_mom = talib.MOM(r.Close, timeperiod=tp_rsi),                
                       
                   ass= a, Datetime= r.Datetime, Close0= r.Close, feep= s.fee / r.Close  # for yxx_ker[2] = more
                   )
                   
               #yxx.tab(f'1.{a=}. {yxx.shape=},  yxx:', nr=3)
                   
                   
               if 'ret' in var:
                   yxx= yxx.appe_col(
                       o= incr(r.Open  , r.Close),
                       h= incr(r.High  , r.Close),
                       l= incr(r.Low   , r.Close),
                       c= incr(r.Close , r.shift().Close ),
                       clh= incr(r.Close - r.Low, r.High - r.Low) / 100,
                       #v= incr(r.Volume , r.shift().Volume )
                       v= r.Volume / r.shift().Volume
                   ) 
                   
               if 'a4' in var:
                   q_norm= q1.apply(lambda r: .25*(r.Open+ r.High+ r.Low+ r.Close), **a1).values[-1]
                   #for c in ['Open','High','Low','Close']: q1[c]= q1[c] / q_norm - 1
                   q1= {c: r[c] / q_norm - 1 for c in ['Open','High','Low','Close']}
                   q1['Volume']=  r['Volume']/ r['Volume'].iloc[-1]  # dup?
                   
                   yxx= yxx.appe_col(**q1)
                   
               if 'ta' in var:
                   yxx= pd.concat([yxx, taa], **a1) 
       
                   
               #yxx.tab(f'2. {a=}. yxx', nr=3)

               #### slice yxx --> u
               #pgre(f'slice yxx --> u {n_hist, yxx.shape, len(yxx)=}')
               
               # insert 2023-07-17
               v= yxx.iloc[- n_hist: ].copy()
               #if len(v.dropna()) != n_hist : continue
               # if any( np.isnan(v)): continue
               if v.isnull().values.any() : continue
               #if v.isin([np.inf, -np.inf]) : continue
               v.tab('v')
               #if  any(np.isinf(npa(v.iloc[:,2:]))) : continue
               #if  np.isinf(v).values.sum() : continue
               if  any(np.isinf(v.drop(['ass','Datetime'],**a1).astype(float).any().values)) : continue
           
               if 0:     pred= Dima_backtest.get_prediction(mo=m, hi1=hi1 )   #zzzzzz

           
          
               #uu.append({a: v})
               uu.append(v)
               continue

               
               
               u1= []    
               for i in range(n_hist, len(yxx)):
                   v= yxx.iloc[i- n_hist: i].copy()
                   #if len(v)== n_hist  and v.max() < np.inf :
                   if len(v.drop(s.cy, **a1).dropna()) != n_hist : continue
               
                   #v.tab('11111 v')
                   if i== n_hist: pr(Y, f'{s.cdxx=}, \n{v.columns=}')
                   #s.cdxx= [c for c in s.cdxx if c in v.columns] # ???
                   #if len(v)== n_hist  and max(v[s.cdxx].max()) < 9e9 :                    
                   if len(v)== n_hist  and max(v.drop(columns=['ass', 'Datetime'] + s.cy).max()) < 9e9 :                    
                      # max(np.max(uu[0][th.cdxx]))
                       uu.append(v.copy())
                       u1.append(v.copy())
                   else: 
                       pr(Y, v.Volume, v.tabb('v:'))
                       pred(f' miss {a=},  {i=}, {len(yxx)=},    {len(v)=},   \n{v=}')
                       pr(Y, 'missed in  v  cols:',  [c for c in s.cdxx if c not in v.columns])
                       #pred(f' \n{PDF(v.max()).ri().tab(nr=99)=} \n{max(v.max())=}')
                       pred(f'{PDF(v[s.cdxx].max()).ri().tab(nr=99)=} \n{max(v[s.cdxx].max())=}')
               uud[a]= u1
               
               if 0:
                   th= Thinker()
                   m.__class__= Model
                   yxx07= th.read_hist_dir(dire='out', list_only= False, patt='hist_07-11', var=['ret','a4','ta'], nLast=1, tp_rsi=2)
                  
                   
                       
           pr(Y, 'len(uu)=', len(uu))
           #pgre('uu=', uu)
           
           if len(uu)==0:
             #pred(f'len(uu)==0, {len(v)=}, {v=}, return []')
             pred(f'len(uu)==0, \n{uud=},  \nreturn []')
             return  [[],[]], [], []
         
           #di('uu[0]=', uu[0].columns, uu[0])
           #return

           ####  transform uu --> yxx_ker  by variable groups for keras:
           lidf2ker= Thinker.lidf2ker
           ns= len(s.csxx) 
           sh= [-1,  ns] if s.nLast==1 else  [-1, nLast, ns]
           #sh= [-1,  *( [] if s.nLast==1 else  [nLast]), ns]
           
           pr(B, f'{len(s.cdxx)=}, {s.cdxx=}, \n{uu[0]=}' )
           
           return uu, 0, (npa([u.loc[:, s.cdxx] for u in uu]),  npa([u[s.csxx].iloc[-1] for u in uu])), \
                          0, \
                          pd.concat([(u[s.more].iloc[-1:]) for u in uu])
           
           yxx_ker=   [lidf2ker(s.cdxx, uu), 
                       lidf2ker(s.csxx, uu, nLast=nLast).reshape(*sh)
                      ], \
                      lidf2ker(s.cy, uu, nLast=1), \
                      pd.concat([(u[s.more].iloc[-1:]) for u in uu])    # fee %
                      #lidf2ker(s.more, uu, nLast=1) #.redundant??   # fee %
                      #.reshape(-1,2) 
                      
           if errs != []: pred('errs=', PDF(errs))           
                       
           return uu, uud, yxx_ker   


   
        
        
    def get_prediction(mo, hi1, verb=0):
        # at="10:00"; fHist='out/hist_4_06-30.pck'
        
        pr, di, pgre, pred= (dummy, dummy, dummy, dummy) if verb==0  else  (print, display, pgreGl, predGl)


        hi1['y_20']= -99;  hi1['y_40']= -99
        #hi1.tab('hi1')
        
        # th= Thinker()
        # _,_, yxx= th.prep_XX_ret(sss='', fHist='', histo=hi1, 
        #                # kma=[5,10,20,30],  var=['ret','a4','ta'], nLast=1, tp_rsi=5, n_pred=[20,40],
        #                 kma=[5,10,20,30],  var=['ret'], nLast=1, tp_rsi=5, n_pred=[20,40],
        #                 toPlot=True, n_hist=10, row0=False, verb=1)
        
        dim= Dima_backtest()
        if 0: 
            _,_, yxx_k= dim.hi2yxx(histo=hi1, 
                        kma=[5,10,20,30],  var=['ret','a4','ta'], nLast=1, tp_rsi=2, n_pred=[20,40],
                        toPlot=False, n_hist=10, row0=False, verb=0, name='hi2yxx', ta=[])
        
        v= dim.hi2xx(histo=hi1, 
                        kma=[5,10,20,30],  var=['ret','a4','ta'], nLast=1, tp_rsi=2, n_pred=[20,40],
                        toPlot=False, n_hist=10, row0=False, verb=0, name='hi2xx', ta=[])
        
        display(f'{B} ----- v=', len(v), v)
        
        _,_, yxx_k, more= v
        
        
        shape_ker(yxx_k) # (((1400, 10, 65), (1400, 3)), (1400, 1, 2), (1400, 4))
                
        (tx1, tx2), ty, more= yxx_k
        tx1[-1], tx2[-1], more.iloc[-1].Datetime 
        
        more['i']= range(len(more))
        t= more.iloc[-1].Datetime 
       
        
        aa= more.query('Datetime == @t')[['i', 'ass']]
        ii= aa.i.values
        
        
        pr(Y, f'{tx1[ii, :].shape=}')
        pr(tx1[ii, :])
        
        t1()
        yy= mo.model.predict((tx1[ii, :], tx2[ ii, :]), verbose = 0)
        t1('keras prediction')
        return PDF( yy, index= aa.ass, columns=dim.cy)
    
#### Test predictions    
if 0:        
    m= Model()
    #m.tp_rsi=2
   # m.load('mo_65f_714_ta_602ep')
    m.load('mo_65f_ta_299ep')  
    m.model.summary()
    
    #fHist='out/hist_4_06-30.pck'    
    fHist='out/hist_07-12_sp500.pck'    
    hi= pilo(fHist)
    pr(G, f'{len(set(hi.level_1.values))=}')  #53
    hi['tmin']= hi.Datetime.apply(lambda dt: dt.hour*60+ dt.minute )  # minute of the day
    
    at="10:00"
    t= ttime.fromisoformat(at) # + 'Z')
    atm= 60*t.hour+  t.minute # minute of the day of prediction 
    hi1= hi.query('tmin <= @atm').copy()
    #for a,r in hi1.groupby('level_1'): pr(G, a, r.shape);  r.tab(f'{B}{a} hi1', nr=10)
        
    t1()
    pred= Dima_backtest.get_prediction(mo=m, hi1=hi1 )   #zzzzzz
    pred.tab('pred', nr=99)
    t1() # Execution time 0:00:10.21




#### test do_backtest()        
if 0:        
    clients= Dima_backtest.prep_clients()      
    #Dima_backtest.do_backtest(clients, fHist='out/hist_4_06-30.pck') # small , 4 assets 
    Dima_backtest.do_backtest(clients, fHist='out/hist_sp500_07-07.pck') # large  
    
    hi= pilo('out/hist_07-14_1015am-fromDima.pck')
    hi= pilo('out/hist_07-13_sp500.pck')
    hi.tabb('hi')
    
    th= Thinker()
    yxxs=  th.prep_XX_ret(sss='SPY AAPL MSFT F', fHist='', histo='', 
                    kma=[5,10,20,30],  var=['ret','a4','ta'], nLast=1, tp_rsi=5, n_pred=[20,40],
                    toPlot=True, n_hist=10, row0=False, verb=1)
    
    
    #### BTC
    #https://coinmarketcap.com/currencies/bitcoin/
    btc= pd.read_csv('ref/BTC_7D_graph_coinmarketcap.csv', sep=';')
    btc= pd.read_csv('ref/BTC_1D_graph_coinmarketcap.csv', sep=';')
    btc.tab('BTC', nr=30)
            
        
        
        
def hi2npa(hi):
    l= []
    for a, r in  hi.groupby('level_1') : 
        l.append(npa(r))     
    return npa(l)#.drop(columns=)        
                
if 0: 
    l=   hi2npa(hi)    
    l[:][2:]                  
    l.shape    # (53, 390, 9)  (53, 31, 11)  a, t, xx
                
    l[:,:,2:]                  
    l[:,:, 0]                  
    aa= l[:,0, 0]      # assets                   
    tt= l[0,:, 1]      # Datetimes            
    xx= l[:, 7, 2:]    # xx in moment 7
    xx.shape           # (53, 7)
    
    xx= l[:, 10:15, 2:]    # xx in moments 10: 15
    xx.shape               #  (53, 5, 7)
    
    t= 40
    xx= l[:, (t-10):t, 2:]    # xx in moments (t-10):t
    xx.shape                  # (53, 10, 7)
    
    
    # data for predict:
    nxd= 5
    nxs= l.shape[2] - nxd
    xxd= l[:, -10:, 2:(2+nxd)]    
    xxs= l[:, -1, (2+nxd):(2+nxd+nxs)]    
    xxd.shape, xxs.shape  # ((53, 10, 5), (53, 2))
    
    # perform of talib
    a= l[0, :, 3] ; a.shape  # (390,)
    ad= a.astype(double)
    ad= a.astype(float)
    t1()
    for i in range(100000): x= talib.stream.MOM(ad, timeperiod=2)
    t1()  # .12s
    for i in range(100000): x= talib.MOM(ad, timeperiod=2)
    t1()  # .18s
    
    
    pidu(hi,'out/hi.pck')  # 1,334 K
    pidu(l,'out/l.pck')    # 1,696 K
    
    
#def add_ta2npa(hil= hi2npa(pilo('./out/hist_4_06-30.pck')), no_ohlcv=True):
def add_ta2npa(hil, no_ohlcv=True):
        cxx_ta_full= pilo('cxx_ta_full.pck')
        sh= hil.shape # (4, 390, 8)
        len(cxx_ta_full)  # 50
        
        code= '\n'.join([f"ta['ta_{r.v}']= {r.t}" for i,r in cxx_ta_full.iterrows()])
        hil2= np.zeros((sh[0], sh[1], 60))
        hil2[:,:,:8]= hil.copy()
        
        for ia in range(hil.shape[0]):
           # hi.columns  ['level_1', 'Datetime', 'Adj Close', 'Close', 'High',   'Low', 'Open',  'Volume', 'tmin']
           close= hil[ia, :, 3].astype(double)
           high=  hil[ia, :, 4].astype(double)
           low=   hil[ia, :, 5].astype(double)
           open=  hil[ia, :, 6].astype(double)
           volume= hil[ia, :, 7].astype(double)
           real, real0,real1= close,close,close
           pr(hil[ia]) 
           if 0: 
               #ctax= [r.v for i,r in cxx_ta_full.iterrows()]
               ctax= cxx_ta_full.v.values
               code2= '\n'.join([f"hil2[ia,:, {10+i}]= {r.t[3:]:50}.astype(double)   # {i}" for i,r in cxx_ta_full.iterrows()])
               pr(code2)
           
           if 1:               
                hil2[ia,:, 10]= talib.ADX(high, low, close  , timeperiod=2  ).astype(double)
                hil2[ia,:, 11]= talib.ADXR(high, low, close  , timeperiod=2  ).astype(double)
                hil2[ia,:, 12]= talib.APO(real  , fastperiod=2, slowperiod=4).astype(double)
                hil2[ia,:, 13]= talib.AROONOSC(high, low  , timeperiod=2  ).astype(double)
                hil2[ia,:, 14]= talib.CCI(high, low, close  , timeperiod=2  ).astype(double)
                hil2[ia,:, 15]= talib.CDL3OUTSIDE(open, high, low, close).astype(double)
                hil2[ia,:, 16]= talib.CDLBELTHOLD(open, high, low, close).astype(double)
                hil2[ia,:, 17]= talib.CDLCLOSINGMARUBOZU(open, high, low, close).astype(double)
                hil2[ia,:, 18]= talib.CDLENGULFING(open, high, low, close).astype(double)
                hil2[ia,:, 19]= talib.CDLHAMMER(open, high, low, close).astype(double)
                hil2[ia,:, 20]= talib.CDLHANGINGMAN(open, high, low, close).astype(double)
                hil2[ia,:, 21]= talib.CDLHARAMI(open, high, low, close).astype(double)
                hil2[ia,:, 22]= talib.CDLHIKKAKE(open, high, low, close).astype(double)
                hil2[ia,:, 23]= talib.CDLINVERTEDHAMMER(open, high, low, close).astype(double)
                hil2[ia,:, 24]= talib.CDLLONGLINE(open, high, low, close).astype(double)
                hil2[ia,:, 25]= talib.CDLMARUBOZU(open, high, low, close).astype(double)
                hil2[ia,:, 26]= talib.CDLSHORTLINE(open, high, low, close).astype(double)
                hil2[ia,:, 27]= talib.CMO(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 28]= talib.CORREL(real0, real1  , timeperiod=2  ).astype(double)
                hil2[ia,:, 29]= talib.DEMA(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 30]= talib.DX(high, low, close  , timeperiod=2  ).astype(double)
                hil2[ia,:, 31]= talib.EMA(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 32]= talib.KAMA(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 33]= talib.LINEARREG(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 34]= talib.LINEARREG_ANGLE(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 35]= talib.LINEARREG_INTERCEPT(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 36]= talib.LINEARREG_SLOPE(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 37]= talib.MA(real  , timeperiod=2).astype(double)
                hil2[ia,:, 38]= talib.MAX(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 39]= talib.MAXINDEX(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 40]= talib.MFI(high, low, close, volume  , timeperiod=2  ).astype(double)
                hil2[ia,:, 41]= talib.MIDPOINT(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 42]= talib.MIN(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 43]= talib.MININDEX(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 44]= talib.MINUS_DI(high, low, close  , timeperiod=2  ).astype(double)
                hil2[ia,:, 45]= talib.MOM(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 46]= talib.PLUS_DI(high, low, close  , timeperiod=2  ).astype(double)
                hil2[ia,:, 47]= talib.PPO(real  , fastperiod=2, slowperiod=4 ).astype(double)
                hil2[ia,:, 48]= talib.ROCP(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 49]= talib.ROCR(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 50]= talib.RSI(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 51]= talib.SMA(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 52]= talib.T3(real  , timeperiod=2).astype(double)
                hil2[ia,:, 53]= talib.TEMA(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 54]= talib.TRIMA(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 55]= talib.TRIX(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 56]= talib.TSF(real  , timeperiod=2  ).astype(double)
                hil2[ia,:, 57]= talib.ULTOSC(high, low, close  , timeperiod1=2, timeperiod2=4, timeperiod3=6  ).astype(double)
                hil2[ia,:, 58]= talib.VAR(real  , timeperiod=2 ).astype(double)
                hil2[ia,:, 59]= talib.WMA(real  , timeperiod=2  ).astype(double)
                
                           
                pr(B, f'{hil2.shape=}, {hil2[2, :8]=}')
                
        return hil2, [r.v for i,r in cxx_ta_full.iterrows()], code2

               
        taa= []      
        try:
            hi= hi.rename(columns={'level_1':'ass'})
        except:pass
        
        for a, h in hi.groupby('ass'):
            #pgre(f'treating {a=}') 
            ta= h[['ass','Datetime']].copy()  if no_ohlcv else h.copy()
            try:
                loca= {'talib':talib, 'x':0, 'high':h.High, 'low':h.Low, 'close':h.Close, \
                      'open':h.Open,'volume':h.Volume,  \
                      'real':h.Close,  'real0':h.Close, 'real1':h.Close, 'h':h, 'ta':ta}
                exec(code, {}, loca)  # add columns to  ta
                taa.append(ta)
            except: pass
        taa= pd.concat(taa)
        hi_ta= hi[['ass','Datetime']].merge(taa, on=['ass','Datetime'])
        #hi_ta.tabb('hi_ta')
        if no_ohlcv: hi_ta= hi_ta[cxx_ta_full.v.apply(lambda x: 'ta_' + x)]
        return hi_ta, code 
    
    
if 0:
        
        hil2, xx_ta, code2= add_ta2npa(hil= hi2npa(pilo('./out/hist_4_06-30.pck')), no_ohlcv=True)
        
        
        taa1, code= Thinker.add_ta2hist(hi= pilo('./out/hist_4_06-30.pck'), no_ohlcv=False)
        #taa1, code= Thinker.add_ta2hist(hi= pilo('./out/hist_4_06-30.pck'), no_ohlcv=True)
        taa1.tabb('taa1',nr1=20)  ; pr(code)   
        

        def update(x, y):
            return np.concatenate([x[1:,:], y.reshape(1,-1)])
        
        def update2(xx, y):
           return [np.concatenate([x[1:,:], y.reshape(1,-1)]) for x in xx]
       
                
        x= npa([[1,2, 3],
                [5,6, 7]])             
        x3= npa([[1,2, 3],
                 [5,6, 7],                
                 [35,36, 37]])                
        x1= npa([[11,12, 13]])
        
        y= npa([8,9,10])
        update(x, y)
        update2([x, x1], y)
        update2([x,x3,  x1], y)


#%% package ta        
if 0:  # https://github.com/bukosabino/ta   >pip install --upgrade ta
    from ta import add_all_ta_features
    from ta.utils import dropna
    
    # Load datas
    #df = pd.read_csv('ta/tests/data/datas.csv', sep=',')
    df = pd.read_csv(r'C:\z\work\Crypto\ref\in/bukosabino_ta_master_test_data_datas.csv')
    
    # Clean NaN values
    df = dropna(df)

    # Add all ta features
    df = add_all_ta_features(  df, open="Open", high="High", low="Low", close="Close", 
                             volume="Volume_BTC", fillna=True, colprefix='taa_')
    df.shape  # [46306 rows x 94 columns]
    df[20:40].tabb('df',nr2=20);
    
    df.iloc[25:33].T.tab('', nr=100);

        
        
