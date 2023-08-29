# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 11:53:41 2023

@author: alexz
"""
import os
os.chdir(r'C:\z\work\Crypto')
os.getcwd()

import keras
from copy import deepcopy

import json


import asyncio
import nest_asyncio
nest_asyncio.apply()

from pyppeteer import launch
import gc


from fun import *
import fun

import fun2_ibkr_nq as fib

def irelo(): reload(fun); reload(fib)
#irelo()


def wait_to_sec(t= 9):
    sleep= (119.65 + t - tnow().second ) % 60
    while sleep >5:
        pr(Y, f'sleeping {sleep} sec')
        time.sleep(sleep-2)
        sleep= (119.65 + t - tnow().second ) % 60

    time.sleep(sleep)    
    pr(tnow())
    
# wait_to_sec(t= 9)    


toTest= 0 #(__name__=="__main__")
print('toTest=', toTest)

conc= pd.concat



#### Get data ♔♕♖♗♘♙♚♛♜♝♞Black Chess Pawn 🙾 🙿 ⚑ ⚐ ⒶⒷⒸⒹⒺⒻ
os.chdir(r'C:\z\work\Crypto')


def get_BTC_quotes():
    for f in os.listdir('in'):
        pr(f)
        
#get_BTC_quotes() 

def rcsv(f):  return pd.read_csv(f, parse_dates=['timestamp']) 

if 0:    
    u= pd.read_csv('in/kaggle archive/bitstampUSD_1-min_data_2012-01-01_to_2021-03-31.csv')  
    u.tab()  ; # (4857377, 8)
    
    
    u1= pd.read_csv('in/06-01 firstratedata frd_complete_sample/BTC_1min.txt_sample.txt', \
                   header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    u2= rcsv('in/07-28 frd_crypto_sample/BTC_1min_sample.csv')
    u2.columns
                   
    u3= rcsv('in/08-05 frd_crypto_sample/BTC_1min_sample.csv')  
    
    #### A. Prep data to train        
    u= rcsv('in/08-05 frd_crypto_sample/ETH_1min_sample.csv') 
    
    x_imp= 'hm high low open close volume clv rule_2_6 rule_4_12 rule_d rule_k ta_momentum_ao ta_momentum_kama ta_momentum_pvo ta_momentum_pvo_hist ta_momentum_pvo_signal ta_momentum_roc ta_momentum_stoch_rsi ta_momentum_stoch_rsi_d ta_momentum_stoch_rsi_k ta_momentum_stoch_signal ta_momentum_tsi ta_momentum_uo ta_momentum_wr ta_trend_adx ta_trend_adx_neg ta_trend_adx_pos ta_trend_cci ta_trend_dpo ta_trend_ema_fast ta_trend_ichimoku_a ta_trend_ichimoku_b ta_trend_ichimoku_conv ta_trend_kst ta_trend_kst_diff ta_trend_kst_sig ta_trend_mass_index ta_trend_stc ta_trend_trix ta_trend_visual_ichimoku_a ta_trend_visual_ichimoku_b ta_trend_vortex_ind_diff ta_trend_vortex_ind_neg ta_trend_vortex_ind_pos ta_volatility_atr ta_volatility_bbh ta_volatility_bbl ta_volatility_bbp ta_volatility_bbw ta_volatility_dch ta_volatility_dcl ta_volatility_dcm ta_volatility_dcp ta_volatility_dcw ta_volatility_kch ta_volatility_kcl ta_volatility_kcp ta_volatility_kcw ta_volatility_ui ta_volume_adi ta_volume_cmf ta_volume_em ta_volume_fi ta_volume_mfi ta_volume_nvi ta_volume_obv ta_volume_sma_em ta_volume_vpt ta_volume_vwap'.split()
    x_imp= 'hm high low open close volume clv rule_2_6 rule_4_12 rule_d rule_k ta_momentum_ao ta_momentum_kama ta_momentum_pvo ta_momentum_pvo_hist ta_momentum_pvo_signal ta_momentum_roc ta_momentum_stoch_rsi ta_momentum_stoch_rsi_d ta_momentum_stoch_rsi_k ta_momentum_stoch_signal ta_momentum_tsi ta_momentum_uo ta_momentum_wr ta_trend_adx ta_trend_adx_neg ta_trend_adx_pos ta_trend_cci ta_trend_dpo ta_trend_ema_fast ta_trend_ichimoku_a ta_trend_ichimoku_b ta_trend_ichimoku_conv ta_trend_kst ta_trend_kst_diff ta_trend_kst_sig ta_trend_mass_index ta_trend_stc ta_trend_trix ta_trend_visual_ichimoku_a ta_trend_visual_ichimoku_b ta_trend_vortex_ind_diff ta_trend_vortex_ind_neg ta_trend_vortex_ind_pos ta_volatility_atr ta_volatility_bbh ta_volatility_bbl ta_volatility_bbp ta_volatility_bbw ta_volatility_dch ta_volatility_dcl ta_volatility_dcm ta_volatility_dcp ta_volatility_dcw ta_volatility_kch ta_volatility_kcl ta_volatility_kcp ta_volatility_kcw ta_volatility_ui ta_volume_adi ta_volume_cmf ta_volume_em ta_volume_fi ta_volume_mfi ta_volume_nvi ta_volume_obv ta_volume_sma_em ta_volume_vpt ta_volume_vwap'.split()
    x_d1_not_imp= 'h hm hmq ta_momentum_ppo_hist ta_momentum_ppo_signal ta_others_cr ta_others_dlr ta_others_dr ta_trend_aroon_down ta_trend_aroon_up ta_trend_macd_diff ta_trend_macd_signal ta_trend_psar_down_indicator ta_trend_psar_up_indicator ta_trend_sma_fast ta_trend_sma_slow ta_volatility_kchi ta_volatility_kcli us'.split()
    x_d2_not_imp= 'h hm hmq ta_momentum_ppo ta_others_cr ta_trend_ema_slow ta_trend_psar_down_indicator ta_trend_psar_up_indicator ta_trend_sma_slow ta_trend_stc ta_volatility_kchi ta_volatility_kcli us'.split()
    
     
#u= hih
def add_columns(u, rmNA=True):   
    #u['h']= u.timestamp.apply(lambda t:t.time().hour )
    #u['hm']= u.timestamp.apply(lambda t: (p:=t.time()).hour + p.minute/60)
    u['hm']= u.ts.apply(lambda t: t % ( 24*3600) / ( 24*3600))  # float time in days
    u['h']= u.hm # float time in days

    u['hmq']= u.hm.apply(lambda t: t // .25 * .25)
    u['us']= u.hmq.apply(lambda t: w(14 <=t and t <= 20, 1,0))
    
    if 0:
        u.tab()
        u.plot.scatter('hm', 'volume');
        u.groupby('hmq').volume.sum().ri().plot.scatter('hmq', 'volume');
        PDF(u.groupby('hmq').volume.sum()).ri().plot.scatter('hmq', 'volume');
        PDF(u.groupby('hmq').volume.sum()).ri().plot('hmq', 'volume');
        PDF(u.groupby('hmq').volume.sum()).ri().plot();
        PDF(u.groupby('hmq').volume.sum()).ri().plot('hmq', 'volume');
        v= PDF(u.groupby(['hmq','us']).volume.sum()).ri()
        v.plot.scatter('hmq', 'volume', c=v.us.map({0:'b',1:'r'}));
    
    
    def incr(a,b): return 100* np.log((1e-8 + a) /(1e-8 + b))  # log increment
    def em(x,k): return x.ewm(com=k).mean()            # exp moving average
    def dt1(x): return x.diff()         # exp moving average
    def dt2(x): return x.diff().diff()         # exp moving average
    
    scale= dict(p= u.close[:10].mean(), v= u.volume[:10].mean())
    
    u_sc= u.copy()
    for x in ['open', 'high', 'low', 'close']: u_sc[x]= u[x] / scale['p']
    u_sc['volume']= np.log(u.volume / scale['v'])
    
    r, cl= u_sc, u_sc.close
    r['y_1']= incr(cl.shift(-1), cl)
    r['y_2']= incr(cl.shift(-2), cl) / 2
    r['y_3']= incr(cl.shift(-3), cl) / 3
    r['yqb_1']= w(r.y_1 >  .001,  1, 0)
    r['yqs_1']= w(r.y_1 < -.001,  1, 0)
    r['yqb_2']= w(r.y_2 >  .001,  1, 0)
    r['yqs_2']= w(r.y_2 < -.001,  1, 0)
    r['yqb_3']= w(r.y_3 >  .001,  1, 0)
    r['yqs_3']= w(r.y_3 < -.001,  1, 0)
    
    
    r['clv']= (cl - r.low) / (.001 + r.high -r.low)
    r['rule_2_6']=  100*(cl.em( 2) / cl.em(6) -1)            
    r['rule_4_12']= 100*(cl.em(4) / cl.em(12) -1 ) 
    
    # for x in ['open', 'high', 'low', 'close']: u_sc[x+'_d1']= dt1(u_sc[x])
    # for x in ['open', 'high', 'low', 'close']: u_sc[x+'_d2']= dt2(u_sc[x])
    
    
    import talib
    u_sc['rule_RSI']= talib.RSI(cl, timeperiod=3) #  rule_RSI(r.Close, dt=20),  # https://www.qmr.ai/relative-strength-index-rsi-in-python/  
    u_sc['rule_k'], u_sc['rule_d'] = talib.STOCHRSI(cl, timeperiod=3)
    
    #u_sc.tabb();
    
    from ta import add_all_ta_features
    from ta.utils import dropna
    
    import warnings
    warnings.filterwarnings("ignore", category=Warning)
    
    df = add_all_ta_features(u_sc, open="open", high="high", low="low", 
           close="close",  volume="volume", fillna=False, colprefix='ta_')
    
    warnings.resetwarnings()
    
    if 0:
        df.shape # (23036, 109)        
        df.tabb();
        nas = np.isnan(df).sum()
        n= PDF(nas).ri(); n.columns=['x','nnan']; 
        n.tab(nr=222);
    
    df= df.drop(columns=['ta_trend_psar_up', 'ta_trend_psar_down',  # drop many na and std==0
                         'ta_volatility_bbhi','ta_volatility_bbli'])  #[28:-3]
    if 0:
        df.shape # (23008, 107)
        df.tabb();
        stat= PDFd(std= df.std(), mi=df.min(), ma=df.max())
        stat[1:].round(3).tab(nr=133, ndig=3);
        
        
    # Filter
    cy= [y for y in df.columns if y[0]=='y']
    #pr(Y, f'{cy + x_imp=}')
    df= df[cy + x_imp]
    
    for x in df.columns[1:]: 
        if x[0] !='y': 
            if x not in x_d1_not_imp : df[x+'_d1']= dt1(df[x])
            if x not in x_d2_not_imp : df[x+'_d2']= dt2(df[x])
            
            
    if rmNA: df= df[28:-3]
    
    # tri & normalization
    mima= dict()
    for c in df.columns[1:]:
        if c[:2] != 'yq':
            x= df[c]
            x[x < -9999]= -9999; x[x > 9999]= 9999
            mima[c]= {'mi': np.nanmin(x), 'ma': np.nanmax(x)}
            df[c]=  (x - mima[c]['mi']) / (.001 +  mima[c]['ma'] -  mima[c]['mi'])
            
    #df.tabb();    
        
    return df, mima
 
#u= hih
def add_columns_btc(u, rmNA=True, toScale=False):   
    if 'vol' in u.columns: u= u.rename(columns={'vol':'volume'})
    u['date']= pd.to_datetime(u.ts, unit='s')
    #u['h']= u.date.apply(lambda t:t.time().hour )
    u['hm']= u.ts.apply(lambda t: t % ( 24*3600) / ( 24*3600))  # float time in days
    #t1('ts')
    # u['hm']= u.date.apply(lambda t: (p:=t.time()).hour + p.minute/60)

    #u['hmq']= u.hm.apply(lambda t: t // .25 * .25)
    u['us']= u.hm.apply(lambda t: w(14/24 <=t and t <= 20/34, 1,0))
    
    if 0:
        u.tab()
        u.plot.scatter('hm', 'volume');
        u.groupby('hmq').volume.sum().ri().plot.scatter('hmq', 'volume');
        PDF(u.groupby('hmq').volume.sum()).ri().plot.scatter('hmq', 'volume');
        PDF(u.groupby('hmq').volume.sum()).ri().plot('hmq', 'volume');
        PDF(u.groupby('hmq').volume.sum()).ri().plot();
        PDF(u.groupby('hmq').volume.sum()).ri().plot('hmq', 'volume');
        v= PDF(u.groupby(['hmq','us']).volume.sum()).ri()
        v.plot.scatter('hmq', 'volume', c=v.us.map({0:'b',1:'r'}));
    
    
    def incr(a,b): return 100* np.log((1e-8 + a) /(1e-8 + b))  # log increment
    def em(x,k): return x.ewm(com=k).mean()            # exp moving average

    
    scale= dict(p= u.close[:10].mean(), v= u.volume[:10].mean())
    
    u_sc= u.drop(columns=['ts','date']).copy()
    for x in ['open', 'high', 'low', 'close']: u_sc[x]= u[x] / scale['p']
    #u_sc['volume']= np.log(u.volume / scale['v'])
    u_sc['volume']= u.volume / scale['v']
    
    r, cl= u_sc, u_sc.close
    r['y_1']= incr(cl.shift(-1), cl)  * 200  # % daily
    r['y_2']= incr(cl.shift(-2), cl) / 2 * 200
    r['y_3']= incr(cl.shift(-3), cl) / 3 * 200
    y_thr= .05  # 1%/day
    r['yq_1']= w(r.y_1 >  y_thr,  1, 0) -  w(r.y_1 < -y_thr,  1, 0)  # 1 - up. -1 = down
    r['yq_2']= w(r.y_2 >  y_thr,  1, 0) - w(r.y_2 < -y_thr,  1, 0)
    r['yq_3']= w(r.y_3 >  y_thr,  1, 0) - w(r.y_3 < -y_thr,  1, 0)
    
    
    r['clv']= (cl - r.low) / (.001 + r.high -r.low)
    r['rule_2_6']=  100*(cl.em( 2) / cl.em(6) -1)            
    r['rule_4_12']= 100*(cl.em(4) / cl.em(12) -1 ) 
    
    # tri & normalization
    df, mima= r, dict()
    if toScale:
        for c in df.columns[1:]:
            if c[:2] != 'yq':
                x= df[c]
                #x[x < -9999]= -9999; x[x > 9999]= 9999
                x= w(x < -9999, -9999, w(x > 9999,  9999, x))
                m= mima[c]= {'mi': np.nanmin(x), 'ma': np.nanmax(x)}
                df[c]=  (x - m['mi']) / (.001 +  m['ma'] -  m['mi'])
    
    return df, mima
    
def add_talib(u_sc, rmNA=False, cx_not_imp=[], with_taz=True, opts=False, opts_taz=False):    
    import talib
    
    def dt1(x): return x.diff()         # 
    def dt2(x): return x.diff().diff()  #
    
    r, cl= u_sc, u_sc.close

    u_sc['rule_RSI']= talib.RSI(cl, timeperiod=3) #  rule_RSI(r.Close, dt=20),  # https://www.qmr.ai/relative-strength-index-rsi-in-python/  
    u_sc['rule_k'], u_sc['rule_d'] = talib.STOCHRSI(cl, timeperiod=3)
    
    #u_sc.tabb();
    
    from ta import add_all_ta_features
    from ta.utils import dropna
    
    import ta
    
    # o2= {'w_fast': 3, 'w_slow': 12, 'w_sign': 2, 'w_': 11, 'w_sm': 8, 
    #      'w1': 2, 'w2': 4, 'w3': 7, 'w4': 11}
    # o2v= [3, 12, 2, 11, 8, 2, 4, 7, 11] # o2.values()
    #o_taz_v= [3, 10, 3, 11, 7, 5, 8, 14, 20]
    
    opts=     opts     or [3, 12, 2, 11, 8,  2, 4,  7, 11]
    #opts_taz= opts_taz or [3, 10, 3, 11, 7,  5, 8, 14, 20]
    #opts_taz= opts_taz or [2, 10, 3, 14, 7,  5, 8, 14, 20]
    opts_taz= opts_taz or [2,  6, 2,  4, 3,   2, 4, 6,  8]
    w_fast,w_slow,w_sign,w_,w_sm,  w1,w2,w3,w4 = opts_taz
    
    if with_taz:  #  https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html#volatility-indicators
        high, low, close= u_sc.high, u_sc.low, u_sc.close
        u_sc['taz_ma_in']= ta.trend.mass_index(high, low, 
                                        window_fast=w_fast, window_slow=w_slow)
        u_sc['taz_atr']= ta.volatility.average_true_range(high, low, close, window=w_)
        u_sc['taz_adx']= ta.trend.adx(high, low, close, window=w_)
        u_sc['taz_kst']= ta.trend.kst(close, roc1=5, roc2= 7, roc3= 10, roc4= 15, 
                                window1=w1, window2=w2, window3=w3, window4=w4)
        u_sc['taz_ichimoku_b']= ta.trend.ichimoku_b(high, low, window2=w2, window3=w3)
        u_sc['taz_kcw']= ta.volatility.keltner_channel_wband(high, low, close, 
                                             window=w_, window_atr=w_sign)
    
    
    import warnings
    warnings.filterwarnings("ignore", category=Warning)
    
    df = add_all_ta_features(u_sc, open="open", high="high", low="low", 
           close="close",  volume="volume", fillna=False, colprefix='ta_', opts=opts)
    
    warnings.resetwarnings()
    
    if 0:
        df.shape # (23036, 109)        
        df.tabb();
        nas = np.isnan(df).sum()
        n= PDF(nas).ri(); n.columns=['x','nnan']; 
        n.tab(nr=222);
    
    df= df.drop(columns=['ta_trend_psar_up', 'ta_trend_psar_down',  # drop many na and std==0
                         'ta_volatility_bbhi','ta_volatility_bbli'])  #[28:-3]
    if 0:
        df.shape # (23008, 107)
        df.tabb();
        stat= PDFd(std= df.std(), mi=df.min(), ma=df.max())
        stat[1:].round(3).tab(nr=133, ndig=3);
        
        
    # Filter
    cy= [y for y in df.columns if y[0]=='y']
    #pr(Y, f'{cy + x_imp=}')
    if 1:
        x_imp=  [x for x in df.columns if x[0] !='y' and x not in cx_not_imp]
        x_d1_not_imp= []
        x_d2_not_imp= []
    
    df= df[cy + x_imp]
    
    # for x in df.columns[1:]: 
    #     if x[0] !='y': 
    #         if x not in x_d1_not_imp : df[x+'_d1']= dt1(df[x])
    #         if x not in x_d2_not_imp : df[x+'_d2']= dt2(df[x])
            
    d1= conc([dt1(df[x]) for x in df.columns[1:] if x[0] !='y' and  x not in x_d1_not_imp ], **a1)        
    d2= conc([dt2(df[x]) for x in df.columns[1:] if x[0] !='y' and  x not in x_d2_not_imp ], **a1) 
    d1.columns= [x+'_d1' for x in d1.columns]
    d2.columns= [x+'_d2' for x in d2.columns]
    df= conc([df, d1, d2] , **a1)      
            

    if rmNA: df= df[28:-3]

    #df.tabb();    
        
    return df

if 0:
    t1(0)
    df, mima= add_columns(u)  # 23005 rows x 301 columns
    t1()  # Execution time 0:00:34.41  Execution time 0:02:19.263
    
    cx= [x for x in df.columns[1:] if x[0] != 'y']
    cy= [x for x in df.columns[1:] if x[0] == 'y'] # ['y_1', 'y_2', 'y_3', 'yqb_1', 'yqs_1', 'yqb_2', 'yqs_2', 'yqb_3', 'yqs_3']
    X= df[cx]  #  x 204 columns  [23005 rows x 291 columns]  
    y= w(df[cy[5]] > .5, 1,0) # 'yqb_2'  Length: 23005

####  GBM
import lightgbm as lgb


import optuna
from optuna import Trial
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from sklearn.ensemble import GradientBoostingClassifier
from optuna.integration import LightGBMPruningCallback


def objective(trial, X, y):  # https://www.kaggle.com/code/bextuychiev/lgbm-optuna-hyperparameter-tuning-w-understanding
    params={
      'n_estimators':trial.suggest_int('n_estimators', 0, 1000),
      'num_leaves':trial.suggest_int('num_leaves', 2, 512),
      'max_depth':trial.suggest_int('max_depth', 2, 128),
      'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.15),
      'min_split_gain': trial.suggest_loguniform('min_split_gain', 0.001, 0.1),
      'feature_fraction':trial.suggest_uniform('feature_fraction',0.1, 1.0),
      'bagging_freq':trial.suggest_int('bagging_freq',0.1,10),
      'verbosity': -1,
      #'random_state':seed
            } # params={'n_estimators': 10}; trial={}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1121218)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = lgb.LGBMClassifier(objective="binary", **params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="binary_logloss",
            early_stopping_rounds=100,
            # callbacks=[
            #     LightGBMPruningCallback(trial, "binary_logloss")
            # ] 
        )
        preds = model.predict_proba(X_test)
        #cv_scores[idx] = preds
        cv_scores[idx] = log_loss(y_test, preds)

    return np.mean(cv_scores)

if 0:
    study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
    func = lambda trial: objective(trial, X, y)
    t1(0)
    study.optimize(func, n_trials=20)
    t1('optim')  # Execution time 0:09:35.16
    
    sbm= study.best_params
    sbm
            # {'n_estimators': 748,
            #  'num_leaves': 98,
            #  'max_depth': 62,
            #  'learning_rate': 0.004997927132712387,
            #  'min_split_gain': 0.005482450149323626,
            #  'feature_fraction': 0.47149259294980617,
            #  'bagging_freq': 2}
    
        # {'n_estimators': 865,
        #  'num_leaves': 200,
        #  'max_depth': 4,
        #  'learning_rate': 0.08213604437640139,
        #  'min_split_gain': 0.0200539312572924,
        #  'feature_fraction': 0.6378533633787903,
        #  'bagging_freq': 2}
    
    model = lgb.LGBMClassifier(objective="binary", **sbm)
    model.fit(X, y)

    u2.info()
    df, mima= add_columns(u3.iloc[:, :6])  # 23005 rows x 301 columns
    X= df_te[cx]  # [23005 rows x 291 columns]
    yb= w(df_te[cy[5]] > .5, 1,0) #
    ys= w(df_te[cy[6]] > .5, 1,0) #
    
    model_b = lgb.LGBMClassifier(objective="binary", **sbm)
    model_b.fit(X, yb)
    
    model_s = lgb.LGBMClassifier(objective="binary", **sbm)
    model_s.fit(X, ys)
    
    #### Importance 
    imp= PDFd(x=X.columns, imp_b= model_b.feature_importances_, imp_s= model_s.feature_importances_)
    imp['imp']= imp.imp_b + imp.imp_s
    impo= imp.svde('imp').rid().tab(nr=333);
    x_imp= list(set([re.sub('_d1', '', 
                     re.sub('_d2', '', x)) for x in impo.x[:150]]))
    
    x_d1_not_imp= [ x[:-3] for x in impo.x[-50:] if x[-3:]=='_d1']
    x_d2_not_imp= [ x[:-3] for x in impo.x[-50:] if x[-3:]=='_d2']

    len(x_imp) # 68
    ' '.join(sorted(x_imp)) # 
    ' '.join(sorted(x_d1_not_imp)) # 
    ' '.join(sorted(x_d2_not_imp)) # 
    x_imp= 'hm high low open close volume clv rule_2_6 rule_4_12 rule_d rule_k ta_momentum_ao ta_momentum_kama ta_momentum_pvo ta_momentum_pvo_hist ta_momentum_pvo_signal ta_momentum_roc ta_momentum_stoch_rsi ta_momentum_stoch_rsi_d ta_momentum_stoch_rsi_k ta_momentum_stoch_signal ta_momentum_tsi ta_momentum_uo ta_momentum_wr ta_trend_adx ta_trend_adx_neg ta_trend_adx_pos ta_trend_cci ta_trend_dpo ta_trend_ema_fast ta_trend_ichimoku_a ta_trend_ichimoku_b ta_trend_ichimoku_conv ta_trend_kst ta_trend_kst_diff ta_trend_kst_sig ta_trend_mass_index ta_trend_stc ta_trend_trix ta_trend_visual_ichimoku_a ta_trend_visual_ichimoku_b ta_trend_vortex_ind_diff ta_trend_vortex_ind_neg ta_trend_vortex_ind_pos ta_volatility_atr ta_volatility_bbh ta_volatility_bbl ta_volatility_bbp ta_volatility_bbw ta_volatility_dch ta_volatility_dcl ta_volatility_dcm ta_volatility_dcp ta_volatility_dcw ta_volatility_kch ta_volatility_kcl ta_volatility_kcp ta_volatility_kcw ta_volatility_ui ta_volume_adi ta_volume_cmf ta_volume_em ta_volume_fi ta_volume_mfi ta_volume_nvi ta_volume_obv ta_volume_sma_em ta_volume_vpt ta_volume_vwap'.split()
    x_d1_not_imp= 'h hm hmq ta_momentum_ppo_hist ta_momentum_ppo_signal ta_others_cr ta_others_dlr ta_others_dr ta_trend_aroon_down ta_trend_aroon_up ta_trend_macd_diff ta_trend_macd_signal ta_trend_psar_down_indicator ta_trend_psar_up_indicator ta_trend_sma_fast ta_trend_sma_slow ta_volatility_kchi ta_volatility_kcli us'.split()
    x_d2_not_imp= 'h hm hmq ta_momentum_ppo ta_others_cr ta_trend_ema_slow ta_trend_psar_down_indicator ta_trend_psar_up_indicator ta_trend_sma_slow ta_trend_stc ta_volatility_kchi ta_volatility_kcli us'.split()
    
    
    #### Confusion
    df_te, mima= add_columns(u2.iloc[:, :6])  # 23005 rows x 301 columns
    X_te= df_te[cx]  # [23005 rows x 291 columns]
    yb_te= w(df_te[cy[5]] > .5, 1,0) # 'yqb_2'  Length: 23005
    ys_te= w(df_te[cy[6]] > .5, 1,0) # 'yqb_2'  Length: 23005
    
    # buy
    yhb= npa([y[1] for y in model_b.predict_proba(X_te)])
    yhh= PDFd(y=yb_te, yh= w(yhb > .6, 1, 0))
    PDF(yhh.vc()).ri().pivot('y','yh')
                # yh      0     1
                # y              
                # 0   12035    73
                # 1    7774  3123
    
    # full model
        # yh        0       1
        # y                  
        # 0   12304.0     NaN
        # 1    7386.0  3315.0
    
    # sell
    yhs= npa([y[1] for y in model_s.predict_proba(X_te)])
    yhh= PDFd(y=ys_te, yh= w(yhs > .6, 1, 0))
    PDF(yhh.vc()).ri().pivot('y','yh')
            # yh      0     1
            # y              
            # 0   12369     3
            # 1    7757  2876
    
    # full model
        # yh      0     1
        # y              
        # 0   12107     1
        # 1    7290  3607
    
    
        
        
        
    
    u.info()
    u.tabb();  
    u.timestamp.diff().vc()
    
    u['ass']= 'btc'
    #u['date']= u.timestamp
    #u['Volume']= u.volume
    u= u[['ass', 'timestamp',  'open', 'high', 'low', 'close', 'volume']]
    u.columns= ['ass', 'date',  'Open', 'High', 'Low', 'Close', 'Volume']
    
    u.date.diff().vc()

if 0:
    irelo()
    th= fib.Thinker_nq()
    th.__class__ = fib.Thinker_nq
    sk= fib.Thinker_nq.shape_ker
    
    # hi68, quali= th68_te.treat_Dima_IBKR(d='18', di='out/fromDima/f-20230721T231824Z-001') #OK        
    # hi68, quali= th68_te.treat_Dima_IBKR(d='19', di='out/fromDima/f-20230721T231824Z-001') #OK        
    
    # def prep(u, high_chng=True):
    #     # u= u2
    #     u['ass']= 'btc'
    #     u= u[['ass', 'timestamp',  'open', 'high', 'low', 'close', 'volume']]
    #     u.columns= ['ass', 'date',  'Open', 'High', 'Low', 'Close', 'Volume']

    #     th= fib.Thinker_nq()

    #     th.hi= u       
    #     hi_scale5, mn5= th.rescale(mn5= False)
    #     # bta,bmi,ata, 
    #     cxd=  'rule_2_6, rule_4_12, rule_t, taa_volume_mfi, taa_momentum_uo, taa_momentum_wr, taa_volume_nvi, taa_trend_stc, taa_trend_kst_sig, taa_trend_psar_up_indicator, taa_volatility_kcli, taa_others_cr, taa_trend_aroon_down, taa_momentum_stoch_signal, clh, taa_trend_cci, taa_trend_aroon_up, taa_momentum_stoch_rsi, taa_trend_adx_neg, taa_trend_adx, taa_volatility_kchi, taa_others_cr, taa_trend_adx_pos, taa_momentum_stoch_rsi_d, taa_volatility_dch, taa_trend_aroon_ind, taa_trend_kst, taa_volume_obv, taa_momentum_tsi, taa_volatility_atr, taa_volatility_dcw, taa_others_cr, taa_momentum_stoch, taa_momentum_stoch_rsi_k, taa_volatility_dcp, taa_volatility_bbhi, taa_momentum_rsi, taa_trend_psar_down_indicator, taa_volume_vwap, taa_volatility_dcl, taa_others_cr, High, taa_trend_ema_slow, v, taa_volatility_kcl, taa_volatility_bbl, taa_volatility_ui, taa_trend_sma_slow, taa_others_cr, taa_volatility_kch, taa_momentum_kama, taa_volatility_bbm, Close, taa_volatility_kcc, h, Open, taa_trend_ichimoku_a'.split(', ')
    #     cxs=  'taa_trend_visual_ichimoku_a, taa_others_cr, taa_trend_ichimoku_conv, taa_volatility_bbh, taa_others_cr, taa_others_cr, taa_others_cr, taa_others_cr'.split(', ')
    #     th.cxd, th.cxs, th.cyn,  th.cyq=  cxd, cxs, ['y_1', 'y_2'], ['yqb_1', 'yqs_1']
        
    #     t1(0)
    #     xxyy= th.add_xx2_1day()
    #     t1('xxyy= th.add_xx2_1day()')  # Execution time 0:00:26.69  Execution time 0:00:35.66
    
    #     # slicing
    #     ttxxyy= th.xxyy2ker()  #  (28717, 10, 58) (28717, 1, 8) (28717, 1, 3) (28717, 1, 2)
    #     ttxxyy_norm= th.trim_n_norm(large=9999) # calk th.ttxxyy --> th.ttxxyy_norm
    #     (xd,xs), (yn,yq)=  ttxxyy_norm= th.reshape() # calk th.ttxxyy --> th.ttxxyy_norm
    #     sk(ttxxyy_norm)
        
    #     if high_chng:
    #         v= fib.Thinker_nq.high_chng(txxyy=ttxxyy_norm, ymin=.01) 
    #     else: v= ttxxyy_norm
    #     sk(v) 
    #     return(v)
    
    v0= fib.Thinker_nq.prep(u, high_chng=True)
    
    
    
    np.sum(v0[1][1], axis=0)
    np.sum(v2[1][1], axis=0)
    
    #### B. Model
    m2= fib.Model_nq()
    m2.create(th, nLSTM=256, nStat=32, nDen2=64, n_mem=10, var= 'ynq', qActiv='LeakyReLU', for_ker=ttxxyy_norm)
  
    #m2.model.load_weights('../ADZ_Trader/out/models/mo_mo68-256mse-4210ep-119.25vl.hdf5')
    #m2.set_weights(fm='../ADZ_Trader/out/models/mo_mo68-256mse-4210ep-119.25vl.hdf5', k= .2)

    m2.model.compile(loss='mean_squared_error')
    m2.name='mo_btc-256mse'
    
    #### C. Train
    for i in range(2): hist= m2.train(ker_good_norm= v0, nEp=500)      # 10000-> Execution time 0:01:13.709539,  1000 --> Execution time 0:01:32.720 Execution time 0:00:40.87  100-> Execution time 0:01:09.99
    # 500 --> Execution time 0:08:37.045
    m2.nEp
    m2.trd2Xy= fun.Thinker.trd2Xy
    m2.save('out/mo_btc-256mse-1534ep')
    
    
    
    #### D. Validate
    v2, v3= [prep(u, high_chng=False) for u in [u2,u3] ]
    np.sum(v2[1][1], axis=0)

    
    mo= m2
    mo.cyn=  ['y_1', 'y_2']
    mo.cyq=  ['yqb_1', 'yqs_1']
    
    for i,v in enumerate([v2, v3]):
        pr(B, f'\n{i=} --------------------------------------------')
        #(xd,xs), (yn,yq)= v       
        yhh= mo.predict(v)            
        mo.plot_lifts(yhh=yhh,  nEp=0, name=f'date {i}, ') 




    #### E. --> test Backtester
    
from dataclasses import dataclass, field
 
@dataclass
class Strategy:
  id:str
  pport:float= .3  # % portf to buy
  dt:int= 1        # time predict in min
  y_min:float= .1  # threshold
  lastTrade:int =-99
      
      
if 0:
   bt= Backtester()

   # Model
   m= Model_nq(name='').load('out/mo_btc-256mse-1534ep')
   
   tf.keras.__version__
   pr(model.summary())
   
   # Data
   yXX= v2
   pr(f'Backtester, {sk(yXX)=}');
   #di(2222, [(s, v.Datetime.values[0]) for s,v in yXX.items() if s>''])
   #all_t= sum([v.Datetime.values for s,v in yXX.items() if s>''], [np.datetime64('2023-05-23T13:30:00.000000000')])

   # for s,v in yXX.items():
   #     if s>'':
   #         pr(f'{s:8}, {len(v):4} points, {v.iloc[0].Datetime} -  {v.iloc[-1].Datetime}')
           
    
   #### Strategies
   st_1= Strategy('st_1', dt=1, y_min=.0001)
   st_2= Strategy('st_2', dt=2, y_min=.0001)
   st_1_4= Strategy('st_1_4', dt=1, pport= .25)
   st_2_4= Strategy('st_2_4', dt=2, pport= .25)
   sts= [st_1, st_2, st_1_4, st_2_4]

   for s in sts: pr(s) 


   #### Clients, set initial portfolio
   def init_clients(sts):
        clients= []
        for sa in sts: 
            cl= Client(val=1000)
            cl.strat= sa
            cl.id= sa.id
            #cl.portf= [Pos(100, 'cash')]
            cl.portf= dict(cash=1000, u=0, d=0)
            cl.lastTrade= -990
            cl.trades= 0
            cl.hunter= cl.strat.id.endswith('hunter')
            clients.append(cl)    
    
        pgre('clients init:')
        for c in clients: pr(c)
        return clients
        
   clients= init_clients(sts)   
   

   help(fib)
   help(fib.Backtester_btc)
   help(Backtester_btc.run)
   
   irelo()
   BT= Backtester_btc
   p, nu, nd =  BT.run(mo, cl=clients[0], txxyy= v2, hi= u2, n=900, verb=1)
   p, nu, nd =  BT.run(mo, cl=clients[1], txxyy= v2, hi= u2, n=900, verb=1)
   
   p, nu, nd =  BT.run_gbm(mo_b=model_b, mo_s=model_s, cl=clients[0], 
                           X=X, yb=yb, ys=ys, hi=[], n=300, verb=0)

   p, nu, nd =  BT.run_gbm(mo_b=model_b, mo_s=model_s, cl=clients[1], 
                          X=X, yb=yb, ys=ys, hi=[], n=300, verb=0)

class Coinbase_API:
    ''' https://analyzingalpha.com/coinbase-api-python-tutorial 
    
        https://rapidapi.com/community/api/coinbase
    '''
    def test():
        import http.client
    
        conn = http.client.HTTPSConnection("community-coinbase.p.rapidapi.com")
        
        headers = {
            'X-RapidAPI-Key': "a8ffbdfcf3msh0d28330e775a632p1d1d13jsn4cc46b3f69dc",
            'X-RapidAPI-Host': "community-coinbase.p.rapidapi.com"
        }
        
        conn.request("POST", "/?api_key=%3CREQUIRED%3E", headers=headers)
        
        res = conn.getresponse()
        data = res.read()
        
        print(data.decode("utf-8"))
        # Blocked host: community-coinbase.p.rapidapi.com
        
        
    
        import http.client
    
        conn = http.client.HTTPSConnection("tardis-dev.p.rapidapi.com")
        
        headers = {
            'X-RapidAPI-Key': "a8ffbdfcf3msh0d28330e775a632p1d1d13jsn4cc46b3f69dc",
            'X-RapidAPI-Host': "tardis-dev.p.rapidapi.com"
        }
        
        conn.request("GET", "/exchanges/coinbase", headers=headers)
        
        '''https://www.coinbase.com/api/v3/brokerage/stream/products/BTC-USDT/candles?granularity=ONE_MINUTE&limit=300&update_interval=ONE_SECOND_CANDLE
           https://www.coinbase.com/api/v3/brokerage/stream/products/BTC-USDT/candles?granularity=ONE_MINUTE&limit=10&update_interval=ONE_SECOND_CANDLE
        '''
        
        res = conn.getresponse()
        data = res.read()
        
        print(data.decode("utf-8"))
        '''  "message": "Invalid 'exchange' param provided: '%7Bexchange%7D'. 
        Allowed values: 'bitmex', 'deribit', 'binance-futures', 'binance-delivery', 
        'binance-options', 'binance', 'ftx', 'okex-futures', 'okex-options', 'okex-swap', 
        'okex', 'huobi-dm', 'huobi-dm-swap', 'huobi-dm-linear-swap', 'huobi-dm-options', 
        'huobi', 'bitfinex-derivatives', 'bitfinex', 'coinbase', 'cryptofacilities', 
        'kraken', 'bitstamp', 'gemini', 'poloniex', 'upbit', 'bybit', 'bybit-spot', 
        'bybit-options', 'phemex', 'ascendex', 'kucoin', 'serum', 'mango', 'dydx', 'delta', 
        'ftx-us', 'binance-us', 'gate-io-futures', 'gate-io', 'okcoin', 'bitflyer', 'hitbtc', 
        'coinflex', 'crypto-com', 'crypto-com-derivatives', 'binance-jersey', 'binance-dex', 
        'star-atlas', 'bitnomial', 'woo-x', 'blockchain-com'."'''
           
       
        u= data.decode("utf-8")
        
        import json
        j= json.loads(u)
        j.keys()  # ['id', 'name', 'enabled', 'availableSince', 'availableChannels', 'availableSymbols', 'datasets', 'incidentReports']
        for k,v in j.items(): pr(B, k, Y, v)
        j['datasets'].keys()  # ['formats', 'exportedFrom', 'exportedUntil', 'stats', 'symbols']
        j['datasets']['stats']
        for s in j['datasets']['symbols']: pr(s['id'], s['stats']['trades'])
        
        v= PDF([(s['id'], s['stats']['trades'])  for s in j['datasets']['symbols']], columns=['x', 'nTr'])
        v.svde('nTr')
        # 0         SPOT  4025301805
        # 3      BTC-USD   489214266
        # 11     ETH-USD   416034247
        # 72    SHIB-USD   101065626
        # 60     ADA-USD    95748408
    
        j['datasets']['symbols']  
        for s in j['datasets']['symbols']:
            if s['id']=='BTC-USD': ppr(s)
         # {'availableSince': '2019-03-30T00:00:00.000Z',
         #  'availableTo': '2023-08-08T00:00:00.000Z',
         #  'dataTypes': ['trades',
         #                'incremental_book_L2',
         #                'quotes',
         #                'book_snapshot_5',
         #                'book_snapshot_25',
         #                'book_ticker'],
         #  'id': 'BTC-USD',
         #  'stats': {'bookChanges': 36194039116, 'trades': 489,214,266},
         #  'type': 'spot'}
     
         
        import requests
    
        url = "https://tardis-dev.p.rapidapi.com/exchanges/gemini"
        
        headers = {
        	"X-RapidAPI-Key": "a8ffbdfcf3msh0d28330e775a632p1d1d13jsn4cc46b3f69dc",
        	"X-RapidAPI-Host": "tardis-dev.p.rapidapi.com"
        }
        
        response = requests.get(url, headers=headers)
        
        print(response.json())
    
        j= response.json()
        j.keys()  # ['id', 'name', 'enabled', 'availableSince', 'availableChannels', 'availableSymbols', 'datasets', 'incidentReports']
    
        for s in j['datasets']['symbols']:
            #pr(s['id'], s['stats']['trades'])
            if s['id']=='BTCUSD': ppr(s)
            
        v= PDF([(s['id'], s['stats']['trades'])  for s in j['datasets']['symbols']], columns=['x', 'nTr'])
        v.svde('nTr')
        
        # {'availableSince': '2019-08-30T00:00:00.000Z',
        #  'availableTo': '2023-08-08T00:00:00.000Z',
        #  'dataTypes': ['trades',
        #                'incremental_book_L2',
        #                'quotes',
        #                'book_snapshot_5',
        #                'book_snapshot_25'],
        #  'id': 'BTCUSD',
        #  'stats': {'bookChanges': 1268191882, 'trades': 30,225,749},
        #  'type': 'spot'}
        
    
    
        import requests
    
        url = "https://bravenewcoin.p.rapidapi.com/ohlcv"
        
        querystring = {"size":"10"}
        
        headers = {
        	"Authorization": "Bearer <append token here>",
        	"X-RapidAPI-Key": "a8ffbdfcf3msh0d28330e775a632p1d1d13jsn4cc46b3f69dc",
        	"X-RapidAPI-Host": "bravenewcoin.p.rapidapi.com"
        }
        
        response = requests.get(url, headers=headers, params=querystring)
        
        print(response.json())
  
    
  
os.getcwdb()    
os.chdir(r'C:\z\work\Crypto') 
 
class Pyppeteer:
    
    
    async def gmail(page):
        # browser = await puppeteer.launch({ headless: false})
        # page = await browser.newPage()        
        navigationPromise = page.waitForNavigation()
        
        if 1:
            await page.goto('https://accounts.google.com/')        
            #await navigationPromise        
            await pidu(page, 'out/page_gmail.pck')  # siz=0
            await page.waitForSelector('input[type="email"]')
            
        else: 
            page= pilo('out/page_gmail.pck')  # EOFError: Ran out of input
        
        await page.click('input[type="email"]')        
        #await navigationPromise       
        await page.waitFor(3500);        

        # TODO : change to your email 
        await page.type('input[type="email"]', 'alex.zolot@gmail.com')
        
        await page.waitForSelector('#identifierNext')
        await page.click('#identifierNext')
        
        await page.waitFor(500);        
        await page.waitForSelector('input[type="password"]')
        await page.click('input[type="password"]')
        await page.waitFor(500);
        
        # TODO : change to your password
        await page.type('input[type="password"]', '13Zzmm..g')
        
        await page.waitForSelector('#passwordNext')
        await page.click('#passwordNext')        
        #await navigationPromise
    
    
    
    def test1():

        
#### def main()
        async def main():
           browserObj =await launch({"headless": False})
           page = await browserObj.newPage()
           await page.setViewport({ 'width': 900, 'height': 700 })
           #await url.goto('https://scrapeme.live/shop/')
           if 0:
               await page.goto('https://facebook.com')
               #await page.waitForNavigation()
               await page.type('[ id = email ]', 'alex.zolot@gmail.com')
               await page.type('[ id = pass ]', '11Zzmm..f')
               await page.click('[type = submit]')
               
           await page.goto('https://www.curve.bet')
           await page.waitForNavigation()

           time.sleep(3)


           await page.type('input', 'alex.zolot@gmail.com')
           await page.type('[type = password]', '11Zzmm..c')
           
           # body > uni-app > uni-page > uni-page-wrapper > uni-page-body > uni-view > uni-view.white-warp > uni-view > uni-view.button
           # /html/body/uni-app/uni-page/uni-page-wrapper/uni-page-body/uni-view/uni-view[3]/uni-view/uni-view[3]
           #     <uni-view data-v-79c6b73d="" class="button">Log in</uni-view>
           time.sleep(3)

           await page.click('uni-view [ class = button ]') 
           gc.collect()          
           
           time.sleep(3)
               
           await page.goto('https://www.curve.bet/#/pages/k/index?id=1&currency=BTC')
           
           col={'up':'green', 'down':'red'}
           def btn_ud(act): return f'uni-view [ class*= {col[act]}_box ]'
           inp_amt= '.uni-input-input[type = "number"]'
           nn={'up':0, 'down':0}
           
           k= .3
           
           for i in range(1000):                
               amt, act, term= pilo('out/aat.pck')
               iterm= int(term / 60)
               pr(B,  f'{i=},  {tnow()},  \n{amt, act, term=},  \n{nn=}')
               
               if 0:
                   amt, act, term= 500, 'up', 60
                   amt, act, term= 1500, 'down', 120
                   amt= 510 if term==60 else 1200
                   pr(G, f'{amt, act, term=}')        
                   
               amts= 'uni-view[class=trade_list_class]'
               await page.waitForSelector(amts);
               val= await (await (await page.querySelector(amts)).getProperty("textContent")).jsonValue()
               pr(B, 'val=', val)
               try:
                    val2= float(val[1:-4])
                    amt= max(amt, round(k*val2, 2))
                    pr(B, 'val2=', val2, amt)
                    #if val2 < 500: break
               except Exception as e: 
                    pr(R, e)
                    pass      


               htmlContent = await page.content()
               pidu(htmlContent, 'out/htmlContent.pck')
               if it >12: break
    
               
               if act in ['up', 'down']:
                   nn[act] += 1
                   await page.waitForSelector(btn_ud(ac));
                   await page.click(btn_ud(act)) # up
                   
                   # click term
                   await page.waitForSelector('uni-view [class = "seconds"]')
                   terms= await page.querySelectorAll('uni-view [class = "seconds"]')
                   await terms[iterm].click()
                   
                   # get val, calc and input  amt                
                   await page.waitForSelector(inp_amt)
                   
                   val= float((await (await (await page.querySelector('uni-view [class = "leftmoney"]')).getProperty("textContent")).jsonValue()).split(':')[1])
                   pr(B, act, G, val, B, amt, term)
                   
                   await page.type(inp_amt, f'{amt}')                   
    
                   
                   await page.waitForSelector('input  [enterkeyhint = "done"]')    
                   #await page.click('input  [enterkeyhint = "done"]')
                   
                   pidu((amt, 'wait', term), 'out/aat.pck')
                   
               await page.waitFor(200)
           
           # <uni-view data-v-6603d1ad="" class="seconds">60<span class="s">s</span></uni-view>
           # <uni-view data-v-6603d1ad="" class="seconds">120<span class="s">s</span></uni-view>
           


           time.sleep(2)
           await page.screenshot({'path' : 'out/curve1.png'})
           
           # htmlContent = await page.content()           
           # titles = await page.querySelectorAll("h2")        
           # await page.waitFor(240000)
           await browserObj.close()
           
           # for t in titles:
           #     pr(t)
           #return htmlContent
       
        

        
        asyncio.get_event_loop().run_until_complete(main())
        
        u= pilo('out/htmlContent.pck')
        u
        print(u.encoding)
        u.write('out/zz.html')
        open('out/zz.html','w', encoding="utf-8").write(u)
        open('out/zz16.html','w', encoding="utf-16").write(u)
        
    # import schedule 
    # help(schedule.every)
    # schedule.every()
    
    async def hist():
           browserObj =await launch({"headless": False})
           page = await browserObj.newPage()
           await page.setViewport({ 'width': 900, 'height': 700 })

           for i in range(200):    
               await page.goto('https://api.cuvrefier.com/pub/getHistory?id=1&type=1min')
               
               hi= await page.querySelector('pre')
               hi= await (await hi.getProperty("textContent")).jsonValue()
               u= eval(hi)
               u= PDF(u['data'][-30:]) #.tail(30)
               u['date']= u.ts.apply(dttime.utcfromtimestamp)
               #u.tab(nr=50)
               t= tnow()
               pr(i, f'{t=},  {u.date.values[-1]=}')
               await page.waitFor(100)
    
           return u
       
    async def await_to_sec(page, t= 9):
        sleep= (119.7 + t - tnow().second ) % 60
        while sleep > 4:
            pr(Y, f'sleeping {sleep} sec')
            time.sleep(sleep-2)
            sleep= (119.7 + t - tnow().second ) % 60
            
        await page.waitFor(1000* sleep)
        pr(tnow(), ' - end of async sleep')   
        
        
    def hist_get(n=0, toSave=False):    
        import requests, json
        # hi['date']= pd.to_datetime(hi.ts, unit='s')
        df= json.loads(requests.get( "https://api.cuvrefier.com/pub/getHistory?id=1&type=1min").text)['data']
        dt= pd.to_datetime(df[-1]['ts'], unit='s')
        if toSave:
            pidu(df, f'out/BTC-hist-{dt}'[:-3].replace(':','-').replace(' ','_') + '.pck')
        pr(B, f'\n{tnow()}  hist_get(): got hist data, last time = \n',  dt)
        if n>0: df= df[-n:]
        return  PDF(df)
    # hist_get(toSave=True);
    
    async def hist_get_coinbase(n=10):  
         browserObj =await launch({"headless": False})
         page = await browserObj.newPage()          
         page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0')
         await page.setViewport({ 'width': 900, 'height': 700 })
         await page.goto('https://www.coinbase.com')
         #await page.goto('https://www.coinbase.com/login')
         await page.goto('https://login.coinbase.com/signin')
         cookie= '''cb_dm=0c73074b-1eed-4d39-bdcd-2ba51a21a21f; 
         oauth2_consent_csrf=MTY5MjA4NzQ2M3xEdi1IQkFFQ180Z0FBUkFCRUFBQVB2LUlBQUVHYzNSeWFXNW5EQVlBQkdOemNtWUdjM1J5YVc1bkRDSUFJREpoWTJObU9UY3pNMlZqTnpRM1lXWTROV05tTURCbFlXUmhOV001T1RVMnz5qBNMPnwlPCMhA3ua9-6fLo1SdbFaum9VCnK1Wd80yg==;
         coinbase_device_id=0d6e3f98-ffdd-4b24-a0ab-9c57607f220c; 
         amplitude_device_id=0d6e3f98-ffdd-4b24-a0ab-9c57607f220c; 
         __cf_bm=N1WIa8cAIupnZ.6dIij6zqSL4PoSgAzARtxA8r3rKkA-1692129941-0-AYzesob2NY/XbzzkX/pacaCYbgj24WRvC2MQ3A5G6DSqJespBU6rGE7Kya7oWKTOZ20jfoiulOQHVzG2ADPGzYA=; 
         advertising_sharing_allowed={%22value%22:true}; _ga=GA1.2.201192285.1692129947; 
         _gid=GA1.2.1656608313.1692129947; _gat_UA-32804181-23=1; 
         _dpm_ses.268f=*; 
         _dpm_id.268f=45c889ab-c45b-4979-a170-43ef0dbe547d.1692129947.1.1692129947.1692129947.2f7166d0-5357-4f85-baf5-bf3cd5934082; 
         _fbp=fb.1.1692129947374.2023260529; _ga_W5Z1BRK56L=GS1.2.1692129947.1.0.1692129947.0.0.0; df2=69e1140a5d25f8f53fb8c9bbe054dd1d; 
         oauth2_authentication_csrf=MTY5MjEyOTk1MXxEdi1GQkFFQ180WUFBUkFCRUFBQVB2LUdBQUVHYzNSeWFXNW5EQVlBQkdOemNtWUdjM1J5YVc1bkRDSUFJR0ZtWWpFeE9EUXpPREUzWVRSaU16QmhZbVl4TmpCa00yVmxNRE5sT0RrM3wQn8NeWOxIFQ4DGPfWUwZHcHnecEsrAP3cj5p9HagNFg==;
         login-session=MTY5MjEyOTk1MXxlaU1wcFVVX2F0VVZZMzlwS2FYNjZldzREaVNKOWJzSmtLendyNHFfNkJaR2lGQ0dkRkpvOGw3eGhvbWpxQjR2OGlDSnpib1dTVnZNRUcyUmFjYVRSbHJXemVUR3lXaVltTlRCVGtSLW5qbU8xWWROOW82XzljYlFVX1QtU095ZHJvWk1MQ1p1RjF4STFoc1g1cEpjU0V1T3B4dnJEMGN4VkhHYnFFMjZCWXZrNFRjLTNqckktVXdROFFRdzFYNXpvRHF1VkNRdG5kVXAwbV95NDhScmNRakxFNGNsWHJDdndwQWkyb2tIMnV2RV9saWRaTWtqVTJ6bl9KMjlkc2E5VE56bEt5eDVrYXNTSVJUNUxmQ25neVRxVFNKemJyckpYcHZPZ0drTndVVFVLTEJMWE8yOUR0YlE4d1owUzctcWlJR2UxU0Vwc1dwZTRKSTRON1BoeWpYT0dNMDg1bXVGNlJ3Qy1zOUVOV3dJTUh4WUhid2F0RXFCQU9BaHA0Rml0ajdQNHRmclpFTm9mNmVXQl9xWURzdTVGUWM2dGRjY3VXRXhsOTQxUElEVzVfRERXczR6N2Y2bExwV0R3SXFGUXcwdVQ1Z3RkZz09fE_lCTVcoyRpQ81JdByWD1m-oTq75guRXBvHb3WLJdI1; 
         _ga_90YJL6R0KZ=GS1.1.1692129947.1.0.1692129951.56.0.0'''.replace('\n','').replace(' ','').replace(' ','')#.split(';')
         cookie= '{ "' + cookie.replace(';', '", \n\n"') .replace('==;', 'yyyy')  .replace('=;', 'zzzz') .replace('=', '":"').replace('zzzz','=;' ) .replace('yyyy', '==;') +'" }'
         cookie= eval('{ "' + cookie.replace(';', '", "') .replace('=', '":"') +'" }')
         cookie= eval('{ """' + cookie.replace(';', '""", \n"""') .replace('=', '""":"""') +'""" }')
         pr(cookie)
         cookie= eval(cookie)
         await page.setCookie(*cookie)
         
        # https://login.coinbase.com/signin
         await page.waitForSelector('[type = submit]');
         await page.type('input[area-label = Email]', 'alex.zolot@gmail.com')
         await page.click('button [ type = submit]') 
         await page.type('input[area-label = Password]', '11Zzmm..c')
         await page.click('button') 


         
         #await page.waitForNavigation()

         #time.sleep(1)
         await page.waitFor(1000)
         
         await page.waitForSelector('[type = password]');
         await page.type('input', 'alex.zolot@gmail.com')
         await page.type('[type = password]', '11Zzmm..c')
         
         #time.sleep(1)
         await page.waitFor(1000)
        
        
         if 0:
            import requests
            from requests.auth import HTTPBasicAuth
            url= f'https://www.coinbase.com/api/v3/brokerage/stream/products/BTC-USDT/candles?granularity=ONE_MINUTE&limit={n}&update_interval=ONE_SECOND_CANDLE'
            
            prk='MHcCAQEEIPsNe0dr1WGYbWi8o0QQ/3y7YRPtW5Sra+ZxuSjztrbkoAoGCCqGSM49\nAwEHoUQDQgAEklLNyIfDicfx3a+iQ7DVm1qtsMrwAJFa3XKWaipFeRBaQagJfsKZ\nJIsPNwQJ/d05VG0UdULOGMLZlT7WHmsJfQ=='
            prk='-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIPsNe0dr1WGYbWi8o0QQ/3y7YRPtW5Sra+ZxuSjztrbkoAoGCCqGSM49\nAwEHoUQDQgAEklLNyIfDicfx3a+iQ7DVm1qtsMrwAJFa3XKWaipFeRBaQagJfsKZ\nJIsPNwQJ/d05VG0UdULOGMLZlT7WHmsJfQ==\n-----END EC PRIVATE KEY-----\n'
            url= f'https://www.coinbase.com/api/v3/brokerage/stream/products/BTC-USDT/candles?granularity=ONE_MINUTE&limit={n}&update_interval=ONE_SECOND_CANDLE&privateKey={prk}'
    
            u= requests.get(url)  #, auth=HTTPBasicAuth('alex.zolot@gmail.com', '11Zzmm..c'))
            u.text # Out[704]: 'Unauthorized\n'
            u.status_code # 401
            
            import http.client
            import json
            
            conn = http.client.HTTPSConnection("api.coinbase.com")
            payload = ''
            headers = {
              'Content-Type': 'application/json',
               "name": "organizations/8e7c799e-cfe6-4a31-be8d-fc04be1c468b/apiKeys/57e2d7bc-2dc7-4e83-a5f9-23ad8db821ff",
               "principal": "38b6bcd5-19ed-5a57-9536-d6e9362b16ee",
               "principalType": "USER",
               "publicKey": "MFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEklLNyIfDicfx3a+iQ7DVm1qtsMrw\nAJFa3XKWaipFeRBaQagJfsKZJIsPNwQJ/d05VG0UdULOGMLZlT7WHmsJfQ",
               "privateKey": "MHcCAQEEIPsNe0dr1WGYbWi8o0QQ/3y7YRPtW5Sra+ZxuSjztrbkoAoGCCqGSM49\nAwEHoUQDQgAEklLNyIfDicfx3a+iQ7DVm1qtsMrwAJFa3XKWaipFeRBaQagJfsKZ\nJIsPNwQJ/d05VG0UdULOGMLZlT7WHmsJfQ",
              # "createTime": "2023-08-15T00:47:54.329600984Z",
               "projectId": "00000000-0000-0000-0000-000000000000"
            }
            conn.request("GET", "/api/v3/brokerage/products/BTC-USDT/candles?granularity=ONE_MINUTE", payload, headers)
            res = conn.getresponse()
            data = res.read()
            print(data.decode("utf-8"))
            
            '''14jfma@
            Save your API key somewhere safe. 
            Your public key can be used for verification purposes. 
            Your private key is used to sign requests and should not be shared with 3rd parties.
            
            {
               "name": "organizations/8e7c799e-cfe6-4a31-be8d-fc04be1c468b/apiKeys/57e2d7bc-2dc7-4e83-a5f9-23ad8db821ff",
               "principal": "38b6bcd5-19ed-5a57-9536-d6e9362b16ee",
               "principalType": "USER",
               "publicKey": "-----BEGIN EC PUBLIC KEY-----\nMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEklLNyIfDicfx3a+iQ7DVm1qtsMrw\nAJFa3XKWaipFeRBaQagJfsKZJIsPNwQJ/d05VG0UdULOGMLZlT7WHmsJfQ==\n-----END EC PUBLIC KEY-----\n",
               "privateKey": "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIPsNe0dr1WGYbWi8o0QQ/3y7YRPtW5Sra+ZxuSjztrbkoAoGCCqGSM49\nAwEHoUQDQgAEklLNyIfDicfx3a+iQ7DVm1qtsMrwAJFa3XKWaipFeRBaQagJfsKZ\nJIsPNwQJ/d05VG0UdULOGMLZlT7WHmsJfQ==\n-----END EC PRIVATE KEY-----\n",
               "createTime": "2023-08-15T00:46:54.329600984Z",
               "projectId": "00000000-0000-0000-0000-000000000000"
            }
            '''
    
    if 0: # zzzz
        t1(0)
        hi= hist_get(0)  
        t1('hist data downloaded')
        df, mima= add_columns_btc(hi.rename(columns={'vol':'volume'}), rmNA=False)
        t1('add_columns_btc done')        
        df= add_talib(df, rmNA=False)
        t1('add_talib done');
        
        df.iloc[-1:].tab(nr=222)
        np.isnan(df).sum()
        np.isinf(df).sum()
        np.isnan(df.head(50)).sum()
        df.loc[:, np.isnan(df.tail(3)).sum() > 0].tab()
        (np.isnan(df.tail(3)).sum() > 0).index
        df.columns[np.isnan(df.tail(3)).sum() > 0]
        df.columns[np.isnan(df.head(50)).sum() > 35]  # 0
        
        df1= df[35:]
        df1.columns[np.isnan(df1.tail(3)).sum() > 0]
        df1.columns[np.isnan(df1.tail(3)).sum() == 0]
        x='ta_volume_adi'
        cx= sorted(list(set([x for x in df1.columns 
                             if sum(sum(np.isnan(df1[x]))) == 0 and x[0] != 'y'])))
        cy= [x for x in df1.columns if  x[0] == 'y']
        sum(sum(np.isnan(df1[cx])))  # 0
        sum(np.isnan(df1[cy]))
        # y_1     1
        # y_2     2
        # y_3     3
        # yq_1    0
        # yq_2    0
        # yq_3    0
        
        u2= rcsv('in/07-28 frd_crypto_sample/BTC_1min_sample.csv')
        
        
        #### read BTC hist json from curve
        h1= PDF(json.loads(open('in/getHistory-1.json').read())['data'])
        h2= PDF(json.loads(open('in/getHistory-2.json').read())['data'])
        h3= PDF(json.loads(open('in/getHistory-3.json').read())['data'])
        u= h4= PDF(json.loads(open('in/getHistory-4.json').read())['data'])
        u= h5= PDF(json.loads(open('in/getHistory-5.json').read())['data'])
        u= h6= PDF(json.loads(open('in/getHistory-6.json').read())['data'])
        h7= PDF(json.loads(open('in/getHistory-7.json').read())['data'])
        h8= PDF(json.loads(open('in/getHistory-8.json').read())['data'])
        h8e= PDF(json.loads(open('in/getHistory-ETH-8.json').read())['data'])
        h9= PDF(json.loads(open('in/getHistory-9.json').read())['data'])
        h10= PDF(json.loads(open('in/getHistory-10.json').read())['data'])
        h11= PDF(json.loads(open('in/getHistory-11.json').read())['data'])
        h12= PDF(json.loads(open('in/getHistory-12.json').read())['data'])
        h13= PDF(json.loads(open('in/getHistory-13.json').read())['data'])
        u['date']= pd.to_datetime(u.ts, unit='s')
        f'{u.date.values[0]},  {u.date.values[-1]}'  #':%Y-%m-%d %H:%M:%S%z}' #' H:M:S}'
        # 2023-08-10T17:28:00.000000000,  2023-08-11T01:47:00.000000000'
        # 2023-08-11T07:56:00.000000000,  2023-08-11T16:15:00.000000000'
        # 2023-08-11T16:32:00.000000000,  2023-08-12T00:51:00.000000000'
        # 2023-08-13T10:11:00.000000000,  2023-08-13T18:30:00.000000000
     
        def calk_yx(h1): # zzzz
            df1, mima= add_columns_btc(h1.rename(columns={'vol':'volume'}), rmNA=False, toScale=False)
            df1= add_talib(df1, rmNA=False, cx_not_imp=cx_not_imp)
            return df1.iloc[35:-3] #[cy+cx]
        
        yx1= calk_yx(h1) # yx1.tab(nr=20);
        yx2= calk_yx(h2)
        yx3= calk_yx(h3)
        yx4= calk_yx(h4)
        yx5= calk_yx(h5)
        yx6= calk_yx(h6) #.drop('win', **a1))  # yx6.tab(nr=1);
        yx7= calk_yx(h7)#.drop('win', **a1))  # yx6.tab(nr=1); yx7.win,  yx1.win ,  yx2.win ,  yx3.win ,  yx4.win ,  yx5.win ,  yx6.win  yx.win
        yx8= calk_yx(h8)#.drop('win', **a1))  # yx6.tab(nr=1); yx7.win,  yx1.win ,  yx2.win ,  yx3.win ,  yx4.win ,  yx5.win ,  yx6.win  yx.win
        yx8e= calk_yx(h8e)#.drop('win', **a1))  # yx6.tab(nr=1); yx7.win,  yx1.win ,  yx2.win ,  yx3.win ,  yx4.win ,  yx5.win ,  yx6.win  yx.win
        yx9= calk_yx(h9)#.drop('win', **a1))  # yx6.tab(nr=1); yx7.win,  yx1.win ,  yx2.win ,  yx3.win ,  yx4.win ,  yx5.win ,  yx6.win  yx.win
        yx10= calk_yx(h10)#.drop('win', **a1))  # yx6.tab(nr=1); yx7.win,  yx1.win ,  yx2.win ,  yx3.win ,  yx4.win ,  yx5.win ,  yx6.win  yx.win
        yx11= calk_yx(h11)#.drop('win', **a1))  # yx6.tab(nr=1); yx7.win,  yx1.win ,  yx2.win ,  yx3.win ,  yx4.win ,  yx5.win ,  yx6.win  yx.win
        yx12= calk_yx(h12)#.drop('win', **a1))  # yx6.tab(nr=1); yx7.win,  yx1.win ,  yx2.win ,  yx3.win ,  yx4.win ,  yx5.win ,  yx6.win  yx.win
        yx13= calk_yx(h12)#.drop('win', **a1))  # yx6.tab(nr=1); yx7.win,  yx1.win ,  yx2.win ,  yx3.win ,  yx4.win ,  yx5.win ,  yx6.win  yx.win
        
        plt.imshow(yx1);
        plt.imshow(np.isnan(yx1));
        plt.imshow(np.isnan(yx2));
        plt.imshow(np.isnan(yx3));
        plt.imshow(np.isnan(yx4));
        plt.imshow(yx4);
        plt.imshow(yx5[cx_good]);
        plt.imshow(yx6);
        plt.imshow(yx6[cx_good]);
        
        yx= conc([yx1, yx2, yx3, yx4, yx5, yx6, yx7]).dropna()
        nn= sum(np.isnan(yx))  # 0
        cx_nn=  yx.columns.astype(str)[nn >20] # 0
        cx_not_imp= []
        cx_good=[x for x in yx.columns if x[0] !='y' and x not in list(cx_nn) + cx_not_imp]; len(cx_good) # 258  261
        
        plt.imshow(yx);
        plt.imshow(yx[cx_nn]);
        plt.imshow(np.isnan(yx[cx_nn]));
        yx[cx_nn].tab(nr=50);
        yx[['volume','ta_volume_em', 'ta_volume_sma_em']].tab(nr=50);
        yx.dropna().shape  #  (1694, 291)  (764, 291)
        PDF(sum(np.isnan(yx))> 80).tab(nr=50);
        PDF(yx.loc[:, nn> 80]).tab(nr=50);
        
        np.quantile(yx.y_1, [.25, .75])  # [-0.52 ,  0.524]

        
        sum(sum(np.isnan(yx1)))  # 0
        sum(sum(np.isnan(yx1[cx])))  # 0
        nn= sum(np.isnan(yx3))  # 0
        cx_nn=  yx3.columns[nn >0] # 0
        #cx_good=[x for x in cx if x not in list(cx_nn) + cx_not_imp]; len(cx_good) # 166
        
        nnn= yx3.loc[:, nn >0] # 0
        
        sum(sum(np.isnan(yx1[cx_good])))  # 0
        sum(sum(np.isnan(yx2[cx_good])))  # 0
        sum(sum(np.isnan(yx3[cx_good])))  # 0
        sum(sum(np.isnan(yx4[cx_good])))  # 0
        sum(sum(np.isnan(yx5[cx_good])))  # 6
        sum(sum(np.isnan(yx6[cx_good])))  # 0
   
        
        pd.set_option('display.max_rows', 600)
        pd.set_option('display.max_rows', 20)
        np.isnan(nnn).sum(**a1) 
        nnn.iloc[138:142].tab();
        nnn[138:142].tab();
        
        sum(sum(np.isnan(yx2)))  # 0
        sum(sum(np.isnan(yx3)))  # 0
        sum(np.isnan(yx3))  # 0
        yx3.loc[:, np.isnan(yx3).sum() > 0].tab();
        
        #### train GBM models
        ii= random.choice(range(len(yx)), size= len(yx)-500, replace=False)
        #tr= conc([yx1, yx2, yx3, yx4[:100]] )[cy + cx_good]
        tr= yx.iloc[ii][cy + cx_good]
        te= yx.iloc[[i for i in range(len(yx)) if i not in ii]][cy + cx_good]
        te= yx11[cy + cx_good]
        
        models= {}
        sbm= {'n_estimators': 852,
         'num_leaves': 221,
         'max_depth': 100,
         'learning_rate': 0.0286,
         'min_split_gain': 0.0247,
         'feature_fraction': 0.300,
         'bagging_freq': 0}
        
        for  y in ['yq_1','yq_2','yq_3']: 
            pr(y)
            m= lgb.LGBMClassifier(objective="multiclass", **sbm) 
            m.fit(tr[cx_good], tr[y])
            models[y]= m
            
        for  y in ['y_1', 'y_2', 'y_3']: 
            pr(y)
            m= lgb.LGBMRegressor( **sbm) 
            m.fit(tr[cx_good], tr[y])
            models[y]= m
            
        pidu(models, 'out/models/m_gbm_3q.pck')   
        models= pilo('out/models/m_gbm_3q.pck')   #zzzz
        models.keys()
        
        #### validate GBM models
        def validate():
            a= PDFd(**{y + 'h': yh[:,2] - yh[:,0] for y in ['yq_1','yq_2','yq_3'] if (yh:= models[y].predict_proba(te[cx]))[0][0] > -99 }) 
            b= PDFd(**{y + 'h': models[y].predict(te[cx]) for y in ['y_1','y_2','y_3'] }) 
            yhh= conc([te[cy].ri(), a, b], **a1)
            return yhh
            
        yhh .tab();
        
        p= figu(title='Keras pred yq_1', x_axis_label='yq_1h', y_axis_label='yq_2h', width=800, height=600)
        p.scatter(x=yhh.yq_1h, y=yhh.yq_2h, color=yhh.yq_1.map({-1:'red', 0:'grey',1:'green'}))  #,  line_color= 'red',    legend_label='training' ) # axes
        show(p)

         
        p= figu(title='Keras pred yq_2', x_axis_label='yq_2h', y_axis_label='yq_3h', width=800, height=600)
        p.scatter(x=yhh.yq_2h, y=yhh.yq_3h, color=yhh.yq_2.map({-1:'red', 0:'grey',1:'green'}))  #,  line_color= 'red',    legend_label='training' ) # axes
        show(p) 

        thr= .93
        yhh['yq_1hq']= w(yhh.yq_1h < -thr, 'd', w(yhh.yq_1h > thr, 'up', 'hold'))
        PDF(yhh[['yq_1', 'yq_1hq']].vc()).ri().pivot( index='yq_1', columns='yq_1hq').fillna(0).astype(int)
            # .8
            # yq_1hq  d hold up
            # yq_1             
            # -1      6   65  0
            #  0      4  204  9
            #  1      2   67  5
            
            #  .85
            #  yq_1hq  d hold up
            #  yq_1             
            #  -1      4   67  0
            #   0      2  207  8
            #   1      1   69  4
         
        def bokeh_win(y='yq_1', yh= 'yq_1h'):   
             k= {'1':.20, '2':.30, '3':.50}[y.split('_')[1][0]]
             u= yhh[[y, yh]].svde(yh) 
             win_u= k * u[y].cumsum() - (u[y] != 1).cumsum()
             p= figu() #title='Keras Learning Curves', x_axis_label='epoch', y_axis_label='Loss', width=900, height=600)
             p.line(x= u[yh][:79], y=win_u[:79],    legend_label='up')  #,  line_color= 'red',    legend_label='training' ) # axes
             #show(p) 
             
             us= yhh[[y, yh]].sv(yh) 
             win_d= k * (us[y]==-1).cumsum() - (us[y] !=-1).cumsum()
             
             h= -us[yh] if yh[1]=='q' else 1-us[yh]  
             p.line(x= h[:79], y=win_d[:79], color='red',    legend_label='down')  #,  line_color= 'red',    legend_label='training' ) # axes
             p.legend.location = "bottom_right"
             show(p) 
        
        yhh.tab();
        yhh['yqh']= .7*yhh.yq_1h + .2*  yhh.yq_2h + .1*  yhh.yq_3h   # worst case yqh > .91  win 17, lost 2 = +1.4
        bokeh_win(y='yq_1', yh= 'yqh')
        
        thr= .91
        yhh['yqh_q']= w(yhh.yqh < -thr, 'd', w(yhh.yqh > thr, 'up', 'hold'))
        PDF(yhh[['yq_1', 'yqh_q']].vc()).ri().pivot( 'yq_1', 'yqh_q').fillna(0).astype(int)
          # yqh_q  d hold  up
          # yq_1             
          # -1     6  169   1
          #  0     0  151   1
          #  1     0  162  10
          
              # yqh_q  d hold  up
              # yq_1             
              # -1     6  170   0
              #  0     0  150   2
              #  1     0  161  11
        
        bokeh_win(y='yq_1', yh= 'yq_1h')
        
        bokeh_win(y='yq_1', yh= 'yq_2h')
        bokeh_win(y='yq_1', yh= 'y_1h')
        
        bokeh_win(y='yq_2', yh= 'yq_2h')
        thr= .92
        yhh['yqh']= .7*yhh.yq_1h + .2*  yhh.yq_2h + .3*  yhh.yq_3h   # worst case yqh > .91  win 17, lost 2 = +1.4
        yhh['yqh_q']= w(yhh.yqh < -thr, 'd', w(yhh.yqh > thr, 'up', 'hold'))
        wm= PDF(yhh[['yq_1', 'yqh_q']].vc()).ri().pivot(index= 'yq_1', columns='yqh_q').fillna(0).astype(int)
        wm
            # yqh_q   d hold  up
            # yq_1              
            # -1     48  155  32
            #  0      6   70   4
            #  1      8  106  71
        w= npa(wm)    
        win= 1.2*(w[0,0] + w[2,2])  - sum(w[:,0]) - sum(w[:,2]) 
            
        yhh['yqh']= .1*yhh.yq_1h + .7*  yhh.yq_2h + .2*  yhh.yq_3h   # worst case yqh > .9  win 36, lost 10= +.8
        bokeh_win(y='yq_2', yh= 'yqh')
        thr= .96  # +36 -10 = 0.8
        yhh['yqh_q']= w(yhh.yqh < -thr, 'd', w(yhh.yqh > thr, 'up', 'hold'))
        wm= PDF(yhh[['yq_2', 'yqh_q']].vc()).ri().pivot( index='yq_2', columns= 'yqh_q').fillna(0).astype(int)
        wm
         # yqh_q   d hold  up
         # yq_2              
         # -1     15  180   5
         #  0      2   91   2
         #  1      1  183  21   
        
        bokeh_win(y='yq_3', yh= 'yq_3h')
        bokeh_win(y='yq_2', yh= 'yq_3h')
        
        def win_1f(pp, k=.2):
            w= np.where
            thr, a,b,c= npa(pp) / pp[0]
            u= yhh.copy()
            u['yqh']= a *u.yq_1h + b *  u.yq_2h + c *  u.yq_3h 
            u['yqh_q']= w(u.yqh < -thr, 'd', w(u.yqh > thr, 'up', 'hold'))
            wd= PDF(u[['yq_1', 'yqh_q']].vc()).ri().pivot( index='yq_1', columns= 'yqh_q').fillna(0).astype(int)
            #wd.tab('wd')
            v= npa(wd)
            pr(f'{wd=}')
            pr(f'{v=}')
            if v.shape[1] !=3: return 4e4
            win= k * (v[0,0] + v[2,2]) - v[0,2]- v[1,2] -v[1,0] -v[2,0]
            pr(f'{win=} after {sum(v[:,0])+ sum(v[:,2])} trials from {sum(v)}')
            return -win

        win_1f((.9, .7, .2,.1))
                # yqh_q   d hold  up
                # yq_1              
                # -1     39  134   2
                #  0      0  146   1
                #  1      2  130  46
        
                # win=12.0
        
        import optuna
        def objective(trial):
            thr = trial.suggest_float("thr", 0, 1)
            a = trial.suggest_float("a", 0, 1)
            b = trial.suggest_float("b", 0, 1)
            c = trial.suggest_float("c", 0, 1)
            return win_1f((thr, a,b,c), k=.2)
        
        study = optuna.create_study()
        study.optimize(objective, n_trials=999)
        sbm= study.best_params
        -study.best_value, sbm  #dir(study)
        # (19.200000000000003,
        #  {'thr': 0.4683696435194862,
        #   'a': 0.10265065745918353,
        #   'b': 0.04257333980205964,
        #   'c': 0.3658354222510839})
        
        # ETH:
        # (19.0,
        #  {'thr': 0.9819720613171979,
        #   'a': 0.41567368235627944,
        #   'b': 0.2396090218976028,
        #   'c': 0.4118474132628479})
        win_1f((94, 20, 8, 72), k=.2)
        # yqh_q   d hold  up
        # yq_1              
        # -1     44  155   0
        #  0      0  138   0
        #  1      0  117  46
        # win=18.0 after 90 trials from 500
            
        win_1f((94, 20, 8, 72), k=.2)   # to try first 
                # yqh_q   d hold  up
                # yq_1              
                # -1     44  155   0
                #  0      0  138   0
                #  1      0  117  46
        
                # win=18.0 after 90 trials from 500
        
        win_1f((92, 20, 8, 72), k=.2)  
                # yqh_q   d hold  up
                # yq_1              
                # -1     49  150   0
                #  0      0  138   0
                #  1      1  110  52
             
                # win=19.2 after 102 trials from 500
        win_1f((46, 10, 4, 37), k=.2)  #  win_1f((68, 55, 17, .1), k=.2) 14
                # yqh_q   d hold  up
                # yq_1              
                # -1     51  148   0
                #  0      0  135   3
                #  1      2  106  55
        
                # win=16.2 after 111 trials from 500
        
        win_1f((.68, .55, .17, .001), k=.2)  #  win_1f((68, 55, 17, .1), k=.2) 14
                # yqh_q   d hold  up
                # yq_1              
                # -1     41  133   1
                #  0      0  146   1
                #  1      0  130  48
                # win=15.8 after 91 trials from 500
        
        #### GBM Importance xx
        imp_r, imp_q= models['y_1'].feature_importances_,  models['yq_1'].feature_importances_
        imp= PDFd(x=cx_good, imp= imp_r + imp_q)
        imp.svde('imp').head(40).tab('best', nr=50);
        imp.svde('imp').tail(40).tab('worst', nr=50);
        cx_most_imp= sorted(set([x.replace('_d1','').replace('_d2','') for x in imp.svde('imp').head(20).x ]))
        cx_not_imp= sorted(set([x for x in imp.svde('imp').tail(30).x if x[-3:] not in ['_d1', '_d2']]))
        
        import ta
        help(ta.trend.mass_index)  # window_fast=w_fast, window_slow=w_slow
        t= ta.trend.mass_index(high, low, window_fast=5, window_slow=10)
        t= ta.volatility.AverageTrueRange(high, low, close, window=10)
        t= ta.trend.adx(high, low, close, window=10)
        t= ta.trend.kst(close, roc1=5, roc2= 7, roc3= 10, roc4= 15, window1=4, window2=4, window3=4, window4=8, nsig=9)
        t= ta.trend.ichimoku_b(high, low, window2=8, window3=16)
        t= ta.volatility.keltner_channel_wband(high, low, window2=8, window3=16)
        help(ta.trend.ichimoku_b)  # window=w_        help(ta.volatility.AverageTrueRange)  # window=w_
        help(ta.volatility.AverageTrueRange)  # window=w_      
        help(ta.trend.adx) # window=w_
        help(ta.trend.kst) # roc1=10, roc2=15, roc3=20, roc4=30, window1=10, window2=10, window3=10, window4=15
        help(ta.volatility.keltner_channel_wband) # roc1=10, roc2=15, roc3=20, roc4=30, window1=10, window2=10, window3=10, window4=15
            # w_fast, w_slow, w_sign, w_, w_sm= 2, 6, 2, 4, 3   #  12, 26, 9, 20, 3
            # w1, w2, w3, w4= 2, 4, 6, 8  # 9, 26, 52

        if 0:
             u['win']= win_u + win_d
         
             p= figu() #title='Keras Learning Curves', x_axis_label='epoch', y_axis_label='Loss', width=900, height=600)
             p.line(x=range(99), y=u.win[:99])  #,  line_color= 'red',    legend_label='training' ) # axes
             show(p) 

            
    #async def hist1(page, model, n=50):
    async def hist1(page,  models, n=50):
        await Pyppeteer.await_to_sec(page, 10)  # update  https://api.cuvrefier.com/....
        t1(0)
        # await page.goto('https://api.cuvrefier.com/pub/getHistory?id=1&type=1min')            
        # hi= PDF(eval(await page.querySelectorEval('pre', 'e => e.textContent')
        #              )['data'])
        # #pr(G, f'{hi=}')
        # hi= hi.tail(n)
        # #pr(B, f'{hi=}')
        
        
        hi= Pyppeteer.hist_get(n=80)
        t1('hist data downloaded')

        #models3= pilo('out/m_3yq_1_taz_GBM.pck') 
        models3= models
        m1= models3['y_1']
        cxi, mima, sbm, cx_impo= m1.cxi, m1.mima,  m1.sbm, m1.cx_impo
        
        yxx= calc_yx(hi, mima= mima, cx2=cxi, opts=False, with_taz=True, 
                      n_cx=270, cx_impo=cx_impo)[0]
        
        #pr(Y, f'{cxi=}')
        te= yxx[list(cxi)].dropna()
        
        if len(te) >0: 
            te= rm_dup_col(te).iloc[-1:] 
           
            yhh= PDFd(**{y +'qh':  yh[:, 1] - yh[:, 0]  for y in ['y_1','y_2','y_3'] 
                               if (yh:= models3[y].predict_proba(te[cxi]))[0][0] > -99 }).iloc[-1]
            pr(G, f'\nin hist1():  {yhh=}  ')
          
            pp= {'thr': 93, 'a': 22, 'b': 74, 'c': 4} 
            #pp= {'thr': 40, 'a': 22, 'b': 74, 'c': 4} ## adj ??
            thr, a,b,c= [pp['thr'], pp['a'], pp['b'], pp['c']]        
            yh= a *yhh.y_1qh + b* yhh.y_2qh + c* yhh.y_3qh        
            pr(B, f'\nin hist1():  {yh=:.4f}  {thr=}  /n', W)
            act= 'down' if yh < -thr else 'up' if yh > thr else 'hold'
            
        else:
            pr(R,'len(te)==0')
            yh, act= -9999,  'hold'

        if 0:
            cx2, mima, o2, sbm= model.cx2, model.mima, model.opts, model.sbm
            
            # model= pilo('out/m_yq_1_GBM.pck')
            # cx2, mima, o2, sbm= sbm= model.cx2, model.mima, model.opts, model.sbm
    
            yxx= calc_yx(hi, mima= mima, cx2=cx2, opts=o2)[0] 
            te= rm_dup_col( yxx).iloc[-1:]      
            yhh= model.predict_proba(te[cx2])        
            pr(G, f'{yhh=} ')
    
            p= .92
            yh= (yhh[:, 1] -yhh[:, 0])[0]  #.yq_1h
            pr(B, f'\n{yh=}')
            act= 'down' if yh < -p else 'up' if yh > p else 'hold'

        if 0:
            df, mima= add_columns_btc(hi, rmNA=False, toScale=True)
                #hi.rename(columns={'date':'timestamp', 'vol':'volume'}), rmNA=False)
            cx_not_imp=['rule_k', 'ta_trend_aroon_down',
                 'ta_trend_aroon_ind', 'ta_trend_aroon_up', 'ta_trend_ema_slow', 'ta_trend_psar_down_indicator',  'ta_trend_psar_up_indicator',
                 'ta_trend_sma_slow', 'ta_volatility_bbm',  'ta_volatility_dcm',
                 'ta_volatility_kcc', 'ta_volatility_kchi',
                 'ta_volatility_kcli', 'ta_volume_vwap', 'us']
            df= add_talib(df, rmNA=False, cx_not_imp=cx_not_imp)[-1:]
            t1('add_columns done')
    
            yhh= {}
                
            for  y in ['yq_1','yq_2','yq_3']:
                yh= models[y].predict_proba(te[cx_good])  # models['yq_1']  y='yq_1'
                yhh[y + 'h']= yh[:,2] - yh[:,0]
                
            # (94, 20, 8, 72)    
            thr= .94
            qh= .2* yhh['yq_1h'] + .08*  yhh['yq_2h'] + .72*  yhh['yq_3h']   # worst case yqh > .91  win 17, lost 2 = +1.4
            act= w(qh < -thr, 'down', w(qh > thr, 'up', 'hold'))
       
            pr(Y, f'{yhh=}')    
            pr(Y, f'{act=}')    
        
        
        #act= 'up' if y1==1 and  y2==1 and y3==1 else 'down'  if y1== -1 and  y2== -1 and y3== -1 else 'hold'     
        #act= 'up' if y1==1  else 'down'  if y1== -1  else 'hold'     
        t1(f'Exit from hist with {act=} \n{yh=}')
        return hi, act
       

    async def hist():
            models= pilo('out/models/m_gbm3.pck')

            browserObj =await launch({"headless": False})
            page = await browserObj.newPage()
            await page.setViewport({ 'width': 900, 'height': 700 })
            
            for i in range(6100):
                t0= tnow()
                if t0.second >= 9:   # change value at ~ 8.6 sec
                    hi, act= await hist1(page, models)
                    pr(i, f'\n{t0=}, \n{tnow()=},  \n{hi.date.values[-1]=}, {act=}')
                    return act
                await page.waitFor(10)
        
    if 0:
        act= asyncio.get_event_loop().run_until_complete(hist()) 
        
        # hih= h4.assign(date= pd.to_datetime(h4.ts, unit='s'))
    
    
        hih.tab();
        # hih['timestamp']= hih.date
        # hih['volume']= hih.vol
        hih= hih.rename(columns={'date':'timestamp', 'vol':'volume'})
        df_te, mima= add_columns(hih, rmNA=False)
        
        df_te.tabb();
        '    '.join([str(x) for x in df_te.iloc[-1]])
        pd.set_option('display.max_rows', 20)
        nna= PDFd(nna= np.isnan(df_te).sum(), 
             nnaH=  np.isnan(df_te.head(50)).sum(),
             nnaT=  np.isnan(df_te.tail(50)).sum(),
             nnaL=  np.isnan(df_te.tail(1)).sum())
        xx= nna[nna.nnaL== 0].index
        df_te[xx].tab(nr=9);
        
        yq1= df_te.yqb_1 - df_te.yqs_1 
        yq2= df_te.yqb_2 - df_te.yqs_2 
        yq3= df_te.yqb_3 - df_te.yqs_3 
        
        df= conc([PDFd(yq1=yq1,  yq2=yq2,  yq3=yq3),  df_te[xx]], **a1)
        #df= conc([PDFd(yq1=yq1,  yq2=yq2,  yq3=yq3),  df_te], **a1)
        # [500 rows x 211 columns]
        df_tr= df.dropna()
        # [450 rows x 211 columns]
        
        tr,te= df_tr[:300], df_tr[300:]
        #te= df_te
    
        models= {}
        sbm= {'n_estimators': 852,
         'num_leaves': 221,
         'max_depth': 100,
         'learning_rate': 0.0286,
         'min_split_gain': 0.0247,
         'feature_fraction': 0.300,
         'bagging_freq': 0}
        
        for  y in ['yq1','yq2','yq3']: 
            pr(y)
            m= lgb.LGBMClassifier(objective="multiclass", **sbm) 
            m.fit(tr[xx], tr[y])
            models[y]= m
            
        pidu(models, 'out/models/m_gbm3.pck')   
        models= pilo('out/models/m_gbm3.pck')   
        models.keys()
    
        #### Confusion
        yhh= te[['yq1','yq2','yq3']].copy()
        
        for y in ['yq1','yq2','yq3']:
            #yh= models[y].predict_proba(te[xx])  # models['yq1'].__dict__.keys()  dir(models['yq1'].feature_name_)
            yh= models[y].predict_proba(te[[x for x in models[y].feature_name_ if x[0] !='y']])
            y= yh[:, 2] - yh[:, 0] 
            yhh[f'{y}_h']= w(y > .8, 1, w(y < - .8, -1, 0))
            # yh= models[y].predict(te[xx])
            # yhh[f'{y}_h']= yh
        
   
        yhh.vc()
        yhh[['yq1','yq2','yq3']].vc()
        yhh[['yq3','yq2']].vc()
        yhh[['yq1','yq1_h']].vc()
        yhh[['yq2','yq2_h']].vc()
        yhh[['yq3','yq3_h']].vc()
        
        import apscheduler
        from apscheduler.triggers.interval import IntervalTrigger
        help(apscheduler.triggers.interval) 
    
           
      
        from apscheduler.schedulers.asyncio import AsyncIOScheduler 
        scheduler = AsyncIOScheduler()
        tr= IntervalTrigger 
        scheduler.add_job(hist1, IntervalTrigger(seconds=60), [page],
                          id='az', second='57')
        scheduler.start()
        scheduler.remove_job('az')
        scheduler.shutdown()  
        
        
        u= eval(hih)
        u= PDF(u['data'])
        type()
        
        
        ts = int('1284101485')
        
        # if you encounter a "year is out of range" error the timestamp
        # may be in milliseconds, try `ts /= 1000` in that case
        print(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
        u['date']= u.ts.apply(dttime.utcfromtimestamp)
        u
    
    
    
    async def init_curve(page):  # www.curve.bet  

           await page.goto('https://www.curve.bet')
           #await page.waitForNavigation()

           #time.sleep(1)
           await page.waitFor(1000)
           
           await page.waitForSelector('[type = password]');
           await page.type('input', 'alex.zolot@gmail.com')
           await page.type('[type = password]', '11Zzmm..c')
           
           #time.sleep(1)
           await page.waitFor(1000)


           await page.click('uni-view [ class = button ]') 
           gc.collect()          
           
           #time.sleep(2)
           await page.waitFor(2000)

               
           await page.goto('https://www.curve.bet/#/pages/k/index?id=1&currency=BTC')
           
           model= pilo('out/m_yq_1_GBM.pck')
           cx2, mima, o2, sbm=  model.cx2, model.mima, model.opts, model.sbm
    
   
    async def trade_curve(page, act, amt, iterm):  # www.curve.bet
        col={'up':'green', 'down':'red'}
        def btn_ud(act): return f'uni-view [ class*= {col[act]}_box ]'
        inp_amt= '.uni-input-input[type = "number"]'
           
        val2= -99
        amts= 'uni-view[class=trade_list_class]'
        await page.waitForSelector(amts);
        val= await (await (await page.querySelector(amts)).getProperty("textContent")).jsonValue()
        pr(B, 'val=', val)
        try:
             val2= float(val[1:-4])
             amt= max(amt, round(k*val2, 2))
             pr(B, 'val2=', val2, amt, act)
             #if val2 < 500: break
        except Exception as e: 
             pr(R, e)
             pass     

        
        if act in ['up', 'down']:
            await page.waitForSelector(btn_ud(act));
            await page.click(btn_ud(act)) # up
            
            # click term
            await page.waitForSelector('uni-view [class = "seconds"]')
            terms= await page.querySelectorAll('uni-view [class = "seconds"]')
            await terms[iterm].click()
            
            # get val, calc and input  amt                
            await page.waitForSelector(inp_amt)
            
            val= float((await (await (await page.querySelector('uni-view [class = "leftmoney"]')).getProperty("textContent")).jsonValue()).split(':')[1])
            pr(B, act, G, val, B, amt, term)
            
            await page.type(inp_amt, f'{amt}')                   
 
            
            await page.waitForSelector('input  [enterkeyhint = "done"]')    
            #await page.click('input  [enterkeyhint = "done"]')    
    
        return val, val2
    
####   def curve      
    async def curve():  # www.curve.bet    
           # models= pilo('out/models/m_gbm3.pck')   # models.keys(); models['yq_1']
               
           # #if 0:   pidu(mima, 'out/mima.pck')
           # #if 0:   pidu(cx_good, 'out/cx_good.pck')
           # mima= pilo('out/mima.pck')           
           # cx_good= pilo('out/cx_good.pck')
           
           #model= pilo('out/m_yq_1_GBM.pck')
           models3= pilo('out/m_3yq_1_taz_GBM.pck') 


        
           chro= 'C:/Program Files/Google/Chrome/Application/chrome.exe'  # Google Chrome v. 115.0.5790.171 (Official Build) (64-bit)
           UsAg= 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0'
           
           browserObj =await launch({"headless": False},  
                                    args={'ignoreDefaultArgs': ['--enable-automation'], 
                                          'disable-blink-features':'AutomationControlled'}, executablePath= chro)
           #page_hist = await browserObj.newPage()
           # page_gmail= await browserObj.newPage() 
           # await page_gmail.setUserAgent(UsAg)
           # await gmail(page_gmail)
           
           # return
       
        
           page = await browserObj.newPage()     
           page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0')

           await page.setViewport({ 'width': 1100, 'height': 700 })
               
           await Pyppeteer.init_curve(page)
           

           nn={'up':0, 'down':0, 'hold':0, 'wait':0}
           
           k= .3
           
           for i in range(1000):                
               amt, act, term, t_wait= pilo('out/aat.pck')
               amt, term= 100, 30
               iterm= int(term / 60)
               pr(B,  f'\n{i=},  {tnow()},  \n{amt, act, term=},  \n\n{nn=}')
               
               
               if tnow() < t_wait: 
                   await page.waitFor(50)
                   continue
               
               if act =='wait': act='hold'
               
                
               pr(Y, tnow(), '\n--------  go to hist')
               t1(0)
               #hi, act= await hist1(page_hist,  models)
               hi, act= await Pyppeteer.hist1(page, models=models3)  #, models)
               t1( f'{tnow()}  hist done, {act=}')
               nn[act] += 1
               
               if nn['up'] + nn['down'] > 1: return    #?? temporary constraint
               #if nn['up'] + nn['down'] > 22: return  #?? temporary constraint
               
               if act=='hold':
                    await page.waitFor(30 * 1000)
                    continue
                
               pr(R if act=='down' else G, act, nn) 
               
               if 0:# debug block
                   t_wait= tnow() + tdelta(seconds=term)                   
                   pidu((amt, 'wait', term, t_wait), 'out/aat.pck')                   
                   await page.waitFor(200)
                   continue
      
                   
               if 0 and act in ['up', 'down']:
                   t1(0)
                   val, val2= await trade_curve(page, act, amt, iterm)  
                   t1( f'{tnow()}  trade done, {act}')
                   pr(Y, f'{val=}, {val2=}')
                   t_wait= tnow() + tdelta(seconds=term)                   
                   pidu((amt, 'wait', term, t_wait), 'out/aat.pck')
                   
               await page.waitFor(30 * 1000)

           await browserObj.close()
           
    if 0:
        act= asyncio.get_event_loop().run_until_complete(Pyppeteer.curve()) 


#class TA_LSTM:
    
    
# def norm(xx, mima=False): 
#     if not mima:
#         mi=[np.min(xx[:,:,i]) for   i in range(xx.shape[2])]
#         ma=[np.max(xx[:,:,i]) for   i in range(xx.shape[2])]
#     else: mi, ma= mima  
    
#     r= xx.copy()
#     for i in range(xx.shape[2]): r[:,:,i]= (xx[:,:,i] - mi[i]) / (.001 + ma[i] - mi[i])
#     #return npa([(xx[:,:,i] - mi[i]) / (.001 + ma[i] - mi[i])  for i in range(txd.shape[2]) ], dtype=float32) , (mi, ma)
#     return r , (mi, ma)
    
    
def norm(xx, mima=False, large= 1e8):  # xx - 2D
    r= xx.to_numpy()
    r= w(r < -large, -large, w(r > large, large, r))
    if not mima:
        mi= np.nanmin(r, axis=0)  # mi.shape  #  107
        ma= np.nanmax(r, axis=0)  # ma.shape  107
    else: mi,ma= mima
    
    pred(f'mi==ma for {len(xx)=}:', np.where(ma==mi)) # [38, 39]
    return PDF((r - mi) / (.001 + ma - mi), index=xx.index, columns=xx.columns),  (mi,ma)    
    
def trim_n_norm(th, large=9999):
    ''' th.ttxxyy --> th.ttxxyy_norm '''
    def lesha(xx): pr(Y, len(xx), xx[0].shape) 
    
    (ttxd, ttxs), (ttyn,ttyq)=  th.ttxxyy
    
    
    
    ttxd_norm, th.xd_mima= norm(Thinker_nq.trim(ttxd, large=large))
    ttxs_norm, th.xs_mima= norm(Thinker_nq.trim(ttxs, large=large))
    

    pr(Y, f'in trim_n_norm: \n{th.xd_mima=}, \n{th.xs_mima=}')
    th.ttxxyy_norm=  ttxxyy= (ttxd_norm ,ttxs_norm), (ttyn, ttyq)
    for u in  (ttxd_norm ,ttxs_norm,  ttyn, ttyq): lesha(u)
    return  th.ttxxyy_norm    
    ''' after
        yx1= calk_yx(h1) # yx1.tab(nr=20);
        yx2= calk_yx(h2)
        yx3= calk_yx(h3)
        yx4= calk_yx(h4)'''
        
    def trim(df, large=9999, mima=False):
        u= df.replace([np.inf, -np.inf], np.nan, inplace=False)
        u= u.dropna()
        m =mima
        if not m: mima= {}
        for c in u.columns:
            if c[:2] != 'yq':
                x= u[c]
                x= w(x < -large, -large, w(x > large,  large, x))
                if not m: 
                    mima[c]= mi,ma= np.nanmin(x), np.nanmax(x)
                else: mi,ma= mima[c]
                x= u[c]= (x-mi) /(.01+ ma-mi)
        u= u.dropna()
        return u, mima  

        
    def cutter(df, nt= 7):
            x,y=[],[]
            for i in range(nt, len(df)):
                u= df.rid()
                x.append(u.iloc[(i-nt):i, :][cx])
                y.append(u.iloc[i][ cy])
            x = npa(x)
            y = npa(y)
            return x,y
     
    def az_run():    
        #yx= PDF()    
            
        df= yx
        cx= [x for x in df.columns if x[0] != 'y']
        cy= [x for x in df.columns if x[0] == 'y']     
        
        i=50
            

        
        yx1.tab();
        yx1n, yx1_mima= norm(fib.Thinker_nq.trim(yx1, large=9999))
        plt.imshow(yx1)
        plt.imshow(trim(yx1))
    

    
        #from sklearn.preprocessing import MinMaxScaler
        #scaler = MinMaxScaler(feature_range=(0,1))
        
        # yx1s= scaler.fit_transform(trim(yx1))
        # yx2s = scaler.transform(trim(yx2))
        # yx3s = scaler.transform(trim(yx3))  # plt.imshow( yx3s)
        # yx4s = scaler.transform(trim(yx4))
        
        yx1s, mima= trim(yx1)
        yx2s = trim(yx2, mima=mima)[0]
        yx3s = trim(yx3, mima=mima)[0] # plt.imshow( yx3s)
        yx4s = trim(yx4, mima=mima)[0]
        yx5s = trim(yx5, mima=mima)[0]
        yx6s = trim(yx6, mima=mima)[0]
        yx7s = trim(yx7, mima=mima)[0]
        yx8s = trim(yx8, mima=mima)[0]
        yx8es = trim(yx8e, mima=mima)[0]
    
        np.min(yx1s),    np.max(yx1s)
        yx8es.tab(nr=0);
        
        
        for y in [yx1s, yx2s, yx3s, yx4s, yx5s, yx6s, yx7s, yx8s, yx8es]: pr(y.shape, sum(sum(np.isnan(y))))
    
        #cx= cx_good
        cx= [x for x in yx1s.columns if x[0] != 'y']

        
        xx,yy=[],[]
        for df in [yx1s,yx2s, yx3s,    yx5s, yx6s, yx7s]:
            x,y= cutter(df)
            xx.append(x)
            yy.append(y)
            #xx += x
            #yy += y
        
        xx_smallxx= np.concatenate(xx)      
        yy= np.concatenate(yy)  
        q1, q3= np.quantile(yy[:,0], [.1, .9])  # min(yy[:,0]), max(yy[:,0])
        q= .5*(q1+ (1-q3))
        xx_small= npa([x  for x,y in zip(xx,yy) if y[0] < q or y[0] > 1-q])
        yy_small= npa([npa((y[0],y[3]))  for x,y in zip(xx,yy) if y[0] < q or y[0] > 1-q])
        xx.shape,           yy.shape          #  (2046, 7, 285), (2046, 6))   ((1247, 7, 285), (1247, 6))
        xx_small.shape,     yy_small.shape    #  (451, 7, 285), (451, 6)   (917, 7, 285), (917, 6)
        
        xx_te, yy_te= cutter(yx4s)
        xx_te, yy_te= cutter(yx8e)  # ETH
        xx_te.shape,     yy_te.shape # ((455, 7, 285), (455, 6))
        
        sum(np.isnan(xx)),         sum(np.isnan(yy))
        sum(np.isnan(xx_te)),      sum(np.isnan(yy_te))
    
    
        #### model
        from keras.models import Sequential, load_model
        from keras.layers import LSTM, Dense, Dropout
        n1= 512 ## 256 #64  #512
        model = Sequential()
        model.add(LSTM(units=n1, return_sequences=True, input_shape=(xx.shape[1], xx.shape[2])))
        #model.add(LSTM(units=512, return_sequences=False, input_shape=(xx.shape[1], xx.shape[2])))
        model.add(Dropout(0.1))
        model.add(LSTM(units=n1 // 2, return_sequences=True))
        model.add(Dropout(0.1))
        model.add(LSTM(units= n1// 4,return_sequences=False))
        #model.add(Dense(units= n1//4, activation= 'LeakyReLU'))
    
        # model.add(Dropout(0.1))
        # model.add(LSTM(units=n1//8))
        model.add(Dropout(0.1))
        #model.add(Dense(units=6, activation= 'LeakyReLU'))
        model.add(Dense(units=2, activation= 'linear'))
        #model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))   
        model.compile(loss='mean_absolute_error', optimizer= keras.optimizers.Adam(learning_rate=1e-4))   
        model.summary()
    
        # m= fib.Model_nq()
        # m.model= model
        # m.set_weights(fm='out/m_ta_ker-64-ep1200.keras', mo=False, k= .2)
        # m.set_weights( mo=m64, k= .2)
        model= set_weights_from_model(model,  m256, k= .1)
    
        hi= model.fit(xx_small, yy_small, epochs=600, batch_size=32, validation_split= 0.3) 
        #Epoch 600/600  - loss: 0.1884 - val_loss: 0.7374   512: loss: 0.0995 - val_loss: 0.7146  loss: 0.0605 - val_loss: 0.7619
        #  loss: 0.1425 - val_loss: 0.6771
        
        model.save('out/m_ta_ker-256-ep600.keras')
        m256= keras.models.load_model('out/m_ta_ker-256-ep600.keras')
        
        model.save('out/m_ta_ker-64-ep1200.keras')
        m64= keras.models.load_model('out/m_ta_ker-64-ep1200.keras')
        model= m64
    
        #hi= model.fit(xx, yy, epochs=600, batch_size=32, validation_split= 0.3) 
        # Epoch 600/600
        # 28/28 [==============================] - 2s 71ms/step - loss: 0.2373 - val_loss: 0.4493
        model.save('out/m_ta_ker1-ep1200.keras')
        if 0:
            model.save('out/m_ta_ker1-ep600.keras')
            #model = keras.models.load_model('out/m_ta_ker1.keras')

        p= figu(title='Keras Learning Curves')
        p.line(range(len( hi.history['loss'])), hi.history['loss'], color='blue')
        p.line(range(len( hi.history['loss'])), hi.history['val_loss'], color='red')
        show(p)  
    
        # eval
        yhh= model.predict(xx_te)
        yhh= conc([PDF(yy_te, columns=['y_1','y_2', 'y_3','yq_1','yq_2', 'yq_3']),        
                   PDF(yhh, columns=['y_1h','yq_1h'])], **a1)
    
        p= figu() #title='Keras Learning Curves', x_axis_label='epoch', y_axis_label='Loss', width=900, height=600)
        p.scatter(x=yhh.yq_1h, y=yhh.y_1h, color=yhh.yq_1.map({-1:'red', 0:'grey',1:'green'}))  #,  line_color= 'red',    legend_label='training' ) # axes
        show(p) 
        
        yhh= conc([PDF(yy_te, columns=['y_1','y_2', 'y_3','yq_1','yq_2', 'yq_3']),        
                   PDF(yhh, columns=['y_1h','y_2h', 'y_3h','yq_1h', 'yq_2h', 'yq_3h'])], **a1)    
        
        p= figu() #title='Keras Learning Curves', x_axis_label='epoch', y_axis_label='Loss', width=900, height=600)
        p.scatter(x=yhh.yq_1h, y=yhh.yq_2h, color=yhh.yq_1.map({-1:'red', 0:'grey',1:'green'}))  #,  line_color= 'red',    legend_label='training' ) # axes
        show(p)    
    
        p= figu() #title='Keras Learning Curves', x_axis_label='epoch', y_axis_label='Loss', width=900, height=600)
        p.scatter(x=yhh.yq_2h, y=yhh.yq_3h, color=yhh.yq_2.map({-1:'red', 0:'grey',1:'green'}))  #,  line_color= 'red',    legend_label='training' ) # axes
        show(p)       
        
     
        p= figu() #title='Keras Learning Curves', x_axis_label='epoch', y_axis_label='Loss', width=900, height=600)
        p.scatter(x=yhh.y_1h, y=yhh.y_2h, color=yhh.yq_1.map({-1:'red', 0:'grey',1:'green'}))  #,  line_color= 'red',    legend_label='training' ) # axes
        show(p)       
    

        p= figu() #title='Keras Learning Curves', x_axis_label='epoch', y_axis_label='Loss', width=900, height=600)
        p.scatter(x=yhh.y_2h, y=yhh.y_3h, color=yhh.yq_2.map({-1:'red', 0:'grey',1:'green'}))  #,  line_color= 'red',    legend_label='training' ) # axes
        show(p)       
        
        bokeh_win(y='yq_1', yh= 'yq_1h')
        bokeh_win(y='yq_1', yh= 'yq_2h')
        bokeh_win(y='yq_1', yh= 'y_1h')
        bokeh_win(y='yq_1', yh= 'y_2h')
        
        bokeh_win(y='yq_2', yh= 'yq_2h')
        bokeh_win(y='yq_2', yh= 'y_2h')
        bokeh_win(y='yq_2', yh= 'yq_1h')
        bokeh_win(y='yq_2', yh= 'yq_3h')    
        
        bokeh_win(y='yq_3', yh= 'yq_3h')
        bokeh_win(y='yq_3', yh= 'y_3h')
        bokeh_win(y='yq_3', yh= 'yq_1h')
        bokeh_win(y='yq_3', yh= 'yq_2h')
    
        #### Compare curve vs. coinbase
        cur=  PDF(json.loads(open('in/getHistory-5.json').read())['data'])
        cob=  PDF(json.loads(open('in/coinbase-5b.json').read())["result"]["snapshot"]["candles"])
        
        cur=  PDF(json.loads(open('in/getHistory-6.json').read())['data'])
        cob=  PDF(json.loads(open('in/coinbase-6.json').read())["result"]["snapshot"]["candles"])
        
        cur['date']= pd.to_datetime(cur.ts, unit='s')
        cob['date']= pd.to_datetime(cob.start, unit='s')
        u= pd.merge(cur, cob, on='date')  # [272 rows x 13 columns]
        
        
        u= pd.merge(h8, h8e, on='ts')  # [272 rows x 13 columns]  # BTC vs ETH
        u.date= pd.to_datetime(u.ts, unit='s')
        
        k=16
        p= figu(title='BTC-USDT quotes') #title='Keras Learning Curves', x_axis_label='epoch', y_axis_label='Loss', width=900, height=600)
        p.scatter(x=u.date, y=u.close_x,  color='red')  #,  line_color= 'red',    legend_label='training' ) # axes
        p.scatter(x=u.date, y= k* u.close_y,  color='green') #, color=yhh.yq_2.map({-1:'red', 0:'grey',1:'green'}))  #,  line_color= 'red',    legend_label='training' ) # axes
        p.line(x=u.date, y=u.close_x,  color='red', legend_label='curve')  #,  line_color= 'red',    legend_label='training' ) # axes
        p.line(x=u.date, y=k* u.close_y,  color='green', legend_label='coinbase') #, color=yhh.yq_2.map({-1:'red', 0:'grey',1:'green'}))  #,  line_color= 'red',    legend_label='training' ) # axes
        show(p) 
        
        
        p= figu(title='BTC-USDT volume') #title='Keras Learning Curves', x_axis_label='epoch', y_axis_label='Loss', width=900, height=600)
        p.scatter(x=u.date, y=u.vol.astype(float) / 2e5,  color='red')  #,  line_color= 'red',    legend_label='training' ) # axes
        p.scatter(x=u.date, y=u.volume,  color='green') #, color=yhh.yq_2.map({-1:'red', 0:'grey',1:'green'}))  #,  line_color= 'red',    legend_label='training' ) # axes
        p.line(x=u.date, y=u.vol.astype(float) / 2e5,  color='red', legend_label='curve')  #,  line_color= 'red',    legend_label='training' ) # axes
        p.line(x=u.date, y=u.volume,  color='green', legend_label='coinbase') #, color=yhh.yq_2.map({-1:'red', 0:'grey',1:'green'}))  #,  line_color= 'red',    legend_label='training' ) # axes
        show(p)   
    
    
        u.info()
        np.corrcoef(u.close_x, u.close_y.astype(float))[0,1], \
        np.corrcoef(u[1:].close_x, u.close_y.astype(float).shift()[1:])[0,1], \
        np.corrcoef(u.close_x[:-1], u.close_y.astype(float).shift(-1)[:-1])[0,1]  # (0.9954, 0.9852, 0.9709)
        

class Test_modul_Backtesting:        
    def test():
        from backtesting import Backtest, Strategy
        from backtesting.lib import crossover
        
        from backtesting.test import SMA, GOOG
        
        
        class SmaCross(Strategy):
            n1 = 10
            n2 = 20
        
            def init(self):
                close = self.data.Close
                self.sma1 = self.I(SMA, close, self.n1)
                self.sma2 = self.I(SMA, close, self.n2)
        
            def next(self):
                if crossover(self.sma1, self.sma2):
                    self.buy()
                elif crossover(self.sma2, self.sma1):
                    self.sell()
                    
        
        class AZ_GBM(Strategy):
  
            def init(self):
                p= .3
                self.qs, self.qb= np.quantile(yx.y_1, [p, 1-p])
                
                self.y1 = self.data.y_1
                self.y2 = self.data.y_2
                self.y3 = self.data.y_3
        
            def next(self):
                #if np.greater(self.y1 , self.qb): 
                if (self.y1 > self.qb) and  (self.y2 > self.qb) and  (self.y3 > self.qb): 
                    #self.position.close()
                    self.buy()
                elif (self.qs > self.y1) and (self.qs > self.y2) and (self.qs > self.y3): 
                    #self.position.close()
                    self.sell()
        
        GOOG
        
        u= h9= PDF(json.loads(open('in/getHistory-9.json').read())['data'])  #  '2023-08-24T16:11:00.000000000,  2023-08-25T00:30:00.000000000'
        h10= PDF(json.loads(open('in/getHistory-10.json').read())['data'])  #  '2023-08-24T16:11:00.000000000,  2023-08-25T00:30:00.000000000'
        h11= PDF(json.loads(open('in/getHistory-10.json').read())['data'])  #  '2023-08-24T16:11:00.000000000,  2023-08-25T00:30:00.000000000'
        u['date']= pd.to_datetime(u.ts, unit='s')
        f'{u.date.values[0]},  {u.date.values[-1]}'  #':%Y-%m-%d %H:%M:%S%z}' #' H:M:S}'
        # 2023-08-10T17:28:00.000000000,  2023-08-11T01:47:00.000000000'
        # 2023-08-11T07:56:00.000000000,  2023-08-11T16:15:00.000000000'
        # 2023-08-11T16:32:00.000000000,  2023-08-12T00:51:00.000000000'
        # 2023-08-13T10:11:00.000000000,  2023-08-13T18:30:00.000000000
     
        def calc_yx(h1, mima=False, cy=[], cx2=[]): # zzzz
            df1, mima= add_columns_btc(h1.rename(columns={'vol':'volume'}), rmNA=False, toScale=False)
            df1= add_talib(df1, rmNA=False, cx_not_imp=cx_not_imp)
            
            #cx= [c for c in df1.columns if c[0] != 'y']
            #cy= [c for c in df1.columns if c[0] == 'y']
            
            u, (mi,ma)= norm(xx= df1[cx2], mima=mima)
            
            #cx_const= [cx[i] for i in npa(np.where(mi==ma)).astype(int)[0]]  #  ['us', 'us_d1', 'us_d2']
            #cx_many_na= PDF(np.isnan(df1[cx]).sum(), columns=['n']).ri().query('n > 40')['index'].values
            #cx2= [c for c in cx if c not in list(cx_many_na) + cx_const]
            
            yx= conc([h9[['ts','close']], df1[cy], u[cx2]], **a1)
            
            return yx, (mi,ma) #.iloc[35:-3] #[cy+cx]
        
        yx1, (mi,ma) = calc_yx(h1= PDF(json.loads(open(f'in/getHistory-{1}.json').read())['data']))
        cx= [c for c in yx1.columns if c[0] != 'y']
        cy= [c for c in yx1.columns if c[0] == 'y']     #  ['y_1', 'y_2', 'y_3', 'yq_1', 'yq_2', 'yq_3']
        cx_const= [cx[i] for i in npa(np.where(mi==ma)).astype(int)[0]]  #  ['us', 'us_d1', 'us_d2']
        cx_many_na= PDF(np.isnan(yx1[cx]).sum(), columns=['n']).ri().query('n > 40')['index'].values
        cx2= sorted(list(set([c for c in cx if c not in list(cx_many_na) + cx_const +['ts']])))
        
        
        yx_tr= conc([calc_yx(PDF(json.loads(open(f'in/getHistory-{i}.json').read())['data']), mima=(mi,ma), cy=cy, cx2=cx2)[0]  for i in range(1, 8)])
        yx_tr2= yx_tr.dropna() # [3055 rows x 258 columns]
        yx_tr2= yx_tr2.loc[:, ~yx_tr2.columns.duplicated()].copy()   # remove dup columns
        yx_tr2.yq_1.vc()
        
        
        yx_te, (mi,ma)= calc_yx(h9, mima=(mi,ma), cy=cy, cx2=cx2)  #.tab('yx', nr=1); # yx1.tab(nr=20);
        yx_te, (mi,ma)= calc_yx(h10, mima=(mi,ma), cy=cy, cx2=cx2)  #.tab('yx', nr=1); # yx1.tab(nr=20);
        yx_te, (mi,ma)= calc_yx(h11, mima=(mi,ma), cy=cy, cx2=cx2)  #.tab('yx', nr=1); # yx1.tab(nr=20);
        yx_te= yx_te.loc[:, ~yx_te.columns.duplicated()].copy()   # remove dup columns

        
        ''' --> models= '''
        models= {}
        for  y in cy:  #['y_10']: # cy: # ['y_1', 'y_2', 'y_3', 'yq_1', 'yq_2', 'yq_3']
            pr(y)
            m= lgb.LGBMClassifier(objective="multiclass", **sbm) if y[:2]=='yq' \
               else lgb.LGBMRegressor(**sbm)
            yv= npa(yx_tr2[y].values)
            #yv= w(yv< .1, -1, w( yv > .7, 1, 0)) if y[:2]=='yq' else yv
            m.fit(yx_tr2[cx2], yv)
            models[y]= m

        models.keys()  # ['y_20', 'y_30', 'yq_10', 'yq_20', 'yq_30', 'y_10']
        #pidu(models, 'out/models/m_gbm6.pck')
        
        #### Confusion
        yhh= conc([yx_te[['yq_1','yq_2','yq_3']], 
                   PDFd(**{y +'h':  yh[:, 2] - yh[:, 0]  for y in ['yq_1','yq_2','yq_3'] 
                           if (yh:= models[y].predict_proba(yx_te[cx2]))[0][0] > -99 })
                  ], **a1)
        
        p, a, b, c= 88, 31, 19, 49
        for y in ['yq_3']: #,'yq_2','yq_3']:
            yh= yhh[y +'h']
            yh= a *yhh.yq_1h + b* yhh.yq_2h + c* yhh.yq_3h
            u= yhh[[y, y +'h']].assign(yhq= w(yh > p, 1, w(yh < - p, -1, 0)))
            pr('\n', y, p, B, '\n', PDF(u[[y, 'yhq']].vc(), columns=['n']).ri().pivot(index=y, columns='yhq', values='n'), W)
                #  p, a, b, c= 88, 31, 19, 49
                #  yhq    -1      0    1
                # yq_1                 
                # -1    1.0  220.0  NaN
                #  0    NaN   54.0  NaN
                #  1    NaN  222.0  3.0 
   
        yhh.vc()
        yhh[['yq1','yq2','yq3']].vc()
        
        # find: def win_1f((.9, .7, .2,.1))
        
        p= {'thr': 0.7978806518526921,
          'a': 0.24183079330331736,
          'b': 0.15029188006684258,
          'c': 0.3792798997357324}
        {k: round(100* v/(p['a'] +p['b'] +p['c']),1)  for k,v in p.items()} #  {'thr': 103.4, 'a': 31.3, 'b': 19.5, 'c': 49.2}
        [round(100* v/(p['a'] +p['b'] +p['c']),1)  for k,v in p.items()] #  103.4, 31.3, 19.5, 49.2
        win_1f((103.4, 31.3, 19.5, 49.2))
       
        
        
        nna= sum(PDF(np.isnan(u.to_numpy()))).ri()
        plt.imshow(u)
        plt.imshow(np.isnan(u))
        
        
        yx= yx.assign(Open=yx.open,    Close=yx.close, Low=yx.low,  High=yx.high, Volume=yx.volume)
        yx.tab(nr=0) ;  
        
        qs,qb= np.quantile(yx.y_1, [.1,.9])
        

        if 0:
            models= pilo('out/models/m_gbm_3q.pck')   
            models.keys()
        
        
        
       # bt = Backtest(GOOG, SmaCross,  cash=10000, commission=.002, exclusive_orders=True)
          
        bt = Backtest(yx, AZ_GBM, cash=10000, commission=.0001,  exclusive_orders=True)        
        output = bt.run()
        # output.keys()
        ['Start', 'End', 'Duration', 'Exposure Time [%]', 'Equity Final [$]',
                'Return [%]', 'Buy & Hold Return [%]',
                'Volatility (Ann.) [%]', 
               '# Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
               'Avg. Trade [%]', 'Max. Trade Duration', 'Avg. Trade Duration',
               'Profit Factor', 'Expectancy [%]', 'SQN', '_strategy', '_equity_curve',
               '_trades']
        for k  in ['Start', 'End', 'Duration', 'Exposure Time [%]', 'Equity Final [$]',
               'Equity Peak [$]', 'Return [%]', 'Buy & Hold Return [%]', '# Trades',
               'Return (Ann.) [%]', 'Volatility (Ann.) [%]', 
               'Avg. Drawdown [%]',  'Avg. Drawdown Duration',
               'Win Rate [%]', 
               'Avg. Trade [%]', 'Max. Trade Duration', 'Avg. Trade Duration',
               'Profit Factor', 'Expectancy [%]',  '_strategy', '_equity_curve']:
            pr(f'{k:22} \t {output[k]}')
       
        bp= bt.plot( filename='out/tmb1.html', open_browser=True)


def calc_yx(h1, mima=False, cy=[], cx2=[], cx_not_imp=[], 
            opts=False, with_taz=False, n_cx=0, cx_impo=[]): # zzzz
    df1, mima= add_columns_btc(h1.rename(columns={'vol':'volume'}), rmNA=False, 
                               toScale=False)
    df1= add_talib(df1, rmNA=False, cx_not_imp=cx_not_imp, with_taz=with_taz, opts=opts)
    
    cy=  cy  or [c for c in df1.columns if c[0] == 'y']
    cx=  [c for c in df1.columns if c[0] != 'y'  and c not in ['ts']]
    cx2= cx2 if len(cx2) == 0 else  cx    
    cx3= cx_impo[:n_cx] if n_cx > 0 else  cx2
    
    u, (mi,ma)= norm(xx= df1[cx3], mima=mima)
    
    #cx_const= [cx[i] for i in npa(np.where(mi==ma)).astype(int)[0]]  #  ['us', 'us_d1', 'us_d2']
    #cx_many_na= PDF(np.isnan(df1[cx]).sum(), columns=['n']).ri().query('n > 40')['index'].values
    #cx2= [c for c in cx if c not in list(cx_many_na) + cx_const]
   
    
    yx= conc([h1[['ts','close']], df1[cy], u[cx3]], **a1)
    
    return yx, (mi,ma) #.iloc[35:-3] #[cy+cx]       
 

def ta_opt_optim():
    # find #### read BTC hist json from curve curve

    
    def objective(trial, opts_taz=True, with_n_cx= True):  # https://www.kaggle.com/code/bextuychiev/lgbm-optuna-hyperparameter-tuning-w-understanding
        if with_n_cx:  # n=3
            opts_taz= False
            n_cx= trial.suggest_int('n_cx', 200, 300) 
            cxx= cx_n= cx_impo[:n_cx]
        else:    
            w_fast= trial.suggest_int('w_fast', 2, 3) # 7)
            w_slow= trial.suggest_int('w_slow', w_fast+7, w_fast+12) #, w_fast+1, w_fast+12)
            w_sign= trial.suggest_int('w_sign', 2, 4) #9)
            w_= trial.suggest_int('w_', 10,14)    #2, 12)
            w_sm= trial.suggest_int('w_sm', 5,9)  # 2, 12)
            w1= trial.suggest_int('w1', 2, 6)  #8)
            w2= trial.suggest_int('w2', w1+1, w1+4)  # w1+16)
            w3= trial.suggest_int('w3', w2+2, w2+6)  #, w2+1, w2+20)
            w4= trial.suggest_int('w4', w3+3, w3+6)  #, w3+1, w3+20)
        
            opts= o2v if opts_taz else [w_fast,w_slow,w_sign,w_,w_sm,  w1,w2,w3,w4]
            if opts_taz: opts_taz=     [w_fast,w_slow,w_sign,w_,w_sm,  w1,w2,w3,w4]
            
            cxx= [c for c in cx2 if c[0] != 'y']


        pr(G, f'{cxx=}')
        yxx= [calc_yx(h, mima= mima, cx2=cxx, opts=opts_taz, 
                      with_taz=True, n_cx=n_cx)[0] for i,h in enumerate(hii)]
        
        tr= conc(yxx[:-2]).dropna() #adj  len(hii)
        te= yxx[-1].dropna()       #adj
        tr, te= rm_dup_col(tr),  rm_dup_col(te)
       # cxx= [c for c in cx2 if c[0] != 'y']

        model = lgb.LGBMClassifier(objective="binary",  **sbm)
        y= w(tr.y_1.values >0, 1, 0)
        #pr(B, f'{y=}')
        pr(B, f'{cxx=}')
        tr.tab('tr', nr=1);
        
        preds = model.predict_proba(te[cxx])
        cv_scores = log_loss(w(te.y_1 >0, 1, 0), preds)

        return cv_scores
    
    #----------------------------------------------------------------------
    hii= [PDF(json.loads(open(f'in/getHistory-{i}.json').read())['data']) 
          for i in range(1, 6)]  #12)]  
    yx1, mima= calc_yx(hii[0], opts=False, with_taz=True )
    cx2= [c for c in rm_dup_col(yx1).columns if c[0] != 'y' and c not in ['ts']]
    
    model= model= pilo('out/m_yq_1_GBM.pck')
    cx2, sbm= model.cx2, model.sbm
    
    study = optuna.create_study()
    study.optimize(objective, n_trials= 199)
    study.best_params,  study.best_value
    
    {'n_cx': 270}, 0.7416856
    
    # value: 0.7143469 and parameters: 
    o2= {'w_fast': 3, 'w_slow': 12, 'w_sign': 2, 'w_': 11, 'w_sm': 8, 
         'w1': 2, 'w2': 4, 'w3': 7, 'w4': 11}
    o2v=   [3, 12, 2, 11, 8,   2, 4, 7, 11] # o2.values()
    oOldv= [2,  6, 2,  4, 3,   2, 4, 6,  8] # o2.values()
    
    
    # taz
    o_taz= {'w_fast': 3,
      'w_slow': 10,
      'w_sign': 3,
      'w_': 11,
      'w_sm': 7,
      'w1': 5,
      'w2': 8,
      'w3': 14,
      'w4': 20}  #,     0.7342)  o_taz.values()
    o_taz_v= [3, 10, 3, 11, 7,  5, 8, 14, 20]
    
    #### Confusion  / start here
    hii= [PDF(json.loads(open(f'in/getHistory-{i}.json').read())['data']) 
              for i in range(1, 12)] 
    yx1, mima= calc_yx(hii[1], opts=o2)
    yxx= [calc_yx(h, mima= mima, cx2=cx2, opts=False)[0] for i,h in enumerate(hii)]
    yxx= [calc_yx(h, mima= mima, cx2=cx2, opts=o1)[0] for i,h in enumerate(hii)]
    yxx= [calc_yx(h, mima= mima, cx2=cx2, opts=o2v)[0] for i,h in enumerate(hii)]
    yxx= [calc_yx(h, mima= mima, cx2=cx2, opts=o2v, with_taz=True, n_cx=270)[0] for i,h in enumerate(hii)]
    
    
    tr= conc(yxx[:-2]).dropna()
    te= yxx[-1].dropna()
    tr, te= rm_dup_col(tr),  rm_dup_col(te)
    
    
    model = lgb.LGBMClassifier(objective="binary",  **sbm)
    y= w(tr.y_1.values >0, 1, 0)
    if 0: cxx= [c for c in cx2 if c[0] != 'y']
    model.fit(tr[cx_impo[:270]], y, eval_metric="binary_logloss" )
    model.cx2= cx2
    model.cxx= cxx
    model.mima= mima
    model.opts= o2v
    model.sbm= sbm
    model.cx_impo= cx_impo
    
    # importance
    imp= PDFd(x=cx_impo[:270], imp= model.feature_importances_)
    imp.svde('imp').head(40).tab('best', nr=50);
    imp.svde('imp').tail(40).tab('worst', nr=50);
    cx_most_imp= sorted(set([x.replace('_d1','').replace('_d2','') for x in imp.svde('imp').head(20).x ]))
    cx_not_imp= sorted(set([x for x in imp.svde('imp').tail(30).x if x[-3:] not in ['_d1', '_d2']]))
    cx_impo= imp.svde('imp').x.values  ; len(cx_impo)
        
      
    if 0:
        pidu(model, 'out/m_yq_1_GBM.pck')
        pidu(model, 'out/m_yq_1_taz_GBM.pck')
        model= pilo('out/m_yq_1_GBM.pck')
        model= pilo('out/m_yq_1_taz_GBM.pck')
        cx2= model.cx2
    
    yhh= conc([PDFd(yq_1=w(te.y_1.values >0, 1, 0)), 
               PDFd(**{y +'h':  yh[:, 1] - yh[:, 0]  for y in ['yq_1'] 
                       if (yh:= model.predict_proba(te[cxx]))[0][0] > -99 })
              ], **a1)
    
    pd, pu= .999999999, .9 #.88
    pd, pu= .9, .85 #.88
    y= 'yq_1'
    yh= yhh['yq_1h']
    #yh= a *yhh.yq_1h + b* yhh.yq_2h + c* yhh.yq_3h
    u= yhh[['yq_1', 'yq_1h']].assign(yhq= w(yh > pu, 1, w(yh < 1- pd, -1, 0)))
    pr('\n', y, pu, B, '\n', PDF(u[['yq_1', 'yhq']].vc(), columns=['n']).ri().pivot(index='yq_1', columns='yhq', values='n'), W)
            #  yq_1 0.9  
            # yhq    -1   0   1
            # yq_1             
            # 0     187  29   1
            # 1     108  49   7 


    #### combo 3 models
    models3= {}
    cxi= cx_impo[:270]
    for  y in ['y_1', 'y_2', 'y_3']:  #['y_10']: # cy: # ['y_1', 'y_2', 'y_3', 'yq_1', 'yq_2', 'yq_3']
        pr(y)
        m= lgb.LGBMClassifier(objective="binary",  **sbm)
        yy= tr[y].values
        yv= w(yy >0, 1, -1)
        m.fit(tr[cxi], yv)
        models3[y]= m
        models3[y].cxi= cxi
        models3[y].mima= mima
        models3[y].sbm= sbm
        models3[y].cx_impo= cx_impo

        
    models3.keys()  # ['y_1', 'y_2', 'y_3']  actually yq_1, ...
    if 0:
        pidu(models3, 'out/m_3yq_1_taz_GBM.pck')
        models3= pilo('out/m_3yq_1_taz_GBM.pck') 
        
        
    #### Confusion
    def q(y): return w(y >0, 1, -1)
    yhh= conc([PDFd(**{y+'q': q(te[y]) for y in ['y_1', 'y_2', 'y_3']}), 
               PDFd(**{y +'qh':  yh[:, 1] - yh[:, 0]  for y in ['y_1','y_2','y_3'] 
                       if (yh:= models3[y].predict_proba(te[cxi]))[0][0] > -99 })
              ], **a1)

    p= figu(title='Keras pred yq_1', x_axis_label='yq_1h', y_axis_label='yq_2h', width=800, height=600)
    p.scatter(x=yhh.y_1qh, y=yhh.y_2qh, color=yhh.y_1q.map({-1:'red', 1:'green'}))  #,  line_color= 'red',    legend_label='training' ) # axes
    show(p)
    
    p= figu(title='Keras pred yq_1', x_axis_label='yq_1h', y_axis_label='yq_3h', width=800, height=600)
    p.scatter(x=yhh.y_1qh, y=yhh.y_3qh, color=yhh.y_1q.map({-1:'red', 1:'green'}))  #,  line_color= 'red',    legend_label='training' ) # axes
    show(p)
    
    def accur(trial, pp=False): 
        #thr, a,b,c= 70 , 40,30,30
        def tsf(s): return pp[s] if pp else trial.suggest_float(s,0,100)
        def ind(df, i): return df[i] if i in df.index else 0
        def nsu(df, i): 
            try: s= np.nansum(df[:,i], initial=0)
            except: s=0
            return s
        
        thr, a,b,c= [tsf('thr'), tsf('a'), tsf('b'), tsf('c')]
        yh= a *yhh.y_1qh + b* yhh.y_2qh + c* yhh.y_3qh
        u= yhh[['y_1q']].assign(yqh= w(yh > thr, 1, w(yh < -thr, -1, 0)))
        conf= u.vc()
        pr('\n', thr, B, '\n', PDF(u.vc(), columns=['n']).ri(). pivot(index='y_1q', columns='yqh', values='n'), W)
       
        n_try= nsu(conf,-1) + nsu(conf, 1) 
        acc= (ind(conf, (-1,-1)) + ind(conf, (1,1)) ) / (.0001+ n_try)
        win= 1.2* (ind(conf, (-1,-1)) + ind(conf, (1,1)) ) -  n_try
        return -win  # -acc
    
    study = optuna.create_study()
    study.optimize(accur, n_trials=199)
    pp= study.best_params
    norrm= 100* npa(list(pp.values())) / (sum(list(pp.values())) - pp['thr'])
    norrm, -study.best_value
    pp= {'thr': 62.74260522474065,
      'a': 4.503258061124157,
      'b': 12.752751826460702,
      'c': 47.12952820975692} 
    0.9999000099990001
     
     # win:
    pp= {'thr': 61.65572187762009,
       'a': 14.406457137069017,
       'b': 49.28869066811066,
       'c': 2.7484611425018093}  # [92.794, 21.682, 74.181,  4.137]
    0.5999999999999996    
     
    100* npa(list(pp.values())) / (sum(list(pp.values())) - pp['thr'])
     
    [94.931, 30.076, 37.754, 32.17 ], 0.399999
     
    pp= {'thr': 93,
       'a': 22,
       'b': 74,
       'c': 4}  # [92.794,  21.682, 74.181,  4.137]  ->win=  $.79

    
    accur(1, pp)
    


    
    
    

    # for i in range(1,12):
    #      hi= PDF(json.loads(open(f'in/getHistory-{i}.json').read())['data'])
    #      hi['date']= pd.to_datetime(hi.ts, unit='s')
    #      f'{hi.date.values[0]},  {hi.date.values[-1]}'  #':%Y-%m-%d %H:%M:%S%z}' #' H:M:S}'
      
    hii= [PDF(json.loads(open(f'in/getHistory-{i}.json').read())['data']) 
              for i in range(1, 12)]  
    # len(hii), hii[0]
    # df1= hii[1]
    # cx= [c for c in df1.columns if c[0] != 'y' and c not in ['ts']]
    # #cy= [c for c in df1.columns if c[0] == 'y']
    
    yx1, mima= calc_yx(hii[1], opts=False)

          
         
    mi,ma= mima
    cx= yx1.columns
    cx_const= [cx[i] for i in npa(np.where(mi==ma)).astype(int)[0]]  #  ['us', 'us_d1', 'us_d2']
    #cx_many_na= PDF(np.isnan(yx1[cx]).sum(), columns=['n']).ri().query('n > 40')['index'].values
    c_many_na= PDF(np.isnan(yx1).sum(), columns=['n']).ri().query('n > 20')['index'].values
    cx2= sorted(list(set([c for c in cx if c not in list(c_many_na) + cx_const +['ts']])))
      # yxx[0].tab(nr=20);
     
    yxx= [calc_yx(h, mima= mima, cx2=cx2, opts=False)[0] for i,h in enumerate(hii)]
    yxx[0].tab();
    
    tr= conc(yxx[:-1]).dropna()
    te= yxx[-1].dropna()
    
    
    study = optuna.create_study()
    study.optimize(objective, n_trials=999)
    sbm= study.best_params
    sbm
    
    
