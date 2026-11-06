import pandas as pd
import numpy as np

'''
Se escoge trabajar con un dataset
total que llega hasta '2026-10-01'
esto para facilitar el uso de algunas
covariables cuyos datos no se 
publican a diario

Se elminia el primer row por datos faltantes 
'''

final_date = '2026-10-01'
initial_date = '2015-01-02'

## DATOS ##

#Time series Ventas
df_bandas = pd.read_csv('raw_data/modeloFlotacion.csv',index_col=0, parse_dates=True)
df_sales = df_bandas[['venta']].loc[initial_date:final_date]

#Covariables:
#Raw data
df_tasa_basica = pd.read_csv('raw_data/tasa_basica_pasiva.csv',index_col=0, parse_dates=True)
#df_imae = pd.read_csv('raw_data/imae.csv',index_col=0, parse_dates=True) 
df_ipc = pd.read_csv('raw_data/ipc.csv',index_col=0, parse_dates=True)
df_volumen_monex = pd.read_csv('raw_data/volumen_monex.csv',index_col=0, parse_dates=True)
df_reservas_inter = pd.read_csv('raw_data/reservas_internacionales.csv',index_col=0, parse_dates=True) #llega hasta el 2 de nov
df_fedfunds = pd.read_csv('raw_data/fedfunds.csv',index_col=0,parse_dates=True) #llega hasta octubre

#Fixes
df_ipc_fix = df_ipc.copy().sort_index().resample("D").ffill().loc[initial_date:final_date] #Fill Forward
df_volumen_monex_fix = df_volumen_monex.copy().fillna(0).loc[initial_date:final_date]
df_tasa_basica_fix = df_tasa_basica.copy().loc[initial_date:final_date]
df_reservas_inter_fix = df_reservas_inter.copy().loc[initial_date:final_date]
df_fedfunds_fix = df_fedfunds.copy().loc[initial_date:final_date]

covariates = {
    #'ipc' : df_ipc_fix,
   # 'volumen monex': df_volumen_monex_fix,
    'reservas internacionales': df_reservas_inter_fix,
    #'Federal Funds': df_fedfunds_fix,
    'Tasa basica' : df_tasa_basica_fix
}


def covar_gen(covariates, target):
    daily_index = target.index
    cov_list = []
    for name,df in covariates.items():
        
        df = df.copy().sort_index()
        df.columns = [name] 
        df = df.reindex(daily_index)
        cov_list.append(df)

    df_covariates = pd.concat(cov_list, axis=1)
    return df_covariates


def data_model_gen(ts,cov):
    ts.to_csv('model_data/ventas.csv')
    cov.to_csv('model_data/covariables.csv')

df_cov = covar_gen(covariates,df_sales)
data_model_gen(df_sales,df_cov)

