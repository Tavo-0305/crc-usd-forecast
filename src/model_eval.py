#Imports
from bccr import SW 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from sklearn.preprocessing import StandardScaler
from darts import TimeSeries
from darts.metrics import mape, rmse, mae
from darts.dataprocessing.transformers import Scaler

from darts.models import BlockRNNModel
from darts.models import XGBModel

#Utilidades:

def scale(train,val):
    scaler = Scaler()
    return scaler(train),scaler(val)

def de_scale(scaled_train,scaled_val):
    scaler = Scaler()
    return scaler.inverse_transform(scaled_train), scaler.inverse_transform(scaled_val)

def data_splitter(main_df,cov_df):
    #Time series principal 
    ts = TimeSeries.from_dataframe(main_df, value_cols="venta",freq="D")
    #Covariables
    ts_cov = TimeSeries.from_dataframe(cov_df,value_cols=['reservas internacionales','Tasa basica'],freq="D")
    #Split
    train, val = ts.split_before(pd.Timestamp('2026-05-31')) #train set -> hasta mayo
    train_cov, val_cov = ts.split_before(pd.Timestamp('2026-05-31')) 

    return train,val,train_cov,val_cov

def model_evaluation(val,forecast):
    #Devuelve las metricas RMSE,MAPE,MAE para cada modelo
    return rmse(val,forecast),mape(val,forecast),mae(val,forecast)

def visualize(val,forecast,model_name,metrics):
    plt.figure(figsize=(12, 6))
    val.plot(label="Valor Real", color="black", lw=2)

    forecast.plot(label=f'Predicción del modelo: {model_name}', color="red", linestyle="--")
    plt.title("Predicción del comportamiento del tipo de Cambio de venta CRC/USD")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(f'Metricas, RMSE: {metrics[0]} - MAPE: {metrics[1]} - MAE: {metrics[2]}')
    plt.savefig(f'model_results/{model_name}_eval.pdf')

def corr_test(df_cov,method='pearson'):
    numeric_df = df_cov.select_dtypes(include='number')
    corr_matrix = numeric_df.corr(method=method)
    return corr_matrix

#Data Pull

sales_df = pd.read_csv('model_data/ventas.csv')
cov_df = pd.read_csv('model_data/covariables.csv')

print(corr_test(cov_df))

#Modelos

def XGBoost(train_data,val_data,cov_train_data,conv_val_data,lags,past_cov_lags,output_chunk_l,rand_s):
    model = XGBModel(
        lags=lags,
        lags_past_covariates= past_cov_lags, 
        output_chunk_length= output_chunk_l,
        random_state= rand_s 
    )

def RNN():
    pass
def LSTM():
    pass
def GRU():
    pass 


model = XGBModel(
    lags=30,
    lags_past_covariates=30,
    output_chunk_length=7,
    random_state=42
)

model.fit(
    series=train,
    past_covariates=cov_train
)

forecast = model.predict(
    n=len(val),
    series=train,
    past_covariates=cov_train
)