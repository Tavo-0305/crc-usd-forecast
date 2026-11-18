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
    train, val = ts.split_before(pd.Timestamp('2026-09-30')) #train set -> hasta setiembre
    train_cov, val_cov = ts_cov.split_before(pd.Timestamp('2026-05-31')) 

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

def save_params(model_name,params,metrics): 
    with open(f"model_results/{model_name}_results.txt", "w", encoding="utf-8") as f:
        f.write(f'Model: {model_name}\n')
        f.write(f'Metricas:\n')
        f.write(f'RMSE: {metrics[0]}\n')
        f.write(f'MAPE: {metrics[1]}\n')
        f.write(f'MAE: {metrics[2]}\n')
        f.write('PARAMETROS:\n')
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

#Modelos

def XGBoost(model_name,train_data,val_data,cov_train_data,cov_val_data,lags,past_cov_lags,output_chunk_l,rand_s):
    '''
    PENDIENTE: voy a doc. params en español
    lags:
    past_cov:
    output_chunk:
    rand_s:

    Para la predicción se puede utilizar el set completo de covariable
    '''
    model = XGBModel(
        lags=lags,
        lags_past_covariates= past_cov_lags, 
        output_chunk_length= output_chunk_l,
        random_state= rand_s
    )
    params = {
        'lags': lags,
        'lags for past covariates': past_cov_lags,
        'ouput chunk length': output_chunk_l,
        'random state': rand_s
    }
    model.fit(series=train_data,past_covariates=cov_train_data)
    cov_total = cov_train_data.append(cov_val_data)
    forecast = model.predict(n=len(val_data),series=train_data,past_covariates=cov_total) 
    metrics = model_evaluation(val_data,forecast)
    visualize(val_data,forecast,model_name,metrics)
    save_params(model_name,params,metrics)

def RNN(
    model_name,train_data, val_data,
    cov_train_data, cov_val_data,
    input_chunk_l, output_chunk_l,
    hidden_dim, n_rnn_layers,
    dropout, n_epochs, batch_size, rand_s
):
    model = BlockRNNModel(
        model="RNN",
        input_chunk_length=input_chunk_l,
        output_chunk_length=output_chunk_l,
        hidden_dim=hidden_dim,
        n_rnn_layers=n_rnn_layers,
        dropout=dropout,
        n_epochs=n_epochs,
        batch_size=batch_size,
        random_state=rand_s
    )

    params = {
        
    }

    model.fit(
        series=scale(train_data),
        past_covariates=scale(cov_train_data)
    )

    cov_total = scale(cov_train_data.append(cov_val_data))

    forecast_ = model.predict(
        n=len(val_data),
        series=train_data,
        past_covariates=cov_total
    )
    forecast = de_scale(forecast_)
    metrics = model_evaluation(val_data,forecast)
    visualize(val_data,forecast,model_name,metrics)

def LSTM(
    train_data, val_data,
    cov_train_data, cov_val_data,
    input_chunk_l, output_chunk_l,
    hidden_dim, n_rnn_layers,
    dropout, n_epochs, batch_size, rand_s
):
    model = BlockRNNModel(
        model="LSTM",
        input_chunk_length=input_chunk_l,
        output_chunk_length=output_chunk_l,
        hidden_dim=hidden_dim,
        n_rnn_layers=n_rnn_layers,
        dropout=dropout,
        n_epochs=n_epochs,
        batch_size=batch_size,
        random_state=rand_s
    )

    model.fit(
        series=scale(train_data),
        past_covariates=scale(cov_train_data)
    )

    cov_total = scale(cov_train_data.append(cov_val_data))

    forecast_ = model.predict(
        n=len(val_data),
        series=train_data,
        past_covariates=cov_total
    )

    forecast = de_scale(forecast_)
    metrics = model_evaluation(val_data,forecast)
    visualize(val_data,forecast,'LSTM',metrics)
    

def GRU(
    train_data, val_data,
    cov_train_data, cov_val_data,
    input_chunk_l, output_chunk_l,
    hidden_dim, n_rnn_layers,
    dropout, n_epochs, batch_size, rand_s
):
    model = BlockRNNModel(
        model="GRU",
        input_chunk_length=input_chunk_l,
        output_chunk_length=output_chunk_l,
        hidden_dim=hidden_dim,
        n_rnn_layers=n_rnn_layers,
        dropout=dropout,
        n_epochs=n_epochs,
        batch_size=batch_size,
        random_state=rand_s
    )

    model.fit(
        series=scale(train_data),
        past_covariates=scale(cov_train_data)
    )

    cov_total = scale(cov_train_data.append(cov_val_data))

    forecast_ = model.predict(
        n=len(val_data),
        series=train_data,
        past_covariates=cov_total
    )
    forecast = de_scale(forecast_)
    metrics = model_evaluation(val_data,forecast)
    visualize(val_data,forecast,'GRU',metrics)
    
    
#Data Pull 
sales_df = pd.read_csv('model_data/ventas.csv')
cov_df = pd.read_csv('model_data/covariables.csv')

data = data_splitter(sales_df,cov_df)

