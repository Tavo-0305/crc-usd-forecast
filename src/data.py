#Librerias a usar:
import numpy as np
import pandas as pd
from bccr import SW
from datetime import date
from pathlib import Path
import os


def actualizar_tipo_cambio():
    '''
    Esta función se encarga de actualizar los datos de interés para este proyecto:
    Tipo de cambio entre dolar/colón y otros indicadores que se usarán como covariables en el modelo
    '''
    df = SW(
        compra=317,
        venta=318,
        FechaInicio="01/01/1983",
        FechaFinal= date.today().strftime("%d/%m/%Y")
    )

    #Separación entre los distintos régimenes usados en Costa Rica:
    df_minidevaluaciones = df.loc[:"2006-10-16"]
    df_bandas = df.loc["2006-10-17":"2015-01-29"]
    df_flotacion = df.loc["2015-01-30":]

    #Crear directorio si no existe
    directorio = Path("raw_data")
    directorio.mkdir(parents=True, exist_ok=True)

    #Se guardan en un csv
    df_minidevaluaciones.to_csv(directorio/ 'modeloMinidevaluaciones.csv', index=False)
    df_bandas.to_csv(directorio/'modeloBandas.csv', index=False)
    df_flotacion.to_csv(directorio/'modeloFlotacion.csv', index=False)

def actualizar_covariables():
    INDICADORES = {
        "tasa_basica_pasiva":        423,
        "reservas_internacionales":  5970,
        "volumen_monex":             3446,
        "ipc":                       98405,
        "imae":                      87703,
    }
    for k,v in INDICADORES.items():
        df_i = SW(
            **{k:v},
            FechaInicio= '2015-01-30',
            FechaFinal= date.today().strftime("%d/%m/%Y")
        )
        ruta_salida = os.path.join('raw_data', f"{k}.csv")
        df_i.to_csv(ruta_salida)

if __name__ == '__main__':
    actualizar_tipo_cambio()
    actualizar_covariables()
    


