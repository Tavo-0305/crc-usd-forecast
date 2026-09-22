import pandas as pd 


def cargar_datos():
    df_minidevaluaciones = pd.read_csv(
        "../src/raw_data/modeloMinidevaluaciones.csv",
        index_col=0,      # usa la primera columna (las fechas) como índice
        parse_dates=True  # la convierte a formato fecha, no texto
    )

    df_bandas = pd.read_csv(
        "../src/raw_data/modeloBandas.csv",
        index_col=0,      # usa la primera columna (las fechas) como índice
        parse_dates=True  # la convierte a formato fecha, no texto
    )

    df_flotacion = pd.read_csv(
        "../src/raw_data/modeloFlotacion.csv",
        index_col=0,      # usa la primera columna (las fechas) como índice
        parse_dates=True  # la convierte a formato fecha, no texto
    )

    df_ipc = pd.read_csv(
        "../src/raw_data/ipc.csv",
        index_col=0,      # usa la primera columna (las fechas) como índice
        parse_dates=True  # la convierte a formato fecha, no texto
    )

    return (
        df_minidevaluaciones,
        df_bandas,
        df_flotacion,
        df_ipc
    )


if __name__ == "__main__":

    datos = cargar_datos()

    print("Datos cargados correctamente.")