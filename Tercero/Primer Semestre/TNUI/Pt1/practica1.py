import os
import urllib.request
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm
import operator
import numpy as np
import matplotlib.pyplot as plt


YEARS = [2019, 2020, 2021]
def descargar_datos(years):
  '''
  Descarga tablas de taxis durante los años seleccionados
  YEARS = [2019, 2020, 2021]
  '''
  for year in tqdm(YEARS):
      if not os.path.exists(f'data/{year}'):
          os.makedirs(f'data/{year}', exist_ok=True)
          for month in tqdm(range(1, 13)):
              urllib.request.urlretrieve(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet', f'data/{year}/{month:02d}.parquet')


def load_table(year, month, sampling = 100):
    """
    Carga y muestrea los datos de taxis desde un archivo Parquet para un año y mes específicos.

    Lee el archivo correspondiente, selecciona un subconjunto de columnas relevantes
    y aplica un muestreo para reducir el tamaño del DataFrame resultante.

    Parámetros
    ----------
    year : int
        El año de los datos a cargar (ej: 2023).
    month : int
        El mes de los datos a cargar (ej: 1 para enero).
    sampling : int, opcional
        Factor de muestreo. Se seleccionará 1 de cada `sampling` filas.
        El valor por defecto es 100.

    Devuelve
    -------
    pd.DataFrame
        Un DataFrame de pandas con los datos muestreados.
    """
    data = pq.read_table(f'data/{year}/{str(month).zfill(2)}.parquet').to_pandas()
    required_data = ['tpep_pickup_datetime',
                 'tpep_dropoff_datetime',
                 'passenger_count',
                 'trip_distance',
                 'PULocationID',
                 'DOLocationID',
                 'payment_type',
                 'fare_amount',
                 'total_amount']
    return data[required_data][::sampling]


def build_full_dataset(years_list, sampling=100):
    """
    Carga, limpia y concatena los datos de taxis para una lista de años.

    Itera a través de cada mes para cada año proporcionado, carga los datos
    utilizando la función `load_table`, los limpia con `clean_data` y finalmente
    combina todos los DataFrames resultantes en uno solo.

    Parámetros
    ----------
    years_list : list of int
        Una lista de los años que se van a procesar (ej: [2019, 2020]).
    sampling : int, opcional
        Factor de muestreo que se pasará a `load_table`. Por defecto es 100.

    Devuelve
    -------
    pd.DataFrame
        Un único DataFrame con todos los datos procesados y combinados.

    Dependencias
    ------------
    Requiere que las funciones `load_table` y `clean_data` estén definidas.
    """

    # Creamos una lista de DataFrames limpios usando una list comprehension
    list_of_dfs = [
        clean_data(load_table(year, month, sampling), year, month)
        for year in tqdm(years_list, desc="Procesando Años")
        for month in tqdm(range(1, 13), leave=False, desc=f"Meses del {year}")
    ]

    # Concatenamos todos los DataFrames de la lista en uno solo
    full_df = pd.concat(list_of_dfs, ignore_index=True, sort=False)

    return full_df
#Funciones independientes


def filter_by_condition(data, column, op, value):
    """
    Filtra un DataFrame basándose en una condición.
    # Ejemplos de uso para reemplazar tus funciones:
    # clean_a_mayor_que_b(data, 'col_A', 'col_B') -> filter_by_condition(data, 'col_A', operator.lt, 'col_B')
    # clean_a_mayor_que_b_int(data, 'col_A', 5) -> filter_by_condition(data, 'col_A', operator.lt, 5)
    # clean_negatives(data, 'fare_amount') -> filter_by_condition(data, 'fare_amount', operator.gt, 0)
    # clean_equals(data, 'PULocationID', 'DOLocationID') -> filter_by_condition(data, 'PULocationID', operator.ne, 'DOLocationID')
    Parameters
    ----------
    data: pd.DataFrame
        El DataFrame a filtrar.
    column: str
        La columna sobre la que aplicar la condición.
    op: operator
        Un operador de comparación (ej: operator.gt, operator.le, operator.ne).
    value: any
        El valor con el que comparar. Puede ser un número o el nombre de otra columna.
    """
    if isinstance(value, str) and value in data.columns:
        # Compara dos columnas
        return data[op(data[column], data[value])]
    else:
        # Compara una columna con un valor estático
        return data[op(data[column], value)]

def filter_by_range(data, column, lower_bound, upper_bound):
    """
    Mantiene las filas donde el valor de una columna está dentro de un rango.
    """
    return data[data[column].between(lower_bound, upper_bound)]

def filter_by_datetime_part(data, dt_column, part, op, value):
    """
    Filtra por una parte específica de una columna de fecha/hora.
    # mantain_año_bajada(data, 2019) -> filter_by_datetime_part(data, 'tpep_dropoff_datetime', 'year', operator.eq, 2019)
    # clean_hora_recogida(data, 10) -> filter_by_datetime_part(data, 'tpep_pickup_datetime', 'hour', operator.ne, 10)
    Parameters
    ----------
    dt_column: str
        Nombre de la columna datetime ('tpep_pickup_datetime', etc.).
    part: str
        Parte de la fecha a extraer ('year', 'month', 'day', 'hour').
    op: operator
        El operador de comparación (operator.eq para mantener, operator.ne para limpiar).
    value: int
        El valor (ej: 2019 para el año).
    """
    # getattr() nos permite acceder al atributo .dt.year, .dt.month, etc. de forma dinámica
    datetime_series = getattr(data[dt_column].dt, part)
    return data[op(datetime_series, value)]


def clean_payment_type(data,s):
    '''
    Elimina el payment_type s
    '''
    data = data[data['payment_type']!=s]
    return data

def mantain_payment_type(data,s,p):
    '''
    Mantiene SOLO los payment type de s a p
    '''
    data = data[data['payment_type'].between(s,p)]
    return data

def mantain_passenger_count(data, s, p):
    '''
    Mantiene SOLO los passenger count de s a p.
    '''
    data = data[data['passenger_count'].between(s,p)]
    return data

def clean_a_mayor_oigual_que_b(data,a,b):
    '''
    Elimina los casos donde a es mayor o igual que b. Donde a y b son columnas

    Parameters
    ----------
    a: str
    b: str
    '''
    data = data[data[a] <= data[b]]
    return data

def clean_a_mayor_que_b(data,a,b):
    '''
    Elimina los casos donde a es MAYOR que b. Donde a y b son columnas

    Parameters
    ----------
    a: str
    b: str
    '''
    data = data[data[a] < data[b]]
    return data

def clean_a_mayor_oigual_que_b_int(data,a,b):
    '''
    Elimina los casos donde a es mayor o igual que b. Donde a y b son columnas

    Parameters
    ----------
    a: str
    b: int
    '''
    data = data[data[a] <= b]
    return data

def clean_a_mayor_que_b_int(data,a,b):
    '''
    Elimina los casos donde a es MAYOR que b.

    Parameters
    ----------
    a: str
    b: int
    '''
    data = data[data[a] < b]
    return data

def clean_negatives(data,s):
    '''
    Elimina los negativos de la columna S
    '''
    data = data[data[s] > 0]
    return data

def clean_equals(data,s,p):
    '''
    Elimina los casos donde, en la misma fila, la columna s es lo mismo que p
    data = data[data['PULocationID'] != data['DOLocationID']]
    '''
    data = data[data[s] != data[p]]
    return data

def mantain_range(data,s,p,q):
    '''
    Mantiene los valores dentro de la columna s entre los valores p y q (ambos incluidos)
    '''
    data = data[data[s].between(p, q)]
    return data

def clean_travels_mayor_que(data, s):
    '''
    Elimina todos los viajes que su duración sea mayor a s horas

    Parameters
    ----------
    s: int
    '''
    data = data[(data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']) <= pd.Timedelta(hours = s)]
    return data

def clean_travels_menor_que(data, s):
    '''
    Elimina todos los viajes que su duración sea menor a s minutos

    Parameters
    ----------
    s: int
    '''
    data = data[(data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']) >= pd.Timedelta(minutes = s)]
    return data

def mantain_año_bajada(data,s):
    '''
    Mantiene solo la recogida que se hayan efectuado en el año s
    '''
    data = data[data['tpep_pickup_datetime'].dt.year == s]
    return data

def mantain_dia_recogida(data,s):
    '''
    Mantiene solo la recogida que se hayan efectuado en el dia s
    '''
    data = data[data['tpep_pickup_datetime'].dt.day == s]
    return data

def mantain_mes_recogida(data,s):
  '''
  Mantiene solo la recogida que se hayan efectuado en el mes s
  '''
  data = data[data['tpep_pickup_datetime'].dt.month == s]
  return data

def mantain_hora_recogida(data,s):
    '''
    Mantiene solo la recogida que se hayan efectuado en la hora s
    '''
    data = data[data['tpep_pickup_datetime'].dt.hour == s]
    return data

def clean_año_recogida(data,s):
    '''
    Elimina todos los casos donde la recogida se haya efectuado en el año s
    '''
    data = data[data['tpep_pickup_datetime'].dt.year != s]
    return data

def clean_dia_recogida(data,s):
  '''
  Elimina todos los casos donde la recogida se haya efectuado en el mes s
  '''
  data = data[data['tpep_pickup_datetime'].dt.month != s]
  return data

def clean_dia_recogida(data,s):
    '''
    Elimina todos los casos donde la recogida se haya efectuado en el dia s
    '''
    data = data[data['tpep_pickup_datetime'].dt.day != s]
    return data

def clean_hora_recogida(data,s):
    '''
    Elimina todos los casos donde la recogida se haya efectuado en la hora s
    '''
    data = data[data['tpep_pickup_datetime'].dt.hour != s]
    return data


def mantain_año_bajada(data,s):

    data = data[data['tpep_dropoff_datetime'].dt.year == s]
    return data

def mantain_dia_bajada(data,s):
    data = data[data['tpep_dropoff_datetime'].dt.day == s]
    return data

def mantain_hora_bajada(data,s):
    data = data[data['tpep_dropoff_datetime'].dt.hour == s]
    return data

def clean_año_bajada(data,s):
    '''
    Elimina todos los casos donde la BAJADA se haya efectuado en el año s
    '''
    data = data[data['tpep_dropoff_datetime'].dt.year != s]
    return data

def clean_mes_bajada(data,s):
    '''
    Elimina todos los casos donde la BAJADA se haya efectuado en el año s
    '''
    data = data[data['tpep_dropoff_datetime'].dt.month != s]
    return data

def clean_dia_bajada(data,s):
    '''
    Elimina todos los casos donde la BAJADA se haya efectuado en el dia s
    s:int
    '''
    data = data[data['tpep_dropoff_datetime'].dt.day != s]
    return data


def clean_hora_bajada(data,s):
    '''
    Elimina todos los casos donde la BAJADA se haya efectuado en el dia s
    '''
    data = data[data['tpep_dropoff_datetime'].dt.hour != s]
    return data

def clean_exceso_velocidad(data,s):
    '''
    Elimina los casos donde la velocidad sea mayor o igual a s data*h
    '''
    pickup = pd.to_datetime(data['tpep_pickup_datetime'], errors='coerce')
    dropoff = pd.to_datetime(data['tpep_dropoff_datetime'], errors='coerce')
    duration_h = (dropoff - pickup).dt.total_seconds() / 3600

    speed = data['trip_distance'] / duration_h
    data = data[
        pickup.notna() &
        dropoff.notna() &
        (duration_h > 0) &
        (speed <= s)
    ]
    return data

def clean_min_velocidad(data,s):
    '''
    Elimina los casos donde la velocidad sea menor o igual a s mph
    '''
    pickup = pd.to_datetime(data['tpep_pickup_datetime'], errors='coerce')
    dropoff = pd.to_datetime(data['tpep_dropoff_datetime'], errors='coerce')
    duration_h = (dropoff - pickup).dt.total_seconds() / 3600

    speed = data['trip_distance'] / duration_h
    data = data[
        pickup.notna() &
        dropoff.notna() &
        (duration_h > 0) &
        (speed >= s)
    ]
    return data



def clean_data_refactored(data, year, month):
    """
    Función que limpia los datos de forma reutilizable y legible.
    """
    # Eliminar nulos
    data = data.dropna()

    # Filtrar por rango
    data = filter_by_range(data, 'payment_type', 1, 2)
    data = filter_by_range(data, 'passenger_count', 1, 5)
    data = filter_by_range(data, 'PULocationID', 1, 263)
    data = filter_by_range(data, 'DOLocationID', 1, 263)

    # Lógica de pagos
    data = filter_by_condition(data, 'fare_amount', operator.le, 'total_amount')
    data = filter_by_condition(data, 'fare_amount', operator.gt, 0)

    # Validar localizaciones
    data = filter_by_condition(data, 'PULocationID', operator.ne, 'DOLocationID')

    # Lógica de fechas y duración de viaje
    data = filter_by_condition(data, 'tpep_pickup_datetime', operator.le, 'tpep_dropoff_datetime')
    data = filter_by_datetime_part(data, 'tpep_pickup_datetime', 'year', operator.eq, year)
    data = filter_by_datetime_part(data, 'tpep_pickup_datetime', 'month', operator.eq, month)

    duration = data['tpep_dropoff_datetime'] - data['tpep_pickup_datetime']
    data = data[duration.between(pd.Timedelta(minutes=3), pd.Timedelta(hours=6))]

    # Lógica de distancia y velocidad
    data = filter_by_condition(data, 'trip_distance', operator.gt, 0.4)

    duration_h = duration.dt.total_seconds() / 3600
    # Evitar división por cero y mantener solo viajes con duración positiva
    valid_duration_mask = duration_h > 0
    data = data[valid_duration_mask]

    # Calcular velocidad solo para duraciones válidas
    speed = data.loc[valid_duration_mask, 'trip_distance'] / duration_h[valid_duration_mask]
    data = data[speed <= 65]

    return data

def mph_to_kmh(data, row):
    '''
    Convierte los datos de la columna row a kmh
    '''
    data[row] = data [row] * 1.60934
    return data

def kmh_to_mph(data, row):
    '''
    Convierte los datos de la columna row a mph
    '''
    data[row] = data[row] / 1.60934
    return data

def percentage(data,a,b, name_new_column):
    '''
    Calcula cual es el porcentaje de la columna b sobre la columna a
    Es como decir, si a = 100 y b = 80 dará 80%
    data [name_new_column] = (data[a] - data[b]) / data[a] * 100
    '''
    data [name_new_column] = (data[a] - data[b]) / data[a] * 100
    return data

def velocidad_promedio_viaje(data, new_column):
    '''
    Calcula la velocidad promedio de un viaje
    '''
    data[new_column] = data["trip_distance"] / data["duration_hours"]
    return data

def calcular_duracion(data, col_inicio, col_fin, nueva_col, unidades='hours'):
    """
    Calcula la duración entre dos columnas de fecha y hora y la guarda en una nueva columna.

    Args:
        data (pd.DataFrame): El DataFrame que contiene los datos.
        col_inicio (str): El nombre de la columna con la fecha/hora de inicio.
        col_fin (str): El nombre de la columna con la fecha/hora de fin.
        nueva_col (str): El nombre de la nueva columna donde se guardará la duración.
        unidades (str): Las unidades para la duración ('hours', 'minutes', 'seconds').
    """
    delta_tiempo = (data[col_fin] - data[col_inicio]).dt.total_seconds()

    if unidades == 'hours':
        data[nueva_col] = delta_tiempo / 3600
    elif unidades == 'minutes':
        data[nueva_col] = delta_tiempo / 60
    elif unidades == 'seconds':
        data[nueva_col] = delta_tiempo
    else:
        raise ValueError("Las unidades deben ser 'hours', 'minutes' o 'seconds'")

    return data

def calcular_ratio(data, col_numerador, col_denominador, nueva_col):
    """
    Calcula un ratio dividiendo dos columnas y lo guarda en una nueva.
    Ej: km/h, $/h, etc.
    """
    data[nueva_col] = data[col_numerador] / data[col_denominador]
    return data

def mapear_valores(data, columna, mapa_valores):
    """
    Traduce o mapea los valores de una columna usando un diccionario.

    mapa_pagos = {1: 'Credit card', 2: 'Cash'}
    data = mapear_valores(data, 'payment_type', mapa_pagos)
    """
    data[columna] = data[columna].map(mapa_valores)
    return data

def extraer_componente_fecha(data, col_fecha, nueva_col, componente):
    """
    Extrae un componente específico (año, mes, día, etc.) de una columna de fecha.

    Componentes válidos: 'year', 'month', 'day_of_month', 'day_of_week',
                         'hour', 'week_of_year'.
    """
    if componente == 'year':
        data[nueva_col] = data[col_fecha].dt.year
    elif componente == 'month':
        data[nueva_col] = data[col_fecha].dt.month
    elif componente == 'day_of_month':
        data[nueva_col] = data[col_fecha].dt.day
    elif componente == 'day_of_week':
        data[nueva_col] = data[col_fecha].dt.isocalendar().day
    elif componente == 'hour':
        data[nueva_col] = data[col_fecha].dt.hour
    elif componente == 'week_of_year':
        data[nueva_col] = data[col_fecha].dt.isocalendar().week
    else:
        raise ValueError(f"Componente '{componente}' no reconocido.")

    return data


def post_processing(data):
    """
    Función donde implementar cualquier tipo de postprocesamiento necesario
    llamando a funciones modulares.
    """
    # Conversión de unidades
    data = mph_to_kmh(data, 'trip_distance')

    # Cálculo de duración del viaje en horas y minutos
    data = calcular_duracion(data, 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration_hours', unidades='hours')
    data = calcular_duracion(data, 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration_minutes', unidades='minutes')

    # Cálculo de ratios (velocidad y ganancias por hora)
    data = calcular_ratio(data, 'trip_distance', 'duration_hours', 'avg_speed_kmh')
    data = calcular_ratio(data, 'total_amount', 'duration_hours', 'dolarperhour')

    # Cálculo de porcentajes
    data = percentage(data, 'total_amount', 'fare_amount', '%tips')

    # Mapeo de valores categóricos
    mapa_pagos = {1: 'Credit card', 2: 'Cash'}
    data = mapear_valores(data, 'payment_type', mapa_pagos)

    # Extracción de componentes de la fecha de recogida (pickup)
    data = extraer_componente_fecha(data, 'tpep_pickup_datetime', 'pickup_year', 'year')
    data = extraer_componente_fecha(data, 'tpep_pickup_datetime', 'pickup_month', 'month')
    data = extraer_componente_fecha(data, 'tpep_pickup_datetime', 'pickup_day_month', 'day_of_month')
    data = extraer_componente_fecha(data, 'tpep_pickup_datetime', 'pickup_day_week', 'day_of_week')
    data = extraer_componente_fecha(data, 'tpep_pickup_datetime', 'pickup_hour', 'hour')
    data = extraer_componente_fecha(data, 'tpep_pickup_datetime', 'pickup_week', 'week_of_year')

    # Extracción de componentes de la fecha de llegada (dropoff)
    data = extraer_componente_fecha(data, 'tpep_dropoff_datetime', 'dropoff_month', 'month')
    data = extraer_componente_fecha(data, 'tpep_dropoff_datetime', 'dropoff_day_month', 'day_of_month')
    data = extraer_componente_fecha(data, 'tpep_dropoff_datetime', 'dropoff_day_week', 'day_of_week')
    data = extraer_componente_fecha(data, 'tpep_dropoff_datetime', 'dropoff_hour', 'hour')
    data = extraer_componente_fecha(data, 'tpep_dropoff_datetime', 'dropoff_week', 'week_of_year')

    return data

def plot_metric_by_category(df, group_by_column, metric_column, agg_func, xlabel, ylabel, title, text_format='{:.0f}'):
    """
    Crea un gráfico de barras agrupando los datos y aplicando una función de agregación.
    # ¿Cómo ha cambiado la distancia promedio de los viajes cada año?
    tu.plot_metric_by_category(
        df=df_processed,
        group_by_column='pickup_year',      # Eje X: Las categorías (años)
        metric_column='trip_distance',      # Eje Y: La columna a medir
        agg_func='mean',                    # Cálculo: La media de la distancia
        xlabel='Año de Recogida',
        ylabel='Distancia Promedio (km)',
        title='Distancia Promedio de Viaje por Año'
    )

    Parámetros
    ----------
    df : pd.DataFrame
        El DataFrame con los datos.
    group_by_column : str
        La columna por la que se agruparán los datos (ej: 'year', 'payment_type').
    metric_column : str
        La columna sobre la que se calculará la métrica (ej: 'trip_distance').
    agg_func : str
        La función de agregación a aplicar: 'mean', 'sum', 'count', 'median', etc.
    xlabel : str
        Etiqueta para el eje X.
    ylabel : str
        Etiqueta para el eje Y.
    title : str
        Título del gráfico.
    text_format : str, opcional
        Formato para el texto que aparece sobre las barras. Por defecto, sin decimales.
    """
    # Caso especial para extraer el año si no existe la columna
    if group_by_column not in df.columns and 'tpep_pickup_datetime' in df.columns:
        df[group_by_column] = df['tpep_pickup_datetime'].dt.year

    # ✅ LA LÓGICA CLAVE: Agrupar y agregar en lugar de solo contar
    # Agrupamos por la columna de categoría (ej: 'year')
    # Seleccionamos la columna de la métrica (ej: 'trip_distance')
    # Aplicamos la función de agregación (ej: 'mean')
    grouped_data = df.groupby(group_by_column)[metric_column].agg(agg_func)

    # Crear la figura
    plt.figure(figsize=(10, 6))
    plt.bar(grouped_data.index, grouped_data.values, color='skyblue')

    # Etiquetas y título
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, weight='bold')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Añadir los valores formateados encima de cada barra
    for i, v in enumerate(grouped_data.values):
        plt.text(grouped_data.index[i], v, text_format.format(v), ha='center', va='bottom', fontsize=10)

    plt.tight_layout() # Ajusta el gráfico para que todo encaje bien
    plt.show()







def create_barchart(df, group_by_column, metric_column, agg_func,
                    xlabel, ylabel, title,
                    facet_by=None, text_format='{:.0f}'):
    """
    Crea un gráfico de barras, con la opción de generar subgráficos (facetas).

    Parámetros
    ----------
    df : pd.DataFrame
        El DataFrame con los datos.
    group_by_column : str
        Columna por la que se agruparán los datos (eje X).
    metric_column : str
        Columna sobre la que se calculará la métrica (eje Y).
    agg_func : str
        Función de agregación a aplicar ('mean', 'sum', 'count', etc.).
    xlabel, ylabel, title : str
        Etiquetas y título para el gráfico.
    facet_by : str, opcional
        Columna para crear subgráficos. Si se especifica, se creará un
        gráfico para cada valor único en esta columna. Por defecto es None.
    text_format : str, opcional
        Formato para el texto sobre las barras.

    # ¿Cuántos viajes se pagaron con tarjeta vs. efectivo, desglosado por año?
    tu.create_barchart(
        df=df_processed,
        group_by_column='payment_type',     # Eje X: Categorías principales (tipo de pago)
        metric_column='total_amount',       # Eje Y: La columna a medir (usamos una cualquiera para contar)
        agg_func='count',                   # Cálculo: Contar el número de viajes
        facet_by='pickup_year',             # Crea un subgráfico para cada año
        xlabel='Tipo de Pago',
        ylabel='Número de Viajes',
        title='Número de Viajes por Tipo de Pago (Desglosado por Año)'
    )
    """
    
    # --- Lógica para preparar las columnas dinámicamente ---
    for col in [group_by_column, facet_by]:
        if col and col not in df.columns and 'tpep_pickup_datetime' in df.columns:
            if col == 'year':
                df[col] = df['tpep_pickup_datetime'].dt.year
            elif col == 'month':
                df[col] = df['tpep_pickup_datetime'].dt.month
            elif col == 'day_of_week':
                 df[col] = df['tpep_pickup_datetime'].dt.day_name()

    # --- CASO 1: Se pide una cuadrícula de gráficos ---
    if facet_by:
        facet_values = sorted(df[facet_by].unique())
        n_facets = len(facet_values)

        # Creamos una cuadrícula de subgráficos, una fila por cada valor de la faceta
        fig, axes = plt.subplots(nrows=n_facets, ncols=1, figsize=(10, 5 * n_facets), sharex=True)
        fig.suptitle(title, fontsize=16, weight='bold') # Título general

        # Si solo hay una faceta, 'axes' no es una lista, lo convertimos
        if n_facets == 1:
            axes = [axes]

        for i, value in enumerate(facet_values):
            ax = axes[i]
            df_subset = df[df[facet_by] == value]

            grouped_data = df_subset.groupby(group_by_column)[metric_column].agg(agg_func)

            ax.bar(grouped_data.index, grouped_data.values, color='steelblue')
            ax.set_title(f'{facet_by.capitalize()}: {value}')
            ax.set_ylabel(ylabel)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            for j, val in enumerate(grouped_data.values):
                ax.text(grouped_data.index[j], val, text_format.format(val), ha='center', va='bottom')

        # Etiqueta X solo en el último gráfico
        axes[-1].set_xlabel(xlabel)

    # --- CASO 2: Se pide un único gráfico  ---
    else:
        grouped_data = df.groupby(group_by_column)[metric_column].agg(agg_func)
        plt.figure(figsize=(10, 6))
        plt.bar(grouped_data.index, grouped_data.values, color='skyblue')
        plt.title(title, fontsize=14, weight='bold')
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for i, v in enumerate(grouped_data.values):
            plt.text(grouped_data.index[i], v, text_format.format(v), ha='center', va='bottom')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajusta para que el título general no se solape
    plt.show()


def analyze_change_over_time(df, group_by_col, baseline_group, metrics_to_calculate):
    """
    Calcula el cambio absoluto y porcentual de varias métricas respecto a un grupo base.

    Parámetros
    ----------
    df : pd.DataFrame
        El DataFrame de entrada.
    group_by_col : str
        La columna por la que agrupar los datos (ej: 'year', 'month').
    baseline_group : any
        El valor en `group_by_col` que se usará como referencia (ej: 2019).
    metrics_to_calculate : list of dict
        Una lista de diccionarios, donde cada uno define una métrica a calcular.
        Ejemplo: [{'name': 'Media_Pasajeros', 'column': 'passenger_count', 'agg_func': 'mean'}]

    Devuelve
    -------
    pd.DataFrame
        Un DataFrame con los cambios absolutos y porcentuales para cada métrica.
    # 1. DEFINIR las métricas que quieres analizar
    #    Para la proporción de 1 pasajero, usamos una lambda que hace el cálculo
    metrics_definition = [
        {
            'name': 'Prop_1_Pax',
            'column': 'passenger_count',
            'agg_func': lambda s: s.value_counts(normalize=True).get(1, 0)
        },
        {
            'name': 'Media_Pax',
            'column': 'passenger_count',
            'agg_func': 'mean'
        }
    ]

    # 2. CALCULAR los cambios usando la función general
    #    Agrupamos por 'year' y usamos 2019 como base
    change_results = analyze_change_over_time(df,
                                              group_by_col='year',
                                              baseline_group=2019,
                                              metrics_to_calculate=metrics_definition)

    # 3. VISUALIZAR los resultados que te interesen
    plots_to_show = [
        {
            'metric_name': 'Prop_1_Pax', 'change_type': 'pct',
            'title': 'Cambio % en Proporción de Viajes Individuales',
            'ylabel': 'Cambio Porcentual (%)'
        },
        {
            'metric_name': 'Media_Pax', 'change_type': 'abs',
            'title': 'Cambio Absoluto en la Media de Pasajeros',
            'ylabel': 'Cambio (Pasajeros)',
            'colors': ['darkred', 'lightcoral']
        }
    ]

    plot_change_analysis(change_results, plots_to_show)
    """
    # --- 1. Calcular el valor de cada métrica para cada grupo (año) ---

    # Preparamos el DataFrame para guardar los indicadores por año
    groups = df[group_by_col].unique()
    df_indicators = pd.DataFrame(index=groups)

    for metric in metrics_to_calculate:
        # Agrupamos y aplicamos la función de agregación especificada
        grouped_data = df.groupby(group_by_col)[metric['column']].agg(metric['agg_func'])
        df_indicators[metric['name']] = grouped_data

    df_indicators.sort_index(inplace=True)

    # --- 2. Calcular los cambios respecto al grupo base ---
    baseline_metrics = df_indicators.loc[baseline_group]
    comparison_metrics = df_indicators.drop(baseline_group)

    # Cambio absoluto (Nuevo - Base)
    absolute_change = comparison_metrics.subtract(baseline_metrics)

    # Cambio porcentual ((Nuevo - Base) / Base) * 100
    percentage_change = absolute_change.div(baseline_metrics).multiply(100)

    # --- 3. Unir los resultados en un DataFrame final ---
    results = pd.concat([
        absolute_change.add_suffix('_abs_change'),
        percentage_change.add_suffix('_pct_change')
    ], axis=1)

    return results.round(2)


def plot_change_analysis(df_results, metrics_to_plot):
    """
   gráficos de barras que muestren claramente el cambio porcentual o absoluto

    Parámetros
    ----------
    df_results : pd.DataFrame
        El DataFrame generado por `analyze_change_over_time`.
    metrics_to_plot : list of dict
        Lista de diccionarios que definen qué y cómo graficar.
        Ej: [{'metric_name': 'Media_Pasajeros', 'change_type': 'abs', 'title': 'Cambio Absoluto...'}]
    
    # ¿Cuántos viajes se pagaron con tarjeta vs. efectivo, desglosado por año?
    tu.create_barchart(
        df=df_processed,
        group_by_column='payment_type',     # Eje X: Categorías principales (tipo de pago)
        metric_column='total_amount',       # Eje Y: La columna a medir (usamos una cualquiera para contar)
        agg_func='count',                   # Cálculo: Contar el número de viajes
        facet_by='pickup_year',             # Crea un subgráfico para cada año
        xlabel='Tipo de Pago',
        ylabel='Número de Viajes',
        title='Número de Viajes por Tipo de Pago (Desglosado por Año)'
    )
    # Paso 1: Definir y calcular los cambios que queremos analizar.
    # Queremos ver el cambio en la propina promedio ('%tips') usando 2019 como referencia.
    metrics_definition = [
        {'name': 'Propina_Media', 'column': '%tips', 'agg_func': 'mean'}
    ]

    change_results = tu.analyze_change_over_time(
        df=df_processed,
        group_by_col='pickup_year',
        baseline_group=2019,  # Año de referencia
        metrics_to_calculate=metrics_definition
    )

    # Paso 2: Visualizar ese cambio porcentual.
    plots_to_show = [
        {
            'metric_name': 'Propina_Media', 'change_type': 'pct',
            'title': 'Cambio Porcentual en la Propina Promedio (vs. 2019)',
            'ylabel': 'Cambio (%)'
        }
    ]

    tu.plot_change_analysis(change_results, plots_to_show)
    """
    n_plots = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    fig.suptitle('Análisis de Cambio Cuantitativo (vs. Baseline)', fontsize=16, weight='bold')

    # Si solo hay un gráfico, 'axes' no es una lista, lo convertimos
    if n_plots == 1:
        axes = [axes]

    for i, plot_info in enumerate(metrics_to_plot):
        ax = axes[i]
        metric_name = plot_info['metric_name']
        change_type = plot_info['change_type']

        # Construimos el nombre de la columna a graficar
        column_to_plot = f"{metric_name}_{change_type}_change"
        data = df_results[column_to_plot]

        data.plot(kind='bar', ax=ax, color=plot_info.get('colors', ['darkblue', 'skyblue']))
        ax.set_title(plot_info['title'])
        ax.set_ylabel(plot_info['ylabel'])
        ax.set_xlabel('Grupo de Comparación')
        ax.tick_params(axis='x', rotation=0)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--') # Línea en cero

        # Añadir etiquetas de valor
        for j, v in enumerate(data.values):
            offset = abs(v) * 0.05 + np.sign(v) * 0.1
            ax.text(j, v + offset, f'{v:.2f}{"%" if change_type == "pct" else ""}',
                    ha='center', va='center', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    plt.show()



def plot_metric_over_time(df, time_unit_col, metric_col, agg_func,
                          title, xlabel, ylabel, create_from_datetime=True):
    """
    Visualiza la evolución de una métrica a lo largo de una unidad de tiempo,
    comparando diferentes años con gráficos de líneas.
    Crea gráficos de líneas para mostrar cómo evoluciona una métrica a lo largo 
    de un período de tiempo (horas, días de la semana, meses). Su punto fuerte es que puede superponer varias 
    líneas (una para cada año, por ejemplo) para comparar patrones fácilmente. Es perfecta para preguntas como
    "¿La hora punta de viajes cambió después de la pandemia?"
    Parámetros
    ----------
    df : pd.DataFrame
        El DataFrame que contiene los datos.
    time_unit_col : str
        La columna que representa la unidad de tiempo en el eje X
        (ej: 'hour', 'day_of_week', 'month').
    metric_col : str
        La columna sobre la que se calculará la métrica (ej: 'trip_distance').
    agg_func : str
        La función de agregación a aplicar ('mean', 'sum', 'count', 'median').
    title, xlabel, ylabel : str
        Etiquetas y título para el gráfico.
    create_from_datetime : bool, opcional
        Si es True, intentará crear la columna `time_unit_col` a partir de
        'tpep_pickup_datetime' si no existe. Por defecto es True.
    
    # ¿Cómo varía la cantidad de viajes a lo largo del día? Comparativa por año.
    tu.plot_metric_over_time(
        df=df_processed,
        time_unit_col='pickup_hour',        # Eje X: Unidad de tiempo (la hora)
        metric_col='total_amount',          # Eje Y: Columna a medir
        agg_func='count',                   # Cálculo: Contar el número de viajes
        xlabel='Hora del Día',
        ylabel='Número de Viajes',
        title='Distribución de Viajes a lo Largo del Día'
    )
    
    
    """
    if time_unit_col not in df.columns and create_from_datetime:
        if time_unit_col == 'hour':
            df[time_unit_col] = df['tpep_pickup_datetime'].dt.hour
        elif time_unit_col == 'day_of_week':
            # .dt.day_name() devuelve el nombre del día
            df[time_unit_col] = df['tpep_pickup_datetime'].dt.day_name()
        elif time_unit_col == 'month':
            df[time_unit_col] = df['tpep_pickup_datetime'].dt.month

    plt.figure(figsize=(12, 7))

    # Obtenemos los años únicos del DataFrame
    YEARS = sorted(df['year'].unique())

    for y in YEARS:
        df_year = df[df["year"] == y]
        grouped_data = df_year.groupby(time_unit_col)[metric_col].agg(agg_func)

        # --- Ordenamiento inteligente para el eje X ---
        # Si son días de la semana, ordena cronológicamente, no alfabéticamente
        if time_unit_col == 'day_of_week':
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            grouped_data = grouped_data.reindex(days_order)
        else:
            grouped_data = grouped_data.sort_index()

        plt.plot(grouped_data.index, grouped_data.values,
                 linestyle="--", marker="o", label=str(y))

    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(title="Año")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_faceted_histograms(df, value_col, facet_by, n_cols=3,
                            title=None, xlabel=None, ylabel='Frecuencia',
                            bins=30, xlim=None, sharey=True, show_mean=False):
    """
    Descripción: Esta función sirve para comparar distribuciones de una variable numérica. Crea 
    una cuadrícula donde cada celda contiene un histograma para una categoría diferente.
    Crea una cuadrícula de histogramas para comparar la distribución de una columna numérica
    a través de las categorías de otra columna.

    Parámetros
    ----------
    df : pd.DataFrame
        El DataFrame que contiene los datos.
    value_col : str
        La columna numérica cuya distribución se va a visualizar.
    facet_by : str
        La columna categórica que se usará para crear los subgráficos (facetas).
    n_cols : int, opcional
        El número de columnas en la cuadrícula de subgráficos. Por defecto es 3.
    title, xlabel, ylabel : str, opcional
        Etiquetas y título para el gráfico.
    bins : int, opcional
        Número de contenedores para el histograma.
    xlim : tuple, opcional
        Límites para el eje X, ej: (0, 100).
    sharey : bool, opcional
        Si es True, todos los subgráficos compartirán el mismo eje Y.
    show_mean : bool, opcional
        Si es True, mostrará una línea vertical en la media de cada histograma.
    
    
    # Queremos ver si la distribución de la duración de los viajes ha cambiado entre años.
    # Limitamos el eje X para una mejor visualización.
    tu.plot_faceted_histograms(
        df=df_processed,
        value_col='duration_minutes',      # Variable numérica a analizar
        facet_by='pickup_year',            # Crear un histograma para cada año
        title='Distribución de la Duración de los Viajes por Año',
        xlabel='Duración del Viaje (Minutos)',
        bins=50,                           # Número de barras en el histograma
        xlim=(0, 60),                      # Limitar el eje X de 0 a 60 minutos
        show_mean=True                     # Mostrar la media en cada gráfico
    )
    """
    # --- Configuración de la cuadrícula ---
    categories = sorted(df[facet_by].unique())
    n_categories = len(categories)
    n_rows = int(np.ceil(n_categories / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=sharey)
    axes = axes.flatten()  # Facilita la iteración sobre la cuadrícula

    # --- Creación de cada subgráfico ---
    for i, category in enumerate(categories):
        ax = axes[i]
        df_subset = df[df[facet_by] == category]

        ax.hist(df_subset[value_col], bins=bins, range=xlim, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_title(f"{facet_by.capitalize()}: {category}")
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Añadir línea vertical para la media (opcional)
        if show_mean:
            mean_val = df_subset[value_col].mean()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Media: {mean_val:.2f}')
            ax.legend()

    # --- Limpieza y etiquetas ---
    # Eliminar subgráficos vacíos si los hay
    for i in range(n_categories, len(axes)):
        axes[i].set_visible(False)

    # Añadir etiquetas de forma inteligente para no saturar
    for i, ax in enumerate(axes):
        if i >= n_categories: continue
        if i % n_cols == 0:  # Primera columna
            ax.set_ylabel(ylabel)
        if i // n_cols == n_rows - 1:  # Última fila
            ax.set_xlabel(xlabel if xlabel else value_col)

    # Título general
    fig.suptitle(title if title else f'Distribución de {value_col} por {facet_by}', fontsize=16, weight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()