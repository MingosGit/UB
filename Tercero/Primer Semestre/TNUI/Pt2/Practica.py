import os
import math
import numpy as np
import pandas as pd
import datetime
import itertools
from tqdm import trange, tqdm
import matplotlib.pyplot as plt

def mat_users_movies_rating():
    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_table('ml-1m/users.dat', sep='::', header=None, names=unames, engine='python')
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_table('ml-1m/ratings.dat', sep='::', header=None, names=rnames, engine='python')
    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_table('ml-1m/movies.dat', sep='::', header=None, names=mnames, engine='python', encoding='latin-1')
    return users, ratings, movies

def fusiona(users, ratings, movies):
    data = pd.merge(pd.merge(ratings, users), movies)
    return data
def taula_de_usuari(data):
    return data[data['user_id'] == 2]

def files(data,a,b):
    return data.iloc[a:b]
def percentatge(data,columna,valor,evaluacio):
    """Calcula el percentatge d'elements en una columna que tenen un valor específic
    columna: Nom de la columna a avaluar (Gender)
    valor: Valor específic a comptar (F)
    evaluacio: Nom de la columna sobre la qual es calcula el percentatge (rating)
    """
    return data[data[columna]==valor][evaluacio].count()/float(data[evaluacio].count())*100, '%'

def quantitatd(data,columna):
    """Calcula la quantitat d'elements únics en una columna específica
    columna: Nom de la columna a avaluar (title)
    :return : Sèrie de Pandas amb la quantitat d'elements únics
    """
    return data.groupby(columna).size()

def data_superior_a(data,B):
    """Retorna els índexs de les pel·lícules amb una valoració mitjana superior o igual a B
"""
    return data.index[data >= B]

def users_per_movie(data,movie_id):
    """Retorna una llista amb els identificadors dels usuaris que han valorat una pel·lícula específica
    movie_id: Identificador de la pel·lícula
    :return : Llista d'identificadors d'usuaris
    """
    return data.columns[data.loc[movie_id].notna()].tolist()
def users_per_movie_count(data,movie_id):
    """Retorna la quantitat d'usuaris que han valorat una pel·lícula específica
    movie_id: Identificador de la pel·lícula
    :return : Quantitat d'usuaris
    """
    return data.loc[movie_id].count()
def users_mean_rating(data):
    """Retorna la valoració mitjana dels usuaris
    :return : Valoració mitjana (float)
    """
    return  data.groupby('user_id')['rating'].mean()
def best_movie(data):
    """Retorna la pel·lícula amb la millor valoració per a un usuari específic
    usr: Identificador de l'usuari
    :return : Títol de la pel·lícula amb la millor valoració
    """
    return data.groupby('title')['rating'].mean().sort_values().tail(1).index[0] 
def best_movie_rating_maxviews(data):
    """Retorna la pel·lícula amb la millor valoració mitjana i el major nombre"""
    best_movie_rating_maxviews = data.groupby('title')['rating']
    return best_movie_rating_maxviews.agg(['mean', 'size']).sort_values(by=['mean', 'size']).tail(1)
def best_movie_by_user(data, usr):
    udata = data[data['user_id'] == usr]
    return udata.sort_values(by=['rating','movie_id'], ascending=[False, True]).iloc[0]["title"]

def build_counts_table(df):
    """
    Retorna un dataframe on les columnes són els `movie_id`, les files `user_id` i els valors
    la valoració que un usuari ha donat a una peli d'un `movie_id`
    
    :param df: DataFrame original 
    :return: DataFrame descrit adalt
    """
    
    # la vostra solució aquí
    df_counts = df.pivot_table(values='rating', index='user_id', columns='movie_id', aggfunc='mean')
    return df_counts
def get_count(df, user_id, movie_id):
    """
    Retorna la valoració que l'usuari 'user_id' ha donat de 'movie_id'
    
    :param df: DataFrame retornat per `build_counts_table`
    :param user_id: ID de l'usuari
    :param movie_id: ID de la peli
    :return: Enter amb la valoració de la peli
    """
    
    # la vostra solució aquí
    return df.loc[user_id, movie_id]
def reindexar(data,col1,col2):
    data[col1] = pd.Categorical(data[col1]).codes
    data[col2] = pd.Categorical(data[col2]).codes
    return data

def load_data(data_folder='ml-1m'):
    """
    Carga los datos de MovieLens 1M desde los ficheros .dat.
    
    :param data_folder: Ruta a la carpeta que contiene los archivos.
    :return: Tres DataFrames: users, ratings, movies.
    """
    # Definición de columnas según la documentación de MovieLens 1M
    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    mnames = ['movie_id', 'title', 'genres']

    # Lectura de Usuarios
    # Se usa engine='python' porque el separador '::' es multi-caracter
    users = pd.read_table(f'{data_folder}/users.dat', 
                          sep='::', 
                          header=None, 
                          names=unames, 
                          engine='python')

    # Lectura de Ratings
    ratings = pd.read_table(f'{data_folder}/ratings.dat', 
                            sep='::', 
                            header=None, 
                            names=rnames, 
                            engine='python')

    # Lectura de Películas
    # Se requiere encoding='latin-1' para títulos con caracteres especiales
    movies = pd.read_table(f'{data_folder}/movies.dat', 
                           sep='::', 
                           header=None, 
                           names=mnames, 
                           engine='python', 
                           encoding='latin-1')

    return users, ratings, movies


def distEuclid(x, y):
    """
    Retorna la distancia euclidiana de dos vectors n-dimensionals.
    
    :param x: Primer vector
    :param y: Segon vector
    :return : Escalar (float) corresponent a la distancia euclidiana
    """
    
    # la vostra solució aquí
    return np.sqrt(np.sum(np.power(x - y, 2)))


def simEuclid(Vec1, Vec2, norm):
    """
    Retorna la sembalça de dos vectors.
    
    :param Vec1: Primer vector
    :param Vec2: Segon vector
    :return : Escalar (float) corresponent a la semblança
    """
    # la vostra solució aquí
    if len(Vec1) == 0 or len(Vec2) == 0:
        return 0
    
    dist = distEuclid(Vec1, Vec2)
    similarity = (1 / (1 + dist)) * (len(Vec1) / norm)
    return similarity


def simUsuaris(DataFrame, User1, User2):
    """
    Retorna un score que representa la similitud entre user1 i user2 basada en la distancia euclidiana
    
    :param DataFrame: dataframe que conté totes les dades
    :param User1: id user1
    :param User2: id user2
    :return : Escalar (float) corresponent al score
    """
    
    # 1. Obtenir num_movies directament de la forma de la matriu (molt ràpid)
    #    Com que DataFrame és la pivot table, el nombre de columnes = nombre de pel·lícules.
    num_movies = DataFrame.shape[1]
    
    # 2. Obtenir els vectors
    vec1 = DataFrame.loc[User1]
    vec2 = DataFrame.loc[User2]
    
    # 3. Màscara de pel·lícules en comú
    common_mask = ~(np.isnan(vec1) | np.isnan(vec2))
    
    # 4. Filtre de mínim de coincidències
    if common_mask.sum() < 8: 
        return 0
        
    # 5. Extreure valors
    vec1_common = vec1[common_mask].values
    vec2_common = vec2[common_mask].values
    
    # 6. Calcular
    return simEuclid(vec1_common, vec2_common, num_movies)

def find_similar_users(DataFrame, sim_mx, userID, m):
    # la vostra solució aquí
    
    # 1. Obtenir el vector de similituds del 'userID' amb TOTS els altres
    #    Això és simplement seleccionar una fila de la matriu. És instantani.
    scores = sim_mx[userID]
    
    # 2. Per evitar seleccionar l'usuari mateix, fem una còpia del vector
    #    i posem la seva auto-similitud a un valor molt baix (-infinit).
    scores_copy = np.copy(scores)
    scores_copy[userID] = -np.inf
    
    # 3. Obtenir els ÍNDEXS (que són els user_id) dels 'm' valors més alts.
    #    np.argsort() ordena de menor a major i retorna els índexs.
    #    Agafem els últims 'm' índexs (els més alts) i els invertim [::-1].
    top_m_indices = np.argsort(scores_copy)[-m:][::-1]
    
    # 4. Obtenir els scores de similitud reals d'aquests 'm' índexs
    top_m_scores = scores[top_m_indices]
    
    # 5. Sumar només aquests 'm' scores per normalitzar
    total_sum_top_m = np.sum(top_m_scores)
    
    # 6. Crear el diccionari final normalitzat
    normalized_similar_users = {}
    
    if total_sum_top_m == 0:
        # Cas extrem: si tots els veïns tenen similitud 0
        for user_idx in top_m_indices:
            normalized_similar_users[user_idx] = 0.0
    else:
        # Cas normal: dividim cada score per la suma total
        for user_idx, score in zip(top_m_indices, top_m_scores):
            normalized_similar_users[user_idx] = score / total_sum_top_m
            
    return normalized_similar_users

def compute_similitude(fixed_arr, var_arr):
    """
    Donats dos vectors, calcula la similitud entre els subvectors formats 
    pels elements en comú (sense fer servir cap iteració!). 
    Normalitzeu la sortida multiplicant pel nombre de pel·lícules vistes en comú i
    dividint pel nombre total de pelis del dataset
    """
    
    # la vostra solució aquí

    # 1. Crear una màscara booleana per trobar els ítems en comú
    #    ~np.isnan(arr) ens diu on NO hi ha un NaN (és a dir, s'ha puntuat)
    #    L'operador '&' (AND lògic) ens dóna els índexs on AMBDÓS usuaris han puntuat
    common_mask = ~np.isnan(fixed_arr) & ~np.isnan(var_arr)
    
    # 2. Comptar quants ítems hi ha en comú
    #    Sumar una màscara booleana tracta 'True' com 1 i 'False' com 0
    num_common = np.sum(common_mask)
    
    # 3. Cas extrem: Si no hi ha pel·lícules en comú, la similitud és 0
    if num_common < 8: #No ens interessa si només tenen 7 pelis en comú
        return 0.0
        
    # 4. Crear els subvectors només amb els ítems en comú
    vec1 = fixed_arr[common_mask]
    vec2 = var_arr[common_mask]
    
    # 5. Calcular la distància Euclidiana dels subvectors
    #    np.linalg.norm(vec1 - vec2) és la forma ràpida i vectoritzada de
    #    np.sqrt(np.sum(np.power(vec1 - vec2, 2)))
    dist = np.linalg.norm(vec1 - vec2)
    
    # 6. Calcular la similitud base (com en l'exercici 3.5)
    base_sim = 1.0 / (1.0 + dist)
    
    # 7. Aplicar la normalització demanada per l'enunciat
    
    return base_sim * (num_common / num_movies)
def similarity_matrix_2(DataFrame):
    """
    Retorna una matriu de mida M x M on cada posició 
    indica la similitud entre usuaris (resp. ítems).
    Substitueix els nand per 0.

    :return : Matriu numpy de mida M x M amb les similituds.
    """
    # la vostra solució aquí
    # Substituir NaN per 0
    data_filled = DataFrame.fillna(0).values
    
    # Calcular la matriu de distàncies euclidianes
    # Utilitzant la fórmula: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    
    # Normes al quadrat de cada vector
    norms_sq = np.sum(data_filled ** 2, axis=1, keepdims=True)
    
    # Producte escalar entre tots els vectors
    dot_product = np.dot(data_filled, data_filled.T)
    
    # Distàncies al quadrat
    dist_sq = norms_sq + norms_sq.T - 2 * dot_product
    
    # Evitar valors negatius per errors numèrics
    dist_sq = np.maximum(dist_sq, 0)
    
    # Distàncies
    distances = np.sqrt(dist_sq)
    
    # Similituds
    sim_matrix = 1 / (1 + distances)
    
    return sim_matrix
def weighted_average(DataFrame, user, sim_mx, m):
    """    
    :param DataFrame: dataframe que conté totes les dades
    :param user: usuari al qual fem la recomanació
    :param sim_mx: similarity_matrix
    :param m: nombre d'usuaris semblants a tenir en compte per les recomanacions
    :return: diccionari {peli_id: score predit}
    """
    # la vostra solució aquí
    # Obtenir els usuaris més similars
    similar_users = find_similar_users(DataFrame, sim_mx, user, m)
    
    # Crear un diccionari per guardar les puntuacions predites
    predictions = {}
    
    # Obtenir les pel·lícules que l'usuari ja ha vist
    user_movies = DataFrame.loc[user]
    watched_movies = user_movies[~user_movies.isna()].index.tolist()
    
    # Per cada pel·lícula
    for movie_id in DataFrame.columns:
        # Si l'usuari ja l'ha vista, saltar
        if movie_id in watched_movies:
            continue
        
        # Calcular la mitjana ponderada de les puntuacions dels usuaris similars
        weighted_sum = 0
        weight_sum = 0
        
        for similar_user, weight in similar_users.items():
            rating = DataFrame.loc[similar_user, movie_id]
            if not np.isnan(rating):
                weighted_sum += rating * weight
                weight_sum += weight
        
        # Si algun usuari similar ha vist la pel·lícula, calcular la predicció
        if weight_sum > 0:
            predictions[movie_id] = weighted_sum / weight_sum
    
    return predictions

def getRecommendationsUser(DataFrame, user, sim_mx, n, m):
    """    
    :param DataFrame: dataframe que conté totes les dades
    :param user: usuari al qual fem la recomanació
    :param sim_mx: similarity_function
    :param n: nombre de pelis a recomanar
    :param m: nombre d'usuaris semblants a tenir en compte per les recomanacions
    :return : dataframe de pel·licules amb els scores.
    """
    
    # la vostra solució aquí
    # Obtenir les prediccions
    predictions = weighted_average(DataFrame, user, sim_mx, m)
    
    # Ordenar per score i agafar les n millors
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]
    
    # Crear un DataFrame amb els resultats
    result = pd.DataFrame(sorted_predictions, columns=['movie_id', 'predicted_rating'])
    
    return result
# Funció auxiliar per manejar subconjunts d'usuaris
def find_similar_users_with_mapping(df_counts, sim_mx, userID, m):
    """
    Versió de find_similar_users que maneja correctament el mapeo d'índexs
    quan el df_counts no té índexs consecutius des de 0.
    
    :param df_counts: DataFrame amb les valoracions (índex = user_id)
    :param sim_mx: matriu de similitud (índexs numèrics consecutius)
    :param userID: ID de l'usuari (pot no ser consecutiu)
    :param m: nombre d'usuaris similars a retornar
    :return: diccionari {user_id: similitud normalitzada}
    """
    # Crear mapeo entre user_id i posició a la matriu
    user_ids = df_counts.index.tolist()
    
    # Trobar la posició del userID a la llista
    if userID not in user_ids:
        return {}
    
    user_pos = user_ids.index(userID)
    
    # Obtenir les similituds de l'usuari amb tots els altres
    user_similarities = sim_mx[user_pos, :]
    
    # Crear un diccionari amb els user_ids reals (no les posicions)
    similarities = {}
    for i, user_id in enumerate(user_ids):
        if user_id != userID:
            similarities[user_id] = user_similarities[i]
    
    # Ordenar per similitud i agafar els m primers
    sorted_similarities = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:m])
    
    # Normalitzar perquè sumin 1
    total = sum(sorted_similarities.values())
    if total > 0:
        normalized_similarities = {k: v/total for k, v in sorted_similarities.items()}
    else:
        normalized_similarities = sorted_similarities
    
    return normalized_similarities
# Versió millorada de evaluateRecommendations que maneja subconjunts d'usuaris
def evaluateRecommendations_v2(train, test, m, n):
    """
    Retorna l'error generat pel model (versió millorada per subconjunts)
    
    :param train: dataframe amb dades d'entrenament
    :param test: dataframe amb dades de test
    :param m: nombre d'usuaris que volem per fer la recomanació
    :param n: nombre de pelis a retornar (no utilitzat aquí)
    :return: Escalar (float) corresponent al MAE
    """
    # Construir la taula de counts del train
    df_counts_train = build_counts_table(train)
    
    # Calcular matriu de similitud
    sim_train = similarity_matrix_2(df_counts_train)
    errors = []
    
    # Per cada interacció del test
    for idx, row in test.iterrows():
        user_id = row['user_id']
        movie_id = row['movie_id']
        true_rating = row['rating']

        # Si el usuario o la película no existen en el train set, no podemos predecir
        if user_id not in df_counts_train.index or movie_id not in df_counts_train.columns:
            continue

        # Usar la funció que maneja correctament el mapeo d'índexs
        similar_users_dict = find_similar_users_with_mapping(df_counts_train, sim_train, user_id, m)
        
        weighted_sum = 0
        weight_sum = 0
        
        for similar_user, weight in similar_users_dict.items():
            rating = df_counts_train.loc[similar_user, movie_id]
            if not np.isnan(rating):
                weighted_sum += rating * weight
                weight_sum += weight
        
        if weight_sum > 0:
            predicted_rating = weighted_sum / weight_sum
            error = abs(true_rating - predicted_rating)
            errors.append(error)
    
    if len(errors) > 0:
        mae = np.mean(errors)
    else:
        mae = np.nan
    
    return mae
##AUXILIARS
def filtrar_por_categoria(data, col, categorias):
    """
    Devuelve las filas de 'data' cuya columna 'col' pertenece a 'categorias'.
    """
    return data[data[col].isin(categorias)]
def rendimiento_por_grupo(data, col_grupo, col_metric, grupos=None):
    """
    Calcula el promedio de 'col_metric' por cada grupo indicado.
    Si grupos es None, usa todos los valores presentes en col_grupo.
    """
    if grupos is None:
        grupos = data[col_grupo].unique()

    resultados = {}
    for g in grupos:
        subset = data[data[col_grupo] == g]
        resultados[g] = subset[col_metric].mean()
    return resultados
def comparar_modelos(resultados, modelos=None):
    """
    resultados: DataFrame con columnas ['modelo','metric']
    modelos: lista de modelos a evaluar (si None usa todos).
    """
    if modelos is None:
        modelos = resultados['modelo'].unique()

    comparativa = {}
    for m in modelos:
        subset = resultados[resultados['modelo'] == m]
        comparativa[m] = subset['metric'].mean()
    return comparativa

def precision_umbral(data, col_score, col_label, umbrales):
    """
    Calcula precisión para varios umbrales.
    """
    precisiones = {}
    for t in umbrales:
        pred = data[col_score] >= t
        precision = (pred == data[col_label]).mean()
        precisiones[t] = precision
    return precisiones
def top_n_recomendaciones(data, col_user, col_item, col_score, user_id, n=5):
    """
    Devuelve los N items con mayor score para el usuario dado.
    """
    subset = data[data[col_user] == user_id]
    return subset.sort_values(col_score, ascending=False).head(n)[col_item].tolist()
def recomendador_por_atributo(data, atributo, valor, col_score, n=5):
    """
    Filtra usuarios por atributo y genera un top-N de recomendaciones promedio.
    """
    subset = data[data[atributo] == valor]
    return subset.groupby('item')[col_score].mean().sort_values(ascending=False).head(n)

def sesgo_recomendador(data, col_grupo, col_score):
    """
    Devuelve la media del score por grupo para ver posibles sesgos.
    """
    return data.groupby(col_grupo)[col_score].mean().to_dict()
def metricas_por_subgrupo(data, col_grupo, col_pred, col_real, metrica_fn, grupos=None):
    if grupos is None:
        grupos = data[col_grupo].unique()

    out = {}
    for g in grupos:
        subset = data[data[col_grupo] == g]
        out[g] = metrica_fn(subset[col_real], subset[col_pred])
    return out
def normalizar_columna(data, col, metodo="minmax"):
    if metodo == "minmax":
        return (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    elif metodo == "zscore":
        return (data[col] - data[col].mean()) / data[col].std()
    else:
        raise ValueError("Método no reconocido")
def split_data(data, frac_train=0.8, seed=None):
    train = data.sample(frac=frac_train, random_state=seed)
    test = data.drop(train.index)
    return train, test
def resumen_multiple(data, cols_group, cols_value, agg_fn="mean"):
    return data.groupby(cols_group)[cols_value].agg(agg_fn)
def matriz_user_item(data, col_user, col_item, col_value):
    return data.pivot_table(index=col_user, columns=col_item, values=col_value)
def top_k_por_grupo(data, col_grupo, col_valor, k=3):
    return data.sort_values(col_valor, ascending=False).groupby(col_grupo).head(k)
def comparar_columnas(data, col1, col2, metrica_fn):
    return metrica_fn(data[col1], data[col2])
def binarizar(data, col, umbral):
    return (data[col] >= umbral).astype(int)
def conteo_categorias(data, col, categorias=None):
    if categorias is None:
        categorias = data[col].unique()
    return {c: (data[col] == c).sum() for c in categorias}
def diferencia_metricas(metricas_dict):
    """
    Devuelve la diferencia entre el valor máximo y mínimo.
    """
    vals = list(metricas_dict.values())
    return max(vals) - min(vals)
def fusionar(data1, data2, on, how="inner"):
    return data1.merge(data2, on=on, how=how)
def eliminar_outliers(data, col, k=1.5):
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - k * iqr
    high = q3 + k * iqr
    return data[(data[col] >= low) & (data[col] <= high)]
def cobertura_recomendador(recs, total_items):
    """
    recs: lista o serie de items recomendados.
    """
    return len(set(recs)) / total_items
def filtrar_por_rango(data, col, minimo=None, maximo=None):
    if minimo is not None:
        data = data[data[col] >= minimo]
    if maximo is not None:
        data = data[data[col] <= maximo]
    return data
def normalizar_por_usuario(data, col_user, col_score):
    return data.assign(
        score_norm = data.groupby(col_user)[col_score].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-9)
        )
    )
def cross_validation(data, k, modelo_fn, metrica_fn):
    folds = np.array_split(data.sample(frac=1), k)
    resultados = []

    for i in range(k):
        test = folds[i]
        train = pd.concat([folds[j] for j in range(k) if j != i])
        pred = modelo_fn(train, test)
        resultados.append(metrica_fn(test, pred))
    return resultados

def reemplazar_valores(data, col, mapa):
    return data[col].replace(mapa)
def serendipia(data, col_pred, col_popularidad, top_n):
    """
    Serendipia = recomendar items NO populares.
    """
    recs = data.nlargest(top_n, col_pred)
    return 1 - recs[col_popularidad].mean() / data[col_popularidad].max()
def guardar_metricas(metricas_dict, nombre, valor):
    metricas_dict[nombre] = valor
    return metricas_dict
def contar_usuarios(data, col_user, condicion_fn):
    """
    Cuenta cuántos usuarios cumplen una condición dada.

    :param data: DataFrame con datos.
    :param col_user: Nombre de la columna de usuario.
    :param condicion_fn: Función que recibe un grupo de usuario y devuelve True/False.
    :return: Número de usuarios que cumplen la condición.
    """
    return sum(condicion_fn(g) for _, g in data.groupby(col_user))

def distribucion_ratings(data, col_rating):
    """
    Calcula la distribución de frecuencias de los ratings.

    :param data: DataFrame con ratings.
    :param col_rating: Columna donde están los ratings numéricos.
    :return: Diccionario rating → frecuencia.
    """
    return data[col_rating].value_counts().to_dict()
def calcular_rmse(data, col_real, col_pred):
    """
    Calcula el RMSE entre valores reales y predicciones.

    :param data: DataFrame con datos.
    :param col_real: Columna con valores reales.
    :param col_pred: Columna con predicciones.
    :return: RMSE como float.
    """
    dif = data[col_real] - data[col_pred]
    return np.sqrt((dif ** 2).mean())
def calcular_mae(data, col_real, col_pred):
    """
    Calcula el MAE entre valores reales y predicciones.

    :param data: DataFrame con datos.
    :param col_real: Columna con valores reales.
    :param col_pred: Columna con predicciones.
    :return: MAE como float.
    """
    return (data[col_real] - data[col_pred]).abs().mean()
def filtrar_peliculas_con_min_ratings(data, col_movie, min_count):
    """
    Filtra las películas que tengan un número mínimo de valoraciones.

    :param data: DataFrame con ratings.
    :param col_movie: Columna que contiene las IDs de película.
    :param min_count: Número mínimo de ratings necesarios.
    :return: DataFrame filtrado.
    """
    counts = data[col_movie].value_counts()
    valid = counts[counts >= min_count].index
    return data[data[col_movie].isin(valid)]
def stats_por_grupo(data, col_group, col_value):
    """
    Calcula estadísticas básicas por grupos.

    :param data: DataFrame general.
    :param col_group: Columna para agrupar (ej: género, edad…).
    :param col_value: Columna numérica sobre la que calcular estadísticos.
    :return: DataFrame con media, std, min y max.
    """
    return data.groupby(col_group)[col_value].agg(['mean','std','min','max'])
def usuarios_de_pelicula(data, col_user, col_movie, movie_id):
    """
    Obtiene los IDs de usuarios que han visto una película dada.

    :param data: DataFrame con ratings.
    :param col_user: Columna de usuario.
    :param col_movie: Columna de películas.
    :param movie_id: ID de película.
    :return: Lista de usuarios que la han visto.
    """
    return data[data[col_movie] == movie_id][col_user].unique().tolist()
def normalizar_minmax(data, col):
    """
    Normaliza una columna a rango [0, 1].

    :param data: DataFrame.
    :param col: Nombre de la columna a normalizar.
    :return: Serie con valores normalizados.
    """
    return (data[col] - data[col].min()) / (data[col].max() - data[col].min())
def top_n(data, col_value, n=5):
    """
    Obtiene los N elementos superiores de una columna.

    :param data: DataFrame con datos.
    :param col_value: Columna numérica por la que ordenar.
    :param n: Número de elementos a devolver.
    :return: DataFrame ordenado con top N.
    """
    return data.sort_values(col_value, ascending=False).head(n)
def crear_id_mapping(ids):
    """
    Crea un diccionario para mapear IDs arbitrarios a índices consecutivos.

    :param ids: Lista o array de IDs originales.
    :return: Diccionario {id_original: nuevo_id}
    """
    return {id_val: i for i, id_val in enumerate(sorted(set(ids)))}
def aplicar_mapping(data, col, mapping):
    """
    Aplica un diccionario de mapeo a una columna.

    :param data: DataFrame.
    :param col: Columna a transformar.
    :param mapping: Diccionario {valor_original: valor_nuevo}.
    :return: Serie transformada con los valores mapeados.
    """
    return data[col].map(mapping)
def densidad_matriz(data, col_user, col_movie):
    """
    Calcula la densidad de la matriz user-item (porcentaje de celdas no vacías).

    :param data: DataFrame con ratings.
    :param col_user: Columna usuario.
    :param col_movie: Columna película.
    :return: Porcentaje de densidad entre 0 y 1.
    """
    n_users = data[col_user].nunique()
    n_movies = data[col_movie].nunique()
    n_entries = len(data)
    return n_entries / (n_users * n_movies)
def items_comunes(data, col_user, col_movie, user1, user2):
    """
    Devuelve las películas que ambos usuarios han valorado.

    :param data: DataFrame con ratings.
    :param col_user: Columna de usuario.
    :param col_movie: Columna de película.
    :param user1: Primer usuario.
    :param user2: Segundo usuario.
    :return: Lista de películas comunes.
    """
    u1 = set(data[data[col_user] == user1][col_movie])
    u2 = set(data[data[col_user] == user2][col_movie])
    return list(u1.intersection(u2))
def precision_recall(y_true, y_pred):
    """
    Calcula precisión y recall en un sistema binario.

    :param y_true: Lista/array con valores reales 0/1.
    :param y_pred: Lista/array con predicciones 0/1.
    :return: (precision, recall)
    """
    tp = sum((y_true == 1) & (y_pred == 1))
    fp = sum((y_true == 0) & (y_pred == 1))
    fn = sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    return precision, recall
def binarizar_ratings(data, col_rating, umbral):
    """
    Convierte ratings en 0/1 usando un umbral.

    :param data: DataFrame con ratings.
    :param col_rating: Columna numérica de rating.
    :param umbral: Valor mínimo para considerar rating positivo.
    :return: Serie binaria 0/1.
    """
    return (data[col_rating] >= umbral).astype(int)
def correlacion(data, col1, col2):
    """
    Calcula correlación de Pearson entre dos columnas.

    :param data: DataFrame.
    :param col1: Primera columna.
    :param col2: Segunda columna.
    :return: Valor de correlación.
    """
    return data[col1].corr(data[col2])
def rating_medio_por_usuario(data, col_user, col_rating):
    """
    Calcula el rating medio dado por cada usuario.

    :param data: DataFrame con ratings.
    :param col_user: Columna usuario.
    :param col_rating: Columna rating.
    :return: Serie user_id → rating_medio.
    """
    return data.groupby(col_user)[col_rating].mean()
def usuarios_con_pocas_valoraciones(data, col_user, min_ratings):
    """
    Devuelve usuarios que han valorado menos de min_ratings.

    :param data: DataFrame con ratings.
    :param col_user: Columna usuario.
    :param min_ratings: Mínimo requerido.
    :return: Lista de usuarios con pocos ratings.
    """
    counts = data[col_user].value_counts()
    return counts[counts < min_ratings].index.tolist()
def comparar_medias(data, col_group, col_value, grupoA, grupoB):
    """
    Compara medias de un valor entre dos grupos (ej. hombres vs mujeres).

    :param data: DataFrame.
    :param col_group: Columna categórica.
    :param col_value: Columna numérica.
    :param grupoA: Primer grupo.
    :param grupoB: Segundo grupo.
    :return: Diferencia de medias grupoA - grupoB.
    """
    return data[data[col_group]==grupoA][col_value].mean() - \
           data[data[col_group]==grupoB][col_value].mean()
def contar_items_en_topk(data, col_user, col_item, col_score, k):
    """
    Cuenta cuántas veces un ítem aparece en el top-K de usuarios.

    :param data: DataFrame con scores.
    :param col_user: Columna de usuario.
    :param col_item: Columna de ítems.
    :param col_score: Columna de score predicho.
    :param k: Valor K del top.
    :return: Diccionario item → número de apariciones.
    """
    topk = data.sort_values(col_score, ascending=False).groupby(col_user).head(k)
    return topk[col_item].value_counts().to_dict()
