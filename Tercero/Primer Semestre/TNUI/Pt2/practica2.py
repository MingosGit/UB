"""
Pràctica 2: Recomanador Heurístic
Sistema de recomanació col·laboratiu basat en usuaris


import practica2 as pt2

Autor Jose Candon

Exemple d'ús:
users, ratings, movies, data = pt2.load_movielens_data('ml-1m')
data = pt2.reindex_data(data)
df_counts = pt2.build_counts_table(data)
sim_matrix = pt2.similarity_matrix_2(df_counts)
recommendations = pt2.getRecommendationsUser(df_counts, user=1, sim_mx=sim_matrix, n=10, m=50)

Exemple us exercici extra:
a = pt2.exercici_extra_recomanacions_usuari(user_id=100, n=10, m=50, data_folder='ml-1m')

"""

import math
import numpy as np
import pandas as pd
import datetime
import itertools
from tqdm.notebook import trange, tqdm
import matplotlib.pyplot as plt


# ============================================================================
# 1. FUNCIONS DE CÀRREGA DE DADES
# ============================================================================

def load_movielens_data(data_folder='ml-1m'):
    """
    Carrega les dades de MovieLens-1M des de fitxers .dat
    
    :param data_folder: carpeta on es troben els fitxers de dades
    :return: tuple (users, ratings, movies, data) amb els DataFrames carregats
    
    Exemple:
        >>> users, ratings, movies, data = load_movielens_data('ml-1m')
        >>> print(f"Usuaris: {len(users)}, Ratings: {len(ratings)}, Pelis: {len(movies)}")
    """
    # Llegir usuaris
    unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_table(f'{data_folder}/users.dat', sep='::', header=None, 
                         names=unames, engine='python')
    
    # Llegir ratings
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_table(f'{data_folder}/ratings.dat', sep='::', header=None, 
                           names=rnames, engine='python')
    
    # Llegir pel·lícules
    mnames = ['movie_id', 'title', 'genres']
    movies = pd.read_table(f'{data_folder}/movies.dat', sep='::', header=None, 
                          names=mnames, engine='python', encoding='latin-1')
    
    # Merge de totes les taules
    data = pd.merge(pd.merge(ratings, users), movies)
    
    return users, ratings, movies, data


def reindex_data(data):
    """
    Re-indexa els user_id i movie_id per tenir índexs consecutius començant per 0
    
    :param data: DataFrame amb les dades originals
    :return: DataFrame amb els índexs re-indexats
    
    Exemple:
        >>> users, ratings, movies, data = load_movielens_data('ml-1m')
        >>> data = reindex_data(data)
        >>> print(f"User IDs: {data['user_id'].min()} - {data['user_id'].max()}")
    """
    # Re-indexar user_id de 0 a N-1
    data['user_id'] = pd.Categorical(data['user_id']).codes
    
    # Re-indexar movie_id de 0 a M-1
    data['movie_id'] = pd.Categorical(data['movie_id']).codes
    
    return data


# ============================================================================
# 2. FUNCIONS D'EXPLORACIÓ I ANÀLISI DE DADES
# ============================================================================

def get_users_mean_rating(data):
    """
    Calcula la puntuació mitjana de cada usuari
    
    :param data: DataFrame amb totes les dades
    :return: Series amb la puntuació mitjana per usuari
    
    Exemple:
        >>> users_mean = get_users_mean_rating(data)
        >>> print(f"Puntuació mitjana usuari 0: {users_mean[0]:.2f}")
    """
    return data.groupby('user_id')['rating'].mean()


def get_best_movie_rating(data):
    """
    Retorna la pel·lícula més ben puntuada (en mitja). 
    En cas d'empat, retorna la que té ID més baix.
    
    :param data: DataFrame amb totes les dades
    :return: string amb el títol de la pel·lícula més ben puntuada
    
    Exemple:
        >>> best_movie = get_best_movie_rating(data)
        >>> print(f"Millor pel·lícula: {best_movie}")
    """
    # Calcular la puntuació mitjana de cada pel·lícula
    movie_mean_rating = data.groupby(['movie_id', 'title'])['rating'].mean().reset_index()
    
    # Trobar la puntuació màxima
    max_rating = movie_mean_rating['rating'].max()
    
    # Filtrar les pel·lícules amb la puntuació màxima
    best_movies = movie_mean_rating[movie_mean_rating['rating'] == max_rating]
    
    # Ordenar per movie_id i agafar la primera (ID més baix)
    best_movies = best_movies.sort_values('movie_id')
    best_movie_rating = best_movies.iloc[0]['title']
    
    return best_movie_rating


def get_best_movie_rating_maxviews(data):
    """
    Retorna la pel·lícula amb puntuació màxima que ha rebut més valoracions
    
    :param data: DataFrame amb totes les dades
    :return: string amb el títol de la pel·lícula
    
    Exemple:
        >>> best_popular = get_best_movie_rating_maxviews(data)
        >>> print(f"Millor pel·lícula (més vista): {best_popular}")
    """
    # Calcular puntuació mitjana
    movie_mean_rating = data.groupby(['movie_id', 'title'])['rating'].mean().reset_index()
    
    # Comptar valoracions
    movie_counts = data.groupby(['movie_id', 'title']).size().reset_index(name='count')
    
    # Fusionar
    movie_stats = movie_mean_rating.merge(movie_counts, on=['movie_id', 'title'])
    
    # Filtrar les que tenen puntuació màxima
    max_rating = movie_stats['rating'].max()
    best_rated = movie_stats[movie_stats['rating'] == max_rating]
    
    # Ordenar per nombre de valoracions (descendent) i després per movie_id (ascendent)
    best_rated = best_rated.sort_values(['count', 'movie_id'], ascending=[False, True])
    
    # Agafar la primera
    best_movie_rating_maxviews = best_rated.iloc[0]['title']
    
    return best_movie_rating_maxviews


def top_movie(dataFrame, usr):
    """
    Retorna la pel·lícula millor puntuada per un usuari. 
    En cas d'empat, retorna la que té ID més baix.
    
    :param dataFrame: DataFrame amb totes les dades
    :param usr: ID de l'usuari
    :return: string amb el títol de la pel·lícula
    
    Exemple:
        >>> top = top_movie(data, 0)
        >>> print(f"Pel·lícula preferida de l'usuari 0: {top}")
    """
    udata = dataFrame[dataFrame['user_id'] == usr]
    best = udata.sort_values(by=['rating', 'movie_id'], ascending=[False, True]).iloc[0]["title"]
    return best


# ============================================================================
# 3. FUNCIONS PER CONSTRUCCIÓ DE TAULES
# ============================================================================

def build_counts_table(df):
    """
    Retorna un dataframe on les columnes són els movie_id, les files user_id 
    i els valors la valoració que un usuari ha donat a una peli
    
    :param df: DataFrame original 
    :return: DataFrame descrit adalt
    
    Exemple:
        >>> df_counts = build_counts_table(data)
        >>> print(f"Shape: {df_counts.shape}")  # (num_users, num_movies)
    """
    df_counts = df.pivot_table(values='rating', index='user_id', 
                               columns='movie_id', aggfunc='mean')
    return df_counts


def get_count(df, user_id, movie_id):
    """
    Retorna la valoració que l'usuari user_id ha donat de movie_id
    
    :param df: DataFrame retornat per build_counts_table
    :param user_id: ID de l'usuari
    :param movie_id: ID de la peli
    :return: Enter amb la valoració de la peli
    
    Exemple:
        >>> rating = get_count(df_counts, 0, 100)
        >>> print(f"Usuari 0 va donar {rating} a la peli 100")
    """
    return df.loc[user_id, movie_id]


# ============================================================================
# 4. FUNCIONS DE SIMILITUD (DISTÀNCIA EUCLIDIANA)
# ============================================================================

def distEuclid(x, y):
    """
    Retorna la distancia euclidiana de dos vectors n-dimensionals.
    
    :param x: Primer vector
    :param y: Segon vector
    :return: Escalar (float) corresponent a la distancia euclidiana
    
    Exemple:
        >>> vec1 = np.array([1, 2, 3])
        >>> vec2 = np.array([4, 5, 6])
        >>> dist = distEuclid(vec1, vec2)
        >>> print(f"Distància: {dist:.2f}")  # 5.20
    """
    return np.sqrt(np.sum(np.power(x - y, 2)))


def simEuclid(Vec1, Vec2, norm):
    """
    Retorna la semblança de dos vectors.
    
    :param Vec1: Primer vector
    :param Vec2: Segon vector
    :param norm: Factor de normalització
    :return: Escalar (float) corresponent a la semblança
    
    Exemple:
        >>> vec1 = np.array([5, 4, 3])
        >>> vec2 = np.array([5, 3, 4])
        >>> sim = simEuclid(vec1, vec2, norm=100)
        >>> print(f"Similitud: {sim:.4f}")
    """
    if len(Vec1) == 0 or len(Vec2) == 0:
        return 0
    
    dist = distEuclid(Vec1, Vec2)
    similarity = (1 / (1 + dist)) * (len(Vec1) / norm)
    return similarity


def simUsuaris(DataFrame, User1, User2, num_movies):
    """
    Retorna un score que representa la similitud entre user1 i user2 
    basada en la distancia euclidiana
    
    :param DataFrame: dataframe que conté totes les dades
    :param User1: id user1
    :param User2: id user2
    :param num_movies: nombre total de pel·lícules (per normalització)
    :return: Escalar (float) corresponent al score
    
    Exemple:
        >>> num_movies = get_num_unique_movies(data)
        >>> similarity = simUsuaris(df_counts, 0, 5, num_movies)
        >>> print(f"Similitud entre usuaris 0 i 5: {similarity:.4f}")
    """
    # Obtenir els vectors de puntuacions dels dos usuaris
    vec1 = DataFrame.loc[User1]
    vec2 = DataFrame.loc[User2]
    
    # Trobar les pel·lícules en comú (no NaN en ambdós vectors)
    common_mask = ~(np.isnan(vec1) | np.isnan(vec2))
    if common_mask.sum() < 8:  # Si no tenen 8 pelis en comú, no es pot fer una bona estimació
        return 0
    
    # Extreure els valors comuns
    vec1_common = vec1[common_mask].values
    vec2_common = vec2[common_mask].values
    
    # Calcular la similitud
    return simEuclid(vec1_common, vec2_common, num_movies)


def compute_similitude(fixed_arr, var_arr, num_movies):
    """
    Donats dos vectors, calcula la similitud entre els subvectors formats 
    pels elements en comú (sense fer servir cap iteració!). 
    Normalitza la sortida multiplicant pel nombre de pel·lícules vistes en comú 
    i dividint pel nombre total de pelis del dataset
    
    :param fixed_arr: Primer vector
    :param var_arr: Segon vector
    :param num_movies: Nombre total de pel·lícules
    :return: Similitud normalitzada
    
    Exemple:
        >>> arr1 = df_counts.iloc[0].values
        >>> arr2 = df_counts.iloc[5].values
        >>> num_movies = get_num_unique_movies(data)
        >>> sim = compute_similitude(arr1, arr2, num_movies)
        >>> print(f"Similitud: {sim:.4f}")
    """
    # Crear una màscara booleana per trobar els ítems en comú
    common_mask = ~np.isnan(fixed_arr) & ~np.isnan(var_arr)
    
    # Comptar quants ítems hi ha en comú
    num_common = np.sum(common_mask)
    
    # Cas extrem: Si no hi ha pel·lícules en comú, la similitud és 0
    if num_common < 8:  # No ens interessa si només tenen 7 pelis en comú
        return 0.0
        
    # Crear els subvectors només amb els ítems en comú
    vec1 = fixed_arr[common_mask]
    vec2 = var_arr[common_mask]
    
    # Calcular la distància Euclidiana dels subvectors
    dist = np.linalg.norm(vec1 - vec2)
    
    # Calcular la similitud base
    base_sim = 1.0 / (1.0 + dist)
    
    # Aplicar la normalització
    return base_sim * (num_common / num_movies)


# ============================================================================
# 5. FUNCIONS PER MATRIUS DE SIMILITUD
# ============================================================================

def similarity_matrix_1(compute_distance, df_counts, num_movies):
    """
    Retorna una matriu de mida M x M on cada posició 
    indica la similitud entre usuaris.
    
    :param compute_distance: funció per calcular la distància
    :param df_counts: df amb els valor que cada usuari li ha donat a una peli
    :param num_movies: nombre total de pel·lícules
    :return: Matriu numpy de mida M x M amb les similituds
    
    Exemple:
        >>> num_movies = get_num_unique_movies(data)
        >>> sim_mx = similarity_matrix_1(compute_similitude, df_counts, num_movies)
        >>> print(f"Matriu de similitud: {sim_mx.shape}")
    """
    num_users = len(df_counts)
    sim_matrix = np.zeros((num_users, num_users))
    
    for i in range(num_users):
        for j in range(num_users):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                sim_matrix[i, j] = compute_distance(df_counts.iloc[i].values, 
                                                   df_counts.iloc[j].values, 
                                                   num_movies)
    
    return sim_matrix


def similarity_matrix_2(DataFrame):
    """
    Retorna una matriu de mida M x M on cada posició 
    indica la similitud entre usuaris.
    Substitueix els NaN per 0 i usa operacions matricials.

    :param DataFrame: DataFrame amb les valoracions
    :return: Matriu numpy de mida M x M amb les similituds
    
    Exemple:
        >>> sim_mx = similarity_matrix_2(df_counts)
        >>> print(f"Similitud usuaris 0-1: {sim_mx[0, 1]:.4f}")
    """
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


# ============================================================================
# 6. FUNCIONS PER TROBAR USUARIS SIMILARS
# ============================================================================

def find_similar_users(DataFrame, sim_mx, userID, m):
    """
    Retorna un diccionari dels m usuaris més similars amb scores normalitzades
    
    :param DataFrame: dataframe que conté totes les dades
    :param sim_mx: matriu de similitud
    :param userID: usuari respecte al qual fem la recomanació
    :param m: nombre d'usuaris que volem per fer la recomanació
    :return: dictionary amb {user_id: similitud normalitzada}
    
    Exemple:
        >>> similar = find_similar_users(df_counts, sim_mx, userID=0, m=10)
        >>> print(f"10 usuaris més similars a 0: {list(similar.keys())}")
    """
    # Obtenir el vector de similituds del userID amb tots els altres
    scores = sim_mx[userID]
    
    # Per evitar seleccionar l'usuari mateix
    scores_copy = np.copy(scores)
    scores_copy[userID] = -np.inf
    
    # Obtenir els índexs dels m valors més alts
    top_m_indices = np.argsort(scores_copy)[-m:][::-1]
    
    # Obtenir els scores de similitud reals
    top_m_scores = scores[top_m_indices]
    
    # Sumar només aquests m scores per normalitzar
    total_sum_top_m = np.sum(top_m_scores)
    
    # Crear el diccionari final normalitzat
    normalized_similar_users = {}
    
    if total_sum_top_m == 0:
        for user_idx in top_m_indices:
            normalized_similar_users[user_idx] = 0.0
    else:
        for user_idx, score in zip(top_m_indices, top_m_scores):
            normalized_similar_users[user_idx] = score / total_sum_top_m
            
    return normalized_similar_users


def find_similar_users_with_mapping(df_counts, sim_mx, userID, m):
    """
    Versió de find_similar_users que maneja correctament el mapeo d'índexs
    quan el df_counts no té índexs consecutius des de 0.
    
    :param df_counts: DataFrame amb les valoracions (índex = user_id)
    :param sim_mx: matriu de similitud (índexs numèrics consecutius)
    :param userID: ID de l'usuari (pot no ser consecutiu)
    :param m: nombre d'usuaris similars a retornar
    :return: diccionari {user_id: similitud normalitzada}
    
    Exemple:
        >>> # Per datasets amb índexs no consecutius
        >>> similar = find_similar_users_with_mapping(df_counts_subset, sim_mx, userID=42, m=5)
        >>> print(f"5 veïns més propers: {similar}")
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
    sorted_similarities = dict(sorted(similarities.items(), 
                                     key=lambda x: x[1], reverse=True)[:m])
    
    # Normalitzar perquè sumin 1
    total = sum(sorted_similarities.values())
    if total > 0:
        normalized_similarities = {k: v/total for k, v in sorted_similarities.items()}
    else:
        normalized_similarities = sorted_similarities
    
    return normalized_similarities


# ============================================================================
# 7. FUNCIONS DE RECOMANACIÓ
# ============================================================================

def weighted_average(DataFrame, user, sim_mx, m):
    """
    Calcula les puntuacions predites per totes les pel·lícules 
    usant una mitjana ponderada dels m usuaris més similars
    
    :param DataFrame: dataframe que conté totes les dades
    :param user: usuari al qual fem la recomanació
    :param sim_mx: similarity_matrix
    :param m: nombre d'usuaris semblants a tenir en compte
    :return: diccionari {peli_id: score predit}
    
    Exemple:
        >>> predictions = weighted_average(df_counts, user=0, sim_mx=sim_mx, m=50)
        >>> print(f"Prediccions per l'usuari 0: {len(predictions)} pel·lícules")
    """
    # Obtenim els m veïns més propers
    scores = sim_mx[user]
    scores_copy = np.copy(scores)
    scores_copy[user] = -np.inf
    
    top_m_indices = np.argsort(scores_copy)[-m:][::-1]
    top_m_scores = scores[top_m_indices].reshape(-1, 1)
    
    if np.sum(top_m_scores) == 0:
        return {}
        
    # Obtenir les puntuacions dels veïns
    ratings_matrix = DataFrame.values
    neighbor_ratings = ratings_matrix[top_m_indices, :]
    
    # Calcular el numerador: Σ [sim(u, v) * rating(v, i)]
    weighted_ratings = neighbor_ratings * top_m_scores
    numerator = np.nansum(weighted_ratings, axis=0)
    
    # Calcular el denominador: Σ [sim(u, v)] només pels veïns que han puntuat
    mask = ~np.isnan(neighbor_ratings)
    applicable_weights = top_m_scores * mask
    denominator = np.sum(applicable_weights, axis=0)
    
    # Càlcul final
    predicted_scores_vec = np.divide(numerator, denominator, 
                                     out=np.full_like(numerator, np.nan), 
                                     where=denominator!=0)
    
    # Retornar com a diccionari, ignorant els NaN
    final_predictions = {}
    for movie_id, score in enumerate(predicted_scores_vec):
        if not np.isnan(score):
            final_predictions[movie_id] = score
            
    return final_predictions


def getRecommendationsUser(DataFrame, user, sim_mx, n, m):
    """
    Retorna les n millors recomanacions per un usuari
    
    :param DataFrame: dataframe que conté totes les dades
    :param user: usuari al qual fem la recomanació
    :param sim_mx: matriu de similitud
    :param n: nombre de pelis a recomanar
    :param m: nombre d'usuaris semblants a tenir en compte
    :return: dataframe de pel·licules amb els scores predits
    
    Exemple:
        >>> recommendations = getRecommendationsUser(df_counts, user=0, sim_mx=sim_mx, n=10, m=50)
        >>> print(recommendations)
        # Output: DataFrame amb movie_id i predicted_rating de les 10 millors
    """
    # Obtenir les prediccions
    predictions = weighted_average(DataFrame, user, sim_mx, m)
    
    # Ordenar per score i agafar les n millors
    sorted_predictions = sorted(predictions.items(), 
                               key=lambda x: x[1], reverse=True)[:n]
    
    # Crear un DataFrame amb els resultats
    result = pd.DataFrame(sorted_predictions, 
                         columns=['movie_id', 'predicted_rating'])
    
    return result


# ============================================================================
# 8. FUNCIONS PER AVALUACIÓ (TRAIN/TEST SPLIT)
# ============================================================================

def create_train_test_split(data, test_size=0.1, random_state=42):
    """
    Crea un train/test split dels usuaris
    
    :param data: DataFrame amb totes les dades
    :param test_size: percentatge d'usuaris pel test (0.1 = 10%)
    :param random_state: seed per reproducibilitat
    :return: tuple (train_set, test_set)
    
    Exemple:
        >>> train, test = create_train_test_split(data, test_size=0.1)
        >>> print(f"Train: {len(train)}, Test: {len(test)}")
    """
    np.random.seed(random_state)
    
    all_users = data['user_id'].unique()
    num_test_users = int(len(all_users) * test_size)
    
    test_users = np.random.choice(all_users, size=num_test_users, replace=False)
    train_users = np.setdiff1d(all_users, test_users)
    
    test_set = data[data['user_id'].isin(test_users)]
    train_set = data[data['user_id'].isin(train_users)]
    
    return train_set, test_set


def add_testdata(traindf, test_set):
    """
    Afegeix el 80% de les interaccions del test_set al train_set
    i retorna el 20% restant com a nou test_set
    
    :param traindf: dataframe que conté les dades de train
    :param test_set: dataframe que conté les dades de test
    :return: tuple (train_final, test_final)
    
    Exemple:
        >>> train, test = create_train_test_split(data)
        >>> train_final, test_final = add_testdata(train, test)
        >>> print(f"Train final: {len(train_final)}, Test final: {len(test_final)}")
    """
    # Calcular el 20% de les interaccions per cada usuari de test
    groupby_count = test_set.groupby('user_id')['movie_id'].count() * 0.2
    
    # Llistes per guardar els frames
    frames_test = []
    frames_train = []
    
    # Per cada usuari del test set
    for idx in range(len(groupby_count)):
        n_test_samples = int(groupby_count.reset_index().iloc[idx]['movie_id'])
        u = groupby_count.reset_index().iloc[idx]['user_id']
        
        # Obtenir totes les interaccions de l'usuari
        test_set_user = test_set[test_set['user_id'] == u]
        
        # Seleccionar aleatòriament el 20% per test
        frame_test = test_set_user.sample(n_test_samples, random_state=42)
        frames_test.append(frame_test)
        
        # La resta va a train
        frame_train = test_set_user[~test_set_user.index.isin(frame_test.index)]
        frames_train.append(frame_train)
    
    # Concatenar tots els frames
    final_test = pd.concat(frames_test)
    train_from_test = pd.concat(frames_train)
    
    # Afegir el 80% del test al train original
    final_train = pd.concat([traindf, train_from_test])
    
    return final_train, final_test


# ============================================================================
# 9. FUNCIONS D'AVALUACIÓ (MAE)
# ============================================================================

def evaluateRecommendations(train, test, m, n):
    """
    Retorna l'error generat pel model usant MAE
    
    :param train: dataframe amb dades d'entrenament
    :param test: dataframe amb dades de test
    :param m: nombre d'usuaris que volem per fer la recomanació
    :param n: nombre de pelis a retornar (no utilitzat)
    :return: Escalar (float) corresponent al MAE
    
    Exemple:
        >>> mae = evaluateRecommendations(train, test, m=50, n=10)
        >>> print(f"MAE del model: {mae:.4f}")
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

        # Si l'usuari o la pel·lícula no existeixen en el train set
        if user_id not in df_counts_train.index or movie_id not in df_counts_train.columns:
            continue

        # Usar la matriu de similitud del train
        similar_users_dict = find_similar_users(df_counts_train, sim_train, user_id, m)
        
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


def evaluateRecommendations_v2(train, test, m, n):
    """
    Versió millorada que maneja subconjunts d'usuaris amb índexs no consecutius
    
    :param train: dataframe amb dades d'entrenament
    :param test: dataframe amb dades de test
    :param m: nombre d'usuaris que volem per fer la recomanació
    :param n: nombre de pelis a retornar (no utilitzat)
    :return: Escalar (float) corresponent al MAE
    
    Exemple:
        >>> # Per subconjunts (ex: només homes o dones)
        >>> mae = evaluateRecommendations_v2(train_males, test_males, m=50, n=10)
        >>> print(f"MAE (homes): {mae:.4f}")
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

        # Si l'usuari o la pel·lícula no existeixen en el train set
        if user_id not in df_counts_train.index or movie_id not in df_counts_train.columns:
            continue

        # Usar la funció que maneja correctament el mapeo d'índexs
        similar_users_dict = find_similar_users_with_mapping(df_counts_train, 
                                                             sim_train, user_id, m)
        
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


# ============================================================================
# 10. FUNCIONS AUXILIARS
# ============================================================================

def filter_by_gender(data, gender):
    """
    Filtra les dades per sexe
    
    :param data: DataFrame amb totes les dades
    :param gender: 'M' o 'F'
    :return: DataFrame filtrat
    
    Exemple:
        >>> data_males = filter_by_gender(data, 'M')
        >>> data_females = filter_by_gender(data, 'F')
        >>> print(f"Homes: {len(data_males)}, Dones: {len(data_females)}")
    """
    return data[data['gender'] == gender].copy()


def get_num_unique_movies(data):
    """
    Retorna el nombre de pel·lícules úniques al dataset
    
    :param data: DataFrame amb les dades
    :return: int amb el nombre de pel·lícules úniques
    
    Exemple:
        >>> num_movies = get_num_unique_movies(data)
        >>> print(f"Nombre de pel·lícules: {num_movies}")
    """
    return data['movie_id'].nunique()


def get_num_unique_users(data):
    """
    Retorna el nombre d'usuaris únics al dataset
    
    :param data: DataFrame amb les dades
    :return: int amb el nombre d'usuaris únics
    
    Exemple:
        >>> num_users = get_num_unique_users(data)
        >>> print(f"Nombre d'usuaris: {num_users}")
    """
    return data['user_id'].nunique()


# ============================================================================
# 11. FUNCIONS PER EXECUTAR ELS EXERCICIS
# ============================================================================

# ----------------------------------------------------------------------------
# EXERCICI A
# ----------------------------------------------------------------------------

def exercici_A(data_folder='ml-1m'):
    """
    Executa l'EXERCICI A: càlcul de puntuació mitjana d'usuaris i millor pel·lícula
    
    :param data_folder: carpeta amb les dades de MovieLens
    """
    print("="*80)
    print("EXERCICI A: Puntuació mitjana d'usuaris i millor pel·lícula")
    print("="*80)
    
    # Carregar dades
    users, ratings, movies, data = load_movielens_data(data_folder)
    
    # Puntuació mitjana de cada usuari
    users_mean_rating = get_users_mean_rating(data)
    print(f"\n1. Puntuació mitjana dels primers 5 usuaris:")
    print(users_mean_rating.head())
    
    # Millor pel·lícula puntuada
    best_movie_rating = get_best_movie_rating(data)
    print(f"\n2. Pel·lícula més ben puntuada: {best_movie_rating}")
    
    # Calcular quantes tenen puntuació màxima
    movie_mean_rating = data.groupby(['movie_id', 'title'])['rating'].mean().reset_index()
    max_rating = movie_mean_rating['rating'].max()
    best_movies = movie_mean_rating[movie_mean_rating['rating'] == max_rating]
    print(f"\n3. Pel·lícules amb puntuació màxima ({max_rating}):")
    print(best_movies.sort_values('movie_id'))
    
    # Millor pel·lícula amb més valoracions
    best_movie_rating_maxviews = get_best_movie_rating_maxviews(data)
    movie_counts = data.groupby(['movie_id', 'title']).size().reset_index(name='count')
    movie_stats = movie_mean_rating.merge(movie_counts, on=['movie_id', 'title'])
    best_rated = movie_stats[movie_stats['rating'] == max_rating].sort_values(['count', 'movie_id'], ascending=[False, True])
    
    print(f"\n4. Pel·lícula més ben puntuada amb més valoracions: {best_movie_rating_maxviews}")
    print(f"   Nombre de valoracions: {best_rated.iloc[0]['count']}")
    print("\n" + "="*80 + "\n")


# ----------------------------------------------------------------------------
# EXERCICI B
# ----------------------------------------------------------------------------

def exercici_B(data_folder='ml-1m'):
    """
    Executa l'EXERCICI B: funció top_movie per un usuari
    
    :param data_folder: carpeta amb les dades de MovieLens
    """
    print("="*80)
    print("EXERCICI B: Top movie per usuari")
    print("="*80)
    
    users, ratings, movies, data = load_movielens_data(data_folder)
    
    # Provar amb usuari 1
    top = top_movie(data, 1)
    print(f"\nPel·lícula preferida de l'usuari 1: {top}")
    print("\n" + "="*80 + "\n")


# ----------------------------------------------------------------------------
# EXERCICI C
# ----------------------------------------------------------------------------

def exercici_C(data_folder='ml-1m'):
    """
    Executa l'EXERCICI C: construcció de la taula df_counts
    
    :param data_folder: carpeta amb les dades de MovieLens
    """
    print("="*80)
    print("EXERCICI C: Construcció de df_counts")
    print("="*80)
    
    users, ratings, movies, data = load_movielens_data(data_folder)
    
    # Construir taula
    df_counts = build_counts_table(data)
    print(f"\nShape de df_counts: {df_counts.shape}")
    print(f"Primeres files i columnes:")
    print(df_counts.iloc[:5, :5])
    
    # Provar get_count
    rating = get_count(df_counts, 1, 1)
    print(f"\nValoració de l'usuari 1 a la pel·lícula 1: {rating}")
    print("\n" + "="*80 + "\n")


# ----------------------------------------------------------------------------
# EXERCICI D
# ----------------------------------------------------------------------------

def exercici_D(data_folder='ml-1m'):
    """
    Executa l'EXERCICI D: re-indexació dels IDs
    
    :param data_folder: carpeta amb les dades de MovieLens
    """
    print("="*80)
    print("EXERCICI D: Re-indexació dels IDs")
    print("="*80)
    
    users, ratings, movies, data = load_movielens_data(data_folder)
    
    print(f"\nAntes de re-indexar:")
    print(f"  User IDs: {data['user_id'].min()} - {data['user_id'].max()}")
    print(f"  Movie IDs: {data['movie_id'].min()} - {data['movie_id'].max()}")
    print(f"  Usuaris únics: {data['user_id'].nunique()}")
    print(f"  Pel·lícules úniques: {data['movie_id'].nunique()}")
    
    # Re-indexar
    data = reindex_data(data)
    
    print(f"\nDesprés de re-indexar:")
    print(f"  User IDs: {data['user_id'].min()} - {data['user_id'].max()}")
    print(f"  Movie IDs: {data['movie_id'].min()} - {data['movie_id'].max()}")
    print(f"  Usuaris únics: {data['user_id'].nunique()}")
    print(f"  Pel·lícules úniques: {data['movie_id'].nunique()}")
    
    # Reconstruir df_counts
    df_counts = build_counts_table(data)
    print(f"\ndf_counts actualitzat - Shape: {df_counts.shape}")
    print("\n" + "="*80 + "\n")


# ----------------------------------------------------------------------------
# EXERCICI E
# ----------------------------------------------------------------------------

def exercici_E(data_folder='ml-1m'):
    """
    Executa l'EXERCICI E: funcions de similitud
    
    :param data_folder: carpeta amb les dades de MovieLens
    """
    print("="*80)
    print("EXERCICI E: Funcions de similitud")
    print("="*80)
    
    users, ratings, movies, data = load_movielens_data(data_folder)
    data = reindex_data(data)
    df_counts = build_counts_table(data)
    num_movies = get_num_unique_movies(data)
    
    # Provar distància euclidiana
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([4, 5, 6])
    dist = distEuclid(vec1, vec2)
    print(f"\nDistància euclidiana entre [1,2,3] i [4,5,6]: {dist:.2f}")
    
    # Provar similitud
    vec1 = np.array([5, 4, 3])
    vec2 = np.array([5, 3, 4])
    sim = simEuclid(vec1, vec2, norm=100)
    print(f"Similitud entre [5,4,3] i [5,3,4] (norm=100): {sim:.4f}")
    
    # Provar similitud entre usuaris
    similarity = simUsuaris(df_counts, 0, 5, num_movies)
    print(f"\nSimilitud entre usuaris 0 i 5: {similarity:.4f}")
    
    similarity2 = simUsuaris(df_counts, 2, 314, num_movies)
    print(f"Similitud entre usuaris 2 i 314: {similarity2:.4f}")
    
    print("\n" + "="*80 + "\n")


# ----------------------------------------------------------------------------
# EXERCICI F - Subejercicios separados
# ----------------------------------------------------------------------------

def exercici_F_1(data_folder='ml-1m', show_matrix_time=False):
    """
    Executa l'EXERCICI F.1: trobar usuaris similars amb matriu de similitud
    
    :param data_folder: carpeta amb les dades de MovieLens
    :param show_matrix_time: si True, mostra el temps de càlcul de la matriu
    :return: tuple (df_counts, sim_mx) per usar en altres funcions
    """
    print("="*80)
    print("EXERCICI F.1 - TROBAR USUARIS SIMILARS")
    print("="*80)
    
    users, ratings, movies, data = load_movielens_data(data_folder)
    data = reindex_data(data)
    df_counts = build_counts_table(data)
    
    # Calcular matriu de similitud
    if show_matrix_time:
        print("\nCalculant matriu de similitud...")
        t = datetime.datetime.now()
    
    sim_mx = similarity_matrix_2(df_counts)
    
    if show_matrix_time:
        t_elapsed = datetime.datetime.now() - t
        print(f"Temps de càlcul de la matriu: {t_elapsed}")
    
    print(f"\nMatriu de similitud: {sim_mx.shape}")
    print(f"Similitud entre usuaris 0 i 1: {sim_mx[0, 1]:.4f}")
    
    # Trobar usuaris similars amb matriu
    print("\nTrobant 10 usuaris més similars a l'usuari 2...")
    t = datetime.datetime.now()
    sim_dict = find_similar_users(df_counts, sim_mx, 2, 10)
    t_elapsed = datetime.datetime.now() - t
    print(f"Temps: {t_elapsed}")
    print(f"\n10 usuaris més similars a l'usuari 2:")
    for user_id, score in list(sim_dict.items())[:5]:
        print(f"  Usuari {user_id}: {score:.4f}")
    print("  ...")
    
    print("\n" + "="*80 + "\n")
    return df_counts, sim_mx


def exercici_F_2(data_folder='ml-1m', df_counts=None, sim_mx=None):
    """
    Executa l'EXERCICI F.2: recomanacions per usuari
    
    :param data_folder: carpeta amb les dades de MovieLens
    :param df_counts: DataFrame precalculat (opcional)
    :param sim_mx: matriu de similitud precalculada (opcional)
    """
    print("="*80)
    print("EXERCICI F.2 - RECOMANACIONS PER USUARI")
    print("="*80)
    
    if df_counts is None or sim_mx is None:
        users, ratings, movies, data = load_movielens_data(data_folder)
        data = reindex_data(data)
        df_counts = build_counts_table(data)
        sim_mx = similarity_matrix_2(df_counts)
    
    t = datetime.datetime.now()
    recommendations = getRecommendationsUser(df_counts, 3, sim_mx, n=10, m=50)
    t_elapsed = datetime.datetime.now() - t
    print(f"\nTemps de càlcul: {t_elapsed}")
    print("\nTop 10 recomanacions per l'usuari 3:")
    print(recommendations)
    
    print("\n" + "="*80 + "\n")


def exercici_F(data_folder='ml-1m', show_matrix_time=False):
    """
    Executa l'EXERCICI F: sistema de recomanació col·laboratiu
    
    :param data_folder: carpeta amb les dades de MovieLens
    :param show_matrix_time: si True, mostra el temps de càlcul de la matriu
    """
    print("="*80)
    print("EXERCICI F: Sistema de recomanació col·laboratiu")
    print("="*80)
    
    users, ratings, movies, data = load_movielens_data(data_folder)
    data = reindex_data(data)
    df_counts = build_counts_table(data)
    num_movies = get_num_unique_movies(data)
    
    # F.1 - Trobar usuaris similars
    print("\n" + "="*60)
    print("F.1 - TROBAR USUARIS SIMILARS")
    print("="*60)
    
    # Calcular matriu de similitud
    if show_matrix_time:
        print("\nCalculant matriu de similitud...")
        t = datetime.datetime.now()
    
    sim_mx = similarity_matrix_2(df_counts)
    
    if show_matrix_time:
        t_elapsed = datetime.datetime.now() - t
        print(f"Temps de càlcul de la matriu: {t_elapsed}")
    
    print(f"\nMatriu de similitud: {sim_mx.shape}")
    print(f"Similitud entre usuaris 0 i 1: {sim_mx[0, 1]:.4f}")
    
    # Trobar usuaris similars amb matriu
    print("\nTrobant 10 usuaris més similars a l'usuari 2...")
    t = datetime.datetime.now()
    sim_dict = find_similar_users(df_counts, sim_mx, 2, 10)
    t_elapsed = datetime.datetime.now() - t
    print(f"Temps: {t_elapsed}")
    print(f"\n10 usuaris més similars a l'usuari 2:")
    for user_id, score in list(sim_dict.items())[:5]:
        print(f"  Usuari {user_id}: {score:.4f}")
    print("  ...")
    
    # F.2 - Recomanacions
    print("\n" + "="*60)
    print("F.2 - RECOMANACIONS PER USUARI")
    print("="*60)
    t = datetime.datetime.now()
    recommendations = getRecommendationsUser(df_counts, 3, sim_mx, n=10, m=50)
    t_elapsed = datetime.datetime.now() - t
    print(f"\nTemps de càlcul: {t_elapsed}")
    print("\nTop 10 recomanacions per l'usuari 3:")
    print(recommendations)
    
    print("\n" + "="*80 + "\n")


# ----------------------------------------------------------------------------
# EXERCICI G - Subejercicios separados
# ----------------------------------------------------------------------------

def exercici_G_1(data_folder='ml-1m'):
    """
    Executa l'EXERCICI G.1: Train/test split inicial (ingenu)
    
    :param data_folder: carpeta amb les dades de MovieLens
    :return: tuple (data, train_set, test_set)
    """
    print("="*80)
    print("EXERCICI G.1 - TRAIN/TEST SPLIT INICIAL (INGENU)")
    print("="*80)
    
    users, ratings, movies, data = load_movielens_data(data_folder)
    data = reindex_data(data)
    
    train_set, test_set = create_train_test_split(data, test_size=0.1, random_state=42)
    print(f"\nNombre d'usuaris en test_set: {test_set['user_id'].nunique()}")
    print(f"Nombre d'usuaris en train_set: {train_set['user_id'].nunique()}")
    print(f"Nombre de registres en test_set: {len(test_set)}")
    print(f"Nombre de registres en train_set: {len(train_set)}")
    print("\nProblema: Els usuaris de test NO estan en train!")
    print("No podem calcular similituds per usuaris desconeguts.")
    
    print("\n" + "="*80 + "\n")
    return data, train_set, test_set


def exercici_G_2(data=None, train_set=None, test_set=None, data_folder='ml-1m'):
    """
    Executa l'EXERCICI G.2: Train/test split millorat (80/20 per usuari)
    
    :param data: DataFrame complet (opcional)
    :param train_set: conjunt train inicial (opcional)
    :param test_set: conjunt test inicial (opcional)
    :param data_folder: carpeta amb les dades de MovieLens
    :return: tuple (train, test)
    """
    print("="*80)
    print("EXERCICI G.2 - TRAIN/TEST SPLIT MILLORAT (80/20 PER USUARI)")
    print("="*80)
    
    if data is None or train_set is None or test_set is None:
        users, ratings, movies, data = load_movielens_data(data_folder)
        data = reindex_data(data)
        train_set, test_set = create_train_test_split(data, test_size=0.1, random_state=42)
    
    print("\nSolució: Afegir 80% de les interaccions de cada usuari de test al train")
    print("i mantenir només el 20% per avaluar.\n")
    
    train, test = add_testdata(train_set, test_set)
    print(f"Train final: {len(train)} interaccions")
    print(f"Test final: {len(test)} interaccions")
    
    # Verificar
    assert train.shape[0] + test.shape[0] == data.shape[0]
    print("\n Verificació OK: train + test = total")
    print(f" Ara tots els usuaris de test també són al train!")
    
    print("\n" + "="*80 + "\n")
    return train, test


def exercici_G_3(train=None, test=None, data_folder='ml-1m'):
    """
    Executa l'EXERCICI G.3: Avaluació del model amb MAE
    
    :param train: conjunt train (opcional)
    :param test: conjunt test (opcional)
    :param data_folder: carpeta amb les dades de MovieLens
    :return: MAE calculat
    """
    print("="*80)
    print("EXERCICI G.3 - AVALUACIÓ DEL MODEL AMB MAE")
    print("="*80)
    
    if train is None or test is None:
        users, ratings, movies, data = load_movielens_data(data_folder)
        data = reindex_data(data)
        train_set, test_set = create_train_test_split(data, test_size=0.1, random_state=42)
        train, test = add_testdata(train_set, test_set)
    
    print("\nCalculant MAE (això pot trigar uns minuts...)")
    
    t = datetime.datetime.now()
    mae = evaluateRecommendations(train, test, m=50, n=10)
    t_elapsed = datetime.datetime.now() - t
    
    print(f"\nMAE del model: {mae:.4f}")
    print(f"Temps de càlcul: {t_elapsed}")
    print("\nInterpretació: Un MAE més baix indica millors prediccions.")
    print("Un MAE de {:.4f} significa que, en mitjana, l'error de predicció".format(mae))
    print("és de {:.2f} estrelles.".format(mae))
    
    print("\n" + "="*80 + "\n")
    return mae


def exercici_G(data_folder='ml-1m'):
    """
    Executa l'EXERCICI G: avaluació amb MAE
    
    :param data_folder: carpeta amb les dades de MovieLens
    """
    print("="*80)
    print("EXERCICI G: Avaluació amb MAE")
    print("="*80)
    
    users, ratings, movies, data = load_movielens_data(data_folder)
    data = reindex_data(data)
    
    # G.1 - Train/test split inicial
    print("\n" + "="*60)
    print("G.1 - TRAIN/TEST SPLIT INICIAL (INGENU)")
    print("="*60)
    train_set, test_set = create_train_test_split(data, test_size=0.1, random_state=42)
    print(f"\nNombre d'usuaris en test_set: {test_set['user_id'].nunique()}")
    print(f"Nombre d'usuaris en train_set: {train_set['user_id'].nunique()}")
    print(f"Nombre de registres en test_set: {len(test_set)}")
    print(f"Nombre de registres en train_set: {len(train_set)}")
    print("\nProblema: Els usuaris de test NO estan en train!")
    print("No podem calcular similituds per usuaris desconeguts.")
    
    # G.2 - Train/test split millorat
    print("\n" + "="*60)
    print("G.2 - TRAIN/TEST SPLIT MILLORAT (80/20 PER USUARI)")
    print("="*60)
    print("\nSolució: Afegir 80% de les interaccions de cada usuari de test al train")
    print("i mantenir només el 20% per avaluar.\n")
    
    train, test = add_testdata(train_set, test_set)
    print(f"Train final: {len(train)} interaccions")
    print(f"Test final: {len(test)} interaccions")
    
    # Verificar
    assert train.shape[0] + test.shape[0] == data.shape[0]
    print("\n Verificació OK: train + test = total")
    print(f" Ara tots els usuaris de test també són al train!")
    
    # G.3 - Avaluació MAE
    print("\n" + "="*60)
    print("G.3 - AVALUACIÓ DEL MODEL AMB MAE")
    print("="*60)
    print("\nCalculant MAE (això pot trigar uns minuts...)")
    
    t = datetime.datetime.now()
    mae = evaluateRecommendations(train, test, m=50, n=10)
    t_elapsed = datetime.datetime.now() - t
    
    print(f"\nMAE del model: {mae:.4f}")
    print(f"Temps de càlcul: {t_elapsed}")
    print("\nInterpretació: Un MAE més baix indica millors prediccions.")
    print("Un MAE de {:.4f} significa que, en mitjana, l'error de predicció".format(mae))
    print("és de {:.2f} estrelles.".format(mae))
    
    print("\n" + "="*80 + "\n")


# ----------------------------------------------------------------------------
# EXERCICI H - Subejercicios separados (OPCIONAL)
# ----------------------------------------------------------------------------

def exercici_H_1(data_folder='ml-1m'):
    """
    Executa l'EXERCICI H.1: MAE del recomanador únic
    
    :param data_folder: carpeta amb les dades de MovieLens
    :return: tuple (data, train, test, mae)
    """
    print("="*80)
    print("EXERCICI H.1 - MAE DEL RECOMANADOR ÚNIC")
    print("="*80)
    
    users, ratings, movies, data = load_movielens_data(data_folder)
    data = reindex_data(data)
    
    train_set, test_set = create_train_test_split(data, test_size=0.1, random_state=42)
    train, test = add_testdata(train_set, test_set)
    mae = evaluateRecommendations(train, test, m=50, n=10)
    print(f"\nMAE recomanador únic: {mae:.4f}")
    
    print("\n" + "="*80 + "\n")
    return data, train, test, mae


def exercici_H_2(data=None, data_folder='ml-1m'):
    """
    Executa l'EXERCICI H.2: Separar les dades per sexe
    
    :param data: DataFrame amb totes les dades (opcional)
    :param data_folder: carpeta amb les dades de MovieLens
    :return: tuple (data_male, data_female)
    """
    print("="*80)
    print("EXERCICI H.2 - SEPARAR LES DADES PER SEXE")
    print("="*80)
    
    if data is None:
        users, ratings, movies, data = load_movielens_data(data_folder)
        data = reindex_data(data)
    
    data_male = filter_by_gender(data, 'M')
    data_female = filter_by_gender(data, 'F')
    
    print(f"\nTotal d'interaccions homes: {len(data_male)}")
    print(f"Total d'interaccions dones: {len(data_female)}")
    print(f"Total usuaris homes: {get_num_unique_users(data_male)}")
    print(f"Total usuaris dones: {get_num_unique_users(data_female)}")
    
    print("\n" + "="*80 + "\n")
    return data_male, data_female


def exercici_H_3(data_male=None, data_folder='ml-1m'):
    """
    Executa l'EXERCICI H.3: Conjunts de dades - HOMES
    
    :param data_male: DataFrame amb dades dels homes (opcional)
    :param data_folder: carpeta amb les dades de MovieLens
    :return: tuple (train_male, test_male)
    """
    print("="*80)
    print("EXERCICI H.3 - CONJUNTS DE DADES - HOMES")
    print("="*80)
    
    if data_male is None:
        users, ratings, movies, data = load_movielens_data(data_folder)
        data = reindex_data(data)
        data_male = filter_by_gender(data, 'M')
    
    np.random.seed(42)
    all_users_male = data_male['user_id'].unique()
    num_test_users_male = int(len(all_users_male) * 0.1)
    test_users_male = np.random.choice(all_users_male, size=num_test_users_male, replace=False)
    train_users_male = np.setdiff1d(all_users_male, test_users_male)
    
    test_set_male = data_male[data_male['user_id'].isin(test_users_male)]
    train_set_male = data_male[data_male['user_id'].isin(train_users_male)]
    train_male, test_male = add_testdata(train_set_male, test_set_male)
    
    print(f"\nTrain homes: {train_male.shape[0]} interaccions")
    print(f"Test homes: {test_male.shape[0]} interaccions")
    
    print("\n" + "="*80 + "\n")
    return train_male, test_male


def exercici_H_4(data_female=None, data_folder='ml-1m'):
    """
    Executa l'EXERCICI H.4: Conjunts de dades - DONES
    
    :param data_female: DataFrame amb dades de les dones (opcional)
    :param data_folder: carpeta amb les dades de MovieLens
    :return: tuple (train_female, test_female)
    """
    print("="*80)
    print("EXERCICI H.4 - CONJUNTS DE DADES - DONES")
    print("="*80)
    
    if data_female is None:
        users, ratings, movies, data = load_movielens_data(data_folder)
        data = reindex_data(data)
        data_female = filter_by_gender(data, 'F')
    
    np.random.seed(42)
    all_users_female = data_female['user_id'].unique()
    num_test_users_female = int(len(all_users_female) * 0.1)
    test_users_female = np.random.choice(all_users_female, size=num_test_users_female, replace=False)
    train_users_female = np.setdiff1d(all_users_female, test_users_female)
    
    test_set_female = data_female[data_female['user_id'].isin(test_users_female)]
    train_set_female = data_female[data_female['user_id'].isin(train_users_female)]
    train_female, test_female = add_testdata(train_set_female, test_set_female)
    
    print(f"\nTrain dones: {train_female.shape[0]} interaccions")
    print(f"Test dones: {test_female.shape[0]} interaccions")
    
    print("\n" + "="*80 + "\n")
    return train_female, test_female


def exercici_H_5(train_male=None, test_male=None, data_folder='ml-1m'):
    """
    Executa l'EXERCICI H.5: Avaluació - Recomanador HOMES
    
    :param train_male: conjunt train homes (opcional)
    :param test_male: conjunt test homes (opcional)
    :param data_folder: carpeta amb les dades de MovieLens
    :return: mae_male
    """
    print("="*80)
    print("EXERCICI H.5 - AVALUACIÓ - RECOMANADOR HOMES")
    print("="*80)
    
    if train_male is None or test_male is None:
        users, ratings, movies, data = load_movielens_data(data_folder)
        data = reindex_data(data)
        data_male = filter_by_gender(data, 'M')
        
        np.random.seed(42)
        all_users_male = data_male['user_id'].unique()
        num_test_users_male = int(len(all_users_male) * 0.1)
        test_users_male = np.random.choice(all_users_male, size=num_test_users_male, replace=False)
        train_users_male = np.setdiff1d(all_users_male, test_users_male)
        
        test_set_male = data_male[data_male['user_id'].isin(test_users_male)]
        train_set_male = data_male[data_male['user_id'].isin(train_users_male)]
        train_male, test_male = add_testdata(train_set_male, test_set_male)
    
    t = datetime.datetime.now()
    mae_male = evaluateRecommendations_v2(train_male, test_male, m=50, n=10)
    t_male = datetime.datetime.now() - t
    print(f"\nMAE homes: {mae_male:.4f}")
    print(f"Temps de càlcul: {str(t_male)}")
    
    print("\n" + "="*80 + "\n")
    return mae_male


def exercici_H_6(train_female=None, test_female=None, data_folder='ml-1m'):
    """
    Executa l'EXERCICI H.6: Avaluació - Recomanador DONES
    
    :param train_female: conjunt train dones (opcional)
    :param test_female: conjunt test dones (opcional)
    :param data_folder: carpeta amb les dades de MovieLens
    :return: mae_female
    """
    print("="*80)
    print("EXERCICI H.6 - AVALUACIÓ - RECOMANADOR DONES")
    print("="*80)
    
    if train_female is None or test_female is None:
        users, ratings, movies, data = load_movielens_data(data_folder)
        data = reindex_data(data)
        data_female = filter_by_gender(data, 'F')
        
        np.random.seed(42)
        all_users_female = data_female['user_id'].unique()
        num_test_users_female = int(len(all_users_female) * 0.1)
        test_users_female = np.random.choice(all_users_female, size=num_test_users_female, replace=False)
        train_users_female = np.setdiff1d(all_users_female, test_users_female)
        
        test_set_female = data_female[data_female['user_id'].isin(test_users_female)]
        train_set_female = data_female[data_female['user_id'].isin(train_users_female)]
        train_female, test_female = add_testdata(train_set_female, test_set_female)
    
    t = datetime.datetime.now()
    mae_female = evaluateRecommendations_v2(train_female, test_female, m=50, n=10)
    t_female = datetime.datetime.now() - t
    print(f"\nMAE dones: {mae_female:.4f}")
    print(f"Temps de càlcul: {str(t_female)}")
    
    print("\n" + "="*80 + "\n")
    return mae_female


def exercici_H_7(mae=None, mae_male=None, mae_female=None, test_male=None, test_female=None, data_folder='ml-1m'):
    """
    Executa l'EXERCICI H.7: Comparació de resultats i anàlisi
    
    :param mae: MAE del recomanador únic (opcional)
    :param mae_male: MAE homes (opcional)
    :param mae_female: MAE dones (opcional)
    :param test_male: conjunt test homes (opcional)
    :param test_female: conjunt test dones (opcional)
    :param data_folder: carpeta amb les dades de MovieLens
    :return: tuple (mae, mae_male, mae_female, weighted_mae_gender)
    """
    print("="*80)
    print("EXERCICI H.7 - COMPARACIÓ DE RESULTATS I ANÀLISI")
    print("="*80)
    
    # Si no tenim els valors, els calculem
    if mae is None or mae_male is None or mae_female is None:
        users, ratings, movies, data = load_movielens_data(data_folder)
        data = reindex_data(data)
        
        # MAE únic
        train_set, test_set = create_train_test_split(data, test_size=0.1, random_state=42)
        train, test = add_testdata(train_set, test_set)
        mae = evaluateRecommendations(train, test, m=50, n=10)
        
        # Homes
        data_male = filter_by_gender(data, 'M')
        np.random.seed(42)
        all_users_male = data_male['user_id'].unique()
        num_test_users_male = int(len(all_users_male) * 0.1)
        test_users_male = np.random.choice(all_users_male, size=num_test_users_male, replace=False)
        train_users_male = np.setdiff1d(all_users_male, test_users_male)
        test_set_male = data_male[data_male['user_id'].isin(test_users_male)]
        train_set_male = data_male[data_male['user_id'].isin(train_users_male)]
        train_male, test_male = add_testdata(train_set_male, test_set_male)
        mae_male = evaluateRecommendations_v2(train_male, test_male, m=50, n=10)
        
        # Dones
        data_female = filter_by_gender(data, 'F')
        all_users_female = data_female['user_id'].unique()
        num_test_users_female = int(len(all_users_female) * 0.1)
        test_users_female = np.random.choice(all_users_female, size=num_test_users_female, replace=False)
        train_users_female = np.setdiff1d(all_users_female, test_users_female)
        test_set_female = data_female[data_female['user_id'].isin(test_users_female)]
        train_set_female = data_female[data_female['user_id'].isin(train_users_female)]
        train_female, test_female = add_testdata(train_set_female, test_set_female)
        mae_female = evaluateRecommendations_v2(train_female, test_female, m=50, n=10)
    
    print(f"\nMAE recomanador únic:        {mae:.4f}")
    print(f"MAE recomanador homes:       {mae_male:.4f}")
    print(f"MAE recomanador dones:       {mae_female:.4f}")
    print()
    
    # Calcular la mitjana ponderada
    total_test_interactions = len(test_male) + len(test_female)
    weighted_mae_gender = (mae_male * len(test_male) + mae_female * len(test_female)) / total_test_interactions
    print(f"MAE ponderat (per sexe):     {weighted_mae_gender:.4f}")
    print()
    
    # Anàlisi
    print("ANÀLISI I CONCLUSIONS:")
    print("="*60)
    if weighted_mae_gender < mae:
        diff = mae - weighted_mae_gender
        millora_percentual = (diff / mae) * 100
        print(f" Els recomanadors separats per sexe són MILLORS")
        print(f"  - Millora absoluta: {diff:.4f}")
        print(f"  - Millora percentual: {millora_percentual:.2f}%")
    else:
        diff = weighted_mae_gender - mae
        empitjorament_percentual = (diff / mae) * 100
        print(f"✗ El recomanador únic és MILLOR")
        print(f"  - Empitjorament absolut: {diff:.4f}")
        print(f"  - Empitjorament percentual: {empitjorament_percentual:.2f}%")
    
    print()
    print("INTERPRETACIÓ:")
    print("-" * 60)
    print("Un MAE més baix indica millors prediccions (menys error).")
    print()
    
    if weighted_mae_gender < mae:
        print("CONCLUSIÓ: Surt més a compte fer recomanadors separats")
        print("per sexe, ja que les preferències cinematogràfiques són")
        print("prou diferents entre homes i dones com per beneficiar-se")
        print("d'una segmentació del dataset.")
    else:
        print("CONCLUSIÓ: Surt més a compte fer un recomanador únic,")
        print("ja que la divisió per sexe redueix la mida del dataset")
        print("i la diversitat d'opinions, empitjorant les prediccions.")
    
    print()
    print("FACTORS A CONSIDERAR:")
    print("-" * 60)
    print("• Mida del dataset: Dividir redueix els usuaris disponibles")
    print("• Similitud intra-grup: Si homes/dones tenen preferències")
    print("  més homogènies dins del seu grup, la segmentació millora")
    print("• Cost computacional: Mantenir dos models és més costós")
    print("• Escalabilitat: Amb més categories (edat, ocupació...)")
    print("  els subconjunts serien massa petits")
    
    print("\n" + "="*80 + "\n")
    return mae, mae_male, mae_female, weighted_mae_gender


def exercici_H_8(mae=None, mae_male=None, mae_female=None, weighted_mae_gender=None, 
                test_male=None, test_female=None, data_folder='ml-1m'):
    """
    Executa l'EXERCICI H.8: Visualització dels resultats
    
    :param mae: MAE del recomanador únic (opcional)
    :param mae_male: MAE homes (opcional)
    :param mae_female: MAE dones (opcional)
    :param weighted_mae_gender: MAE ponderat (opcional)
    :param test_male: conjunt test homes (opcional)
    :param test_female: conjunt test dones (opcional)
    :param data_folder: carpeta amb les dades de MovieLens
    """
    print("="*80)
    print("EXERCICI H.8 - VISUALITZACIÓ DELS RESULTATS")
    print("="*80)
    
    # Si no tenim els valors, els calculem
    if mae is None or mae_male is None or mae_female is None or weighted_mae_gender is None:
        mae, mae_male, mae_female, weighted_mae_gender = exercici_H_7(data_folder=data_folder)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gràfic de barres per comparar MAE
    models = ['Recomanador\nÚnic', 'Recomanador\nHomes', 'Recomanador\nDones', 'MAE Ponderat\n(per sexe)']
    maes = [mae, mae_male, mae_female, weighted_mae_gender]
    colors = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']
    
    bars = ax1.bar(models, maes, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('MAE (Mean Absolute Error)', fontsize=12, fontweight='bold')
    ax1.set_title('Comparació de MAE entre Models', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Afegir valors sobre les barres
    for bar, mae_val in zip(bars, maes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{mae_val:.4f}',
                 ha='center', va='bottom', fontweight='bold')
    
    # Gràfic de millora/empitjorament percentual
    if weighted_mae_gender < mae:
        millora = ((mae - weighted_mae_gender) / mae) * 100
        label = f'Millora: {millora:.2f}%'
        color = '#2ecc71'
    else:
        millora = ((weighted_mae_gender - mae) / mae) * 100
        label = f'Empitjorament: {millora:.2f}%'
        color = '#e74c3c'
    
    ax2.barh(['Recomanadors\nper Sexe vs\nÚnic'], [millora], color=color, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Diferència Percentual (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Millora/Empitjorament Percentual', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    ax2.text(millora/2, 0, label, ha='center', va='center', 
             fontweight='bold', fontsize=11, color='white')
    
    plt.tight_layout()
    plt.show()
    
    print("\nVisualització completada!")
    print("\n" + "="*80 + "\n")


def exercici_H(data_folder='ml-1m'):
    """
    Executa l'EXERCICI H: comparació recomanador únic vs per sexe (OPCIONAL)
    
    :param data_folder: carpeta amb les dades de MovieLens
    """
    print("="*80)
    print("EXERCICI H (OPCIONAL): Recomanador únic vs per sexe")
    print("="*80)
    
    users, ratings, movies, data = load_movielens_data(data_folder)
    data = reindex_data(data)
    
    # H.1 - MAE del recomanador únic
    print("\n" + "="*60)
    print("H.1 - MAE DEL RECOMANADOR ÚNIC")
    print("="*60)
    train_set, test_set = create_train_test_split(data, test_size=0.1, random_state=42)
    train, test = add_testdata(train_set, test_set)
    mae = evaluateRecommendations(train, test, m=50, n=10)
    print(f"MAE recomanador únic: {mae:.4f}")
    print()
    
    # H.2 - Separar les dades per sexe
    print("="*60)
    print("H.2 - SEPARAR LES DADES PER SEXE")
    print("="*60)
    data_male = filter_by_gender(data, 'M')
    data_female = filter_by_gender(data, 'F')
    
    print(f"Total d'interaccions homes: {len(data_male)}")
    print(f"Total d'interaccions dones: {len(data_female)}")
    print(f"Total usuaris homes: {get_num_unique_users(data_male)}")
    print(f"Total usuaris dones: {get_num_unique_users(data_female)}")
    print()
    
    # H.3 - Crear train/test per homes
    print("="*60)
    print("H.3 - CONJUNTS DE DADES - HOMES")
    print("="*60)
    np.random.seed(42)
    all_users_male = data_male['user_id'].unique()
    num_test_users_male = int(len(all_users_male) * 0.1)
    test_users_male = np.random.choice(all_users_male, size=num_test_users_male, replace=False)
    train_users_male = np.setdiff1d(all_users_male, test_users_male)
    
    test_set_male = data_male[data_male['user_id'].isin(test_users_male)]
    train_set_male = data_male[data_male['user_id'].isin(train_users_male)]
    train_male, test_male = add_testdata(train_set_male, test_set_male)
    
    print(f"Train homes: {train_male.shape[0]} interaccions")
    print(f"Test homes: {test_male.shape[0]} interaccions")
    print()
    
    # H.4 - Crear train/test per dones
    print("="*60)
    print("H.4 - CONJUNTS DE DADES - DONES")
    print("="*60)
    all_users_female = data_female['user_id'].unique()
    num_test_users_female = int(len(all_users_female) * 0.1)
    test_users_female = np.random.choice(all_users_female, size=num_test_users_female, replace=False)
    train_users_female = np.setdiff1d(all_users_female, test_users_female)
    
    test_set_female = data_female[data_female['user_id'].isin(test_users_female)]
    train_set_female = data_female[data_female['user_id'].isin(train_users_female)]
    train_female, test_female = add_testdata(train_set_female, test_set_female)
    
    print(f"Train dones: {train_female.shape[0]} interaccions")
    print(f"Test dones: {test_female.shape[0]} interaccions")
    print()
    
    # H.5 - Avaluació recomanador HOMES
    print("="*60)
    print("H.5 - AVALUACIÓ - RECOMANADOR HOMES")
    print("="*60)
    t = datetime.datetime.now()
    mae_male = evaluateRecommendations_v2(train_male, test_male, m=50, n=10)
    t_male = datetime.datetime.now() - t
    print(f"MAE homes: {mae_male:.4f}")
    print(f"Temps de càlcul: {str(t_male)}")
    print()
    
    # H.6 - Avaluació recomanador DONES
    print("="*60)
    print("H.6 - AVALUACIÓ - RECOMANADOR DONES")
    print("="*60)
    t = datetime.datetime.now()
    mae_female = evaluateRecommendations_v2(train_female, test_female, m=50, n=10)
    t_female = datetime.datetime.now() - t
    print(f"MAE dones: {mae_female:.4f}")
    print(f"Temps de càlcul: {str(t_female)}")
    print()
    
    # H.7 - Comparació de resultats
    print("="*60)
    print("H.7 - COMPARACIÓ DE RESULTATS")
    print("="*60)
    print(f"MAE recomanador únic:        {mae:.4f}")
    print(f"MAE recomanador homes:       {mae_male:.4f}")
    print(f"MAE recomanador dones:       {mae_female:.4f}")
    print()
    
    # Calcular la mitjana ponderada dels MAE per sexe
    total_test_interactions = len(test_male) + len(test_female)
    weighted_mae_gender = (mae_male * len(test_male) + mae_female * len(test_female)) / total_test_interactions
    print(f"MAE ponderat (per sexe):     {weighted_mae_gender:.4f}")
    print()
    
    # Anàlisi
    print("="*60)
    print("H.7 - ANÀLISI I CONCLUSIONS")
    print("="*60)
    if weighted_mae_gender < mae:
        diff = mae - weighted_mae_gender
        millora_percentual = (diff / mae) * 100
        print(f" Els recomanadors separats per sexe són MILLORS")
        print(f"  - Millora absoluta: {diff:.4f}")
        print(f"  - Millora percentual: {millora_percentual:.2f}%")
    else:
        diff = weighted_mae_gender - mae
        empitjorament_percentual = (diff / mae) * 100
        print(f"✗ El recomanador únic és MILLOR")
        print(f"  - Empitjorament absolut: {diff:.4f}")
        print(f"  - Empitjorament percentual: {empitjorament_percentual:.2f}%")
    
    print()
    print("INTERPRETACIÓ:")
    print("-" * 60)
    print("Un MAE més baix indica millors prediccions (menys error).")
    print()
    
    if weighted_mae_gender < mae:
        print("CONCLUSIÓ: Surt més a compte fer recomanadors separats")
        print("per sexe, ja que les preferències cinematogràfiques són")
        print("prou diferents entre homes i dones com per beneficiar-se")
        print("d'una segmentació del dataset.")
    else:
        print("CONCLUSIÓ: Surt més a compte fer un recomanador únic,")
        print("ja que la divisió per sexe redueix la mida del dataset")
        print("i la diversitat d'opinions, empitjorant les prediccions.")
    
    print()
    print("FACTORS A CONSIDERAR:")
    print("-" * 60)
    print("• Mida del dataset: Dividir redueix els usuaris disponibles")
    print("• Similitud intra-grup: Si homes/dones tenen preferències")
    print("  més homogènies dins del seu grup, la segmentació millora")
    print("• Cost computacional: Mantenir dos models és més costós")
    print("• Escalabilitat: Amb més categories (edat, ocupació...)")
    print("  els subconjunts serien massa petits")
    print()
    
    # H.8 - Visualització dels resultats
    print("="*60)
    print("H.8 - VISUALITZACIÓ DELS RESULTATS")
    print("="*60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gràfic de barres per comparar MAE
    models = ['Recomanador\nÚnic', 'Recomanador\nHomes', 'Recomanador\nDones', 'MAE Ponderat\n(per sexe)']
    maes = [mae, mae_male, mae_female, weighted_mae_gender]
    colors = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']
    
    bars = ax1.bar(models, maes, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('MAE (Mean Absolute Error)', fontsize=12, fontweight='bold')
    ax1.set_title('Comparació de MAE entre Models', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Afegir valors sobre les barres
    for bar, mae_val in zip(bars, maes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{mae_val:.4f}',
                 ha='center', va='bottom', fontweight='bold')
    
    # Gràfic de millora/empitjorament percentual
    if weighted_mae_gender < mae:
        millora = ((mae - weighted_mae_gender) / mae) * 100
        label = f'Millora: {millora:.2f}%'
        color = '#2ecc71'
    else:
        millora = ((weighted_mae_gender - mae) / mae) * 100
        label = f'Empitjorament: {millora:.2f}%'
        color = '#e74c3c'
    
    ax2.barh(['Recomanadors\nper Sexe vs\nÚnic'], [millora], color=color, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Diferència Percentual (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Millora/Empitjorament Percentual', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    ax2.text(millora/2, 0, label, ha='center', va='center', 
             fontweight='bold', fontsize=11, color='white')
    
    plt.tight_layout()
    plt.show()
    
    print("\nVisualització completada!")
    print("\n" + "="*80 + "\n")


def executar_tots_els_exercicis(data_folder='ml-1m', incluir_opcional=False):
    """
    Executa tots els exercicis de la pràctica
    
    :param data_folder: carpeta amb les dades de MovieLens
    :param incluir_opcional: si True, executa també l'exercici H (opcional)
    """
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*20 + "PRÀCTICA 2: RECOMANADOR HEURÍSTIC" + " "*25 + "█")
    print("█" + " "*25 + "EXECUCIÓ DE TOTS ELS EXERCICIS" + " "*24 + "█")
    print("█" + " "*78 + "█")
    print("█"*80 + "\n")
    
    try:
        exercici_A(data_folder)
        exercici_B(data_folder)
        exercici_C(data_folder)
        exercici_D(data_folder)
        exercici_E(data_folder)
        exercici_F(data_folder, show_matrix_time=True)
        exercici_G(data_folder)
        
        if incluir_opcional:
            exercici_H(data_folder)
        
        print("\n" + "█"*80)
        print("█" + " "*78 + "█")
        print("█" + " "*25 + "TOTS ELS EXERCICIS COMPLETATS!" + " "*24 + "█")
        print("█" + " "*78 + "█")
        print("█"*80 + "\n")
        
    except Exception as e:
        print(f"\nERROR durant l'execució: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# MAIN - Exemple d'ús
# ============================================================================

#if __name__ == "__main__":
    """
    Exemple d'ús del mòdul.
    
    Per executar tots els exercicis:
        python pt2_prueba.py
    
    Per usar des d'un notebook:
        import pt2_prueba as pt2
        
        # Executar tots els exercicis
        pt2.executar_tots_els_exercicis('ml-1m', incluir_opcional=False)
        
        # O executar exercicis individuals
        pt2.exercici_A('ml-1m')
        pt2.exercici_B('ml-1m')
        # etc.
        
        # O executar subejercicios específics
        pt2.exercici_F_1('ml-1m')  # Només F.1
        pt2.exercici_F_2('ml-1m')  # Només F.2
        
        pt2.exercici_G_1('ml-1m')  # Només G.1
        pt2.exercici_G_2('ml-1m')  # Només G.2
        pt2.exercici_G_3('ml-1m')  # Només G.3
        
        pt2.exercici_H_1('ml-1m')  # Només H.1
        pt2.exercici_H_2('ml-1m')  # Només H.2
        # ... fins a H.8
        
        # Per encadenar subejercicios de manera eficient:
        data, train, test, mae = pt2.exercici_H_1('ml-1m')
        data_male, data_female = pt2.exercici_H_2(data)
        train_male, test_male = pt2.exercici_H_3(data_male)
        # etc.
    """
    # Executar tots els exercicis (sense l'opcional per defecte)
#    executar_tots_els_exercicis('ml-1m', incluir_opcional=False)
    
    # Si vols executar només un exercici:
    # exercici_A('ml-1m')
    # exercici_B('ml-1m')
    # exercici_C('ml-1m')
    # exercici_D('ml-1m')
    # exercici_E('ml-1m')
    # exercici_F('ml-1m')
    # exercici_G('ml-1m')
    # exercici_H('ml-1m')  # Opcional


# ============================================================================
# 12. EXERCICIS EXTRA
# ============================================================================

def exercici_extra_recomanacions_usuari(user_id=100, n=10, m=50, data_folder='ml-1m'):
    """
    EXERCICI EXTRA: Generar les 10 millors recomanacions per un usuari específic
    utilitzant la matriu de similitud euclidiana calculada vectorialment (mètode ràpid).
    
    :param user_id: ID de l'usuari per al qual generar recomanacions
    :param n: nombre de pel·lícules a recomanar
    :param m: nombre d'usuaris similars a considerar
    :param data_folder: carpeta amb les dades de MovieLens

    # Usar con valores por defecto (usuario 100, 10 recomendaciones, 50 vecinos)
    pt2.exercici_extra_recomanacions_usuari()

    # Personalizar parámetros
    pt2.exercici_extra_recomanacions_usuari(user_id=250, n=15, m=100)

    # Obtener el DataFrame de recomendaciones
    recommendations = pt2.exercici_extra_recomanacions_usuari(user_id=42, n=20)
    """
    print("="*80)
    print(f"EXERCICI EXTRA: Recomanacions per l'Usuari {user_id}")
    print("="*80)
    
    # Carregar dades
    print("\n1. Carregant dades...")
    users, ratings, movies, data = load_movielens_data(data_folder)
    data = reindex_data(data)
    
    print(f"    Dades carregades: {len(data)} interaccions")
    print(f"    Usuaris: {get_num_unique_users(data)}")
    print(f"    Pel·lícules: {get_num_unique_movies(data)}")
    
    # Construir taula de counts
    print("\n2. Construint taula de valoracions...")
    df_counts = build_counts_table(data)
    print(f"    Taula de valoracions: {df_counts.shape}")
    
    # Calcular matriu de similitud (mètode vectorial ràpid)
    print("\n3. Calculant matriu de similitud euclidiana (mètode vectorial)...")
    t_start = datetime.datetime.now()
    sim_mx = similarity_matrix_2(df_counts)
    t_elapsed = datetime.datetime.now() - t_start
    print(f"    Matriu de similitud calculada en: {t_elapsed}")
    print(f"    Dimensions: {sim_mx.shape}")
    
    # Verificar que l'usuari existeix
    if user_id not in df_counts.index:
        print(f"\nERROR: L'usuari {user_id} no existeix al dataset!")
        print(f"   Els IDs d'usuari van de 0 a {df_counts.index.max()}")
        return
    
    # Generar recomanacions
    print(f"\n4. Generant {n} recomanacions per l'usuari {user_id}...")
    print(f"   Utilitzant els {m} usuaris més similars")
    
    t_start = datetime.datetime.now()
    recommendations = getRecommendationsUser(df_counts, user_id, sim_mx, n=n, m=m)
    t_elapsed = datetime.datetime.now() - t_start
    print(f"    Recomanacions generades en: {t_elapsed}")
    
    # Mostrar resultats
    print(f"\n5. TOP {n} RECOMANACIONS PER L'USUARI {user_id}:")
    print("="*80)
    
    # Afegir els títols de les pel·lícules
    # Primer, crear un mapeo de movie_id reindexat a títol original
    movie_mapping = data[['movie_id', 'title']].drop_duplicates().set_index('movie_id')
    
    print(f"\n{'Posició':<10} {'Movie ID':<12} {'Puntuació':<15} {'Títol'}")
    print("-" * 80)
    
    for idx, row in recommendations.iterrows():
        movie_id = int(row['movie_id'])
        predicted_rating = row['predicted_rating']
        
        # Obtenir el títol
        if movie_id in movie_mapping.index:
            title = movie_mapping.loc[movie_id, 'title']
        else:
            title = "Títol desconegut"
        
        print(f"{idx+1:<10} {movie_id:<12} {predicted_rating:<15.4f} {title}")
    
    # Estadístiques addicionals
    print("\n" + "="*80)
    print("ESTADÍSTIQUES DE LES RECOMANACIONS:")
    print("-" * 80)
    print(f"Puntuació mitjana predita: {recommendations['predicted_rating'].mean():.4f}")
    print(f"Puntuació màxima predita:  {recommendations['predicted_rating'].max():.4f}")
    print(f"Puntuació mínima predita:  {recommendations['predicted_rating'].min():.4f}")
    print(f"Desviació estàndard:       {recommendations['predicted_rating'].std():.4f}")
    
    # Mostrar informació sobre l'usuari
    print("\n" + "="*80)
    print(f"INFORMACIÓ DE L'USUARI {user_id}:")
    print("-" * 80)
    
    user_ratings = df_counts.loc[user_id]
    num_rated = user_ratings.notna().sum()
    avg_rating = user_ratings.mean()
    
    print(f"Pel·lícules valorades:     {num_rated}")
    print(f"Puntuació mitjana:         {avg_rating:.4f}")
    
    # Trobar les pel·lícules que l'usuari ja ha valorat millor
    user_top_rated = user_ratings.dropna().sort_values(ascending=False).head(5)
    print(f"\nTop 5 pel·lícules que ja ha valorat:")
    for movie_id, rating in user_top_rated.items():
        movie_id_int = int(movie_id)
        if movie_id_int in movie_mapping.index:
            title = movie_mapping.loc[movie_id_int, 'title']
        else:
            title = "Títol desconegut"
        print(f"  - {title}: {rating:.1f} estrelles")
    
    print("\n" + "="*80 + "\n")
    
    return recommendations


def exercici_extra_2_vei_mes_proper(user_id=150, data_folder='ml-1m'):
    """
    EXERCICI EXTRA 2: Identificació del "Veí més Proper" (Anàlisi de Similitud)
    
    Calcula la matriu de similitud completa usant el mètode matricial ràpid.
    Per l'usuari donat, identifica quin és l'usuari més similar (el que té el 
    score de similitud més alt, excloent-se a si mateix) i mostra el títol de 
    la pel·lícula que aquest veí ha puntuat més alt.
    
    :param user_id: ID de l'usuari per analitzar
    :param data_folder: carpeta amb les dades de MovieLens
    """
    print("="*80)
    print(f"EXERCICI EXTRA 2: Veí Més Proper de l'Usuari {user_id}")
    print("="*80)
    
    # Carregar dades
    print("\n1. Carregant dades...")
    users, ratings, movies, data = load_movielens_data(data_folder)
    data = reindex_data(data)
    
    # Construir taula de counts
    print("\n2. Construint taula de valoracions...")
    df_counts = build_counts_table(data)
    
    # Calcular matriu de similitud
    print("\n3. Calculant matriu de similitud euclidiana (mètode matricial ràpid)...")
    t_start = datetime.datetime.now()
    sim_mx = similarity_matrix_2(df_counts)
    t_elapsed = datetime.datetime.now() - t_start
    print(f"   ✓ Matriu calculada en: {t_elapsed}")
    
    # Verificar que l'usuari existeix
    if user_id not in df_counts.index:
        print(f"\n ERROR: L'usuari {user_id} no existeix al dataset!")
        return
    
    # Trobar el veí més proper
    print(f"\n4. Identificant el veí més proper a l'usuari {user_id}...")
    
    # Obtenir totes les similituds de l'usuari
    user_similarities = sim_mx[user_id].copy()
    # Excloure l'usuari mateix
    user_similarities[user_id] = -np.inf
    
    # Trobar l'usuari més similar
    most_similar_user = np.argmax(user_similarities)
    max_similarity = user_similarities[most_similar_user]
    
    print(f"\n   ✓ Veí més proper: Usuari {most_similar_user}")
    print(f"   ✓ Similitud: {max_similarity:.6f}")
    
    # Trobar la pel·lícula millor puntuada pel veí
    print(f"\n5. Trobant la pel·lícula millor puntuada pel veí {most_similar_user}...")
    
    neighbor_ratings = df_counts.loc[most_similar_user]
    best_movie_id = neighbor_ratings.idxmax()
    best_rating = neighbor_ratings[best_movie_id]
    
    # Obtenir el títol de la pel·lícula
    movie_mapping = data[['movie_id', 'title']].drop_duplicates().set_index('movie_id')
    if best_movie_id in movie_mapping.index:
        best_title = movie_mapping.loc[best_movie_id, 'title']
    else:
        best_title = "Títol desconegut"
    
    print(f"\n   ✓ Pel·lícula millor puntuada: {best_title}")
    print(f"   ✓ Movie ID: {best_movie_id}")
    print(f"   ✓ Puntuació: {best_rating:.1f} estrelles")
    
    # Informació addicional
    print("\n" + "="*80)
    print("INFORMACIÓ COMPARATIVA:")
    print("-" * 80)
    
    # Estadístiques de l'usuari original
    user_ratings = df_counts.loc[user_id]
    user_num_rated = user_ratings.notna().sum()
    user_avg = user_ratings.mean()
    
    # Estadístiques del veí
    neighbor_num_rated = neighbor_ratings.notna().sum()
    neighbor_avg = neighbor_ratings.mean()
    
    print(f"Usuari {user_id}:")
    print(f"  - Pel·lícules valorades: {user_num_rated}")
    print(f"  - Puntuació mitjana: {user_avg:.4f}")
    
    print(f"\nVeí {most_similar_user}:")
    print(f"  - Pel·lícules valorades: {neighbor_num_rated}")
    print(f"  - Puntuació mitjana: {neighbor_avg:.4f}")
    
    # Pel·lícules en comú
    common_movies = (~user_ratings.isna()) & (~neighbor_ratings.isna())
    num_common = common_movies.sum()
    print(f"\nPel·lícules en comú: {num_common}")
    
    print("\n" + "="*80 + "\n")
    
    return most_similar_user, max_similarity, best_title


def exercici_extra_3_prediccio_rating(user_id=10, movie_title='Toy Story (1995)', 
                                      m=50, data_folder='ml-1m'):
    """
    EXERCICI EXTRA 3: Predicció de Rating Específic (Validació puntual)
    
    El sistema prediu quina nota li posaria l'usuari a una pel·lícula específica.
    Utilitza la matriu de similitud ràpida i considera els m vecins més propers
    per al càlcul de la mitjana ponderada.
    
    :param user_id: ID de l'usuari
    :param movie_title: títol de la pel·lícula
    :param m: nombre de veïns a considerar
    :param data_folder: carpeta amb les dades de MovieLens
    """
    print("="*80)
    print(f"EXERCICI EXTRA 3: Predicció de Rating")
    print("="*80)
    print(f"Usuari: {user_id}")
    print(f"Pel·lícula: {movie_title}")
    print(f"Veïns considerats: {m}")
    print("="*80)
    
    # Carregar dades
    print("\n1. Carregant dades...")
    users, ratings, movies, data = load_movielens_data(data_folder)
    data = reindex_data(data)
    
    # Trobar el movie_id de la pel·lícula
    movie_info = data[data['title'] == movie_title][['movie_id', 'title']].drop_duplicates()
    
    if len(movie_info) == 0:
        print(f"\n ERROR: No s'ha trobat la pel·lícula '{movie_title}'")
        return
    
    movie_id = movie_info.iloc[0]['movie_id']
    print(f"   ✓ Movie ID trobat: {movie_id}")
    
    # Construir taula de counts
    print("\n2. Construint taula de valoracions...")
    df_counts = build_counts_table(data)
    
    # Verificar si l'usuari ja ha valorat la pel·lícula
    if user_id in df_counts.index and movie_id in df_counts.columns:
        actual_rating = df_counts.loc[user_id, movie_id]
        if not pd.isna(actual_rating):
            print(f"\n   ⚠ ATENCIÓ: L'usuari {user_id} ja ha valorat aquesta pel·lícula!")
            print(f"   Valoració real: {actual_rating:.1f} estrelles")
    
    # Calcular matriu de similitud
    print("\n3. Calculant matriu de similitud...")
    t_start = datetime.datetime.now()
    sim_mx = similarity_matrix_2(df_counts)
    t_elapsed = datetime.datetime.now() - t_start
    print(f"   ✓ Calculada en: {t_elapsed}")
    
    # Trobar els m veïns més propers
    print(f"\n4. Trobant els {m} veïns més propers...")
    similar_users_dict = find_similar_users(df_counts, sim_mx, user_id, m)
    
    # Calcular la predicció
    print(f"\n5. Calculant predicció per la pel·lícula '{movie_title}'...")
    
    weighted_sum = 0
    weight_sum = 0
    neighbors_with_rating = 0
    
    for neighbor_id, weight in similar_users_dict.items():
        neighbor_rating = df_counts.loc[neighbor_id, movie_id]
        if not np.isnan(neighbor_rating):
            weighted_sum += neighbor_rating * weight
            weight_sum += weight
            neighbors_with_rating += 1
    
    if weight_sum > 0:
        predicted_rating = weighted_sum / weight_sum
        print(f"\n   ✓ Predicció calculada: {predicted_rating:.4f} estrelles")
        print(f"   ✓ Veïns que han valorat aquesta pel·lícula: {neighbors_with_rating}/{m}")
        
        # Comparar amb la valoració real si existeix
        if user_id in df_counts.index and movie_id in df_counts.columns:
            actual_rating = df_counts.loc[user_id, movie_id]
            if not pd.isna(actual_rating):
                error = abs(actual_rating - predicted_rating)
                print(f"\n   COMPARACIÓ AMB VALORACIÓ REAL:")
                print(f"   - Valoració real: {actual_rating:.1f}")
                print(f"   - Predicció: {predicted_rating:.4f}")
                print(f"   - Error absolut: {error:.4f}")
    else:
        predicted_rating = None
        print(f"\n    No s'ha pogut calcular la predicció.")
        print(f"   Cap dels {m} veïns ha valorat aquesta pel·lícula.")
    
    # Informació addicional
    print("\n" + "="*80)
    print("ESTADÍSTIQUES DE LA PEL·LÍCULA:")
    print("-" * 80)
    
    movie_ratings = df_counts[movie_id].dropna()
    print(f"Nombre total de valoracions: {len(movie_ratings)}")
    print(f"Puntuació mitjana global: {movie_ratings.mean():.4f}")
    print(f"Puntuació màxima: {movie_ratings.max():.1f}")
    print(f"Puntuació mínima: {movie_ratings.min():.1f}")
    
    print("\n" + "="*80 + "\n")
    
    return predicted_rating


def exercici_extra_4_sensibilitat_m(user_id=100, movie_id=50, m_values=[5, 100], 
                                   data_folder='ml-1m'):
    """
    EXERCICI EXTRA 4: Anàlisi de Sensibilitat del Paràmetre 'm' (Veïns)
    
    Calcula la predicció de rating per un usuari i una pel·lícula variant el nombre
    de veïns considerats. Compara el resultat obtingut usant diferents valors de m.
    
    :param user_id: ID de l'usuari
    :param movie_id: ID de la pel·lícula
    :param m_values: llista de valors de m a provar
    :param data_folder: carpeta amb les dades de MovieLens
    """
    print("="*80)
    print(f"EXERCICI EXTRA 4: Anàlisi de Sensibilitat del Paràmetre 'm'")
    print("="*80)
    print(f"Usuari: {user_id}")
    print(f"Pel·lícula ID: {movie_id}")
    print(f"Valors de m a provar: {m_values}")
    print("="*80)
    
    # Carregar dades
    print("\n1. Carregant dades...")
    users, ratings, movies, data = load_movielens_data(data_folder)
    data = reindex_data(data)
    
    # Obtenir títol de la pel·lícula
    movie_mapping = data[['movie_id', 'title']].drop_duplicates().set_index('movie_id')
    if movie_id in movie_mapping.index:
        movie_title = movie_mapping.loc[movie_id, 'title']
    else:
        movie_title = "Títol desconegut"
    
    print(f"   ✓ Pel·lícula: {movie_title}")
    
    # Construir taula de counts
    print("\n2. Construint taula de valoracions...")
    df_counts = build_counts_table(data)
    
    # Verificar valoració real
    actual_rating = None
    if user_id in df_counts.index and movie_id in df_counts.columns:
        actual_rating = df_counts.loc[user_id, movie_id]
        if not pd.isna(actual_rating):
            print(f"   ✓ Valoració real de l'usuari: {actual_rating:.1f} estrelles")
    
    # Calcular matriu de similitud
    print("\n3. Calculant matriu de similitud...")
    sim_mx = similarity_matrix_2(df_counts)
    
    # Calcular prediccions per diferents valors de m
    print(f"\n4. Calculant prediccions per diferents valors de m...")
    print("\n" + "="*80)
    
    results = []
    
    for m in m_values:
        print(f"\nPROVANT m = {m}:")
        print("-" * 40)
        
        # Trobar veïns
        similar_users_dict = find_similar_users(df_counts, sim_mx, user_id, m)
        
        # Calcular predicció
        weighted_sum = 0
        weight_sum = 0
        neighbors_with_rating = 0
        
        for neighbor_id, weight in similar_users_dict.items():
            neighbor_rating = df_counts.loc[neighbor_id, movie_id]
            if not np.isnan(neighbor_rating):
                weighted_sum += neighbor_rating * weight
                weight_sum += weight
                neighbors_with_rating += 1
        
        if weight_sum > 0:
            predicted_rating = weighted_sum / weight_sum
            
            # Calcular error si tenim valoració real
            error = None
            if actual_rating is not None:
                error = abs(actual_rating - predicted_rating)
            
            results.append({
                'm': m,
                'predicted_rating': predicted_rating,
                'neighbors_used': neighbors_with_rating,
                'error': error
            })
            
            print(f"Predicció: {predicted_rating:.4f} estrelles")
            print(f"Veïns utilitzats: {neighbors_with_rating}/{m}")
            if error is not None:
                print(f"Error absolut: {error:.4f}")
        else:
            results.append({
                'm': m,
                'predicted_rating': None,
                'neighbors_used': 0,
                'error': None
            })
            print(f"No s'ha pogut calcular (cap veí ha valorat la pel·lícula)")
    
    # Anàlisi comparatiu
    print("\n" + "="*80)
    print("COMPARACIÓ I CONCLUSIONS:")
    print("="*80)
    
    valid_results = [r for r in results if r['predicted_rating'] is not None]
    
    if len(valid_results) >= 2:
        predictions = [r['predicted_rating'] for r in valid_results]
        diff = max(predictions) - min(predictions)
        
        print(f"\nDiferència entre prediccions: {diff:.4f} estrelles")
        print(f"Desviació estàndard: {np.std(predictions):.4f}")
        
        print(f"\nCONCLUSIONS:")
        print("-" * 80)
        if diff < 0.5:
            print("✓ El sistema és ESTABLE: Les prediccions varien poc amb diferents m.")
            print("  Això indica que els veïns més propers tenen preferències similars.")
        elif diff < 1.0:
            print("⚠ El sistema té SENSIBILITAT MODERADA al paràmetre m.")
            print("  Les prediccions varien moderadament. Considerar ajustar m segons el cas.")
        else:
            print(" El sistema és INESTABLE: Les prediccions varien molt amb diferents m.")
            print("  Això pot indicar que hi ha pocs veïns amb valoracions per aquesta pel·lícula,")
            print("  o que els gustos dels usuaris són molt heterogenis.")
        
        # Recomanació
        print(f"\nRECOMANACIÓ:")
        if len(valid_results) >= 2:
            # Trobar el m amb menys error (si hi ha valoració real)
            if all(r['error'] is not None for r in valid_results):
                best_result = min(valid_results, key=lambda x: x['error'])
                print(f"Usar m = {best_result['m']} (menor error: {best_result['error']:.4f})")
            else:
                # Recomanar basat en nombre de veïns utilitzats
                best_result = max(valid_results, key=lambda x: x['neighbors_used'])
                print(f"Usar m = {best_result['m']} (més veïns utilitzats: {best_result['neighbors_used']})")
    
    print("\n" + "="*80 + "\n")
    
    return results


def exercici_extra_5_mae_usuari(user_id=45, test_ratio=0.2, m=50, data_folder='ml-1m'):
    """
    EXERCICI EXTRA 5: Avaluació d'Error (MAE) per un Usuari (Mini-Test)
    
    Separa les interaccions d'un usuari en un conjunt d'entrenament (80%) i test (20%).
    Entrena el model usant similarity_matrix_2 amb el conjunt d'entrenament i calcula
    l'Error Absolut Mitjà (MAE) de les prediccions sobre el 20% de test d'aquest únic usuari.
    
    :param user_id: ID de l'usuari a avaluar
    :param test_ratio: percentatge de test (0.2 = 20%)
    :param m: nombre de veïns a considerar
    :param data_folder: carpeta amb les dades de MovieLens
    """
    print("="*80)
    print(f"EXERCICI EXTRA 5: Avaluació MAE per Usuari Únic")
    print("="*80)
    print(f"Usuari: {user_id}")
    print(f"Train/Test split: {int((1-test_ratio)*100)}/{int(test_ratio*100)}")
    print(f"Veïns considerats: {m}")
    print("="*80)
    
    # Carregar dades
    print("\n1. Carregant dades...")
    users, ratings, movies, data = load_movielens_data(data_folder)
    data = reindex_data(data)
    
    # Filtrar interaccions de l'usuari
    user_data = data[data['user_id'] == user_id].copy()
    
    if len(user_data) == 0:
        print(f"\n ERROR: L'usuari {user_id} no existeix al dataset!")
        return
    
    print(f"   ✓ Interaccions de l'usuari: {len(user_data)}")
    
    # Verificar que té suficients interaccions
    min_interactions = 10
    if len(user_data) < min_interactions:
        print(f"\n ERROR: L'usuari té menys de {min_interactions} interaccions!")
        print(f"   No és possible fer un split fiable.")
        return
    
    # Separar en train i test per aquest usuari
    print(f"\n2. Separant interaccions en train ({int((1-test_ratio)*100)}%) i test ({int(test_ratio*100)}%)...")
    
    np.random.seed(42)
    test_size = max(1, int(len(user_data) * test_ratio))
    test_indices = np.random.choice(user_data.index, size=test_size, replace=False)
    
    user_test = user_data.loc[test_indices]
    user_train_indices = user_data.index.difference(test_indices)
    
    # Crear conjunt de train: totes les dades EXCEPTE el test de l'usuari
    train_data = data[~data.index.isin(test_indices)].copy()
    
    print(f"   ✓ Interaccions de l'usuari en train: {len(user_train_indices)}")
    print(f"   ✓ Interaccions de l'usuari en test: {len(user_test)}")
    print(f"   ✓ Total interaccions en train: {len(train_data)}")
    
    # Construir matriu amb el train
    print("\n3. Construint matriu de similitud amb dades de train...")
    df_counts_train = build_counts_table(train_data)
    sim_mx = similarity_matrix_2(df_counts_train)
    print(f"   ✓ Matriu creada: {sim_mx.shape}")
    
    # Calcular prediccions per cada pel·lícula del test
    print(f"\n4. Calculant prediccions per les {len(user_test)} pel·lícules del test...")
    
    errors = []
    predictions_made = 0
    
    for idx, row in user_test.iterrows():
        movie_id = row['movie_id']
        true_rating = row['rating']
        
        # Verificar que la pel·lícula existeix en el train
        if movie_id not in df_counts_train.columns:
            continue
        
        # Trobar veïns
        similar_users_dict = find_similar_users(df_counts_train, sim_mx, user_id, m)
        
        # Calcular predicció
        weighted_sum = 0
        weight_sum = 0
        
        for neighbor_id, weight in similar_users_dict.items():
            neighbor_rating = df_counts_train.loc[neighbor_id, movie_id]
            if not np.isnan(neighbor_rating):
                weighted_sum += neighbor_rating * weight
                weight_sum += weight
        
        if weight_sum > 0:
            predicted_rating = weighted_sum / weight_sum
            error = abs(true_rating - predicted_rating)
            errors.append(error)
            predictions_made += 1
    
    # Calcular MAE
    if len(errors) > 0:
        mae = np.mean(errors)
        
        print(f"\n   ✓ Prediccions realitzades: {predictions_made}/{len(user_test)}")
        
        print("\n" + "="*80)
        print("RESULTATS:")
        print("="*80)
        print(f"\nMAE (Mean Absolute Error): {mae:.4f} estrelles")
        print(f"\nInterpretació:")
        print(f"  - L'error mitjà de predicció és de {mae:.4f} estrelles")
        print(f"  - Errors individuals: min={min(errors):.4f}, max={max(errors):.4f}")
        print(f"  - Desviació estàndard dels errors: {np.std(errors):.4f}")
        
        # Classificació de la qualitat
        print(f"\nQUALITAT DE LES PREDICCIONS:")
        if mae < 0.5:
            print("  ✓ EXCEL·LENT: Prediccions molt precises")
        elif mae < 0.75:
            print("  ✓ BONA: Prediccions força precises")
        elif mae < 1.0:
            print("  ⚠ ACCEPTABLE: Prediccions moderadament precises")
        else:
            print("   POBRA: Prediccions poc precises")
        
    else:
        mae = None
        print(f"\n No s'han pogut fer prediccions!")
        print(f"   Cap veí ha valorat les pel·lícules del test.")
    
    print("\n" + "="*80 + "\n")
    
    return mae


def exercici_extra_6_comparativa_genere(user_id=100, movie_id=50, m=50, data_folder='ml-1m'):
    """
    EXERCICI EXTRA 6: Comparativa de Segmentació (Basat en Exercici H)
    
    Utilitzant la matriu ràpida, calcula la predicció per un usuari usant només veïns
    del mateix gènere (Home/Dona) i compara-la amb la predicció usant tots els usuaris.
    
    :param user_id: ID de l'usuari
    :param movie_id: ID de la pel·lícula
    :param m: nombre de veïns a considerar
    :param data_folder: carpeta amb les dades de MovieLens
    """
    print("="*80)
    print(f"EXERCICI EXTRA 6: Comparativa de Segmentació per Gènere")
    print("="*80)
    print(f"Usuari: {user_id}")
    print(f"Pel·lícula ID: {movie_id}")
    print(f"Veïns considerats: {m}")
    print("="*80)
    
    # Carregar dades
    print("\n1. Carregant dades...")
    users, ratings, movies, data = load_movielens_data(data_folder)
    data = reindex_data(data)
    
    # Obtenir informació de la pel·lícula
    movie_mapping = data[['movie_id', 'title']].drop_duplicates().set_index('movie_id')
    if movie_id in movie_mapping.index:
        movie_title = movie_mapping.loc[movie_id, 'title']
    else:
        movie_title = "Títol desconegut"
    
    print(f"   ✓ Pel·lícula: {movie_title}")
    
    # Obtenir el gènere de l'usuari
    user_info = data[data['user_id'] == user_id][['user_id', 'gender']].drop_duplicates()
    
    if len(user_info) == 0:
        print(f"\n ERROR: L'usuari {user_id} no existeix!")
        return
    
    user_gender = user_info.iloc[0]['gender']
    gender_name = "Home" if user_gender == 'M' else "Dona"
    
    print(f"   ✓ Gènere de l'usuari: {gender_name} ({user_gender})")
    
    # PREDICCIÓ 1: Usant tots els usuaris
    print("\n2. PREDICCIÓ AMB TOTS ELS USUARIS:")
    print("-" * 80)
    
    df_counts_all = build_counts_table(data)
    sim_mx_all = similarity_matrix_2(df_counts_all)
    
    # Verificar valoració real
    actual_rating = None
    if user_id in df_counts_all.index and movie_id in df_counts_all.columns:
        actual_rating = df_counts_all.loc[user_id, movie_id]
        if not pd.isna(actual_rating):
            print(f"   Valoració real: {actual_rating:.1f} estrelles")
    
    similar_users_all = find_similar_users(df_counts_all, sim_mx_all, user_id, m)
    
    weighted_sum_all = 0
    weight_sum_all = 0
    neighbors_used_all = 0
    
    for neighbor_id, weight in similar_users_all.items():
        neighbor_rating = df_counts_all.loc[neighbor_id, movie_id]
        if not np.isnan(neighbor_rating):
            weighted_sum_all += neighbor_rating * weight
            weight_sum_all += weight
            neighbors_used_all += 1
    
    if weight_sum_all > 0:
        prediction_all = weighted_sum_all / weight_sum_all
        print(f"   ✓ Predicció: {prediction_all:.4f} estrelles")
        print(f"   ✓ Veïns utilitzats: {neighbors_used_all}/{m}")
    else:
        prediction_all = None
        print(f"    No s'ha pogut calcular la predicció")
    
    # PREDICCIÓ 2: Usant només usuaris del mateix gènere
    print(f"\n3. PREDICCIÓ AMB USUARIS DEL MATEIX GÈNERE ({gender_name}):")
    print("-" * 80)
    
    data_same_gender = filter_by_gender(data, user_gender)
    print(f"   Usuaris del mateix gènere: {get_num_unique_users(data_same_gender)}")
    
    df_counts_gender = build_counts_table(data_same_gender)
    sim_mx_gender = similarity_matrix_2(df_counts_gender)
    
    similar_users_gender = find_similar_users(df_counts_gender, sim_mx_gender, user_id, m)
    
    weighted_sum_gender = 0
    weight_sum_gender = 0
    neighbors_used_gender = 0
    
    for neighbor_id, weight in similar_users_gender.items():
        if neighbor_id in df_counts_gender.index and movie_id in df_counts_gender.columns:
            neighbor_rating = df_counts_gender.loc[neighbor_id, movie_id]
            if not np.isnan(neighbor_rating):
                weighted_sum_gender += neighbor_rating * weight
                weight_sum_gender += weight
                neighbors_used_gender += 1
    
    if weight_sum_gender > 0:
        prediction_gender = weighted_sum_gender / weight_sum_gender
        print(f"   ✓ Predicció: {prediction_gender:.4f} estrelles")
        print(f"   ✓ Veïns utilitzats: {neighbors_used_gender}/{m}")
    else:
        prediction_gender = None
        print(f"    No s'ha pogut calcular la predicció")
    
    # COMPARACIÓ
    print("\n" + "="*80)
    print("COMPARACIÓ I CONCLUSIONS:")
    print("="*80)
    
    if prediction_all is not None and prediction_gender is not None:
        diff = abs(prediction_all - prediction_gender)
        
        print(f"\nPredicció amb tots els usuaris:      {prediction_all:.4f} estrelles")
        print(f"Predicció amb mateix gènere:          {prediction_gender:.4f} estrelles")
        print(f"Diferència absoluta:                  {diff:.4f} estrelles")
        
        # Errors si tenim valoració real
        if actual_rating is not None:
            error_all = abs(actual_rating - prediction_all)
            error_gender = abs(actual_rating - prediction_gender)
            
            print(f"\nComparació amb valoració real ({actual_rating:.1f}):")
            print(f"  Error amb tots els usuaris:         {error_all:.4f}")
            print(f"  Error amb mateix gènere:            {error_gender:.4f}")
            
            if error_gender < error_all:
                print(f"\n  ✓ La segmentació per gènere MILLORA la predicció!")
                print(f"    Millora: {(error_all - error_gender):.4f} estrelles")
            elif error_gender > error_all:
                print(f"\n   La segmentació per gènere EMPITJORA la predicció!")
                print(f"    Empitjorament: {(error_gender - error_all):.4f} estrelles")
            else:
                print(f"\n  = No hi ha diferència en la qualitat de la predicció")
        
        print(f"\nCONCLUSIONS:")
        print("-" * 80)
        if diff < 0.3:
            print("✓ DIFERÈNCIA MÍNIMA: La segmentació per gènere no afecta significativament.")
            print("  Els gustos cinematogràfics són similars entre gèneres per aquesta pel·lícula.")
        elif diff < 0.7:
            print("⚠ DIFERÈNCIA MODERADA: Hi ha alguna diferència en les preferències.")
            print("  Pot valer la pena considerar la segmentació per gènere en alguns casos.")
        else:
            print(" DIFERÈNCIA SIGNIFICATIVA: Els gustos varien molt entre gèneres!")
            print("  Recomanable usar segmentació per gènere per aquest tipus de pel·lícules.")
        
        print(f"\nVeïns utilitzats:")
        print(f"  - Tots els usuaris: {neighbors_used_all}/{m}")
        print(f"  - Mateix gènere: {neighbors_used_gender}/{m}")
        
        if neighbors_used_gender < neighbors_used_all * 0.5:
            print(f"\n  ⚠ ADVERTÈNCIA: La segmentació redueix molt els veïns disponibles.")
            print(f"    Això pot afectar la qualitat de les prediccions.")
    
    elif prediction_all is not None and prediction_gender is None:
        print(f"\n⚠ Només s'ha pogut calcular la predicció amb tots els usuaris.")
        print(f"   Predicció: {prediction_all:.4f} estrelles")
        print(f"\n   Possible causa: Pocs usuaris del mateix gènere han valorat la pel·lícula.")
    
    elif prediction_all is None and prediction_gender is not None:
        print(f"\n⚠ Només s'ha pogut calcular la predicció amb usuaris del mateix gènere.")
        print(f"   Predicció: {prediction_gender:.4f} estrelles")
    
    else:
        print(f"\n No s'han pogut calcular prediccions en cap dels dos casos.")
    
    print("\n" + "="*80 + "\n")
    
    return prediction_all, prediction_gender
