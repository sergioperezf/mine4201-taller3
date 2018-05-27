from flurs.data.entity import User, Item, Event

import os
import time
import numpy as np
from calendar import monthrange
from datetime import datetime, timedelta
import pickle
import csv

from sklearn.utils import Bunch


def load_movies(data_home, size):
    """Load movie genres as a context.
    Returns:
        dict of movie vectors: item_id -> numpy array (n_genre,)
    """
    all_genres = ['Action',
                  'Adventure',
                  'Animation',
                  "Children's",
                  'Comedy',
                  'Crime',
                  'Documentary',
                  'Children',
                  'Drama',
                  'Fantasy',
                  'Film-Noir',
                  'Horror',
                  'IMAX',
                  'Musical',
                  'Mystery',
                  'Romance',
                  'Sci-Fi',
                  'Thriller',
                  'War',
                  'Western',
                  '(no genres listed)']
    n_genre = len(all_genres)

    movies = {}

    if size == 'latest':
        with open(os.path.join(data_home, 'movies.csv'), encoding='ISO-8859-1') as f:
            lines = csv.reader(f, quotechar='"', delimiter=',',
                     quoting=csv.QUOTE_ALL, skipinitialspace=True)

            for item_id_str, title, genres in lines:
                movie_vec = np.zeros(n_genre)
                for genre in genres.split('|'):
                    i = all_genres.index(genre)
                    movie_vec[i] = 1.
                item_id = int(item_id_str)
                movies[item_id] = movie_vec

    return movies


def load_ratings(data_home, size):
    """Load all samples in the dataset.
    """

    if size == 'latest':
        with open(os.path.join(data_home, 'ratings2.csv'), encoding='ISO-8859-1') as f:
            lines = csv.reader(f, quotechar='"', delimiter=',',
                     quoting=csv.QUOTE_ALL, skipinitialspace=True)

            ratings = []

            for l in lines:
                # Since we consider positive-only feedback setting, ratings <= 4 will be excluded.
                if float(l[2]) > 4:
                    ratings.append(l)

    ratings = np.asarray(ratings)

    # sorted by timestamp
    return ratings[np.argsort(ratings[:, 3])]


def delta(d1, d2, opt='d'):
    """Compute difference between given 2 dates in month/day.
    """
    delta = 0

    if opt == 'm':
        while True:
            mdays = monthrange(d1.year, d1.month)[1]
            d1 += timedelta(days=mdays)
            if d1 <= d2:
                delta += 1
            else:
                break
    else:
        delta = (d2 - d1).days

    return delta


def fetch_movielens(data_home=None, size='100k'):
    assert data_home is not None
    print('Loading ratings.')
    ratings = pickle.load(open('ratings.pkl', 'rb'))
    print('Loading movies.')
    movies = pickle.load(open('movies.pkl', 'rb'))

    print(len(ratings))

    samples = []

    user_ids = []
    item_ids = []

    head_date = datetime(*time.localtime(int(ratings[0, 3]))[:6])
    dts = []
    user_ids_keyed = {}
    item_ids_keyed = {}
    last = {}
    print('creating dataset')
    i = 1
    last_item_vec_zeros = np.zeros(0)
    for user_id, item_id, rating, timestamp in ratings:
        a = datetime.now()

        #if i%100000 == 0:
            #print('rating ' + str(i))
        item_id = int(item_id)
        # give an unique user index
        if user_id in user_ids_keyed:
            u_index = user_ids_keyed[user_id]
        else:
            u_index = len(user_ids_keyed) + 1
            user_ids_keyed[user_id] = u_index
        
        #if user_id not in user_ids:
        #    user_ids.append(user_id)
        #u_index = user_ids.index(user_id)
        b = datetime.now()
        # give an unique item index
        if item_id in item_ids_keyed:
            i_index = item_ids_keyed[item_id]
        else:
            i_index = len(item_ids_keyed) + 1
            item_ids_keyed[item_id] = i_index
        #if item_id not in item_ids:
        #    item_ids.append(item_id)
        #i_index = item_ids.index(item_id)
        c = datetime.now()
        # delta days
        date = datetime(*time.localtime(int(timestamp))[:6])
        dt = delta(head_date, date)
        dts.append(dt)
        d = datetime.now()
        weekday_vec = np.zeros(7)
        weekday_vec[date.weekday()] = 1
        e = datetime.now()

        if user_id in last:
            last_item_vec = last[user_id]['item']
            last_weekday_vec = last[user_id]['weekday']
        else:
            last_item_vec = np.zeros(18)
            last_weekday_vec = np.zeros(7)
        f = datetime.now()
        others = np.concatenate((weekday_vec, last_item_vec, last_weekday_vec))
        g = datetime.now()
        user = User(u_index, [])
        item = Item(i_index, movies[item_id])
        h = datetime.now()
        sample = Event(user, item, 1., others)
        samples.append(sample)
        ii = datetime.now()

        #print('to:')

        if i%100000 == 0:
            print (i)
            print (len(item_ids_keyed))
            #print(((ii - a).microseconds))
        # if i % 1000000 == 0:
        #     f = open('samples' + str(i) + '.pkl', 'wb')
        #     pickle.dump(samples, f)
        #     f.close()
        #     samples = []

        # record users' last rated movie features
        last[user_id] = {'item': movies[item_id], 'weekday': weekday_vec}
        i = i+1
        
    
    file = open('item_ids.pkl', 'wb')
    pickle.dump(item_ids, file)
    file.close()

    file = open('user_ids.pkl', 'wb')
    pickle.dump(user_ids, file)
    file.close()

    file = open('samples.pkl', 'wb')
    pickle.dump(samples, file)
    file.close()

    # contexts in this dataset
    # 1 delta time, 18 genres, and 23 demographics (1 for M/F, 1 for age, 21 for occupation(0-20))
    # 7 for day of week, 18 for the last rated item genres, 7 for the last day of week
    return Bunch(samples=samples,
                 can_repeat=False,
                 contexts={'others': 7 + 18 + 7, 'item': 18, 'user': 23},
                 n_user=len(user_ids_keyed),
                 n_item=len(item_ids_keyed),
                 n_sample=len(samples))