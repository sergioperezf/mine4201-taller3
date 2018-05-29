"""USAGE: %(program)s PATH_TO_MOVIELENS_1M_DIR
"""

from datasets_mv.movielens import load_movies
from flurs.recommender.fm import FMRecommender
from flurs.evaluator import Evaluator
from flurs.data.entity import User, Item, Event
import numpy as np
import logging
import os
import sys
import pickle
from datetime import datetime, timedelta
import random
max_bytes = 2**31 - 1

def recommend_service(chosen):
    evaluator = pickle.load(open('evaluator.pckl', 'rb'))
    user_id =  273882713
    last = pickle.load(open('last.pckl', 'rb'))
    tfidfs = pickle.load(open('movies.pckl', 'rb'))
    item_ids_keyed = pickle.load(open('item_ids.pckl', 'rb'))
    # 70% incremental evaluation and updating
    logging.info('incrementally predict, evaluate and update the recommender')
    movie_names = pickle.load(open('movies_names.pckl', 'rb'))
    items = []
    user = User(len(evaluator.rec.users), np.zeros(0))

    if evaluator.rec.is_new_user(user.index):
        evaluator.rec.register_user(user)


    items_in_order = list(item_ids_keyed)
    for item_id in chosen:
        index = items_in_order.index(int(item_id))
        item = Item(index, tfidfs[int(item_id)])
        if evaluator.rec.is_new_item(item.index):
            evaluator.rec.register_item(item)
        items.append(item)

    events = []

    # Calculate time of the week
    date = datetime.now()
    weekday_vec = np.zeros(7)
    weekday_vec[date.weekday()] = 1

    if user_id in last:
        last_item_vec = last[user_id]['item']
        last_weekday_vec = last[user_id]['weekday']
    else:
        last_item_vec = np.zeros(49)
        last_weekday_vec = np.zeros(7)
        
    for item in items:
        others = np.concatenate((weekday_vec, last_item_vec, last_weekday_vec))
        events.append(Event(user, item, 1, others))
        last[user_id] = {'item': item.feature, 'weekday': weekday_vec}
        
    for e in events:
        evaluator.rec.update(e)

    # Re save pickles
    pickle.dump(evaluator, open('evaluator.pckl', 'wb'))
    pickle.dump(last, open('last.pckl', 'wb'))

    candidates = list(set(evaluator.item_buffer))
    recommendations = evaluator.rec.recommend(user, np.array(candidates), [0 for x in range(0, 63)])
    top_rec = recommendations[0][-10:]
    for top in reversed(top_rec):
        print(movie_names[list(item_ids_keyed)[top]])
recommend_service()