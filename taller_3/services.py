from collections import defaultdict
import pickle
import operator

def load_dataset():
    dataset_4 = pd.read_pickle('matriz_final.pickle')
    def get_item(key):
        return dataset_4[key[0]]
    sorted_d = sorted(dataset_4.items(), key=get_item)
    return sorted_d

def get_possible_movies():
    movies = pickle.load(open('movies_names.pkl', 'rb'))
    
    def get_item(key):
        return movies[key[0]]
    movies = sorted(movies.items(), key=get_item)
    
    return movies

def get_recommendation_by_movies(artists, user):
    if (not len(artists)):
        return {}
    dataset = load_dataset()

    dataset_4_test = dataset.copy(deep = True)
    data = {
            'USERID': ['user_001001' for x in artists],
            'ARTIST_NAME': artists,
            'VALOR': rankings}
    x = pd.DataFrame(data = data)
    dataset_4_test = dataset_4_test.append(x, ignore_index = True)


    kf = KFold(n_splits = 2)
    reader = Reader(rating_scale = (1, 5))
    data = Dataset.load_from_df(dataset_4_test[['USERID', 'ARTIST_NAME', 'VALOR']], reader)

    for trainset, testset in kf.split(data):
        sim_options = {'name': 'pearson', 'user_based': False}#'min_k': 1, 'k': 1000, 
        algo = KNNBasic(sim_options = sim_options)
        algo.fit(trainset)
        
        test_set = trainset.build_anti_testset()
        
        test = [x for x in test_set if x[0] is "user_001001"]

        predictions = algo.test(test)

    def get_top_n(predictions, n = 10):
        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key = lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n
    
    def get_top_n(predictions, n = 10):
        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key = lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n

    top_n = get_top_n(predictions, n = 10)
    top_n = dict(top_n)
    try:
        return top_n['user_001001']
    except KeyError:
        print('recommendations not found')
        return {}
