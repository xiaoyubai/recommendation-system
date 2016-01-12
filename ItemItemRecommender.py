mport numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from time import time

class ItemItemRecommender(object):
    def __init__(self, neighborhood_size=75):
        '''
        Initialize the parameters of the model.
        '''
        self.neighborhood_size = neighborhood_size
        self.ratings_mat = None
        self.neighborhoods = None
        self.items_cos_sim = None
        self.relevent_items_dict = {}

    def fit(self, ratings_mat):
        '''
        Implement the model and fit it to the data passed as an argument.
        Store objects for describing model fit as class attributes.
        '''
        self.ratings_mat = ratings_mat
        self.items_cos_sim = cosine_similarity(self.ratings_mat.T)
        least_to_most_sim_indexes = np.argsort(self.items_cos_sim, 1)
        self.neighborhoods = self._set_neighborhoods(self.items_cos_sim)
        self._find_relevent_items()

    def _set_neighborhoods(self, items_cos_sim):
        '''
        Get the items most similar to each other item.
        Should set a class attribute with a matrix that is has
        number of rows equal to number of items and number of
        columns equal to neighborhood size. Entries of this matrix
        will be indexes of other items.
        You will call this in your fit method.
        '''
        least_to_most_sim_indexes = np.argsort(items_cos_sim, 1)
        neighborhoods = least_to_most_sim_indexes[:, -self.neighborhood_size:]
        return neighborhoods

    def _find_relevent_items(self):

        n_users = self.ratings_mat.shape[0]
        n_items = self.ratings_mat.shape[1]
        for user_id in range(n_users):
            items_rated_by_this_user = self.ratings_mat[user_id].nonzero()[1]
            output = np.zeros(n_items)
            relevant_items = []
            for item_to_rate in range(n_items):
                relevant_items.append(np.intersect1d(self.neighborhoods[item_to_rate],
                                                items_rated_by_this_user,
                                                assume_unique=True))
            self.relevent_items_dict[user_id] = relevant_items

    def pred_one_user(self, user_id=1, needtime=False):
        '''
        Accept user id as arg. Return the predictions for a single user.
        Optional argument to specify whether or not timing should be provided
        on this operation.
        '''
        if needtime:
            start_time = time()
        n_items = self.ratings_mat.shape[1]
        output = np.zeros(n_items)
        for item_to_rate in range(n_items):
            relevant_items = self.relevent_items_dict[user_id][item_to_rate]
            output[item_to_rate] = self.ratings_mat[user_id, relevant_items] * \
                self.items_cos_sim[item_to_rate, relevant_items] / \
                self.items_cos_sim[item_to_rate, relevant_items].sum()
            #Modify pred_one_user to replace the missing values with something numerical. numpy.nan_to_num is a good option for this.
            output[item_to_rate] = np.nan_to_num(output[item_to_rate])
        if needtime:
            end_time = time()
            duration = end_time - start_time
            print "duration for one user: %fs." %duration
        return output

    def pred_all_users(self):
        '''
        Repeated calls of pred_one_user, are combined into a single matrix.
        Return value is matrix of users (rows) items (columns) and predicted
        ratings (values).
        Optional argument to specify whether or not timing should be provided
        on this operation.
        '''
        output = np.zeros(self.ratings_mat.shape)
        num_users = self.ratings_mat.shape[0]
        for user_id in range(num_users):
            output[user_id] = self.pred_one_user(user_id=user_id)


    def top_n_recs(self, user_id, n):
        '''
        Take user_id argument and number argument.
        Return that number of items with the highest predicted ratings, after
        removing items that user has already rated.
        '''
        preds = self.pred_one_user(user_id=user_id)
        user_ratings = self.ratings_mat.todense()[user_id] #.reshape(preds.shape)[0]
        print user_ratings[0]
        print preds
        preds[user_ratings!=0] = 0
        return np.argsort(preds)[:-(n+1):-1]

def get_ratings_data():
    ratings_contents = pd.read_table("data/u.data",
                                     names=["user", "movie", "rating",
                                            "timestamp"])
    highest_user_id = ratings_contents.user.max()
    highest_movie_id = ratings_contents.movie.max()
    ratings_as_mat = sparse.lil_matrix((highest_user_id, highest_movie_id))
    for _, row in ratings_contents.iterrows():
        # subtract 1 from id's due to match 0 indexing
        ratings_as_mat[row.user - 1, row.movie - 1] = row.rating
    return ratings_contents, ratings_as_mat


if __name__ == '__main__':
    ratings_data_contents, ratings_mat = get_ratings_data()
    recom = ItemItemRecommender(neighborhood_size=75)
    recom.fit(ratings_mat)
    user_1_preds = recom.pred_one_user(user_id=2, needtime=True)
    print user_1_preds
    print recom.top_n_recs(1, 3)
    # all_user_preds = recom.pred_all_users()
    # print all_user_preds[:2]
