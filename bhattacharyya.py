import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
from math import gamma, pi, sin, exp, sqrt, pow, log
from random import normalvariate, randint, random
import copy

def rmse( prediction, ground_truth ):  #root mean square error
    prediction = prediction[ ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ ground_truth.nonzero() ].flatten()
    return sqrt( mean_squared_error( prediction, ground_truth ) )

def mea( prediction, ground_truth ):   #mean absolute error
    prediction = prediction[ ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ ground_truth.nonzero() ].flatten()
    return mean_absolute_error( prediction, ground_truth )

def precision_recall_F1( prediction, avg_user_rating, ground_truth, N ): #precision, recall and F1 calculation
    for i in range(n_users):
        for j in range(n_items):
            if ground_truth[i][j] == 0:
                prediction[i][j] = 0
                
    recommend = np.fliplr( np.argsort( prediction ) )
    precision = np.zeros( n_users )
    recall = np.zeros( n_users )
    F1 = np.zeros( n_users )
    for i in range( n_users ):
        tp = 0  #true positive
        fp = 0   #false positive
        fn = 0  #false negative
        for j in range(N):
            if ground_truth[i][ recommend[i][j] ] >= 3 :
                tp += 1
            else:
                fp += 1

        for p in range(n_items):
            if ground_truth[i][p] >= 3:
                if  np.nonzero( recommend[i] == p )[0][0] < N :
                    pass
                else:
                    fn += 1

        precision[i] = tp/( tp + fp )
        if tp + fn != 0 :
            recall[i] = tp/( tp + fn )
        if precision[i] != 0 and recall[i] != 0 :
            F1[i] = ( 2*precision[i]*recall[i] )/( precision[i] + recall[i] )

    return np.average(precision), np.average(recall), np.average(F1)



class similarity:
    def __init__( self, n_users=943, n_items=1682):
        pass

    def measure(self, data ):
        '''avg_item_rating = np.zeros((n_items))
        for j in range(n_items):
            s = 0
            count = 0
            for i in range(n_users):
                if data[i][j] > 0:
                    count += 1
                    s += data[i][j]
            if count != 0:
                avg_item_rating[j] = s/count'''

        avg_user_rating = np.zeros((n_users))
        count_movies_per_user = np.zeros((n_users))
        movieset_for_user = []   #movies that the user have watched
        for i in range(n_users):
            s = 0
            count = 0
            temp = []
            for j in range(n_items):
                if data[i][j] > 0:
                    count_movies_per_user[i] += 1
                    s += data[i][j]
                    temp.append(j)
            avg_user_rating[i] = s/count_movies_per_user[i]
            movieset_for_user.append(temp)

        user_dev = np.zeros((n_users))
        for i in range(n_users):
            num = 0
            for j in movieset_for_user[i]:
                num += pow( data[i][j] - avg_user_rating[i], 2 )
            user_dev[i] = sqrt( num/count_movies_per_user[i] )
            
        #calculating bhattacharyya coefficient for item pairs
        bch_coeff_matrix = np.zeros(( n_items, n_items ))
        item_ratings = np.zeros(( n_items, 6 ))
        for j in range(n_items):
            for i in range(n_users):
                d = int(data[i][j])
                if data[i][j] > 0:
                    item_ratings[j][0] += 1
                    item_ratings[j][ d ] += 1

        for i in range(n_items):
            for j in range(i+1):
                if i == j :
                    bch_coeff_matrix[i][j] = 1
                else:
                    term = 0
                    if item_ratings[i][0] != 0 and item_ratings[j][0] != 0 :
                        for p in range(1,5):
                            term += sqrt( (item_ratings[i][p]/item_ratings[i][0])*(item_ratings[j][p]/item_ratings[j][0]) )
                    bch_coeff_matrix[i][j] = term
                    bch_coeff_matrix[j][i] = term
                                                 
        #calculating similarity matrix
        sim_matrix = np.zeros((n_users, n_users))
        for i in range(n_users):
            j = 0
            #print(i)
            while j < i:
                jac_num = 0
                term = 0
                for p in movieset_for_user[i]:
                    for q in movieset_for_user[j]:
                        loc_num = (data[i][p] - avg_user_rating[i]) * (data[j][q] - avg_user_rating[j])
                        loc_den = user_dev[i]*user_dev[j]
                        loc = loc_num / loc_den
                        term += bch_coeff_matrix[p][q] * loc

                        if p == q :
                            jac_num += 1

                jac_den = count_movies_per_user[i] + count_movies_per_user[j] - jac_num
                jac = jac_num / jac_den
                sim_matrix[i][j] = jac + term
                sim_matrix[j][i] = jac + term

                j += 1
            sim_matrix[i][i] = 0

        return sim_matrix, avg_user_rating


    def prediction(self, sim_matrix, avg_user_rating, data, k):       #predicting the missing ratings
        #print('predicting')
        knn = np.zeros((n_users, n_users))
        for i in range(n_users):                                      # finding k nearest neighbors
            temp = np.argsort( sim_matrix[i] )
            #temp = np.flipud( temp )
            j = n_users-1
            p = 0
            while j >= 0:
                knn[i][p] = temp[j]
                j -= 1
                p += 1


        pred = np.zeros((n_users, n_items))  #initializing the prediction matrix
        #pred = copy.deepcopy(data)
        #print('predicting')
        for i in range(n_users):
            for j in range(n_items):
                num = 0
                den = 0
                if data[i][j] == 0:
                    p = 0
                    q = 0
                    while p < k and q < n_users:    #prediction using k nearest neighbors
                        index = knn[i][q]
                        if data[int(index)][j] > 0:
                            den += sim_matrix[i][int(index)]
                            num += sim_matrix[i][int(index)]*( data[int(index)][j] - avg_user_rating[int(index)] )
                            p += 1
                        q += 1
                    if den != 0:
                        val = avg_user_rating[i] + num/den       # final prediction
                    else:
                        val = avg_user_rating[i]
                    pred[i][j] = val
                    
        return pred

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u1.base', sep='\t', names=header)  #loading the ratings

n_users = 943
n_items = 1682
#print( n_users, n_items)

#train_data, test_data = cv.train_test_split( df, test_size = 0.20)

train_data_matrix = np.zeros( (n_users, n_items ) )
for line in df.itertuples():
    train_data_matrix[ line[1]-1, line[2]-1 ] = line[3]   #creating the training rating matrix



header = ['user_id', 'item_id', 'rating', 'timestamp']
dft = pd.read_csv('ml-100k/u1.test', sep='\t', names=header)

test_data_matrix = np.zeros( ( n_users, n_items ))
for line in dft.itertuples():
    test_data_matrix[ line[1]-1, line[2]-1 ] = line[3]  #creating the testing rating matrix


clf = similarity()
sim_matrix, avg_user_rating = clf.measure(train_data_matrix )
