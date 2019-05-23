import pandas as pd
import numpy as np
from random import uniform, random, randint, choice
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt, log
import sys
import copy


def fitness( prediction, ground_truth, data, N=30):      #fitness function is 1/rmse
    new_prediction = prediction[ ground_truth.nonzero()].flatten()
    new_ground_truth = ground_truth[ ground_truth.nonzero() ].flatten()
    
    rmse = sqrt( mean_squared_error( new_prediction, new_ground_truth ) )
    mae = mean_absolute_error( new_prediction, new_ground_truth )


    #print('rmse=', rmse, 'mae=', mae)
    final_val = 1/rmse  
    return final_val


def calculate_similarity( weights, profile, k =30 ):       #distance between every pair of user is being calculated not similarity
    sim_matrix = np.zeros(( n_users, n_users ))
    m = 0
    knn = np.zeros(( n_users, k))
    p = 0
    for i in range(n_users):
        for j in range(n_users):
            if i==j:
                sim_matrix[i][j] = 0
            else:
                dist = 0
                
                age_diff = abs( profile[i][0]**2 - profile[j][0]**2 ) 
                dist += weights[0]*sqrt( age_diff ) / 5     # age difference of 5 has been taken as 1 unit

                dist += weights[1]*sqrt( abs( profile[i][1]**2 - profile[j][1]**2) )   #gender difference
                
                occ_diff = 0
                if profile[i][2] != profile[j][2] :      #occupation difference
                    occ_diff = 1
                dist += weights[2]*occ_diff

                for k in range(3, 21):
                    dist += weights[3] * sqrt( abs(profile[i][k]**2 - profile[j][k]**2) )       #difference in genre preference

                sim_matrix[i][j] = dist
                sim_matrix[j][i] = dist
                #if p < k :
                    
                '''if dist > m:
                    m = dist
    print('m = ', m)'''

    return sim_matrix
                
def prediction( sim_matrix, data, ground_truth, k=30):   #predicting the missing ratings
    #print('predicting')
    knn = np.zeros((n_users, n_users))       # finding k nearest neighbors
    for i in range(n_users):
        temp = np.argsort( sim_matrix[i] )
        #print(i)
        #temp = np.flipud( temp )
        j = 0
        while j < n_users:
            #print(j)
            if temp[j] != i:
                knn[i][j] = temp[j]
            j += 1


    pred = np.zeros((n_users, n_items))   #initializing the prediction matrix
    #pred = copy.deepcopy(data)
    #print('predicting')

    pred_ele = np.transpose( np.nonzero( ground_truth ) )  #finding the non-zero elements in the test data
    #print( pred_ele.shape )
    for row in pred_ele:
        i = int( row[0] )
        j = int( row[1] )
        num = 0
        den = 0
        p = 0
        q = 0
        while p < k and q < (n_users-1): #prediction using k nearest neighbors
            index = int( knn[i][q])
            if data[index][j] > 0:
                den += ( 1/sim_matrix[i][index] )
                num += ( data[index][j] - avg_user_rating[index] )*( 1/sim_matrix[i][index] )
                p += 1
            q += 1

        if den  != 0:
            val = avg_user_rating[i] + num/den    # final prediction
        else:
            val = avg_user_rating[i]

        pred[i][j] = val

    return pred


           


def jaya_and_evaluation( profile, data, test_data ):

    pop_size = 40
    dimensions = 4  #four parameters
    gens = 50
    min_value = -100
    max_value = 100

    #generating weights
    population = np.zeros(( pop_size, dimensions ))  
    for i in range(pop_size):
        for j in range(dimensions):
            population[i][j] = uniform( min_value, max_value )    

    sim_matrix = np.zeros(( pop_size, n_users, n_users ))
    for i in range(pop_size):
        sim_matrix[i] = calculate_similarity( population[i], profile )  #finding similarity/distance for every user pair in every element of population

    pred_matrix = np.zeros(( pop_size, n_users, n_items ))
    for i in range(pop_size):
        pred_matrix[i] = prediction( sim_matrix[i], data, test_data, 30 ) #calculating prediction  
        
    
    best_fitness = 0
    best_index = -1
    best_ind = np.zeros(( dimensions ))
    worst_fitness = sys.maxsize
    worst_index = -1
    worst_ind = np.zeros(( dimensions ))
    fit_matrix = np.zeros(( pop_size ))

    for i in range(pop_size):
        fitness_value = fitness( pred_matrix[i], test_data, data )  #calculating fitness
        #print(fitness_value)
        fit_matrix[i] = fitness_value
        if fitness_value > best_fitness :  #finding best and worst fitness, here its an maximization function
            best_fitness = fitness_value
            best_index = i
            best_ind = population[i]

        if fitness_value < worst_fitness :
            worst_fitness = fitness_value
            worst_index = i
            worst_ind = population[i]

    #modifying the weights of every feature (demographic and genre preference) using modified jaya

    for g in range( gens ):
        print( 'best_fitness =', 1/best_fitness )
        print('generations=', g)
        for i in range(pop_size):
            new_value = np.zeros(( dimensions ))
            for j in range(dimensions):
                new_value[j] = population[i][j] + random()*( best_ind[j] - abs(population[i][j]) ) - random()*( worst_ind[j] - abs(population[i][j])) #jaya equation
            temp_sim_matrix = calculate_similarity( new_value, profile )
            temp_pred_matrix = prediction( temp_sim_matrix, data, test_data, 30 )
            new_value_fit = fitness( temp_pred_matrix, test_data, data )
            current_fit = fit_matrix[i]
            if new_value_fit > current_fit: #replacing current element with new element if it has better fitness
                population[i] = new_value
                fit_matrix[i] = new_value_fit

        index = np.argmax( fit_matrix )
        index = int( index )
        if fit_matrix[ index ] > best_fitness :  # finding the best and worst element
            best_fitness = fit_matrix[ index ]
            best_index = index
            best_ind = population[ index ]

        index = np.argmin( fit_matrix )
        index = int( index )
        if fit_matrix[ index ] < worst_fitness :
            worst_fitness = fit_matrix[index]
            worst_index = index
            worst_ind = population[index]


        #modification in jaya
        sign = [ -1, 1 ]
        temp = np.argsort( fit_matrix ) #the most fit are in the end
        #print(temp)
        new_inds = (int)(pop_size/10)  
        for i in range(new_inds):
            for j in range(dimensions):
                if g > gens/2:   #earlier generations
                    if randint(1,100) <= 75: # 75% chance 
                        population[ temp[i] ][j] = best_ind[j] + uniform(min_value/3, max_value/3)*abs( best_ind[j] )*choice( sign ) #new value should be far away from best individual 
                    else:
                        population[ temp[i] ][j] = best_ind[j] + random()*abs( best_ind[j] )*choice( sign ) # new value should be closer to best individual, randint() produces number between 0 & 1
                else:  later 50% of generationgs the ratio is reversed
                    if randint(1,100) > 75:
                        population[ temp[i] ][j] = best_ind[j] + uniform(min_value/3, max_value/3)*abs( best_ind[j] )*choice( sign )
                    else:
                        population[ temp[i] ][j] = best_ind[j] + random()*abs( best_ind[j] )*choice( sign )

            sim_matrix[temp[i]] = calculate_similarity( population[temp[i]], profile )
            pred_matrix[temp[i]] = prediction( sim_matrix[temp[i]], data, test_data, 30 )


            
            
    

item_header = [ 'movie_id', 'movie title', 'release date', 'video release date',
           'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

idf = pd.read_csv('ml-100k/u.item', sep='|', names=item_header, encoding='latin-1' )  #getting item information (genre)
n_items = idf.movie_id.unique().shape[0]

genre = np.zeros(( n_items, 18))
for line in idf.itertuples():    #storing genre of every movie
    for j in range(18):
        genre[line[1]-1][j] = line[j+7]


user_header = ['user_id', 'item_id', 'rating', 'timestamp']
udf = pd.read_csv('ml-100k/u1.base', sep='\t', names=user_header ) #loading the ratings
n_users = 943

train_data_matrix = np.zeros((n_users, n_items))
for line in udf.itertuples():   #making the training rating matrix
    train_data_matrix[ line[1]-1 ][ line[2]-1 ] = line[3]

test_header = ['user_id', 'item_id', 'rating', 'timestamp']
tdf = pd.read_csv('ml-100k/u1.test', sep='\t', names=user_header )

test_data_matrix = np.zeros((n_users, n_items))
for line in tdf.itertuples():  #making the testing rating matrix
    test_data_matrix[ line[1]-1 ][ line[2]-1 ] = line[3]

demo_header = ['user_id', 'age', 'gender', 'occupation', 'pincode' ]
ddf = pd.read_csv( 'ml-100k/u.user', sep='|', names=demo_header )   #loading user information

gender = { 'F': 0, 'M': 1 }

occupation = {}
integer = 0
for line in ddf.itertuples():
    job = line[4]
    try:
        _ = occupation[job]
    except:
        occupation[job] = integer
        integer += 1


demography = np.zeros((n_users, 3))
for line in ddf.itertuples():  #preparing user demographic profile
    demography[ line[1]-1 ][0] = line[2]
    demography[ line[1]-1 ][1] = gender[ line[3] ]
    demography[ line[1]-1 ][2] = occupation[ line[4] ]

avg_user_rating = np.zeros((n_users))
for i in range(n_users):  #calculating average rating of every user
    s = 0
    count = 0
    for j in range(n_items):
        if train_data_matrix[i][j] > 0:
            count += 1
            s += train_data_matrix[i][j]
    avg_user_rating[i] =  s/count
    

profile = np.zeros(( n_users, 21 ))
for i in range(n_users):
    for j in range(3):
        profile[i][j] = demography[i][j]

for i in range(n_users):
    for j in range(n_items):
        curr_rating = train_data_matrix[i][j]
        if curr_rating > 0 :
            diff = -avg_user_rating[i] + curr_rating + 0.5  #generating genre preference for user profile
            profile[i][3:] += diff*genre[j]


jaya_and_evaluation( profile, train_data_matrix, test_data_matrix)


            
            


































        





    
