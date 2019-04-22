import math
import numpy as np
import heapq as hq
import pandas as pd
#-------------------------------------------------------------------------
'''
    Problem 2: User-based recommender systems
    In this problem, you will implement a version of the recommender system using user-based method.
    You could test the correctness of your code by typing `nosetests test2.py` in the terminal.
'''

#--------------------------
def cosine_similarity(RA, RB):
    '''
        compute the cosine similarity between user A and user B. 
        The similarity values between users are measured by observing all the items which have been rated by BOTH users. 
        If an item is only rated by one user, the item will not be involved in the similarity computation. 
        You need to first remove all the items that are not rated by both users from RA and RB. 
        If the two users don't share any item in their ratings, return 0. as the similarity.
        Then the cosine similarity is < RA, RB> / (|RA|* |RB|). 
        Here <RA, RB> denotes the dot product of the two vectors (see here https://en.wikipedia.org/wiki/Dot_product). 
        |RA| denotes the L-2 norm of the vector RA (see here for example: http://mathworld.wolfram.com/L2-Norm.html). 
        For more details, see here https://en.wikipedia.org/wiki/Cosine_similarity.
        Input:
            RA: the ratings of user A, a float python vector of length m (the number of movies). 
                If the rating is unknown, the number is 0. For example the vector can be like [0., 0., 2.0, 3.0, 0., 5.0]
            RB: the ratings of user B, a float python vector
                If the rating is unknown, the number is 0. For example the vector can be like [0., 0., 2.0, 3.0, 0., 5.0]
        Output:
            S: the cosine similarity between users A and B, a float scalar value between -1 and 1.
        Hint: you could use math.sqrt() to compute the square root of a number
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    # assuming cosine(RA, RB) = P/(sqrt(NA)*sqrt(NB))

    # loop through two lists

        # if both users rated the item 
    RRA=[]
    RRB=[]
    for i in range(len(RA)):
        if RA[i] == 0 or RB[i] == 0:
            RRA.append(0)
            RRB.append(0)

        else:
            RRA.append(RA[i])
            RRB.append(RB[i])
    # if the two user share no item in their ratings

    # compute cosine similarity on the shared items
    

    # if the two user share no item in their ratings

    if any(RRA) == True:
        P = np.dot(RRA, RRB)
        SA = np.linalg.norm(RRA, ord=2)
        SB = np.linalg.norm(RRB, ord=2)
        S = P / (SA * SB)
        S = np.float(S)
    else:
        S = 0.
    # compute cosine similarity on the shared items 
    
    #########################################
    return S 


#--------------------------
def find_users(R, i):
    '''
        find the all users who have rated the i-th movie.  
        Input:
            R: the rating matrix, a float numpy matrix of shape m by n. Here m is the number of movies, n is the number of users.
                If a rating is unknown, the number is 0. 
            i: the index of the i-th movie, an integer python scalar (Note: the index starts from 0)
        Output:
            idx: the indices of the users, a python list of integer values 
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    idx = []
    for j in range(len(R[0])):
        if R[i][j] != 0:
            idx.append(j)
    
    #########################################
    return idx

#--------------------------
def user_similarity(R, j, idx):
    '''
        compute the cosine similarity between a collection of users in idx list and the j-th user.  
        Input:
            R: the rating matrix, a float numpy matrix of shape m by n. Here m is the number of movies, n is the number of users.
                If a rating is unknown, the number is 0. 
            j: the index of the j-th user, an integer python scalar (Note: the index starts from 0)
            idx: a list of user indices, a python list of integer values 
        Output:
            sim: the similarity between any user in idx list and user j, a python list of float values. It has the same length as idx.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    RA = R[:,j]
    sim = []
    for i in idx:
        RB = R[:,i]
        S = cosine_similarity(RA,RB)
        sim.append(S)



    #########################################
    return sim 


#--------------------------
def user_based_prediction(R, i_movie, j_user, K=5):
    '''
        Compute a prediction of the rating of the j-th user on the i-th movie using user-based approach.  
        First we take all the users who have rated the i-th movie, and compute their similarities to the target user j. 
        If there is no user who has rated the i-th movie, predict 3.0 as the default rating.
        From these users, we pick top K similar users. 
        If there are less than K users who has rated the i-th movie, use all these users.
        We weight the user's ratings on i-th movie by the similarity between that user and the target user. 
        Finally, we rescale the prediction by the sum of similarities to get a reasonable value for the predicted rating.
        Input:
            R: the rating matrix, a float numpy matrix of shape m by n. Here m is the number of movies, n is the number of users.
                If the rating is unknown, the number is 0. 
            i_movie: the index of the i-th movie, an integer python scalar
            j_user: the index of the j-th user, an integer python scalar
            K: the number of similar users to compute the weighted average rating.
        Output:
            p: the predicted rating of user j on movie i, a float scalar value between 1. and 5.
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    # find all other users who have rated movie i.
    i = i_movie
    j = j_user
    rate_index=[ x for x in range(len(R[i])) if R[i][x] != 0]
    movie_rate = []
    #rate_index is a list
    if len(rate_index)==0:
        p = 3.
    # if there are less than K users who have rated the movie, change K to the number of users
    else:

        if len(rate_index) <= K:
            K = len(rate_index)

    # compute the similarity between all of these users with user j
        sim = user_similarity(R,j,rate_index)
        index_sim = dict(zip(rate_index,sim))
        sim_K = hq.nlargest(K,sim)
        sorted_dic =sorted(index_sim.items(), key=lambda d: d[1],reverse=True)
        sorted_list = sorted_dic[:K]

    #R_rating = R[i]
        p_sum = []
        f = 0
        for d in range(K):
            p_times = (R[i][sorted_list[d][0]])* sorted_list[d][1]
            p_sum.append(p_times)
            f += 1
            p = sum(p_sum)/sum(sim_K)









    #########################################
    return p 


#--------------------------
def compute_RMSE(ratings_pred, ratings_real):
    '''
        Compute the root of mean square error of the rating prediction.
        Input:
            ratings_pred: predicted ratings, a float python list
            ratings_real: real ratings, a float python list
        Output:
            RMSE: the root of mean squared error of the predicted rating, a float scalar.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    sum_square_dif = 0
    for i in range(len(ratings_pred)):
        square_dif = pow(ratings_pred[i]-ratings_real[i],2)
        sum_square_dif += square_dif
    RMSE = math.sqrt(sum_square_dif/len(ratings_pred))

    #########################################
    return RMSE



#--------------------------
def load_rating_matrix(filename = 'movielens_train.csv'):
    '''
        Load the rating matrix from a CSV file.  In the CSV file, each line represents (user id, movie id, rating).
        Note the ids start from 1 in this dataset.
        Input:
            filename: the file name of a CSV file, a string
        Output:
            R: the rating matrix, a float numpy array of shape m by n. Here m is the number of movies, n is the number of users.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    data = pd.read_csv('movielens_train.csv')
    data_mat = np.loadtxt(open('movielens_train.csv'),delimiter=',')

    m = data['1'].max()
    #943 movies
    n = data['1.1'].max()
    #1682 users
    R = np.zeros( (n,m) )  

    for e in data_mat:
        R[int(e[1])-1,int(e[0])-1] = e[2]








    #########################################
    return R


#--------------------------
def load_test_data(filename = 'movielens_test.csv'):
    '''
        Load the test data from a CSV file.  In the CSV file, each line represents (user id, movie id, rating).
        Note the ids in the CSV file start from 1. But the indices in u_ids and m_ids start from 0.
        Input:
            filename: the file name of a CSV file, a string
        Output:
            m_ids: the list of movie ids, an integer python list of length n. Here n is the number of lines in the test file. (Note indice should start from 0)
            u_ids: the list of user ids, an integer python list of length n. 
            ratings: the list of ratings, a float python list of length n. 
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    dt = np.loadtxt(open('movielens_test.csv'),delimiter=',')
    #get user_number & movie_number
    m_ids =[]
    u_ids =[]
    ratings=[]
    for e in dt:
        u_ids.append(int(e[0])-1)
        m_ids.append(int(e[1])-1)
        ratings.append(float(e[2]))





    #########################################
    return m_ids, u_ids, ratings


#--------------------------
def movielens_user_based(train_file='movielens_train.csv', test_file ='movielens_test.csv', K = 5):
    '''
        Compute movie ratings in movielens dataset. Based upon the training ratings, predict all values in test pairs (movie-user pair).
        In the training file, each line represents (user id, movie id, rating).
        Note the ids start from 1 in this dataset.
        Input:
            train_file: the train file of the dataset, a string.
            test_file: the test file of the dataset, a string.
            K: the number of similar users to compute the weighted average rating.
        Output:
            RMSE: the root of mean squared error of the predicted rating, a float scalar.
    Note: this function may take 1-5 minutes to run.
    '''
   
    # load training set
    R = load_rating_matrix(train_file)

    # load test set
    m_ids, u_ids,ratings_real = load_test_data(test_file)

    # predict on test set
    #########################################
    ## INSERT YOUR CODE HERE
    ratings_pred = []
    for n in range(len(m_ids)):
        i_movie = m_ids[n]
        j_user  = u_ids[n]
        p = user_based_prediction(R, i_movie, j_user, K)
        ratings_pred.append(p)
    



    #########################################
    # compute RMSE 
    RMSE = compute_RMSE(ratings_pred,ratings_real)
    return  RMSE 


