import numpy as np
import math
#-------------------------------------------------------------------------
'''
    Problem 3: optimization-based recommender systems (collaborative filtering)
    In this problem, you will implement a version of the recommender system using optimization-based method.
    You could test the correctness of your code by typing `nosetests test3.py` in the terminal.
'''

#--------------------------
def update_U(R, V, U, beta=.001, mu=1.):
    '''
        Update the matrix U (movie factors) by fixing matrix V using gradient descent. 
        Input:
            R: the rating matrix, a float numpy matrix of shape m by n. Here m is the number of movies, n is the number of users.
                If the rating is unknown, the number is 0. 
            V: the user factor matrix, a numpy float matrix of shape k X n. Here n is the number of users. 
            U: the current item (movie) factor matrix, a numpy float matrix of shape m X k. Here m is the number of movies (items).
            beta: step parameter for , a float scalar 
            mu: the parameter for reggradient descentularization term, a float scalar 
        Output:
            U: the updated item (movie) factor matrix, a numpy float matrix of shape m X k. Here m is the number of movies (items).
    '''

    #########################################
    ## INSERT YOUR CODE HERE
    s_R = np.shape(R)
    B = np.zeros((s_R))

    # compute a binary matrix, representing the elements with known ratings
    for i in range(len(R)):
        for j in range(len(R[0])):
            if R[i][j] == 0:
                B [i][j] = 0
            else:
                B [i][j] = 1
    # compute the gradient of matrix U 
    L = np.multiply((R - U.dot(V)),B)
    landa_U =(-2)* L.dot(V.T)+ 2*mu*U
    # compute the updated matrix U
    
    U = U- beta * landa_U

    #########################################
    return U

#--------------------------
def update_V(R, U, V, beta=.001, mu=1.):
    '''
        Update the matrix V (user factors) by fixing matrix U using gradient descent. 
        Input:
            R: the rating matrix, a float numpy matrix of shape m by n. Here m is the number of movies, n is the number of users.
                If the rating is unknown, the number is 0. 
            U: the item (movie) factor matrix, a numpy float matrix of shape m X k. Here m is the number of movies (items).
            V: the current user factor matrix, a numpy float matrix of shape k X n. Here n is the number of users. 
            beta: step parameter for gradient descent, a float scalar 
            mu: the parameter for regularization term, a float scalar 
        Output:
            V: the updated item (movie) factor matrix, a numpy float matrix of shape m X k. Here m is the number of movies (items).
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    # compute a binary matrix, representing the elements with known ratings
    s_R = np.shape(R)
    B = np.zeros((s_R))
    for i in range(len(R)):
        for j in range(len(R[0])):
            if R[i][j] == 0:
                B [i][j] = 0
            else:
                B [i][j] = 1
    # compute the gradient of matrix V
    L = np.multiply((R - U.dot(V)),B)
    landa_V = (-2)* U.T.dot(L)+2*mu*V

    # compute the updated matrix U
    V = V- beta * landa_V

    #########################################
    return V 
 

#--------------------------
def matrix_decoposition(R, k=5, max_steps=1000000, beta=.01, mu=.01):
    '''
        Compute the matrix decomposition for optimization-based recommender system.  
        Input:
            R: the rating matrix, a float numpy matrix of shape m by n. Here m is the number of movies, n is the number of users.
                If the rating is unknown, the number is 0. 
            k: the number of latent factors for users and items.
            max_steps: the maximium number of steps for gradient descent.
            beta: step parameter for gradient descent, a float scalar 
        Output:
            U: the item (movie) factor matrix, a numpy float matrix of shape m X k. Here m is the number of movies (items).
            V: the user factor matrix, a numpy float matrix of shape k X n. Here n is the number of users. 
    '''
    
    # initialize U and V with random values
    n_movies, n_users = R.shape
    U = np.random.rand(n_movies, k)
    V = np.random.rand(k, n_users)
    
    #########################################
    ## INSERT YOUR CODE HERE
    # gradient descent

    step = 0

    while step < max_steps:
        # fix U, update V
        V = update_V(R,U,V,beta,mu)
                
        # fix V, update U
        U = update_U(R,V,U ,beta,mu)
        
        step = step+1  





    #########################################
    return U, V 


