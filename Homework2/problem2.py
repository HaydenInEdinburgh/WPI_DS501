from problem1 import elo_rating
import numpy as np
#-------------------------------------------------------------------------
'''
    Problem 2: 
    In this problem, you will use the Elo rating algorithm in problem 1 to rank the NCAA teams.
    You could test the correctness of your code by typing `nosetests test2.py` in the terminal.
'''

#--------------------------
def import_W(filename ='ncaa_results.csv'):
    '''
        import the matrix W of game results from a CSV file.
        In the CSV file, each line contains the result of one game. For example, (i,j) in the line represents the i-th team won, the j-th team lost in the game.
        Input:
                filename: the name of csv file, a string 
        Output: 
                W: the game result matrix, a numpy integer matrix of shape (n by 2)
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    W = np.loadtxt(open('ncaa_results.csv'),delimiter=',',skiprows=0)
    W = np.array(W,dtype=int)

    
    #########################################
    return W

#--------------------------
def import_team_names(filename ='ncaa_teams.txt'):
    '''
        import a list of team names from a txt file. Each line of text in the file is a team name.
        Each line of the file contains one team name. For example, the first line contains the name of the 0-th team.
        Input:
                filename: the name of txt file, a string 
        Output: 
                team_names: the list of team names, a python list of string values, such as ['team a', 'team b','team c'].
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    team_names = []
    with open('ncaa_teams.txt') as f:
        line=f.readline()

        while line:
            line=line.strip('\n')
            team_names.append(line)
            line = f.readline()
        f.close()
    #########################################
    return team_names

#--------------------------
def team_rating(resultfile = 'ncaa_results.csv', 
                teamfile='ncaa_teams.txt',
                K=16.):
    ''' 
        Rate the teams in the game results imported from a CSV file.
        (1) import the W matrix from `resultfile` file.
        (2) compute Elo ratings of all the teams
        (3) return a list of team names sorted by descending order of Elo ratings 

        Input: 
                resultfile: the csv filename for the game result matrix, a string.
                teamfile: the text filename for the team names, a string.
                K: a float scalar value, which is the k-factor of Elo rating system

        Output: 
                top_teams: the list of team names in descending order of their Elo ratings, a python list of string values, such as ['team a', 'team b','team c'].
                top_ratings: the list of elo ratings in descending order, a python list of float values, such as ['600.', '500.','300.'].
        
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    # load game results
    W = import_W(filename ='ncaa_results.csv')
    # load team names
    team_names = import_team_names(filename ='ncaa_teams.txt')
    # number of teams
    n_player = len(team_names)
    # compute Elo rating of the teams
    R = elo_rating(W, n_player, K= 16.)
    # sort team names
    top_ratings = sorted(R,reverse=True)

    top = sorted(range(len(R)), key=lambda k: R[k],reverse=True)
    top_teams = []
    for i in top:
        top_teams.append(team_names[i])

    #########################################
    return top_teams, top_ratings

