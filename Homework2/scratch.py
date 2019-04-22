from problem1 import elo_rating
import numpy as np


def import_W(filename='ncaa_results.csv'):
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
    W = np.loadtxt(open('ncaa_results.csv'), delimiter=',', skiprows=0)
    W = np.array(W, dtype=int)

    #########################################
    return W


def import_team_names(filename='ncaa_teams.txt'):
    team_names = []
    with open(filename) as f:
        line = f.readline()

        while line:
            line = line.strip('\n')
            team_names.append(line)
            line = f.readline()
        f.close()
    return team_names


# load game results
W = import_W(filename='ncaa_results.csv')
# load team names
team_names = import_team_names(filename='ncaa_teams.txt')
# number of teams
n_player = len(team_names)
# compute Elo rating of the teams
R = elo_rating(W, n_player, K=16.)

top_ratings = sorted(R,reverse=True)
print n_player
print R
print top_ratings

top = sorted(range(len(R)), key=lambda k: R[k],reverse=True)
top_teams = []
for i in top:
    top_teams.append(team_names[i])
print top_teams