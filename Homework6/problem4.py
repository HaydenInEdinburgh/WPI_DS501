import math
import numpy as np
from problem3 import Tree
# -------------------------------------------------------------------------
'''
    Problem 4: Decision Tree (with continous attributes)
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `nosetests -v test4.py` in the terminal.
'''

# --------------------------


class Node:
    '''
        Decision Tree Node (with continous attributes)
        Inputs:
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar
            th: the threshold on the attribute, a float scalar.
            C1: the child node for values smaller than threshold
            C2: the child node for values larger than threshold
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''

    def __init__(self, X=None, Y=None, i=None, th=None, C1=None, C2=None, isleaf=False, p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.th = th
        self.C1 = C1
        self.C2 = C2
        self.isleaf = isleaf
        self.p = p
        self.C = None

# -----------------------------------------------


class DT(Tree):
    '''
        Decision Tree (with contineous attributes)
        Hint: DT is a subclass of Tree class in problem1. So you can reuse and overwrite the code in problem 1.
    '''

    # --------------------------
    @staticmethod
    def cutting_points(X, Y):
        '''
            Find all possible cutting points in the continous attribute of X.
            (1) sort unique attribute values in X, like, x1, x2, ..., xn
            (2) consider splitting points of form (xi + x(i+1))/2
            (3) only consider splitting between instances of different classes
            (4) if there is no candidate cutting point above, use -inf as a default cutting point.
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                cp: the list of  potential cutting points, a float numpy vector.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        from collections import defaultdict

        KV = [[y, x] for (x, y) in sorted(zip(X, Y))]
        x_u = np.unique(X)
        cp = []
        d = defaultdict(list)
        for y, x in KV:
            if y not in d[x]:
                d[x].append(y)
        for i in range(len(x_u) - 1):
            if d[x_u[i]] != d[x_u[i + 1]]:
                cp.append((x_u[i] + x_u[i + 1]) / 2)
        if cp == []:
            cp = [-np.inf]
        cp = np.array(cp)
        #########################################
        return cp

    # --------------------------
    @staticmethod
    def best_threshold(X, Y):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X.
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                th: the best threhold, a float scalar.
                g: the information gain by using the best threhold, a float scalar.
            Hint: you can reuse your code in problem 1.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        cp = DT.cutting_points(X, Y)
        KV = [[y, x] for (x, y) in sorted(zip(X, Y))]
        # cp = [2.5 3.5 4.5]
        if - float('inf') in cp:
            th = - float('inf')
            g = -1.
        else:
            g_list = []
            g_ini = Tree.entropy(Y)
            for thres in cp:
                x1 = [i for i in X if i < thres]
                x2 = [i for i in X if i >= thres]
                y1 = [i[0] for i in KV if i[1] < thres]
                y2 = [i[0] for i in KV if i[1] >= thres]
                # calculate entropy
                g1 = (len(y1) / len(Y)) * Tree.entropy(y1)
                g2 = (len(y2) / len(Y)) * Tree.entropy(y2)
                g = g_ini - g1 - g2
                g_list.append(g)

            g = max(g_list)
            th = cp[g_list.index(g)]
        #########################################
        return th, g

    # --------------------------
    def best_attribute(self, X, Y):
        '''
            Find the best attribute to split the node. The attributes have continous values (int/float).
            Here we use information gain to evaluate the attributes.
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''
        #########################################
        # INSERT YOUR CODE HERE

        gList = []
        thList = []
        for index in range(len(X)):
            th, g = self.best_threshold(X[index], Y)
            gList.append(g)
            thList.append(th)
        if (len(set(gList)) == 1) and(-1. in gList):
            i = 1.
            th = 1.5
        else:
            g = max(gList)
            i = gList.index(g)
            th = thList[i]
        print('gList', gList)
        print('thList', thList)
        #########################################
        return i, th

    # --------------------------

    @staticmethod
    def split(X, Y, i, th):
        '''
            Split the node based upon the i-th attribute and its threshold.
            (1) split the matrix X based upon the values in i-th attribute and threshold
            (2) split the labels Y
            (3) build children nodes by assigning a submatrix of X and Y to each node

            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C1: the child node for values smaller than threshold
                C2: the child node for values larger than (or equal to) threshold
        '''
        #########################################
        # INSERT YOUR CODE HERE
        print('X[i]', X[i])
        r_1, = np.where(X[i] < th)
        print(r_1)
        r_2, = np.where(X[i] >= th)
        x_C1 = X[:, r_1]
        x_C2 = X[:, r_2]
        y_C1 = Y[r_1, ]
        y_C2 = Y[r_2, ]
        # create the node
        C1 = Node(X=x_C1, Y=y_C1)
        C2 = Node(X=x_C2, Y=y_C2)
        #########################################
        return C1, C2

    # --------------------------

    def build_tree(self, t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape n by p.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C1: the child node for values smaller than threshold
                t.C2: the child node for values larger than (or equal to) threshold
        '''
        #########################################
        # INSERT YOUR CODE HERE
        n = Tree.most_common(t.Y)
        # if Condition 1 or 2 holds, stop recursion
        if Tree.stop1(t.Y) == True or Tree.stop2(t.X) == True:
            t.isleaf = True
            t.p = n
            t.C1 = None
            t.C2 = None
        # find the best attribute to split
        else:
            t.p = n
            X = t.X
            Y = t.Y
            t.i, t.th = DT().best_attribute(t.X, t.Y)
            t.C1, t.C2 = DT().split(t.X, t.Y, t.i, t.th)
            t.C = [t.C1, t.C2]
        # recursively build subtree on each child node
            for CNode in [t.C1, t.C2]:
                DT().build_tree(CNode)
        #########################################

        # --------------------------
    @staticmethod
    def inference(t, x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively.
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        t.X = x
        while not t.isleaf:
            print(t.X[t.i], t.th)
            if t.X[t.i] < t.th:
                t = t.C1
                break
            else:
                t = t.C2
                break
        y = t.p
        # predict label, if the current node is a leaf node

        #########################################
        return y

    # --------------------------

    @staticmethod
    def predict(t, X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset.
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        Y_list = [DT().inference(t, X[:, i]) for i in range(len(X.T))]
        Y = np.array(Y_list)
        #########################################
        return Y

    # --------------------------

    def train(self, X, Y):
        '''
            Given a training set, train a decision tree.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        t = Node(X=X, Y=Y)
        DT().build_tree(t)
        #########################################
        return t

    # --------------------------

    @staticmethod
    def load_dataset(filename='data2.csv'):
        '''
            Load dataset 2 from the CSV file: data2.csv.
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element is a float scalar.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element is a string.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        Load = np.genfromtxt(filename, delimiter=",",
                             dtype=str, skip_header=1)
        Y = Load[:, 0]
        X = (Load[:, 1:].T).astype(np.float)
        #########################################
        return X, Y
