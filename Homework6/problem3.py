import math
import numpy as np
from collections import Counter
# -------------------------------------------------------------------------
'''
    Problem 3: Decision Tree (with Descrete Attributes)
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''

# -----------------------------------------------


class Node:
    '''
        Decision Tree Node (with discrete attributes)
        Inputs:
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float/string.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar
            C: the dictionary of attribute values and children nodes.
               Each (key, value) pair represents an attribute value and its corresponding child node.
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''

    def __init__(self, X, Y, i=None, C=None, isleaf=False, p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.C = C
        self.isleaf = isleaf
        self.p = p

# -----------------------------------------------


class Tree(object):
    '''
        Decision Tree (with discrete attributes).
        We are using ID3(Iterative Dichotomiser 3) algorithm. So this decision tree is also called ID3.
    '''
    # --------------------------
    @staticmethod
    def entropy(Y):
        '''
            Compute the entropy of a list of values.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                e: the entropy of the list of values, a float scalar
            Hint: you could use collections.Counter.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        e = 0
        c = Counter()
        for i in Y:
            c[i] = c[i] + 1
        f = dict(c)
        s = sum(c.values())
        for j in f:
            l = float(f[j]) / s
            e -= l * math.log(l, 2)

        #########################################
        return e

    # --------------------------

    @staticmethod
    def conditional_entropy(Y, X):
        '''
            Compute the conditional entropy of y given x.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
                X: a list of values, a numpy array of int/float/string values.
            Output:
                ce: the conditional entropy of y given x, a float scalar
        '''
        #########################################
        # INSERT YOUR CODE HERE
        ce = 0
        x_list = []
        for i in range(len(X)):
            x_list.append(X[i])
        x_set = set(x_list)
        for v in x_set:
            y = Y[X == v]
            ce += Tree.entropy(y) * float(len(y) / len(Y))
        #########################################
        return ce

    # --------------------------

    @staticmethod
    def information_gain(Y, X):
        '''
            Compute the information gain of y after spliting over attribute x
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                g: the information gain of y after spliting over x, a float scalar
        '''
        #########################################
        # INSERT YOUR CODE HERE
        g = Tree.entropy(Y) - Tree.conditional_entropy(Y, X)
        #########################################
        return g

    # --------------------------

    @staticmethod
    def best_attribute(X, Y):
        '''
            Find the best attribute to split the node.
            Here we use information gain to evaluate the attributes.
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
        '''
        #########################################
        # INSERT YOUR CODE HERE
        X_list = [Tree.information_gain(Y, X[i]) for i in range(len(X))]
        i = X_list.index(max(X_list))
        #########################################
        return i

    # --------------------------

    @staticmethod
    def split(X, Y, i):
        '''
            Split the node based upon the i-th attribute.
            (1) split the matrix X based upon the values in i-th attribute
            (2) split the labels Y based upon the values in i-th attribute
            (3) build children nodes by assigning a submatrix of X and Y to each node
            (4) build the dictionary to combine each  value in the i-th attribute with a child node.

            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C: the dictionary of attribute values and children nodes.
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        X_uni = np.unique(X[i])
        C = dict()
        for j in X_uni:
            r, = np.where(X[i] == j)
            C[j] = Node(X[:, r], Y[r])
        #########################################
        return C

    # --------------------------
    @staticmethod
    def stop1(Y):
        '''
            Test condition 1 (stop splitting): whether or not all the instances have the same label.

            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                s: whether or not Conidtion 1 holds, a boolean scalar.
                True if all labels are the same. Otherwise, false.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        Y_list = list(Y)
        s = (Y_list.count(Y[0]) == len(Y_list))
        #########################################
        return s

    # --------------------------
    @staticmethod
    def stop2(X):
        '''
            Test condition 2 (stop splitting): whether or not all the instances have the same attributes.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
            Output:
                s: whether or not Conidtion 2 holds, a boolean scalar.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        n = 0
        for i in range(len(X)):
            x_unit = list(X[i])
            if x_unit.count(X[i, 0]) == len(x_unit):
                n += 1
        s = n == len(X)
        #########################################
        return s

    # --------------------------

    @staticmethod
    def most_common(Y):
        '''
            Get the most-common label from the list Y.
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        y = max(list(Y), key=list(Y).count)
        #########################################
        return y

    # --------------------------

    @staticmethod
    def build_tree(t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape n by p.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C: the dictionary of attribute values and children nodes.
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        # INSERT YOUR CODE HERE

        # if Condition 1 or 2 holds, stop recursion
        if Tree.stop1(t.Y) == True or Tree.stop2(t.X) == True:
            # find the best attribute to split
            t.isleaf = True
            t.p = Tree.most_common(t.Y)
            t.C = None
        # recursively build subtree on each child node
        else:
            t.i = Tree.best_attribute(t.X, t.Y)
            t.p = Tree.most_common(t.Y)
            t.C = Tree.split(t.X, t.Y, t.i)
            for i in t.C:
                Tree.build_tree(t.C[i])
        #########################################

    # --------------------------
    @staticmethod
    def train(X, Y):
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
        t = Node(X, Y)
        Tree.build_tree(t)
        #########################################
        return t

    # --------------------------

    @staticmethod
    def inference(t, x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively.
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float/string.
            Output:
                y: the class label, a scalar, which can be int/float/string.
        '''
        #########################################
        # INSERT YOUR CODE HERE

        while not t.isleaf:
            if x[t.i] in t.C:
                t = t.C[x[t.i]]
            else:
                break
        y = t.p
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
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        Y_list = [Tree().inference(t, X[:, i]) for i in range(len(X.T))]
        Y = np.array(Y_list)
        #########################################
        return Y

    # --------------------------

    @staticmethod
    def load_dataset(filename='data1.csv'):
        '''
            Load dataset 1 from the CSV file: 'data1.csv'.
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element is a string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element is a string.
            Note: Here you can assume the data type is always str.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        Load = np.genfromtxt(("data1.csv"), delimiter=",",
                             dtype=str, skip_header=1)
        Y = Load[:, 0]
        X = Load[:, 1:].T
        #########################################
        return X, Y
