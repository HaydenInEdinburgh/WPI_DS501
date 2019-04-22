# -------------------------------------------------------------------------
# Note: please don't use any additional package except the following packages
import numpy as np
from problem1 import *
from problem2 import UCBplayer
# -------------------------------------------------------------------------
'''
    Problem 3: Monte Carlo Tree Search (MCTS)
    In this problem, you will implement a MCTS player for TicTacToe.
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
'''

# -----------------------------------------------


class Node():
    '''
        Search Tree Node
        Inputs:
            s: the current state of the game, an integer matrix of shape 3 by 3.
                s[i,j] = 0 denotes that the i-th row and j-th column is empty
                s[i,j] = 1 denotes that the i-th row and j-th column is taken by you. (for example, if you are the "O" player, then i, j-th slot is taken by "O")
                s[i,j] = -1 denotes that the i-th row and j-th column is taken by the opponent.
            isleaf: whether or not this node is a leaf node (an end of the game), a boolean scalar
    '''

    def __init__(self, s, parent=None, isleaf=False):
        self.s = s
        self.isleaf = isleaf
        self.parent = parent
        self.children = []
        self.N = 0  # number of times being selected
        self.w = 0  # sum of results
        self.b = None

# -------------------------------------------------------


class PlayerMCTS:
    '''a player, that chooses optimal moves by Monte Carlo tree search. '''

    # ----------------------------------------------
    @staticmethod
    def expand(node):
        '''
         Expand the current tree node.
         Add one child node for each possible next move in the game.
         Inputs:
                node: the current tree node to be expanded
         Outputs:
                c.children: a list of children nodes.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        g = TicTacToe()
        node.children = []
        # check if the game has already ende
        # if the game has already ended,  update node.isleaf and return
        e = g.check(node.s)
        if e != None:
            node.isleaf = True
        # if the game has not ended yet, expand the current node with one child node for each valid move
        else:
            # get a list of all valid moves
            r, c = np.where(node.s == 0)
            valid_index = list(zip(r, c))
        # if the game has already ended in the child node,  update isleaf
            for r, c in valid_index:
                sc = np.copy(node.s)
                n = Node(sc)
                n.s[r, c] = 1
                n.s = -n.s
                e = g.check(sc)
                if e != None:
                    n.isleaf = True
                n.parent = node
                node.children.append(n)
        #########################################
        # ----------------------------------------------

    @staticmethod
    def rollout(s):
        '''
         Monte Carlo simulation (rollout).
          Starting from the state s, simulates a randomized game from the current node until it reaches an end of the game.
          Inputs:
                s: the current state of the game, an integer matrix of shape 3 by 3.
                    s[i,j] = 0 denotes that the i-th row and j-th column is empty
                    s[i,j] = 1 denotes that the i-th row and j-th column is taken by you. (for example, if you are the "O" player, then i, j-th slot is taken by "O")
                    s[i,j] = -1 denotes that the i-th row and j-th column is taken by the opponent.
            Outputs:
                w: the result of the game (win:1, tie:0, lose: -1), an integer scalar.
            Hint: you could use PlayerRandom in problem 1.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        def randomplay(s):
            w = TicTacToe().check(s)
            if w == None:
                r, c = pr.play(s)
                sc = np.copy(s)
                sc[r, c] = 1
                w = TicTacToe().check(sc)
                if w == None:
                    r, c = pr.play(sc)
                    sc[r, c] = -1
                    w = TicTacToe().check(sc)
                    w = randomplay(sc)
            return w
        pr = PlayerRandom()
        w = randomplay(s)
        #########################################
        return w

    # ----------------------------------------------

    @staticmethod
    def backprop(c, w):
        '''
         back propagation, update the game result in parent nodes recursively until reaching the root node.
          Inputs:
                c: the current tree node to be updated
                w: the result of the game (win:1, tie:0, lose: -1), an integer scalar.
        '''
        #########################################
        # INSERT YOUR CODE HERE

        c.w += w
        c.N += 1
        if c.parent != None:
            w = (-1) * w
            PlayerMCTS.backprop(c.parent, w)

        #########################################

    # ----------------------------------------------

    @staticmethod
    def selection(c):
        '''
         select the child node with the highest bound recursively until reaching a leaf node or an unexpanded node.
          Inputs:
                c: the current tree node to be updated
          Outputs:
                node: the leaf/unexpanded node

        '''
        #########################################
        # INSERT YOUR CODE HERE
        if c.isleaf or c.children == []:
            return c

        queue = []
        queue.append(c)

        ansNode = 0
        while queue != [0]:
            n = queue.pop(0)
            children = n.children
            NParent = n.N

            maxUCB = -100
            maxChild = 0
            for child in children:
                curUCB = UCBplayer.UCB(child.w, child.N, NParent)
                maxUCB = max(maxUCB, curUCB)
                if curUCB == maxUCB:
                    maxChild = child
                    ansNode = maxChild
            queue = [maxChild]

        node = ansNode
        # return the node
        #########################################
        return node

    # ----------------------------------------------

    @staticmethod
    def build_tree(s, n=100):
        '''
        Given the current state of a game, build a search tree by n iteration of (selection->expand(selection)->rollout->backprop).
        After expanding a node, you need to run another selection operation starting from the expanded node, before performing rollout.
        Inputs:
            s: the current state of the game, an integer matrix of shape 3 by 3.
                s[i,j] = 0 denotes that the i-th row and j-th column is empty
                s[i,j] = 1 denotes that the i-th row and j-th column is taken by you. (for example, if you are the "O" player, then i, j-th slot is taken by "O")
                s[i,j] = -1 denotes that the i-th row and j-th column is taken by the opponent.
            n: number of iterations, an interger scalar
          Outputs:
                root: the root node of the search tree

        '''
        #########################################
        # INSERT YOUR CODE HERE

        # create a root node
        root = Node(s)
        # iterate n times
        for i in range(n):
            # selection from root node
            node = PlayerMCTS.selection(root)
        # expand
            expandNodes = PlayerMCTS.expand(node)
        # selection from expanded node
            sel_node = PlayerMCTS.selection(node)
        # rollout
            w = -1 * PlayerMCTS.rollout(sel_node.s)
        # backprop
            PlayerMCTS.backprop(sel_node, w)

        #########################################
        return root

    # ----------------------------------------------
    def play(self, s, n=100):
        '''
           the policy function of the MCTS player, which chooses one move in the game.
           Build a search tree with the current state as the root. Then find the most visited child node as the next action.
           Inputs:
                s: the current state of the game, an integer matrix of shape 3 by 3.
                    s[i,j] = 0 denotes that the i-th row and j-th column is empty
                    s[i,j] = 1 denotes that the i-th row and j-th column is taken by you. (for example, if you are the "O" player, then i, j-th slot is taken by "O")
                    s[i,j] = -1 denotes that the i-th row and j-th column is taken by the opponent.
            n: number of iterations when building the tree, an interger scalar
           Outputs:
                r: the row number, an integer scalar with value 0, 1, or 2.
                c: the column number, an integer scalar with value 0, 1, or 2.
          Hint: you could solve this problem using 3 lines of code.
        '''
        #########################################
        # INSERT YOUR CODE HERE
        node = self.build_tree(s)
        ini_child = [child.N for child in node.children]
        max_Child = np.argmax(ini_child)
        # choose next one
        r_next, c_next = np.where(s == 0)
        max_Index = list(zip(r_next, c_next))[max_Child]

        r = max_Index[0]
        c = max_Index[1]

        #########################################
        return r, c
