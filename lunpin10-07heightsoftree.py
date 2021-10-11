#!/usr/bin/python3

import sys

class Node (object):
    def __init__ (self, data=None):
        self.data = data
        self.left = None
        self.right = None
        # self.children = list ()
    def getHeight (self):
        leftHeight = self.left.getHeight () if self.left is not None else 0
        rightHeight = self.right.getHeight () if self.right is not None else 0
        # rightHeight = 0 if self.right is None else self.right.getHeight ()
        return 1 + max (leftHeight, rightHeight)
        # return 1 + max ([child.getHeight() for child in children])

def getHeight (node):
    if node is None: return 0
    else: # DFS: Depth-First Search
        return 1 + max (getHeight (node.left), getHeight (node.right))
        # return 1 + max ([getHeight (child) for child in children])

def main ():
    # parsing inputs
    stdin = sys.stdin.read ().split ()
    n = int (stdin [0])
    inputs = [int (i) for i in stdin [1: n+1]]
    # first for-loop: create node instances for each input datapoint
    nodes = []
    for i in inputs: nodes.append (Node (i))
    # second for-loop: link nodes
    root = None
    for i in range (0, len (inputs)):
        if inputs [i] == -1: root = nodes [i]
        else:
            # nodes [inputs [i]] <------ nodes [i]
            node = nodes [inputs [i]]
            if node.left is None: node.left = nodes [i]
            elif node.right is None: node.right = nodes [i]
            else: raise Exception () # both left and right are used
            # nodes [inputs [i]].children.append (nodes [i])
    # print (getHeight (root))
    if root is None: print (0)
    else: print (root.getHeight ())


if __name__ == '__main__': main ()
