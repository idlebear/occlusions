from math import inf

class Action:
    def __init__(self, idx, reward = inf):
        self.action_idx = idx
        self.reward = reward

class Node:
    def __init__(self, parent, idx, prob):
        self.children = []
        self.parent = parent
        self.idx = idx
        self.prob = prob

    def add_child(self, child, action):
        self.children.append([child, action])

    
        

class DecisionTree:
    def __init__(self):
        self.nodes = []
        self.cost_to_node = []

    def add_node(self, node):
        self.nodes.append(node)
        self.cost_to_node.append(inf)

    @property
    def num_nodes(self):
        return len(self.nodes)

    def DFS(self, visited=[], current_node = 0)->list:
        """
        DFS algorithm to search on the tree
        """
        for child_idx, action in self.nodes[current_node].children:
            if child_idx in visited:
                continue
            child = self.nodes[child_idx]
            visited.append(current_node)
            self.cost_to_node[current_node] = self.cost_to_node[child.parent] + action.reward*child.prob
            self.DFS(visited, current_node = child.idx)



def test():
    """
    test the tree construction and simple dfs search
    """
    from random import choice
    action_set = {"a":1, "b":2, "c":20, "d":inf}
    tree = DecisionTree()
    
    node1 = Node(-1, 0, 1)
    node2 = Node(0, 1, 0.1)
    node3 = Node(0, 2, 0.3)
    node4 = Node(0, 3, 0.9)
    node5 = Node(1, 4, 0.2)
    
    action_idx = choice(list(action_set.items()))
    node1.add_child(1, Action(action_idx[0], action_idx[1]))
    tree.add_node(node1)
    tree.add_node(node2)
    tree.DFS()
if __name__ == "__main__":
    test()
    try:
        test()
    except Exception as e:
        print("error in testing with message: {0}".format(str(e)))
    
