'''
@File    :   tree_search.py
@Time    :   2022/11/16
@Author  :   Armin Sadeghi 
@Version :   0.1
@Contact :   a6sadegh@uwaterloo.ca
@Desc    :   None
'''


from policies.random_policy import RandomPolicy
from tree_search_util import DecisionTree, Node, Action
from queue import Queue
from policies.VelocityGrid import VelocityGrid
from config import *

class TreeSearchPolicy(RandomPolicy):
    def __init__(self, generator, policy_args=None) -> None:
        super().__init__(policy_args)
        self.action_set = policy_args['action_set']
        

    def _get_next_state(self, current_state : list, action : list):
        """returns the next state of the ego vehicle after taking action

        Args:
            current_state (list): the current location and orientation of the ego vehicle
            action (list): the speed input
        """
        # for now I assume simple speed input for the ego vehicle
        # TODO: maybe better if we define the action set with a set of accelerations 
        
        return [
            current_state[0] + action[0] * TICK_TIME,
            current_state[1] + action[1] * TICK_TIME
        ]
        
        
    def create_decision_tree(self, ego, velocity_grids):
        """
        create the decision tree given the velocity grids
        """
        max_tree_depth = len(velocity_grids)
        
        current_state = [
            ego.pos[0], ego.pos[1]
        ]
        
        tree = DecisionTree()
        root = Node (parent = -1, idx = 0, prob = 1, state=current_state)
        
        tree.add_node(root)
        visited = Queue()
        visited.put(root)
        
        tree_depth = 0
        
        
        while visited.qsize() > 0:
            current_node = visited.pop(0)
            for action in self.action_set:
                # get the future state of the ego vehicle after taking 
                # action 
                future_state = self._get_next_state(
                    current_node.state,
                    action
                )
                
                # TODO: find the visibility for the future state
                
                
                node = Node(
                    current_node.idx, 
                    tree.num_nodes, 
                    
                    )
                tree.add_node(node)

            tree_depth += 1
            if tree_depth > max_tree_depth:
                break
    
    def execute(
            self, 
            ego, 
            actors,
            velocity_grids,
            visibility=None,
            current_time=0, 
            max_solver_time=30, dt=0.1):
        """tree search policy

        Args:
            ego (_type_): the actor vehicle
            actors (_type_): other actors in the environment
            current_time (int, optional): _description_. Defaults to 0.
            max_solver_time (int, optional): _description_. Defaults to 30.
            dt (float, optional): _description_. Defaults to 0.1.
        """
        self.create_decision_tree(ego, velocity_grids)       
        
        
        
