'''
@File    :   tree_search.py
@Time    :   2022/11/16
@Author  :   Armin Sadeghi 
@Version :   0.1
@Contact :   a6sadegh@uwaterloo.ca
@Desc    :   None
'''


from policies.random_policy import RandomPolicy
from policies.tree_search_util import DecisionTree, Node, Action, State
from queue import Queue
from policies.VelocityGrid import VelocityGrid
from config import *
from Actor import Actor
from policies.flow.flow import flow

from config import *

GRID_HEIGHT = 0.2
GRID_WIDTH = 1.0
GRID_RESOLUTION = 0.01
GRID_ORIGIN_Y_OFFSET = -GRID_HEIGHT / 2
GRID_ORIGIN_X_OFFSET = EGO_X_OFFSET

DEBUG = True
FORECAST_COUNT = 5
FORECAST_DT = 0.1

class TreeSearchPolicy(RandomPolicy):
    def __init__(self, generator, policy_args=None) -> None:
        super().__init__(policy_args)
        
        try:
            self.action_set = policy_args['action_set']
        except:
            pass
        
        try:
            self.max_depth = policy_args['max_depth']
        except:
            self.max_depth = 2
            
        self.generator = generator
        
        self.grid = VelocityGrid( 
                                height=GRID_HEIGHT, 
                                width=GRID_WIDTH, 
                                resolution=GRID_RESOLUTION, 
                                origin=(GRID_ORIGIN_Y_OFFSET, GRID_ORIGIN_Y_OFFSET) )
        

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
        
        
    def create_decision_tree(self, ego : Actor, velocity_grid : VelocityGrid) -> DecisionTree:
        """
        create the decision tree given the velocity grids
        """

        current_state = State(
            ego.pos,
            ego.speed,
            velocity_grid
        )
        
        current_state.get_children()
        return
        # create the root node 
        tree = DecisionTree()
        root = Node (
            parent = -1, 
            idx = 0, 
            prob = 1, 
            state=current_state)
        
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
            if tree_depth > max_depth:
                break
    
    def execute(
            self, 
            ego, 
            actors,
            velocity_grid,
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
        
        # update the location of the grid
        self.grid.move_origin( (ego.pos[0]+GRID_ORIGIN_X_OFFSET, GRID_ORIGIN_Y_OFFSET) )
        self.grid.decay( 0.8 )


        forecast = flow( 
                        self.grid.get_probability_map(), 
                        self.grid.get_velocity_map(), 
                        scale=GRID_RESOLUTION, 
                        timesteps=FORECAST_COUNT, 
                        dt=FORECAST_DT, 
                        mode='bilinear')
        
        self.create_decision_tree(ego, velocity_grid)       
        
        
        
def get_policy_fn(generator, policy_args=None):
    return TreeSearchPolicy(generator, policy_args=policy_args)
