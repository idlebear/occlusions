'''
@File    :   tree_search.py
@Time    :   2022/11/16
@Author  :   Armin Sadeghi 
@Version :   0.1
@Contact :   a6sadegh@uwaterloo.ca
@Desc    :   None
'''


from policies.random import RandomPolicy



class TreeSearch(RandomPolicy):
    def __init__(self, generator, policy_args=None) -> None:
        super().__init__(policy_args)
        
    
    def execute(self, ego, actors, current_time=0, max_solver_time=30, dt=0.1):
        """tree search policy

        Args:
            ego (_type_): the actor vehicle
            actors (_type_): other actors in the environment
            current_time (int, optional): _description_. Defaults to 0.
            max_solver_time (int, optional): _description_. Defaults to 30.
            dt (float, optional): _description_. Defaults to 0.1.
        """
        
        
        
        