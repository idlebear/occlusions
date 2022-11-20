import sys 
sys.path.append('../')

from policies.tree_search import TreeSearchPolicy
from policies.VelocityGrid import VelocityGrid



if __name__ == "__main__":
    policy = TreeSearchPolicy()
    policy.grid = VelocityGrid(2, 2)
    policy.grid.probabilityMap[0][0] = 0.2
    policy.grid.probabilityMap[0][1] = 0.5
    policy.grid.probabilityMap[1][0] = 0
    policy.grid.probabilityMap[1][1] = 1
    
    policy.execute(None, None)
