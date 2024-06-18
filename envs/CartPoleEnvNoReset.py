import numpy as np
from typing import Optional
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

MAX_REWARD = 1.0
def get_cost(reward, gamma):
    return (MAX_REWARD - reward) * (1-gamma) / MAX_REWARD

'''
We add an additional action pointing terminated states to themselves.
This allows us to call the step after termination without resetting the environment.
'''
class CartPoleEnvNoReset(CartPoleEnv):      
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(render_mode)
    
    def step(self, action):
        # we define this as a vacuous step, nothing will change.
        if action == self.action_space.n:   
            return np.array(self.state, dtype=np.float32), 0.0, True, False, {}
        return super().step(action)
    