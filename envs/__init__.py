from envs.CartPoleEnvNoReset import CartPoleEnvNoReset

from gymnasium.envs.registration import register

try:
    register(
        id='CartPole-noreset',
        entry_point='envs:CartPoleEnvNoReset',
        # vector_entry_point=CartPoleVectorEnvNoReset, 
        max_episode_steps=200
    )
    # print('Registration success.')
except:
    print('Registration failed!')
