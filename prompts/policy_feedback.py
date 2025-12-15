prompt = """
We trained a RL policy from scratch using the provided reward function code & hyperparameters and tracked the values of the individual components in the reward function as well as global policy metrics such as success rates and episode lengths after every {feedback_timestep_freq} timesteps and the maximum, mean, minimum values encountered:

{reflection}
"""
