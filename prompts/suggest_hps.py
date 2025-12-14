prompt = """
Please recommend a configuration of hyperparameters to optimize the agent's performance.

Refer to the algorithm's signature here: <signature> {signature} </signature>

Guidance on Rewards:
- The magnitude and frequency of your rewards influence how parameters like 'learning_rate' and 'entropy_coefficient' behave.
- For dense, high-value rewards: Decrease the learning rate or increase gradient clipping.
- For sparse rewards: Set the discount factor (gamma) closer to 0.999.

Constraint: Provide only values that directly affect training. Do not include setup arguments such as `env`, `device`, `verbose`, or `tensorboard_log`.
"""
