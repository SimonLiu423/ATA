prompt = """
Write a reward function for the environment that will help the agent learn the task described in text.

Your reward function should use useful variables from the environment as inputs. As an example,
the reward function signature can be: 

<signature>
def compute_reward(self, obs: Any, action: Any) -> Tuple[float, Dict[str, float]]:
    ...
    return reward, {}
</signature>

Since the reward function will be executed in a standard Python environment:
1. Please ensure the code is valid Python 3.
2. Use `numpy` (imported as `np`) for array operations.
3. The input `obs` is a numpy array. 
4. The input `action` depends on the environment:
   - For Continuous Control (e.g., robots), it is a numpy array.
   - For Discrete Control (e.g., CartPole), it is an integer.
   Please handle the action variable accordingly based on the task description.
"""
