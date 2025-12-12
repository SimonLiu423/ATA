import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from eureka_wrapper import ReflectionCallback


def train_and_eval(
    env_id: str,
    n_envs: int,
    wrapper_class: type,
    reward_code: str,
    total_timesteps: int,
    feedback_freq: int,
):
    """
    This function returns the score (float) or -infinity if it fails.
    """
    train_env = make_vec_env(env_id, n_envs=n_envs, wrapper_class=wrapper_class)
    train_env.env_method("edit_reward", reward_code)

    callback = ReflectionCallback()
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        device="cpu",
        tensorboard_log="./tensorboard_logs",
    )
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    # Evaluate
    eval_env = gym.make(env_id)
    obs, _ = eval_env.reset()
    total_reward = 0
    terminated = False
    while not terminated:
        action, _ = model.predict(obs)
        obs, reward, terminated, _, _ = eval_env.step(action)
        total_reward += reward

    return total_reward, callback.get_reflection_summary(feedback_freq)
