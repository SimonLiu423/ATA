import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from eureka_wrapper import ReflectionCallback


def train_and_eval(
    env_id: str,
    n_envs: int,
    wrapper_class: type,
    wrapper_kwargs: dict,
    reward_code: str,
    total_timesteps: int,
    feedback_freq: int,
    tb_log_name: str,
):
    """
    This function returns the score (float) or -infinity if it fails.
    """
    train_env = make_vec_env(
        env_id,
        n_envs=n_envs,
        wrapper_class=wrapper_class,
        wrapper_kwargs=wrapper_kwargs,
    )
    train_env.env_method("edit_reward", reward_code)

    callback = ReflectionCallback()
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        device="cpu",
        tensorboard_log=f"./tensorboard_logs/{env_id}",
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
        tb_log_name=tb_log_name,
    )

    return callback.get_reflection_summary(feedback_freq)
