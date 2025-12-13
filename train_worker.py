import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
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

    eval_env = gym.make(env_id)

    reflection_callback = ReflectionCallback()
    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=total_timesteps // 100,
        best_model_save_path=f"./best_models/{tb_log_name}/",
    )
    callback_list = CallbackList([reflection_callback, eval_callback])
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        device="cpu",
        tensorboard_log=f"./tensorboard_logs/{env_id}",
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        progress_bar=True,
        tb_log_name=tb_log_name,
    )

    train_env.close()

    eval_feedback = """
    <evaluation>
    Triggered at: {timesteps} (timesteps)
    Episode length: {episode_lengths}
    Scores: {scores}
    Max: {max_score}, Mean: {mean_score}, Min: {min_score}
    </evaluation>
    """.format(
        timesteps=eval_callback.evaluations_timesteps,
        episode_lengths=np.mean(eval_callback.evaluations_lengths, axis=0),
        scores=np.mean(eval_callback.evaluations_rewards, axis=0),
        max_score=np.max(eval_callback.evaluations_rewards),
        mean_score=np.mean(eval_callback.evaluations_rewards),
        min_score=np.min(eval_callback.evaluations_rewards),
    )

    return (
        eval_callback.best_mean_reward,
        reflection_callback.get_reflection_summary(feedback_freq) + eval_feedback,
    )
