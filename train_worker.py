import os

import gymnasium as gym
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from eureka_wrapper import ReflectionCallback


def train_and_eval(
    env_id: str,
    env_kwargs: dict,
    algorithm: BaseAlgorithm,
    hyperparameters: dict,
    n_envs: int,
    wrapper_class: type,
    wrapper_kwargs: dict,
    reward_code: str,
    total_timesteps: int,
    feedback_freq: int,
    model_save_dir: str,
    eval_log_dir: str,
    tb_log_dir: str,
    tb_log_name: str,
    success_threshold: float,
    eval_freq: int,
):
    """
    This function returns the score (float) or -infinity if it fails.
    """
    train_env = make_vec_env(
        env_id,
        env_kwargs=env_kwargs,
        n_envs=n_envs,
        wrapper_class=wrapper_class,
        wrapper_kwargs=wrapper_kwargs,
    )
    train_env.env_method("edit_reward", reward_code)

    eval_env = gym.make(env_id, **env_kwargs)

    reflection_callback = ReflectionCallback()
    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        best_model_save_path=os.path.join(model_save_dir, tb_log_name),
        log_path=os.path.join(eval_log_dir, tb_log_name),
    )
    callback_list = CallbackList([reflection_callback, eval_callback])
    model = algorithm(
        env=train_env,
        **hyperparameters,
        device="cpu",
        tensorboard_log=os.path.join(tb_log_dir, env_id),
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        progress_bar=True,
        tb_log_name=tb_log_name,
    )

    train_env.close()

    results_np = np.array(eval_callback.evaluations_results)
    success_rate = np.mean((results_np > success_threshold).astype(int)).item()

    eval_feedback = """
<Evaluation>
Triggered at: {timesteps} (timesteps)
Episode length: {episode_length}
<Score>
Scores: {scores}
Max: {max_score: .2f}, Mean: {mean_score: .2f}, Min: {min_score: .2f}
Success rate (reward > {success_threshold}): {success_rate: .2f}
</Score>
</Evaluation>
""".format(
        timesteps=["{}k".format(x / 1000) for x in eval_callback.evaluations_timesteps],
        episode_length=[
            "{:.2f}k".format(x)
            for x in np.mean(eval_callback.evaluations_length, axis=1)
        ],
        scores=[
            "{:.2f}".format(x)
            for x in np.mean(eval_callback.evaluations_results, axis=1)
        ],
        max_score=np.max(eval_callback.evaluations_results),
        mean_score=np.mean(eval_callback.evaluations_results),
        min_score=np.min(eval_callback.evaluations_results),
        success_threshold=success_threshold,
        success_rate=success_rate,
    )

    return (
        success_rate,
        np.mean(eval_callback.evaluations_results),
        reflection_callback.get_reflection_summary(feedback_freq) + eval_feedback,
    )
