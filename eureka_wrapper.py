import logging
import textwrap
import types
from collections import defaultdict
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class ReflectionCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.ai_rewards_all = []
        self.gt_rewards_all = []
        self.extras = defaultdict(list)

    def _on_step(self) -> bool:
        assert "rewards" in self.locals
        assert "infos" in self.locals
        assert "dones" in self.locals

        infos = self.locals["infos"]
        dones = self.locals["dones"]
        ai_rewards = self.locals["rewards"]

        # Store each environment's reward and reward components
        ai_rew_buf = []
        gt_rew_buf = []
        extras_buf = defaultdict(list)

        # Iterate through each environment and only store non-terminated ones
        for idx, done in enumerate(dones):
            if not done:
                ai_rew_buf.append(ai_rewards[idx])
                gt_rew_buf.append(infos[idx]["gt_reward"])
                for key, val in infos[idx]["reward_components"].items():
                    extras_buf[key].append(val)

        ai_mean_rew = np.mean(ai_rew_buf)
        gt_mean_rew = np.mean(gt_rew_buf)
        self.ai_rewards_all.append(ai_mean_rew)
        self.gt_rewards_all.append(gt_mean_rew)
        self.logger.record("extras/ai_reward", ai_mean_rew)
        self.logger.record("extras/gt_reward", gt_mean_rew)
        for key, val in extras_buf.items():
            mean_val = np.mean(val)
            self.extras[key].append(np.mean(val))
            self.logger.record(f"extras/{key}", mean_val)

        return True

    def get_reflection_summary(self, feedback_freq: int):
        template = """
        <{metric_name}>
        Values: {metric_cur}
        Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f}
        </{metric_name}>
        """
        summary = ""
        summary += template.format(
            metric_name="Ground-Truth Rewards",
            metric_cur=self.gt_rewards_all[::feedback_freq],
            metric_cur_max=max(self.gt_rewards_all),
            metric_cur_mean=sum(self.gt_rewards_all) / len(self.gt_rewards_all),
            metric_cur_min=min(self.gt_rewards_all),
        )
        summary += template.format(
            metric_name="Your Rewards",
            metric_cur=self.ai_rewards_all[::feedback_freq],
            metric_cur_max=max(self.ai_rewards_all),
            metric_cur_mean=sum(self.ai_rewards_all) / len(self.ai_rewards_all),
            metric_cur_min=min(self.ai_rewards_all),
        )
        for metric_name, val in self.extras.items():
            metric_cur = ["{:.2f}".format(x) for x in val[::feedback_freq]]
            metric_cur_max = max(val)
            metric_cur_mean = np.mean(val)
            metric_cur_min = min(val)
            summary += template.format(
                metric_name=metric_name,
                metric_cur=metric_cur,
                metric_cur_max=metric_cur_max,
                metric_cur_mean=metric_cur_mean,
                metric_cur_min=metric_cur_min,
            )

        return summary


class EurekaWrapper(gym.Wrapper):
    def __init__(self, env, is_eval=False):
        super().__init__(env)
        self.is_eval = is_eval

    def step(self, action):
        obs, gt_reward, terminated, truncated, info = self.env.step(action)
        if self.is_eval:
            info["reward_components"] = {}
            info["gt_reward"] = gt_reward
            return obs, gt_reward, terminated, truncated, info

        ai_reward, components = self.compute_reward(obs, action)

        info["reward_components"] = components
        info["gt_reward"] = gt_reward

        return obs, ai_reward, terminated, truncated, info

    def compute_reward(self, obs: Any, action: Any) -> Tuple[float, Dict[str, float]]:
        pass

    def edit_reward(self, code_str: str) -> dict:
        local_scope = {}
        clean_code = textwrap.dedent(code_str)
        try:
            exec(clean_code, globals=globals(), locals=local_scope)

            if "compute_reward" in local_scope:
                bound_method = types.MethodType(local_scope["compute_reward"], self)
                self.compute_reward = bound_method
                return {"success": True}
            else:
                return {
                    "success": False,
                    "error": "compute_reward function not found in provided code",
                }
        except Exception as e:
            logging.error(f"Error in modify_step_fn: {e}")
            return {"success": False, "error": str(e)}
