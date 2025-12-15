import asyncio
import concurrent
import inspect
import json
import logging
import multiprocessing
import os
from copy import deepcopy
from typing import Dict, List

# Import tqdm
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

# STRICTLY limit threads before importing torch/numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gymnasium as gym
import litellm
from agents import (
    Agent,
    Runner,
    Session,
    SQLiteSession,
    function_tool,
    set_trace_processors,
)
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv

import prompts
from eureka_wrapper import EurekaWrapper
from train_worker import train_and_eval

set_trace_processors([])
import mlflow  # noqa: E402

litellm.suppress_debug_info = True
litellm.DEFAULT_REQUEST_TIMEOUT = 90

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("eureka-experiment")
mlflow.openai.autolog()

# --- CONFIGURATION ---
EUREKA_ITERATIONS = 5
HPO_ITERATIONS = 5
SAMPLES_PER_ITER = 16
N_ENVS = 4
OUTPUT_DIR = "eureka_outputs"
MAX_PARALLEL_JOBS = 16
TOTAL_TIMESTEPS = 1_000_000
FEEDBACK_FREQ = TOTAL_TIMESTEPS // (N_ENVS * 10)
RETRY_COUNT = 3
# --------------------

ENV_ID = "BipedalWalker-v3"
ENV_KWARGS = {"hardcore": True}
DEVICE = "cpu"
TASK_DESC = """
## Task
Your task is to train a 2D two-legged robot to coordinate its limbs and successfully walk from the left side of the screen to the right side without falling over

## Action Space
__dict__: {action_space_dict}

Actions are motor speed values in the [-1, 1] range for each of the 4 joints at both hips and knees.

## Observation Space
__dict__: {observation_space_dict}

State consists of hull angle speed, angular velocity, horizontal speed, vertical speed, position of joints and joints angular speed, legs contact with ground, and 10 lidar rangefinder measurements. There are no coordinates in the state vector.

## Starting State
The walker starts standing at the left end of the terrain with the hull horizontal, and both legs in the same position with a slight knee angle.

## Episode Termination
The episode will terminate if the hull gets in contact with the ground or if the walker exceeds the right end of the terrain length.
"""


load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY") or ""

RL_ALGORITHMS = {
    "A2C": A2C,
    "DDPG": DDPG,
    "DQN": DQN,
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}


class TrainingConfig:
    def __init__(self):
        self.algorithm = None
        self.hyperparameters = None
        self.reward_code = None
        self.new_hp = False
        self.new_code = False
        self.can_train = True

    def __str__(self):
        return f"Algorithm: {self.algorithm}, Hyperparameters: {self.hyperparameters}, Reward Code: {self.reward_code}"

    def get_tools(self):
        @function_tool
        def edit_reward(code_str: str) -> dict:
            """
            Edit the environment's gym.wrapper `compute_reward` method with the provided code dynamically.
            the code is executed with `exec` and the wrapper's `compute_reward` method will be replaced with the
            edited version.

            args:
                    code_str (str): a string containing valid python code defining a function
                    `compute_reward(self, obs, action)`.

            returns:
                    dict: success status or error message.
            """
            self.reward_code = code_str
            self.new_code = True
            return {"success": True}

        @function_tool
        def select_algorithm(algorithm: str) -> dict:
            """
            Select an algorithm to use for training the agent.
            Available algorithms: A2C, DDPG, DQN, PPO, SAC, TD3.

            args:
                    algorithm (str): the name of the algorithm to use.

            returns:
                    dict: success status or error message.
            """
            if algorithm not in RL_ALGORITHMS.keys():
                return {
                    "success": False,
                    "error": f"invalid algorithm, available algorithms: {RL_ALGORITHMS.keys()}",
                }
            self.algorithm = RL_ALGORITHMS[algorithm]
            return {"success": True}

        @function_tool
        def suggest_hyperparameters(
            policy: str,
            learning_rate: float,
            n_steps: int,
            batch_size: int,
            n_epochs: int,
            gamma: float,
            gae_lambda: float,
            clip_range: float,
            clip_range_vf: str | float,
            normalize_advantage: bool,
            ent_coef: float,
            vf_coef: float,
            max_grad_norm: float,
            use_sde: bool,
            sde_sample_freq: int,
            target_kl: float | str,
        ):
            """
            Suggest a set of hyperparameters for the selected algorithm.
            The suggested hyperparameters will be used to train the RL agent.

            args:
                    policy (str): The policy model to use.
                    learning_rate (float): The learning rate (from 1 to 0).
                    n_steps (int): The number of steps to run for each environment per update (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
                    batch_size (int): Minibatch size.
                    n_epochs (int): Number of epoch when optimizing the surrogate loss
                    gamma (float): Discount factor
                    gae_lambda (float): Factor for trade-off of bias vs variance for Generalized Advantage Estimator
                    clip_range (float): Clipping parameter (from 1 to 0). IMPORTANT: this clipping depends on the reward scaling.
                    clip_range_vf (str | float): Clipping parameter for the value function (from 1 to 0). This is a parameter specific to the OpenAI implementation. If string value "None" is passed, no clipping will be done on the value function. IMPORTANT: this clipping depends on the reward scaling.
                    normalize_advantage (bool): Whether to normalize or not the advantage
                    ent_coef (float): Entropy coefficient for the loss calculation
                    vf_coef (float): Value function coefficient for the loss calculation
                    max_grad_norm (float): The maximum value for the gradient clipping
                    use_sde (bool): Whether to use generalized State Dependent Exploration (gSDE) instead of action noise exploration (default: False)
                    sde_sample_freq (int): Sample a new noise matrix every n steps when using gSDE Default: -1 (only sample at the beginning of the rollout)
                    target_kl (float | str): Limit the KL divergence between updates, because the clipping is not enough to prevent large update. If string value "None" is passed, there is no limit on the kl div.

            returns:
                    dict: Success status or error message.
            """
            if type(clip_range_vf) is str:
                if clip_range_vf == "None":
                    clip_range_vf = None
                else:
                    raise ValueError(
                        "Only 'None' is supported for 'clip_range_vf' string value"
                    )
            if type(target_kl) is str:
                if target_kl == "None":
                    target_kl = None
                else:
                    raise ValueError(
                        "Only 'None' is supported for 'target_kl' string value"
                    )

            hyperparameters = {
                "policy": policy,
                "learning_rate": learning_rate,
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_range": clip_range,
                "clip_range_vf": clip_range_vf,
                "normalize_advantage": normalize_advantage,
                "ent_coef": ent_coef,
                "vf_coef": vf_coef,
                "max_grad_norm": max_grad_norm,
                "use_sde": use_sde,
                "sde_sample_freq": sde_sample_freq,
                "target_kl": target_kl,
            }
            sig = inspect.signature(self.algorithm)
            try:
                sig.bind(env=ENV_ID, device=DEVICE, **hyperparameters)
                self.hyperparameters = hyperparameters
                self.new_hp = True
                return {"success": True}
            except TypeError as e:
                return {"success": False, "error": str(e)}

        return {
            "algorithm": select_algorithm,
            "reward": edit_reward,
            "hyperparameters": suggest_hyperparameters,
        }


class AgentTrainerAgent:
    def __init__(
        self,
        train_config: TrainingConfig,
        session_id: str,
        max_turns: int = 10,
    ):
        self.agent = Agent(
            name="AgentTrainer",
            instructions=prompts.system_role.prompt,
            model=LitellmModel(
                base_url="https://openrouter.ai/api/v1",
                model="openrouter/kwaipilot/kat-coder-pro:free",
                api_key=api_key,
            ),
            tools=list(train_config.get_tools().values()),
        )
        self.max_turns = max_turns
        self.train_config = train_config
        self.session = SQLiteSession(session_id)

    async def clear_history(self):
        await self.session.clear_session()

    async def load_history(self, history: List[Dict]) -> None:
        await self.session.add_items(history)

    async def load_background_context(self):
        env = gym.make(ENV_ID, **ENV_KWARGS)
        task_desc = TASK_DESC.format(
            action_space_dict=env.action_space.__dict__,
            observation_space_dict=env.observation_space.__dict__,
        )
        context = prompts.initial_user.prompt.format(
            task_obs_code_string=inspect.getsource(env.unwrapped.step),
            task_description=task_desc,
        )
        env.close()
        item = {"role": "user", "content": context}
        await self.session.add_items([item])

    async def select_algorithm(self):
        self.agent.tools = [self.train_config.get_tools()["algorithm"]]
        await Runner.run(
            self.agent,
            prompts.select_algorithm.prompt,
            session=self.session,
            max_turns=self.max_turns,
        )

        if self.train_config.algorithm is None:
            await Runner.run(
                self.agent,
                "Please use the provided tool to select an algorithm.",
                session=self.session,
                max_turns=self.max_turns,
            )
        if self.train_config.algorithm is None:
            raise Exception("No algorithm selected.")

    async def edit_reward(self, prompt: str, save_path: str):
        self.agent.tools = [self.train_config.get_tools()["reward"]]
        # TODO: experiment with 1 agent generating 16 reward functions
        await Runner.run(
            self.agent, prompt, session=self.session, max_turns=self.max_turns
        )
        if not self.train_config.new_code:
            await Runner.run(
                self.agent,
                "Please use the provided tool to provide a new reward function.",
                session=self.session,
                max_turns=self.max_turns,
            )

        if self.train_config.new_code:
            self.train_config.new_code = False
            write_str_to_file(
                self.train_config.reward_code,
                save_path,
            )
        else:
            raise Exception("No new code generated.")

    async def suggest_hyperparameters(self):
        self.agent.tools = [self.train_config.get_tools()["hyperparameters"]]
        await Runner.run(
            self.agent,
            prompts.suggest_hps.prompt.format(
                signature=inspect.signature(self.train_config.algorithm)
            ),
            session=self.session,
            max_turns=self.max_turns,
        )

        if not self.train_config.new_hp:
            await Runner.run(
                self.agent,
                "Please use the provided tool to suggest hyperparameters.",
                session=self.session,
                max_turns=self.max_turns,
            )

        if self.train_config.new_hp:
            self.train_config.new_hp = False
        else:
            raise Exception("No new hyperparameters generated.")

    async def generate_init_config(self, reward_save_path: str):
        try:
            await self.edit_reward(
                prompt=prompts.write_init_reward.prompt, save_path=reward_save_path
            )
            await self.suggest_hyperparameters()
        except Exception as e:
            self.train_config.can_train = False

    async def add_feedback(self, reflection: str):
        feedback = prompts.policy_feedback.prompt.format(
            feedback_timestep_freq=FEEDBACK_FREQ,
            reflection=reflection,
        )
        await self.session.add_items([{"role": "user", "content": feedback}])

    async def generate_new_config(self, reward_save_path: str):
        try:
            await self.edit_reward(
                prompt=prompts.modify_reward.prompt, save_path=reward_save_path
            )
            # await self.suggest_hyperparameters()
        except Exception as e:
            self.train_config.can_train = False


def write_str_to_file(string: str, file_path: str):
    with open(file_path, "w") as f:
        f.write(string)


def train_baseline():
    train_and_eval(
        ENV_ID,
        ENV_KWARGS,
        PPO,
        {},
        N_ENVS,
        EurekaWrapper,
        {"is_eval": True},
        "",
        TOTAL_TIMESTEPS,
        FEEDBACK_FREQ,
        "baseline",
    )


async def train_eureka(main_agent: AgentTrainerAgent):
    # Eureka
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    best_iter_idx = None
    best_reward_session_history = None
    best_train_config = None
    best_score = -float("inf")

    # --- INITIAL GENERATION ---
    tqdm.write("--- Generating Initial Population ---")

    conversation_history = await main_agent.session.get_items()
    agents: List[AgentTrainerAgent] = []
    tasks = []
    # Create multiple Envs and generate reward functions
    for sample_idx in range(SAMPLES_PER_ITER):
        config = deepcopy(main_agent.train_config)
        agent = AgentTrainerAgent(config, f"session_{sample_idx}")
        agents.append(agent)

        await agent.load_history(conversation_history)
        task = agent.generate_init_config(
            reward_save_path=f"{OUTPUT_DIR}/iter{0}_response{sample_idx}.txt"
        )
        tasks.append(task)

    # Wrap gather with async_tqdm
    await async_tqdm.gather(*tasks, desc="Initial Generation")

    # --- EUREKA LOOP ---
    # Outer Loop: Evolution Iterations (position=0 keeps it at the bottom)
    outer_bar = tqdm(range(EUREKA_ITERATIONS), desc="Evolution Iterations", position=0)

    for iter_idx in outer_bar:
        tqdm.write(f"\n--- Iteration {iter_idx} ---")

        candidates = []

        ctx = multiprocessing.get_context("spawn")

        # --- PARALLEL EXECUTION BLOCK ---
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=MAX_PARALLEL_JOBS, mp_context=ctx
        ) as executor:
            future_to_idx = {
                executor.submit(
                    train_and_eval,
                    ENV_ID,
                    ENV_KWARGS,
                    agents[i].train_config.algorithm,
                    agents[i].train_config.hyperparameters,
                    N_ENVS,
                    EurekaWrapper,
                    {"is_eval": False},
                    agents[i].train_config.reward_code,
                    TOTAL_TIMESTEPS,
                    FEEDBACK_FREQ,
                    f"iter{iter_idx}_sample{i}",
                ): i
                for i in range(SAMPLES_PER_ITER)
                if agents[i].train_config.can_train
            }

            # Inner Loop: Training Jobs (leave=False clears it after this generation is done)
            for future in tqdm(
                concurrent.futures.as_completed(future_to_idx),
                total=SAMPLES_PER_ITER,
                desc=f"Training Gen {iter_idx}",
                position=1,
                leave=False,
            ):
                idx = future_to_idx[future]
                try:
                    score, reflection = future.result()
                    tqdm.write(f"  > Finished Sample {idx}: Score {score:.2f}")
                    candidates.append((idx, score, reflection))
                except Exception as e:
                    tqdm.write(f"  > Failed Sample {idx}: {e}")
                    write_str_to_file(
                        str(e), f"{OUTPUT_DIR}/iter{iter_idx}_failed{idx}.txt"
                    )

        # --------------------------------
        # 3. EVOLUTION
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            winner_idx, winner_score, winner_reflection = candidates[0]

            tqdm.write(
                f"Best of Iteration {iter_idx}: #{winner_idx} with score {winner_score:.2f}"
            )

            winner_session_history = await agents[winner_idx].session.get_items()

            if float(winner_score) > float(best_score):
                tqdm.write(f"New Global Best Score: {winner_score:.2f}")
                best_score = float(winner_score)
                best_reward_session_history = winner_session_history
                best_train_config = deepcopy(agents[winner_idx].train_config)
                best_iter_idx = (iter_idx, winner_idx)

            if iter_idx == EUREKA_ITERATIONS - 1:
                break

            # Mutation Step
            tasks = []
            for i, agent in enumerate(agents):
                if i != winner_idx:
                    await agent.clear_history()
                    await agent.load_history(winner_session_history)

                agent.train_config.can_train = True
                await agent.add_feedback(
                    reflection=winner_reflection,
                )
                task = agent.generate_new_config(
                    reward_save_path=f"{OUTPUT_DIR}/iter{iter_idx + 1}_response{i}.txt",
                )
                tasks.append(task)

            # Using async_tqdm for the LLM generation phase
            await async_tqdm.gather(
                *tasks, desc=f"Mutating Gen {iter_idx}", leave=False
            )

    tqdm.write(
        f"Best iteration: iter{best_iter_idx[0]}, response{best_iter_idx[1]} with score {best_score}"
    )
    return best_score, best_iter_idx, best_reward_session_history, best_train_config


async def main():
    # # Train baseline
    # train_baseline()

    train_config = TrainingConfig()
    agent = AgentTrainerAgent(train_config, "main_session")

    await agent.load_background_context()
    await agent.select_algorithm()

    # Train Eureka
    (
        best_score,
        best_iter_idx,
        best_reward_session_history,
        best_train_config,
    ) = await train_eureka(agent)
    agent.train_config = best_train_config

    # HPO
    await agent.clear_history()
    await agent.load_history(best_reward_session_history)

    for i in tqdm(range(HPO_ITERATIONS)):
        while True:
            # Retry until successful hyperparameter suggestion
            try:
                await agent.suggest_hyperparameters()
                break
            except Exception as e:
                pass

        score, reflection = train_and_eval(
            ENV_ID,
            ENV_KWARGS,
            agent.train_config.algorithm,
            agent.train_config.hyperparameters,
            N_ENVS,
            EurekaWrapper,
            {"is_eval": False},
            agent.train_config.reward_code,
            TOTAL_TIMESTEPS,
            FEEDBACK_FREQ,
            f"HPO{i}",
        )

        if score > best_score:
            best_score = score
            best_train_config = deepcopy(agent.train_config)
            tqdm.write(f"New Global Best Score: {best_score:.2f}")

        await agent.add_feedback(reflection)

    tqdm.write(f"Best score: {best_score:.2f}, Best config: {best_train_config}")

    # Load models
    # models_path = [
    #     # os.path.join("./best_models", "baseline", "best_model.zip"),
    #     os.path.join("./best_models", "iter4_sample12", "best_model.zip"),
    # ]

    # env = gym.make(ENV_ID, render_mode="human", **ENV_KWARGS)

    # for model_path in models_path:
    #     model = PPO.load(model_path)
    #     obs, _ = env.reset()
    #     for _ in range(1000):
    #         action, _states = model.predict(obs, deterministic=True)
    #         obs, rewards, terminated, truncated, info = env.step(action)
    #         if terminated or truncated:
    #             obs, _ = env.reset()
    #     env.close()


if __name__ == "__main__":
    asyncio.run(main())
