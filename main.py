import asyncio
import concurrent
import inspect
import logging
import os

from tqdm import tqdm

# Import tqdm
from tqdm.asyncio import tqdm as async_tqdm

# STRICTLY limit threads before importing torch/numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gymnasium as gym
import litellm
import mlflow
from agents import Agent, ModelSettings, Runner, Session, SQLiteSession, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv

import prompts
from eureka_wrapper import EurekaWrapper
from train_worker import train_and_eval

litellm.suppress_debug_info = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("eureka-experiment")
mlflow.openai.autolog()

# --- CONFIGURATION ---
ITERATIONS = 5
SAMPLES_PER_ITER = 4
N_ENVS = 4
OUTPUT_DIR = "eureka_outputs"
MAX_PARALLEL_JOBS = 4
TOTAL_TIMESTEPS = 100_000
FEEDBACK_FREQ = TOTAL_TIMESTEPS // (N_ENVS * 10)
# --------------------

ENV_ID = "LunarLander-v3"
TASK_DESC = """
## Task
Your task is to control the lander to land on the landing pad smoothly without crashing.

## Action Space
__dict__: {action_space_dict}

There are four discrete actions available:
- 0: do nothing
- 1: fire left orientation engine
- 2: fire main engine
- 3: fire right orientation engine

## Observation Space
__dict__: {observation_space_dict}

The state is an 8-dimensional vector: the coordinates of the lander in `x` & `y`, its linear velocities in `x` & `y`, its angle, its angular velocity, and two booleans that represent whether each leg is in contact with the ground or not.

## Starting State
The lander starts at the top center of the viewport with a random initial force applied to its center of mass.

## Episode Termination
The episode finishes if:
1. the lander crashes (the lander body gets in contact with the moon);
2. the lander gets outside of the viewport (x coordinate is greater than 1);
3. the lander is not awake. From the Box2D docs, a body which is not awake is a body which doesn’t move and doesn’t collide with any other body:
"""


load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY") or ""


class RewardGenerator:
    def __init__(self):
        self.agent = Agent(
            name="Eureka",
            instructions=prompts.initial_system.prompt,
            model=LitellmModel(
                base_url="https://openrouter.ai/api/v1",
                model="openrouter/kwaipilot/kat-coder-pro:free",
                api_key=api_key,
            ),
            tools=[self.get_tool()],
            # model_settings=ModelSettings(include_usage=True),
        )
        self.code = None
        self.new_code = False

    async def generate(self, session: Session, prompt: str) -> str:
        # TODO: experiment with 1 agent generating 16 reward functions
        await Runner.run(
            self.agent,
            prompt,
            session=session,
        )
        if self.new_code:
            self.new_code = False
            return self.code
        else:
            raise (Exception("No new reward function generated"))

    def get_tool(self):
        @function_tool
        def edit_reward(code_str: str) -> dict:
            """
            Edit the environment's gym.Wrapper `compute_reward` method with the provided code dynamically.
            The code is executed with `exec` and the wrapper's `compute_reward` method will be replaced with the
            edited version.

            Args:
                    code_str (str): A string containing valid Python code defining a function
                    `compute_reward(self, obs, action)`.

            Returns:
                    dict: Success status or error message.
            """
            self.code = code_str
            self.new_code = True
            return {"success": True}

        return edit_reward


def write_str_to_file(string: str, file_path: str):
    with open(file_path, "w") as f:
        f.write(string)


async def generate_reward(session, user_prompt, iter_idx, sample_idx):
    reward_generator = RewardGenerator()
    try:
        reward_code = await reward_generator.generate(session, user_prompt)
    except Exception as e:
        tqdm.write(f"Failed to generate reward code: {e}")
        return "", session

    write_str_to_file(
        reward_code, f"{OUTPUT_DIR}/iter{iter_idx}_response{sample_idx}.txt"
    )

    return reward_code, session


def train_baseline():
    train_and_eval(
        ENV_ID,
        N_ENVS,
        EurekaWrapper,
        {"is_eval": True},
        "",
        TOTAL_TIMESTEPS,
        FEEDBACK_FREQ,
        "baseline",
    )


async def main():
    # Train baseline
    train_baseline()

    # Eureka
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    best_iter_idx = None
    best_score = -float("inf")

    # --- INITIAL GENERATION ---
    tqdm.write("--- Generating Initial Population ---")

    tasks = []
    # Create multiple Envs and generate reward functions
    for sample_idx in range(SAMPLES_PER_ITER):
        session = SQLiteSession(f"session_{sample_idx}")
        env = gym.make(ENV_ID)
        task_desc = TASK_DESC.format(
            action_space_dict=env.action_space.__dict__,
            observation_space_dict=env.observation_space.__dict__,
        )
        user_prompt = prompts.initial_user.prompt.format(
            task_obs_code_string=inspect.getsource(env.unwrapped.step),
            task_description=task_desc,
        )
        env.close()
        task = generate_reward(session, user_prompt, 0, sample_idx)
        tasks.append(task)

    # Wrap gather with async_tqdm
    results = await async_tqdm.gather(*tasks, desc="Initial Generation")
    reward_codes = [result[0] for result in results]
    sessions = [result[1] for result in results]

    # --- EUREKA LOOP ---
    # Outer Loop: Evolution Iterations (position=0 keeps it at the bottom)
    outer_bar = tqdm(range(ITERATIONS), desc="Evolution Iterations", position=0)

    for iter_idx in outer_bar:
        tqdm.write(f"\n--- Iteration {iter_idx} ---")

        candidates = []

        # --- PARALLEL EXECUTION BLOCK ---
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=MAX_PARALLEL_JOBS
        ) as executor:
            future_to_idx = {
                executor.submit(
                    train_and_eval,
                    ENV_ID,
                    N_ENVS,
                    EurekaWrapper,
                    {"is_eval": False},
                    reward_codes[i],
                    TOTAL_TIMESTEPS,
                    FEEDBACK_FREQ,
                    f"iter{iter_idx}_sample{i}",
                ): i
                for i in range(SAMPLES_PER_ITER)
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

            if winner_score > best_score:
                tqdm.write(f"New Global Best Score: {winner_score:.2f}")
                best_score = winner_score
                best_iter_idx = (iter_idx, winner_idx)

            if iter_idx == ITERATIONS - 1:
                break

            # Mutation Step
            winner_session_items = await sessions[winner_idx].get_items()
            tasks = []
            for i, session in enumerate(sessions):
                if i != winner_idx:
                    await session.clear_session()
                    await session.add_items(winner_session_items)

                prompt = prompts.policy_feedback.prompt.format(
                    feedback_timestep_freq=FEEDBACK_FREQ,
                    feedback=winner_reflection,
                )
                task = generate_reward(session, prompt, iter_idx + 1, i)
                tasks.append(task)

            # Using async_tqdm for the LLM generation phase
            results = await async_tqdm.gather(
                *tasks, desc=f"Mutating Gen {iter_idx}", leave=False
            )
            reward_codes = [result[0] for result in results]
            sessions = [result[1] for result in results]

    tqdm.write(
        f"Best iteration: iter{best_iter_idx[0]}, response{best_iter_idx[1]} with score {best_score}"
    )


if __name__ == "__main__":
    asyncio.run(main())
