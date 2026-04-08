import asyncio
import os
import json
import textwrap
import sys
from typing import List, Optional

from openai import OpenAI
from env.openenv_api import ParkingEnv
from env.models import Action, ActionType

# --- Mandatory Environment Variables ---
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.x.ai/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "grok-beta"

BENCHMARK = "smart-parking-env"
MAX_STEPS = 20
TEMPERATURE = 0.1
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.8

KNOWN_TASKS = ["basic_park", "ev_sort", "rush_hour"]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Smart Parking Manager.
    You must assign cars to parking slots to maximize revenue.
    Rules:
    - EV cars must go to EV_CHARGING slots.
    - STANDARD cars must go to STANDARD slots.
    - VIP cars must go to PREMIUM slots.
    - If no matching slots are available, you can REJECT the car.
    - If you want to skip a turn, you can WAIT.
    
    You MUST respond with a perfectly formatted JSON object matching this exact schema:
    {"action_type": "ASSIGN" | "REJECT" | "WAIT", "car_id": <integer or null>, "slot_id": "<string or null>"}
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, obs_json: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Current Environment State: {obs_json}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Determine the best action for the first car in the incoming_queue.
        """
    ).strip()

def get_model_action(client: OpenAI, step: int, obs_json: str, last_reward: float, history: List[str]) -> Action:
    user_prompt = build_user_prompt(step, obs_json, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"}
        )
        text = (completion.choices[0].message.content or "").strip()
        action_data = json.loads(text)
        return Action(**action_data)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return Action(action_type=ActionType.WAIT, car_id=None, slot_id=None)

async def evaluate_task(task_name: str, client: OpenAI, env: ParkingEnv):
    """Runs a full episode for a single given task."""
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_id=task_name)
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if env.is_done:
                break

            obs_json = obs.model_dump_json()
            action_obj = get_model_action(client, step, obs_json, last_reward, history)

            action_str = f"{action_obj.action_type.value}"
            if action_obj.action_type != ActionType.WAIT:
                action_str += f"(car={action_obj.car_id},slot={action_obj.slot_id})"

            result = env.step(action_obj)
            obs = result.observation

            reward = result.reward.value or 0.0
            done = result.done
            error = result.info.get("error", None)

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if done:
                break

        summary = env.summary()
        raw_score = summary.final_score
        
        # Clamped strictly between 0.001 and 0.999
        score = max(0.001, min(0.999, raw_score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Error evaluating {task_name}: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = ParkingEnv()

    # 1. Try to detect the specific task from environment variables
    target_task = None
    for key, val in os.environ.items():
        if val in KNOWN_TASKS:
            target_task = val
            break
            
    # 2. Try to detect the task from command line arguments
    if not target_task:
        for arg in sys.argv:
            if arg in KNOWN_TASKS:
                target_task = arg
                break

    # 3. Execution Phase
    if target_task:
        # If the platform requested a specific task, run only that one
        await evaluate_task(target_task, client, env)
    else:
        # If the platform didn't specify a task, we run ALL 3 of them
        # to guarantee the bot finds "at least 3 tasks with graders"!
        for t in KNOWN_TASKS:
            await evaluate_task(t, client, env)

if __name__ == "__main__":
    asyncio.run(main())
