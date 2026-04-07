"""
scripts/baseline.py
───────────────────
Baseline agent: Groq (Llama-3.3-70b) solves all three Smart Parking tasks via the
OpenEnv step/reset/grade API. Reads OPENAI_API_KEY from the environment.
"""

import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import OpenAI
from env.openenv_api import ParkingEnv
from env.models import Action, ActionType

# ── constants ──────────────────────────────────────────────────────────────────
TASKS        = ["basic_park", "ev_sort", "rush_hour"]
MAX_STEPS    = 30
MODEL        = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert Parking Manager AI controlling a smart parking lot simulation.

### Observation format (JSON)
{
  "current_time_step": <int>,
  "current_revenue":   <float>,
  "available_slots": [
    {"id": "A1", "slot_type": "STANDARD|EV_CHARGING|PREMIUM", "is_occupied": false, "occupant_id": null}
  ],
  "incoming_queue": [
    {"id": <int>, "car_type": "STANDARD|EV|VIP", "entry_time": <int>}
  ]
}

### Action format (JSON)
{
  "action_type": "ASSIGN" | "REJECT" | "WAIT",
  "car_id":  <int or null>,
  "slot_id": "<string or null>"
}

### Strategy
1. Match types: EV -> EV_CHARGING, VIP -> PREMIUM.
2. STANDARD cars fill STANDARD slots.
3. Reject only when no valid slot exists.
4. WAIT only if the queue is empty.
""").strip()

def obs_to_json(obs) -> str:
    return obs.model_dump_json(indent=2)

def llm_action(client: OpenAI, obs_json: str) -> Action:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Observation:\n{obs_json}\n\nAction:"},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return Action(
        action_type=ActionType(data.get("action_type", "WAIT")),
        car_id=data.get("car_id", None),
        slot_id=data.get("slot_id", None),
    )

def run_task(client: OpenAI, env: ParkingEnv, task_id: str) -> float:
    print(f"\n{'═'*60}\n  Task: {task_id.upper()}\n{'═'*60}")
    obs = env.reset(task_id)
    total_reward = 0.0
    step = 0

    while step < MAX_STEPS:
        try:
            action = llm_action(client, obs_to_json(obs))
        except Exception as exc:
            print(f"  [step {step:02d}] LLM error: {exc} — WAIT")
            action = Action(action_type=ActionType.WAIT)

        result = env.step(action)
        
        # 1. Safely extract variables
        if isinstance(result, tuple):
            obs, raw_reward, done, info = result
        else:
            obs = getattr(result, 'observation', getattr(result, 'obs', obs))
            raw_reward = getattr(result, 'reward', 0.0)
            done = getattr(result, 'done', getattr(result, 'is_done', False))
            info = getattr(result, 'info', {})

        # 2. X-RAY EXTRACTION: If Claude made Reward a Pydantic object, bust it open
        if hasattr(raw_reward, 'model_dump'):
            dump = raw_reward.model_dump()
            raw_reward = next(iter(dump.values())) if dump else 0.0
        elif hasattr(raw_reward, 'value'):
            raw_reward = raw_reward.value
        elif hasattr(raw_reward, 'root'):
            raw_reward = raw_reward.root
        
        # Fallback for the tuple glitch
        if isinstance(raw_reward, tuple):
            raw_reward = raw_reward[0]
            
        # 3. Force float math
        reward = float(raw_reward)
        total_reward += reward
        step += 1

        msg = info.get('message', '') if isinstance(info, dict) else ''
        print(f"  [step {step:02d}] {action.action_type.value:6s} car={str(action.car_id):>4s} slot={str(action.slot_id):>3s} → reward={reward:+.2f} {msg}")

        if done: break

    # --- INDESTRUCTIBLE GRADE LOGIC ---
    try:
        summary = env.summary()
        score = float(getattr(summary, 'score', getattr(summary, 'final_score', 0.0)))
    except AttributeError:
        score = float(env.grade())

    print(f"\n  ✅  Cumulative reward : {total_reward:.3f}")
    print(f"  🏆  Grader score      : {score:.3f}  (0.0 – 1.0)")
    return score

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    env = ParkingEnv()
    scores = {t: run_task(client, env, t) for t in TASKS}

    print(f"\n{'═'*60}\n  BASELINE SUMMARY\n{'═'*60}")
    for t, s in scores.items():
        print(f"  {t:<20} {s:>7.3f}  {'█' * int(s * 20)}")
    print(f"  {'─'*20} {'─'*8}\n  AVERAGE              {sum(scores.values())/len(scores):>7.3f}\n{'═'*60}\n")

if __name__ == "__main__":
    main()