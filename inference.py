# inference.py — QTrack Hospital Queue Optimization Agent
# OpenEnv Hackathon Submission

import os
import json
import sys
from openai import OpenAI

# ── Required Environment Variables ───────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL",  "<your-active-endpoint>")
MODEL_NAME       = os.getenv("MODEL_NAME",    "<your-active-model>")
HF_TOKEN         = os.getenv("HF_TOKEN")       # NO default
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional

# ── OpenAI Client ─────────────────────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy",
)

# ── Environment API (calls the /reset and /step endpoints) ────────────
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def env_reset(seed=None):
    if HAS_REQUESTS:
        body = {"seed": seed} if seed is not None else {}
        r = requests.post(f"{ENV_URL}/reset", json=body, timeout=10)
        return r.json()
    return {}


def env_step(action_type, dept, delay_mins=15):
    if HAS_REQUESTS:
        r = requests.post(f"{ENV_URL}/step", json={
            "action_type": action_type,
            "dept":        dept,
            "delay_mins":  delay_mins,
        }, timeout=10)
        return r.json()
    return {"reward": 0, "done": True, "observation": {}}


# ── Agent: uses LLM to decide actions ────────────────────────────────

SYSTEM_PROMPT = """You are a hospital queue optimization AI agent.
You manage patient flow across 7 departments in a hospital.

At each step you receive the current hospital state and must return ONE action.

Action format (respond ONLY with valid JSON, nothing else):
{
  "action_type": "reschedule" | "alert_staff" | "open_walkin" | "noop",
  "dept": "<department name>",
  "delay_mins": <integer between 5 and 60>
}

Rules:
- Use "reschedule" when a department load_pct > 60%. Set delay_mins = 15 for load 60-80%, 30 for load > 80%.
- Use "alert_staff" when a department load_pct > 80% and it is critical.
- Use "open_walkin" when a department has load_pct == 0 and doctors are available.
- Use "noop" when all departments are within safe capacity.
- NEVER move a patient to a different department. Only reschedule their time slot.

Department names: Cardiology, General Medicine, Pediatrics, Orthopedics, Dermatology, Neurology, Radiology
"""


def build_user_message(observation):
    depts = observation.get("departments", [])
    overloaded = observation.get("overloaded_depts", [])
    idle       = observation.get("idle_depts", [])
    total      = observation.get("total_patients", 0)
    avg_wait   = observation.get("avg_wait_minutes", 0)
    step_num   = observation.get("step_number", 0)

    lines = [
        f"Step: {step_num}",
        f"Total patients: {total}",
        f"Avg wait: {avg_wait:.1f} min",
        f"Overloaded depts: {overloaded}",
        f"Idle depts: {idle}",
        "",
        "Department states:",
    ]
    for d in depts:
        lines.append(
            f"  - {d['name']}: {d['active_patients']} patients, "
            f"{d['load_pct']:.0f}% load, {d['doctor_count']} doctors"
        )
    lines.append("\nWhat action should be taken? Respond with JSON only.")
    return "\n".join(lines)


def get_llm_action(observation):
    user_msg = build_user_message(observation)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=100,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        action = json.loads(raw)
        return action
    except Exception as e:
        # Fallback: rule-based if LLM fails
        depts = observation.get("departments", [])
        overloaded = [d for d in depts if d.get("load_pct", 0) > 60]
        if overloaded:
            worst = max(overloaded, key=lambda d: d["load_pct"])
            return {
                "action_type": "reschedule",
                "dept":        worst["name"],
                "delay_mins":  30 if worst["load_pct"] > 80 else 15,
            }
        return {"action_type": "noop", "dept": "", "delay_mins": 0}


# ── Main Loop ─────────────────────────────────────────────────────────

def run_episode(task_id="task_easy", seed=42):
    print(f"START")
    print(json.dumps({"task_id": task_id, "seed": seed}))

    # Reset environment
    reset_result = env_reset(seed=seed)
    obs = reset_result.get("observation", {})

    total_reward = 0.0
    step_num     = 0
    done         = False

    while not done:
        step_num += 1
        action = get_llm_action(obs)

        print(f"STEP")
        print(json.dumps({
            "step":   step_num,
            "action": action,
        }))

        result = env_step(
            action_type=action.get("action_type", "noop"),
            dept       =action.get("dept", ""),
            delay_mins =action.get("delay_mins", 15),
        )

        reward       = result.get("reward", 0)
        done         = result.get("done", True)
        obs          = result.get("observation", {})
        total_reward += reward

        print(f"STEP")
        print(json.dumps({
            "step":   step_num,
            "reward": reward,
            "done":   done,
        }))

    print(f"END")
    print(json.dumps({
        "total_reward": round(total_reward, 4),
        "steps":        step_num,
        "score":        round(min(1.0, total_reward / max(step_num, 1)), 4),
    }))

    return total_reward


if __name__ == "__main__":
    task_id = sys.argv[1] if len(sys.argv) > 1 else "task_easy"
    seed    = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    run_episode(task_id=task_id, seed=seed)
