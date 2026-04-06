# grader.py — QTrack Agent Grader
# 3 tasks: easy → medium → hard, scores 0.0–1.0

from env import QTrackEnv
from dataclasses import dataclass
from typing import Callable, Dict, Any
import json

@dataclass
class TaskResult:
    task_name: str
    difficulty: str
    score: float        # 0.0 – 1.0
    steps_taken: int
    actions_log: list
    passed: bool
    feedback: str


# ── Task Definitions ──────────────────────────────────────────────────

def run_task(
    task_name: str,
    difficulty: str,
    agent_fn: Callable[[Dict], Dict],
    seed: int = 42,
    pass_threshold: float = 0.5,
) -> TaskResult:
    """Run one task and return scored result."""
    env = QTrackEnv(seed=seed)
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    actions_log = []

    done = False
    while not done:
        action = agent_fn(obs)
        result = env.step(action)
        total_reward += result.reward
        steps += 1
        actions_log.append({
            "step":   steps,
            "action": action,
            "reward": result.reward,
            "info":   result.info.get("effects", []),
        })
        obs  = result.observation
        done = result.done

    # Normalise score to 0.0–1.0
    score   = round(min(1.0, max(0.0, total_reward / max(steps, 1))), 4)
    passed  = score >= pass_threshold

    feedback_parts = [f"Steps taken: {steps}", f"Raw reward sum: {total_reward:.3f}", f"Score: {score:.4f}"]
    if passed:
        feedback_parts.append("✅ PASSED")
    else:
        feedback_parts.append(f"❌ FAILED (threshold: {pass_threshold})")

    return TaskResult(
        task_name   =task_name,
        difficulty  =difficulty,
        score       =score,
        steps_taken =steps,
        actions_log =actions_log,
        passed      =passed,
        feedback    =" | ".join(feedback_parts),
    )


# ── Task 1: EASY — Single Overload Response ───────────────────────────
# Agent must detect ONE overloaded department and reschedule it.
# Seed 1 guarantees exactly one overloaded dept.

def task_easy(agent_fn: Callable) -> TaskResult:
    """
    Scenario: One department is overloaded.
    Goal: Detect it and reschedule patients within 5 steps.
    Pass threshold: 0.35
    """
    return run_task(
        task_name      ="Single Overload Response",
        difficulty     ="easy",
        agent_fn       =agent_fn,
        seed           =1,
        pass_threshold =0.35,
    )


# ── Task 2: MEDIUM — Multi-Department Balancing ───────────────────────
# Agent must handle 2–3 overloaded departments and use idle resources.

def task_medium(agent_fn: Callable) -> TaskResult:
    """
    Scenario: 2–3 departments overloaded, some idle.
    Goal: Reschedule overloaded depts AND open walk-ins at idle depts.
    Pass threshold: 0.50
    """
    return run_task(
        task_name      ="Multi-Department Balancing",
        difficulty     ="medium",
        agent_fn       =agent_fn,
        seed           =7,
        pass_threshold =0.50,
    )


# ── Task 3: HARD — Full Hospital Crisis ──────────────────────────────
# Agent must triage 4+ overloaded depts, alert staff for critical ones,
# use idle resources, and keep avg wait below 20 min.

def task_hard(agent_fn: Callable) -> TaskResult:
    """
    Scenario: 4+ departments overloaded, critical load across hospital.
    Goal: Triage all overloaded depts with correct action types,
          minimise avg wait, and use idle resources efficiently.
    Pass threshold: 0.65
    """
    return run_task(
        task_name      ="Full Hospital Crisis Management",
        difficulty     ="hard",
        agent_fn       =agent_fn,
        seed           =13,
        pass_threshold =0.65,
    )


# ── Grader Runner ─────────────────────────────────────────────────────

def grade_agent(agent_fn: Callable) -> Dict[str, Any]:
    """
    Run all 3 tasks and return a full grading report.

    Args:
        agent_fn: A callable that takes an Observation dict and returns
                  an action dict:
                  {
                    "action_type": "reschedule"|"alert_staff"|"open_walkin"|"noop",
                    "dept": "<dept name>",
                    "delay_mins": <int>
                  }

    Returns:
        Dict with per-task results and overall score.
    """
    results = {
        "easy":   task_easy(agent_fn),
        "medium": task_medium(agent_fn),
        "hard":   task_hard(agent_fn),
    }

    overall = round(
        results["easy"].score   * 0.2 +
        results["medium"].score * 0.3 +
        results["hard"].score   * 0.5,
        4
    )

    report = {
        "overall_score": overall,
        "passed":        all(r.passed for r in results.values()),
        "tasks": {
            diff: {
                "task_name":   r.task_name,
                "score":       r.score,
                "steps":       r.steps_taken,
                "passed":      r.passed,
                "feedback":    r.feedback,
            }
            for diff, r in results.items()
        }
    }
    return report


# ── Baseline Agent (for reproducible baseline scores) ─────────────────

def baseline_agent(obs) -> Dict[str, Any]:
    """
    Simple rule-based baseline agent.
    Always reschedules the most overloaded department by 15 min.
    Reproducible — no randomness.
    """
    depts = obs.departments if hasattr(obs, "departments") else obs.get("departments", [])

    # Find most overloaded dept
    overloaded = [d for d in depts if d.get("load_pct", 0) > 60]
    if overloaded:
        worst = max(overloaded, key=lambda d: d["load_pct"])
        return {
            "action_type": "reschedule",
            "dept":        worst["name"],
            "delay_mins":  15,
        }

    # Check for idle depts
    idle = [d for d in depts if d.get("load_pct", 0) == 0 and d.get("doctor_count", 0) > 0]
    if idle:
        return {
            "action_type": "open_walkin",
            "dept":        idle[0]["name"],
            "delay_mins":  0,
        }

    return {"action_type": "noop", "dept": "", "delay_mins": 0}


# ── Run baseline when executed directly ──────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("QTrack Baseline Agent — Grading Report")
    print("=" * 60)

    report = grade_agent(baseline_agent)

    print(f"\n📊 Overall Score : {report['overall_score']:.4f}")
    print(f"✅ All Passed    : {report['passed']}\n")

    for diff, task in report["tasks"].items():
        print(f"[{diff.upper()}] {task['task_name']}")
        print(f"  Score   : {task['score']:.4f}")
        print(f"  Steps   : {task['steps']}")
        print(f"  Passed  : {task['passed']}")
        print(f"  Feedback: {task['feedback']}")
        print()

    print("=" * 60)
    print(json.dumps(report, indent=2))
