# inference.py — QTrack Baseline Inference Script
# Runs the baseline agent against all 3 tasks and prints reproducible scores

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from env import QTrackEnv
from grader import grade_agent, baseline_agent


def main():
    print("=" * 60)
    print("QTrack — Baseline Inference Script")
    print("OpenEnv Hackathon | Hospital Queue Optimization")
    print("=" * 60)

    report = grade_agent(baseline_agent)

    print(f"\n{'TASK':<40} {'SCORE':>6}  {'PASS?'}")
    print("-" * 60)
    for diff, task in report["tasks"].items():
        status = "✅ PASS" if task["passed"] else "❌ FAIL"
        print(f"[{diff.upper():<6}] {task['task_name']:<33} {task['score']:>6.4f}  {status}")
        print(f"         Steps: {task['steps']}  |  {task['feedback']}")
        print()

    print("-" * 60)
    print(f"Overall Score : {report['overall_score']:.4f}")
    print(f"All Passed    : {report['passed']}")
    print("=" * 60)
    print("\nFull JSON Report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
