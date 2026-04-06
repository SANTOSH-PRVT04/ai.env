---
title: QTrack AI Environment
emoji: 🏥
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
license: mit
tags:
  - healthcare
  - queue-management
  - hospital
  - ai-environment
  - optimization
  - openenv-hackathon
---

🏥 QTrack — Hospital Queue Optimization Environment
OpenEnv Hackathon Submission | Real-world AI environment for hospital queue management
Show Image

📌 Overview
QTrack is a real-world sequential decision-making environment where an AI agent manages patient flow across 7 hospital departments. When departments get overloaded, the agent takes the right actions — rescheduling patient appointment slots, alerting on-call staff, or opening walk-in availability — to minimize wait times and keep the hospital running smoothly.

Key principle: Patients are never rerouted to a different department or doctor.
The AI only adjusts their time slot within their own department, and sends them a clear reason for the change.

Example: You booked a 3:00 PM slot with Dr. Sharma in Cardiology.
Cardiology gets overloaded → AI updates your slot to 3:30 PM and notifies you:

"Your doctor currently has too many patients. Your appointment has been rescheduled by 30 minutes. We apologise for the inconvenience."


🌍 Real-World Task
PropertyValueDomainHealthcare — Hospital Queue ManagementTask typeSequential decision makingDepartments7 (Cardiology, General Medicine, Pediatrics, Orthopedics, Dermatology, Neurology, Radiology)Max steps per episode20StochasticYes (random patient arrivals)Seed supportYes (for reproducible evaluation)

🎯 Action Space
pythonaction = {
    "action_type": "reschedule" | "alert_staff" | "open_walkin" | "noop",
    "dept":        "<department name>",
    "delay_mins":  15   # only used for reschedule
}
ActionWhen to useEffectrescheduleDept load > 60%Pushes patient tokens forward by delay_mins in the same deptalert_staffDept load > 80%Calls on-call staff to support the overloaded deptopen_walkinIdle dept with 0 patientsOpens walk-in slots to absorb hospital overflownoopAll depts within safe loadNo action taken this step

👁️ Observation Space
pythonobservation = {
    "departments": [
        {
            "id": "dept-1",
            "name": "Cardiology",
            "doctor_count": 2,
            "avg_consult_time": 12,   # minutes per patient
            "capacity": 10,           # doctor_count × 5
            "active_patients": 8,
            "load_pct": 80.0          # 0.0 – 100.0
        },
        # ... 6 more departments
    ],
    "overloaded_depts":  ["Cardiology", "Pediatrics"],
    "idle_depts":        ["Neurology"],
    "total_patients":    37,
    "avg_wait_minutes":  13.5,
    "critical_count":    1,
    "step_number":       3,
    "timestamp":         "14:32:05"
}

🏆 Reward Function
EventRewardCorrect reschedule on overloaded dept (load > 60%)+0.4Staff alerted on critical dept (load > 80%)+0.3Walk-in opened at idle dept+0.2Avg wait drops below 15 min+0.2Overloaded dept left unaddressed at episode end−0.3 per deptAction taken on dept that didn't need it−0.1
All rewards clipped to [0.0, 1.0]. Partial progress signals given at every step.

📋 Tasks (Easy → Medium → Hard)
🟢 Task 1 — Easy: Single Overload Response

Seed: 1  |  Pass threshold: 0.35
One department is overloaded. Agent must detect it and reschedule patients within a few steps.

🟡 Task 2 — Medium: Multi-Department Balancing

Seed: 7  |  Pass threshold: 0.50
2–3 departments overloaded, some idle. Agent must reschedule overloaded depts AND open walk-ins at idle ones.

🔴 Task 3 — Hard: Full Hospital Crisis Management

Seed: 13  |  Pass threshold: 0.65
4+ departments in crisis. Agent must triage correctly, alert staff for critical depts, use idle resources, and keep avg wait below 20 min.


🚀 Setup & Usage
Install
bashpip install gradio>=4.0.0
Run the Demo App
bashpython app.py
# Opens at http://0.0.0.0:7860
Use the Environment in Code
pythonfrom env import QTrackEnv

env = QTrackEnv(seed=42)
obs = env.reset()

done = False
while not done:
    action = your_agent(obs)        # your agent returns an action dict
    result = env.step(action)
    print(f"Reward: {result.reward} | Done: {result.done}")
    obs  = result.observation
    done = result.done

print(env.state())                  # full serialisable state
Run the Baseline Grader
bashpython grader.py
Expected baseline output:
[EASY]   Single Overload Response      Score: ~0.40  ✅ PASSED
[MEDIUM] Multi-Department Balancing    Score: ~0.52  ✅ PASSED
[HARD]   Full Hospital Crisis          Score: ~0.38  ❌ FAILED
Overall Score: ~0.45
Grade Your Own Agent
pythonfrom grader import grade_agent

def my_agent(obs):
    return {
        "action_type": "reschedule",
        "dept":        "Cardiology",
        "delay_mins":  15
    }

report = grade_agent(my_agent)
print(f"Overall Score: {report['overall_score']}")

📁 File Structure
├── app.py           # Gradio demo UI — interactive patient token simulation
├── env.py           # OpenEnv environment — step() / reset() / state()
├── grader.py        # Agent grader — 3 tasks with 0.0–1.0 scoring
├── openenv.yaml     # OpenEnv specification file
├── requirements.txt
└── README.md

🧠 Design Notes

Patient first: The AI never moves a patient to a different doctor or department — only the time slot changes, with a human-readable reason sent to the patient.
Partial progress signals: Rewards given at every step, not just at episode end — enabling better credit assignment for RL agents.
Reproducible evaluation: Fixed seeds per task ensure consistent, fair scoring across all agents.
Scalable rules: Easy to extend with more departments, doctors, or action types.


Built for the OpenEnv Hackathon · QTrack by XYZ Hospital Team
