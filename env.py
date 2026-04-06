# env.py — QTrack OpenEnv Environment
# Implements full OpenEnv spec: typed models, step()/reset()/state()

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import random
import json

# ── Typed Models ──────────────────────────────────────────────────────

@dataclass
class Department:
    id: str
    name: str
    doctor_count: int
    avg_consult_time: int   # minutes
    capacity: int           # doctor_count * 5
    active_patients: int = 0
    load_pct: float = 0.0

@dataclass
class PatientToken:
    token_id: str
    dept: str
    doctor: str
    original_time: str      # HH:MM
    current_time: str       # HH:MM (updated by AI actions)
    wait_minutes: int = 0
    rescheduled: bool = False
    reschedule_reason: str = ""

@dataclass
class Observation:
    departments: List[Dict]
    overloaded_depts: List[str]
    idle_depts: List[str]
    total_patients: int
    avg_wait_minutes: float
    critical_count: int
    step_number: int
    timestamp: str

@dataclass
class StepResult:
    observation: Observation
    reward: float           # 0.0 – 1.0
    done: bool
    info: Dict[str, Any]


# ── QTrack Environment ────────────────────────────────────────────────

class QTrackEnv:
    """
    QTrack Hospital Queue Optimization Environment.

    Action Space (dict):
        - action_type : str  → "reschedule" | "alert_staff" | "open_walkin" | "noop"
        - dept        : str  → target department name
        - delay_mins  : int  → minutes to push schedule (for reschedule)

    Observation Space (Observation dataclass):
        - departments     : list of dept load states
        - overloaded_depts: depts above 60% capacity
        - idle_depts      : depts at 0% with available doctors
        - total_patients  : int
        - avg_wait_minutes: float
        - critical_count  : int
        - step_number     : int
        - timestamp       : str

    Reward:
        +0.4  per overloaded dept correctly rescheduled
        +0.2  per idle dept utilised
        +0.2  for avg wait time reduced below threshold
        -0.3  per overloaded dept left unaddressed at episode end
        -0.1  per unnecessary action on non-overloaded dept
        Clipped to [0.0, 1.0]
    """

    DEPARTMENTS_CONFIG = [
        {"id": "dept-1", "name": "Cardiology",       "doctor_count": 2, "avg_consult_time": 12},
        {"id": "dept-2", "name": "General Medicine",  "doctor_count": 2, "avg_consult_time": 8},
        {"id": "dept-3", "name": "Pediatrics",        "doctor_count": 1, "avg_consult_time": 10},
        {"id": "dept-4", "name": "Orthopedics",       "doctor_count": 1, "avg_consult_time": 15},
        {"id": "dept-5", "name": "Dermatology",       "doctor_count": 1, "avg_consult_time": 8},
        {"id": "dept-6", "name": "Neurology",         "doctor_count": 1, "avg_consult_time": 18},
        {"id": "dept-7", "name": "Radiology",         "doctor_count": 1, "avg_consult_time": 20},
    ]

    DOCTORS = {
        "Cardiology":       ["Dr. Sharma", "Dr. Kapoor"],
        "General Medicine": ["Dr. Mehta",  "Dr. Singh"],
        "Pediatrics":       ["Dr. Rao"],
        "Orthopedics":      ["Dr. Verma"],
        "Dermatology":      ["Dr. Nair"],
        "Neurology":        ["Dr. Iyer"],
        "Radiology":        ["Dr. Patel"],
    }

    MAX_STEPS = 20

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._rng  = random.Random(seed)
        self._departments: List[Department] = []
        self._tokens: List[PatientToken] = []
        self._step_num = 0
        self._addressed_depts: set = set()
        self._unnecessary_actions = 0
        self.reset()

    # ── Core API ──────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset environment to initial state. Returns first observation."""
        self._rng = random.Random(self.seed)
        self._step_num = 0
        self._addressed_depts = set()
        self._unnecessary_actions = 0

        self._departments = []
        for cfg in self.DEPARTMENTS_CONFIG:
            cap = cfg["doctor_count"] * 5
            pts = self._rng.randint(0, cap + 3)
            load = min(100, round((pts / cap) * 100)) if cap > 0 else 0
            self._departments.append(Department(
                id=cfg["id"],
                name=cfg["name"],
                doctor_count=cfg["doctor_count"],
                avg_consult_time=cfg["avg_consult_time"],
                capacity=cap,
                active_patients=pts,
                load_pct=load,
            ))

        self._tokens = self._generate_tokens()
        return self._get_observation()

    def step(self, action: Dict[str, Any]) -> StepResult:
        """
        Execute one action in the environment.

        Args:
            action: {
                "action_type": "reschedule" | "alert_staff" | "open_walkin" | "noop",
                "dept": "<department name>",
                "delay_mins": <int>   # only for reschedule
            }

        Returns:
            StepResult(observation, reward, done, info)
        """
        self._step_num += 1
        reward = 0.0
        info   = {"action": action, "effects": []}

        action_type = action.get("action_type", "noop")
        dept_name   = action.get("dept", "")
        delay_mins  = action.get("delay_mins", 15)

        dept = self._get_dept(dept_name)

        if action_type == "noop":
            info["effects"].append("No action taken.")

        elif action_type == "reschedule" and dept:
            if dept.load_pct > 60:
                # Correct action — reschedule tokens for this dept
                affected = self._reschedule_tokens(dept_name, delay_mins)
                dept.load_pct = max(0, dept.load_pct - 15)
                self._addressed_depts.add(dept_name)
                reward += 0.4
                info["effects"].append(
                    f"Rescheduled {affected} token(s) in {dept_name} by {delay_mins} min. "
                    f"Load reduced to {dept.load_pct:.0f}%."
                )
            else:
                # Unnecessary reschedule
                self._unnecessary_actions += 1
                reward -= 0.1
                info["effects"].append(
                    f"{dept_name} is not overloaded ({dept.load_pct:.0f}%). "
                    f"Unnecessary reschedule penalised."
                )

        elif action_type == "alert_staff" and dept:
            if dept.load_pct > 80:
                dept.load_pct = max(0, dept.load_pct - 10)
                self._addressed_depts.add(dept_name)
                reward += 0.3
                info["effects"].append(
                    f"On-call staff alerted for {dept_name}. Load eased slightly."
                )
            else:
                self._unnecessary_actions += 1
                reward -= 0.1
                info["effects"].append(f"Staff alert not needed for {dept_name}.")

        elif action_type == "open_walkin" and dept:
            idle_depts = [d for d in self._departments if d.load_pct == 0]
            if dept in idle_depts:
                busiest = max(self._departments, key=lambda d: d.load_pct)
                busiest.load_pct = max(0, busiest.load_pct - 20)
                reward += 0.2
                info["effects"].append(
                    f"Walk-in opened at {dept_name}. "
                    f"Overflow absorbed from {busiest.name}."
                )
            else:
                self._unnecessary_actions += 1
                reward -= 0.1
                info["effects"].append(f"{dept_name} is not idle. Walk-in not useful.")

        # Partial progress: reward if avg wait improving
        avg_wait = self._avg_wait()
        if avg_wait < 15:
            reward += 0.2

        # Clip reward
        reward = round(max(0.0, min(1.0, reward)), 4)

        done = (self._step_num >= self.MAX_STEPS) or (len(self._get_overloaded()) == 0)
        if done:
            # End-of-episode penalty for unaddressed overloads
            unaddressed = [
                d for d in self._departments
                if d.load_pct > 60 and d.name not in self._addressed_depts
            ]
            penalty = len(unaddressed) * 0.3
            reward  = round(max(0.0, reward - penalty), 4)
            info["end_penalties"] = f"-{penalty:.1f} for {len(unaddressed)} unaddressed overload(s)"

        obs = self._get_observation()
        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> Dict[str, Any]:
        """Return full serialisable environment state."""
        return {
            "step":         self._step_num,
            "max_steps":    self.MAX_STEPS,
            "departments":  [asdict(d) for d in self._departments],
            "tokens":       [asdict(t) for t in self._tokens],
            "addressed":    list(self._addressed_depts),
            "unnecessary":  self._unnecessary_actions,
            "avg_wait":     self._avg_wait(),
        }

    # ── Helpers ───────────────────────────────────────────────────────

    def _get_observation(self) -> Observation:
        overloaded = self._get_overloaded()
        idle       = [d.name for d in self._departments if d.load_pct == 0 and d.doctor_count > 0]
        total_pts  = sum(d.active_patients for d in self._departments)
        return Observation(
            departments    =[asdict(d) for d in self._departments],
            overloaded_depts=overloaded,
            idle_depts     =idle,
            total_patients =total_pts,
            avg_wait_minutes=self._avg_wait(),
            critical_count =sum(1 for d in self._departments if d.load_pct > 80),
            step_number    =self._step_num,
            timestamp      =datetime.now().strftime("%H:%M:%S"),
        )

    def _get_overloaded(self) -> List[str]:
        return [d.name for d in self._departments if d.load_pct > 60]

    def _get_dept(self, name: str) -> Optional[Department]:
        for d in self._departments:
            if d.name == name:
                return d
        return None

    def _avg_wait(self) -> float:
        total_pts = sum(d.active_patients for d in self._departments)
        if total_pts == 0:
            return 0.0
        return sum(d.active_patients * d.avg_consult_time for d in self._departments) / total_pts

    def _generate_tokens(self) -> List[PatientToken]:
        tokens = []
        base_hour = 9  # 9 AM
        for i, dept in enumerate(self._departments):
            for j in range(min(dept.active_patients, 3)):
                slot_min = ((i * 3 + j) * 15) % 60
                slot_hr  = base_hour + ((i * 3 + j) * 15) // 60
                time_str = f"{slot_hr:02d}:{slot_min:02d}"
                doctor   = self._rng.choice(self.DOCTORS.get(dept.name, ["Dr. Unknown"]))
                tokens.append(PatientToken(
                    token_id =f"TKN-{100 + i * 10 + j}",
                    dept     =dept.name,
                    doctor   =doctor,
                    original_time=time_str,
                    current_time =time_str,
                ))
        return tokens

    def _reschedule_tokens(self, dept_name: str, delay_mins: int) -> int:
        count = 0
        for token in self._tokens:
            if token.dept == dept_name and not token.rescheduled:
                hr, mn = map(int, token.current_time.split(":"))
                total  = hr * 60 + mn + delay_mins
                token.current_time    = f"{total // 60:02d}:{total % 60:02d}"
                token.rescheduled     = True
                token.reschedule_reason = (
                    f"Your doctor in {dept_name} has too many patients. "
                    f"Your slot has been moved by {delay_mins} minutes."
                )
                token.wait_minutes   += delay_mins
                count += 1
        return count
