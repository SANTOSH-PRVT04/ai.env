# app.py — QTrack AI Environment
# Runs FastAPI (OpenEnv REST API) + Gradio UI together on port 7860

# ── Patch 1: Fix missing audioop/pyaudioop in Python 3.13 ────────────
import sys, types
if "audioop" not in sys.modules:
    _audioop = types.ModuleType("audioop")
    sys.modules["audioop"] = _audioop
if "pyaudioop" not in sys.modules:
    sys.modules["pyaudioop"] = sys.modules["audioop"]

# ── Patch 2: Fix missing HfFolder in newer huggingface_hub ───────────
import huggingface_hub
if not hasattr(huggingface_hub, "HfFolder"):
    class _HfFolder:
        @staticmethod
        def get_token(): return None
        @staticmethod
        def save_token(token): pass
        @staticmethod
        def delete_token(): pass
    huggingface_hub.HfFolder = _HfFolder

import gradio as gr
import random
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import threading

from env import QTrackEnv

# ── FastAPI app (OpenEnv REST API) ────────────────────────────────────
api = FastAPI(title="QTrack OpenEnv API", version="1.0.0")
api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_env: Optional[QTrackEnv] = None

class ResetRequest(BaseModel):
    seed: Optional[int] = None

class StepRequest(BaseModel):
    action_type: str = Field(default="noop")
    dept:        str = Field(default="")
    delay_mins:  int = Field(default=15)

@api.post("/reset")
def api_reset(request: ResetRequest = ResetRequest()):
    global _env
    _env = QTrackEnv(seed=request.seed)
    obs  = _env.reset()
    return {"observation": obs.__dict__, "status": "reset_ok"}

@api.post("/step")
def api_step(request: StepRequest = StepRequest()):
    global _env
    if _env is None:
        _env = QTrackEnv(); _env.reset()
    result = _env.step({"action_type": request.action_type, "dept": request.dept, "delay_mins": request.delay_mins})
    return {"observation": result.observation.__dict__, "reward": result.reward, "done": result.done, "info": result.info}

@api.get("/state")
@api.post("/state")
def api_state():
    global _env
    if _env is None:
        _env = QTrackEnv(); _env.reset()
    return _env.state()

@api.get("/health")
def api_health():
    return {"status": "ok", "env": "QTrack-HospitalQueue-v1"}

@api.get("/")
def api_root():
    return {"name": "QTrack Hospital Queue Environment", "version": "1.0.0", "endpoints": ["/reset", "/step", "/state", "/health"]}

# ── Start FastAPI in background thread on port 8000 ──────────────────
def run_api():
    uvicorn.run(api, host="0.0.0.0", port=8000, log_level="warning")

api_thread = threading.Thread(target=run_api, daemon=True)
api_thread.start()

# ── Simulated Hospital State ──────────────────────────────────────────
DEPARTMENTS = [
    {"id": "dept-1", "name": "Cardiology",      "avg_consult_time": 12, "doctor_count": 2},
    {"id": "dept-2", "name": "General Medicine", "avg_consult_time": 8,  "doctor_count": 2},
    {"id": "dept-3", "name": "Pediatrics",       "avg_consult_time": 10, "doctor_count": 1},
    {"id": "dept-4", "name": "Orthopedics",      "avg_consult_time": 15, "doctor_count": 1},
    {"id": "dept-5", "name": "Dermatology",      "avg_consult_time": 8,  "doctor_count": 1},
    {"id": "dept-6", "name": "Neurology",        "avg_consult_time": 18, "doctor_count": 1},
    {"id": "dept-7", "name": "Radiology",        "avg_consult_time": 20, "doctor_count": 1},
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

def run_ai_engine(card_p, gen_p, peds_p, ortho_p, derm_p, neuro_p, radio_p):
    active_patients = [card_p, gen_p, peds_p, ortho_p, derm_p, neuro_p, radio_p]
    dept_loads = []
    for i, dept in enumerate(DEPARTMENTS):
        capacity = dept["doctor_count"] * 5
        load = min(100, round((active_patients[i] / capacity) * 100)) if capacity > 0 else 0
        dept_loads.append({**dept, "active": active_patients[i], "load": load})

    recommendations = []
    severity_icons = {"critical": "🚨", "high": "⚠️", "medium": "📊", "low": "💡"}

    for dept in dept_loads:
        if dept["load"] > 60:
            sev = "critical" if dept["load"] > 80 else "high"
            delay_mins = 30 if dept["load"] > 80 else 15
            display = (
                f"{severity_icons[sev]} **{sev.upper()} — {dept['name']} Overloaded**\n\n"
                f"**{dept['name']}** is at **{dept['load']}%** capacity "
                f"({dept['active']} patients, {dept['doctor_count']} doctors).\n\n"
                f"🤖 *Patient slots in {dept['name']} will be pushed by ~{delay_mins} min.*"
            )
            action = {
                "dept": dept["name"], "delay_mins": delay_mins,
                "reason": (
                    f"Your doctor in {dept['name']} currently has too many patients. "
                    f"To ensure you receive proper care, your appointment has been "
                    f"rescheduled by {delay_mins} minutes. We apologise for the inconvenience."
                ),
            }
            recommendations.append((display, action))

    for dept in dept_loads:
        per_doctor = dept["active"] / dept["doctor_count"] if dept["doctor_count"] > 0 else 0
        if per_doctor >= 4:
            delay_mins = 20
            display = (
                f"📊 **MEDIUM — Queue Imbalance in {dept['name']}**\n\n"
                f"Avg **{per_doctor:.1f} patients/doctor** — queue building up.\n\n"
                f"🤖 *Scheduled slots extended by ~{delay_mins} min.*"
            )
            action = {
                "dept": dept["name"], "delay_mins": delay_mins,
                "reason": (
                    f"The queue in {dept['name']} is longer than usual. "
                    f"Your appointment has been updated by {delay_mins} minutes "
                    f"so you are not kept waiting unnecessarily."
                ),
            }
            recommendations.append((display, action))

    for dept in dept_loads:
        if dept["load"] == 0 and dept["doctor_count"] > 0:
            busiest = max(dept_loads, key=lambda d: d["load"])
            if busiest["load"] > 30:
                display = (
                    f"💡 **LOW — Idle Doctors in {dept['name']}**\n\n"
                    f"{dept['doctor_count']} doctor(s) available with 0 patients.\n\n"
                    f"🤖 *Staff notified to support {busiest['name']} if needed.*"
                )
                recommendations.append((display, None))

    if not recommendations:
        recommendations.append(("✅ **ALL CLEAR** — All departments within optimal capacity.", None))

    load_bars = []
    for d in dept_loads:
        filled = round(d["load"] / 5)
        bar    = "█" * filled + "░" * (20 - filled)
        color  = "🔴" if d["load"] > 80 else ("🟠" if d["load"] > 60 else ("🟡" if d["load"] > 30 else "🟢"))
        load_bars.append(f"{color} {d['name']:<20} [{bar}] {d['load']:>3}%  ({d['active']} pts)")

    load_summary = "### 📊 Department Load\n```\n" + "\n".join(load_bars) + "\n```"
    total_patients = sum(active_patients)
    avg_wait = (
        sum(d["active"] * d["avg_consult_time"] for d in dept_loads) / total_patients
        if total_patients > 0 else 0
    )
    critical_count = sum(1 for d in dept_loads if d["load"] > 80)
    action_count   = len([r for r in recommendations if r[1] is not None])
    stats = (
        f"🧑‍⚕️ **Total Active Patients:** {total_patients} &nbsp;|&nbsp; "
        f"⏱ **Avg Est. Wait:** {avg_wait:.1f} min &nbsp;|&nbsp; "
        f"🚨 **Critical Depts:** {critical_count} &nbsp;|&nbsp; "
        f"📋 **Actions Needed:** {action_count}"
    )
    return load_summary, stats, recommendations

def random_scenario():
    return [random.randint(0, 10) for _ in range(7)]

def generate_token(dept_name):
    now = datetime.now()
    minutes = (now.minute // 15 + 1) * 15
    base_time = now.replace(second=0, microsecond=0) + timedelta(minutes=(minutes - now.minute))
    token_num = random.randint(100, 999)
    doctor    = random.choice(DOCTORS.get(dept_name, ["Dr. Unknown"]))
    return {
        "token": f"TKN-{token_num}", "dept": dept_name, "doctor": doctor,
        "orig_time": base_time, "curr_time": base_time,
        "updated": False, "update_reason": None,
    }

def format_token_card(token_data, updated=False):
    t = token_data
    orig_str = t["orig_time"].strftime("%I:%M %p")
    curr_str = t["curr_time"].strftime("%I:%M %p")
    if updated:
        badge     = "🔴 UPDATED"
        time_line = f"~~{orig_str}~~ &nbsp;→&nbsp; **{curr_str}** ⏰"
    else:
        badge     = "🟢 CONFIRMED"
        time_line = f"**{curr_str}**"
    card = (
        f"### 🎟️ Your Token\n\n"
        f"| | |\n|---|---|\n"
        f"| **Token No.**   | `{t['token']}` |\n"
        f"| **Department**  | {t['dept']} |\n"
        f"| **Doctor**      | {t['doctor']} |\n"
        f"| **Status**      | {badge} |\n"
        f"| **Your Slot**   | {time_line} |\n"
    )
    if updated and t["update_reason"]:
        card += f"\n\n> 📢 **Notice:** {t['update_reason']}"
    return card

def get_token(dept_name):
    if not dept_name:
        return gr.update(visible=False), None, ""
    token_data = generate_token(dept_name)
    card = format_token_card(token_data)
    return gr.update(visible=True), token_data, card

def apply_action(action_dict, token_data):
    if token_data is None:
        return None, "", "⚠️ No token found. Please generate a token first."
    current_card = format_token_card(token_data, updated=token_data.get("updated", False))
    if action_dict is None:
        return token_data, current_card, "ℹ️ No schedule change needed for this alert."
    if token_data["dept"] != action_dict["dept"]:
        return token_data, current_card, (
            f"ℹ️ This alert is for **{action_dict['dept']}**. "
            f"Your appointment is with **{token_data['dept']}** — your schedule is unchanged. ✅"
        )
    delay = action_dict["delay_mins"]
    token_data["curr_time"]     = token_data["curr_time"] + timedelta(minutes=delay)
    token_data["updated"]       = True
    token_data["update_reason"] = action_dict["reason"]
    new_card = format_token_card(token_data, updated=True)
    msg = f"✅ Schedule updated! Your new slot: **{token_data['curr_time'].strftime('%I:%M %p')}**"
    return token_data, new_card, msg

# ── Gradio UI ─────────────────────────────────────────────────────────
with gr.Blocks(
    title="QTrack AI Environment | XYZ Hospital",
    theme=gr.themes.Soft(),
    css=".gradio-container { max-width: 1300px !important; } footer { display: none !important; }",
) as demo:

    token_state = gr.State(None)

    gr.Markdown("""
    # 🏥 QTrack AI Environment
    ## Hospital Queue Optimization Engine — XYZ Hospital
    > Simulate hospital patient load and see how the AI automatically updates
    > **your appointment schedule** when your department gets overloaded —
    > keeping you with your same doctor, just at a better time.
    """)

    gr.Markdown("---\n### 🎟️ Step 1: Generate Your Token")
    with gr.Row():
        dept_dropdown = gr.Dropdown(choices=[d["name"] for d in DEPARTMENTS], label="Select Your Department", value="Cardiology")
        get_token_btn = gr.Button("🎟️ Get Token", variant="primary", size="lg")

    with gr.Group(visible=False) as token_card_group:
        token_card_out = gr.Markdown("")

    gr.Markdown("---\n### ⚙️ Step 2: Simulate Patient Load")
    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("*Set active patients per department:*")
            card  = gr.Slider(0, 15, value=6, step=1, label="🫀 Cardiology  ·  2 doctors  ·  cap 10")
            gen   = gr.Slider(0, 15, value=2, step=1, label="🩺 General Medicine  ·  2 doctors  ·  cap 10")
            peds  = gr.Slider(0, 10, value=1, step=1, label="👶 Pediatrics  ·  1 doctor  ·  cap 5")
            ortho = gr.Slider(0, 10, value=1, step=1, label="🦴 Orthopedics  ·  1 doctor  ·  cap 5")
            derm  = gr.Slider(0, 10, value=1, step=1, label="🔬 Dermatology  ·  1 doctor  ·  cap 5")
            neuro = gr.Slider(0, 10, value=0, step=1, label="🧠 Neurology  ·  1 doctor  ·  cap 5")
            radio = gr.Slider(0, 10, value=0, step=1, label="📡 Radiology  ·  1 doctor  ·  cap 5")
            with gr.Row():
                run_btn    = gr.Button("▶ Run AI Engine",    variant="primary",   size="lg")
                random_btn = gr.Button("🎲 Random Scenario", variant="secondary", size="lg")
        with gr.Column(scale=2):
            stats_out = gr.Markdown("*← Set patient counts and click Run AI Engine.*")
            load_out  = gr.Markdown()

    gr.Markdown("---\n### 🤖 Step 3: AI Recommendations & Actions")
    gr.Markdown("*Click **📅 Update My Schedule** to reschedule your slot if your department is overloaded.*")

    MAX_RECS = 8
    groups, rec_mds, act_dicts, act_btns, act_msgs = [], [], [], [], []
    for i in range(MAX_RECS):
        with gr.Group(visible=False) as grp:
            with gr.Row():
                with gr.Column(scale=4):
                    rec_md = gr.Markdown("")
                with gr.Column(scale=1, min_width=180):
                    act_btn = gr.Button("📅 Update My Schedule", variant="primary", size="sm")
            act_msg        = gr.Markdown("")
            act_dict_state = gr.State(None)
        groups.append(grp); rec_mds.append(rec_md)
        act_dicts.append(act_dict_state); act_btns.append(act_btn); act_msgs.append(act_msg)

    gr.Markdown("""
    ---
    ### 🧠 How This Works
    | Situation | What AI Does |
    |-----------|-------------|
    | Your dept > 60% full | Delays your slot by 15 min with reason |
    | Your dept > 80% full | Delays your slot by 30 min with reason |
    | Queue too long | Delays your slot by 20 min with reason |
    | Different dept alert | No change — your schedule is untouched ✅ |

    *You always stay with your same doctor & department. Only the time is adjusted.*

    *Built for the OpenEnv Hackathon · QTrack by XYZ Hospital Team*
    """)

    get_token_btn.click(fn=get_token, inputs=[dept_dropdown], outputs=[token_card_group, token_state, token_card_out])

    def update_ui(card_p, gen_p, peds_p, ortho_p, derm_p, neuro_p, radio_p):
        load_summary, stats, recommendations = run_ai_engine(card_p, gen_p, peds_p, ortho_p, derm_p, neuro_p, radio_p)
        out = [load_summary, stats]
        for i in range(MAX_RECS):
            if i < len(recommendations):
                disp, action = recommendations[i]
                out += [gr.update(visible=True), gr.update(value=disp), action, gr.update(value="")]
            else:
                out += [gr.update(visible=False), gr.update(value=""), None, gr.update(value="")]
        return out

    all_outputs = [load_out, stats_out]
    for i in range(MAX_RECS):
        all_outputs += [groups[i], rec_mds[i], act_dicts[i], act_msgs[i]]

    run_btn.click(fn=update_ui, inputs=[card, gen, peds, ortho, derm, neuro, radio], outputs=all_outputs)
    random_btn.click(fn=random_scenario, inputs=[], outputs=[card, gen, peds, ortho, derm, neuro, radio])

    for i in range(MAX_RECS):
        act_btns[i].click(fn=apply_action, inputs=[act_dicts[i], token_state], outputs=[token_state, token_card_out, act_msgs[i]])

# ── Launch Gradio on port 7860 ────────────────────────────────────────
demo.launch(server_name="0.0.0.0", server_port=7860)
