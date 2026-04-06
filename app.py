# app.py — QTrack AI Environment Demo on Hugging Face Spaces

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
    "Cardiology":      ["Dr. Sharma", "Dr. Kapoor"],
    "General Medicine":["Dr. Mehta",  "Dr. Singh"],
    "Pediatrics":      ["Dr. Rao"],
    "Orthopedics":     ["Dr. Verma"],
    "Dermatology":     ["Dr. Nair"],
    "Neurology":       ["Dr. Iyer"],
    "Radiology":       ["Dr. Patel"],
}

# ── Token Generator ───────────────────────────────────────────────────
def generate_token(dept_name):
    now = datetime.now()
    # round up to next 15-min slot
    minutes = (now.minute // 15 + 1) * 15
    base_time = now.replace(second=0, microsecond=0) + timedelta(minutes=(minutes - now.minute))
    token_num = random.randint(100, 999)
    doctor    = random.choice(DOCTORS.get(dept_name, ["Dr. Unknown"]))
    return {
        "token":      f"TKN-{token_num}",
        "dept":       dept_name,
        "doctor":     doctor,
        "orig_time":  base_time,
        "curr_time":  base_time,
        "updated":    False,
        "update_reason": None,
    }

# ── AI Engine ─────────────────────────────────────────────────────────
def run_ai_engine(card_p, gen_p, peds_p, ortho_p, derm_p, neuro_p, radio_p):
    active_patients = [card_p, gen_p, peds_p, ortho_p, derm_p, neuro_p, radio_p]

    dept_loads = []
    for i, dept in enumerate(DEPARTMENTS):
        capacity = dept["doctor_count"] * 5
        load = min(100, round((active_patients[i] / capacity) * 100)) if capacity > 0 else 0
        dept_loads.append({**dept, "active": active_patients[i], "load": load})

    recommendations = []
    severity_icons = {"critical": "🚨", "high": "⚠️", "medium": "📊", "low": "💡"}

    # Rule 1: Overload Detection — delay patients, don't reroute
    for dept in dept_loads:
        if dept["load"] > 60:
            sev        = "critical" if dept["load"] > 80 else "high"
            delay_mins = 30 if dept["load"] > 80 else 15
            display = (
                f"{severity_icons[sev]} **{sev.upper()} — {dept['name']} Overloaded**\n\n"
                f"**{dept['name']}** is at **{dept['load']}%** capacity "
                f"({dept['active']} patients, {dept['doctor_count']} doctors).\n\n"
                f"🤖 *Patients with upcoming slots will be pushed by ~{delay_mins} min.*"
            )
            action = {
                "dept":       dept["name"],
                "delay_mins": delay_mins,
                "reason":     (
                    f"Your doctor in {dept['name']} currently has too many patients. "
                    f"To ensure you receive proper care, your appointment has been "
                    f"rescheduled by {delay_mins} minutes. We apologise for the inconvenience."
                ),
            }
            recommendations.append((display, action))

    # Rule 2: Queue Imbalance
    for dept in dept_loads:
        per_doctor = dept["active"] / dept["doctor_count"] if dept["doctor_count"] > 0 else 0
        if per_doctor >= 4:
            delay_mins = 20
            display = (
                f"📊 **MEDIUM — Queue Imbalance in {dept['name']}**\n\n"
                f"Avg **{per_doctor:.1f} patients/doctor** — queue building up.\n\n"
                f"🤖 *Scheduled slots will be extended by ~{delay_mins} min.*"
            )
            action = {
                "dept":       dept["name"],
                "delay_mins": delay_mins,
                "reason":     (
                    f"The queue in {dept['name']} is longer than usual right now. "
                    f"Your appointment time has been updated by {delay_mins} minutes "
                    f"so you are not kept waiting unnecessarily at the hospital."
                ),
            }
            recommendations.append((display, action))

    # Rule 3: Idle Resource (info only, no patient delay needed)
    for dept in dept_loads:
        if dept["load"] == 0 and dept["doctor_count"] > 0:
            busiest = max(dept_loads, key=lambda d: d["load"])
            if busiest["load"] > 30:
                display = (
                    f"💡 **LOW — Idle Doctors in {dept['name']}**\n\n"
                    f"{dept['doctor_count']} doctor(s) available with 0 patients.\n\n"
                    f"🤖 *Staff notified to support {busiest['name']} if needed.*"
                )
                recommendations.append((display, None))  # no patient action needed

    if not recommendations:
        recommendations.append((
            "✅ **ALL CLEAR** — All departments within optimal capacity.", None
        ))

    # Load bar chart
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
        f"🧑⚕️ **Total Active Patients:** {total_patients} &nbsp;|&nbsp; "
        f"⏱ **Avg Est. Wait:** {avg_wait:.1f} min &nbsp;|&nbsp; "
        f"🚨 **Critical Depts:** {critical_count} &nbsp;|&nbsp; "
        f"📋 **Actions Needed:** {action_count}"
    )

    return load_summary, stats, recommendations


def random_scenario():
    return [random.randint(0, 10) for _ in range(7)]


def format_token_card(token_data, updated=False):
    t = token_data
    orig_str = t["orig_time"].strftime("%I:%M %p")
    curr_str = t["curr_time"].strftime("%I:%M %p")

    if updated:
        badge = "🔴 UPDATED"
        time_line = f"~~{orig_str}~~ &nbsp;→&nbsp; **{curr_str}** ⏰"
    else:
        badge = "🟢 CONFIRMED"
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
    """Apply a schedule delay to the user's token if dept matches."""
    # Always return 3 values: (token_state, token_card_md, message)

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

    # Apply delay to matching department
    delay = action_dict["delay_mins"]
    token_data["curr_time"]      = token_data["curr_time"] + timedelta(minutes=delay)
    token_data["updated"]        = True
    token_data["update_reason"]  = action_dict["reason"]

    new_card = format_token_card(token_data, updated=True)
    msg = f"✅ Schedule updated! Your new slot: **{token_data['curr_time'].strftime('%I:%M %p')}**"
    return token_data, new_card, msg


# ── Gradio UI ─────────────────────────────────────────────────────────
with gr.Blocks(
    title="QTrack AI Environment | XYZ Hospital",
    theme=gr.themes.Soft(),
    css="""
        .gradio-container { max-width: 1300px !important; }
        footer { display: none !important; }
    """,
) as demo:

    token_state = gr.State(None)

    gr.Markdown("""
    # 🏥 QTrack AI Environment
    ## Hospital Queue Optimization Engine — XYZ Hospital
    > Simulate hospital patient load and see how the AI automatically updates
    > **your appointment schedule** when your department gets overloaded — keeping
    > you with your same doctor, just at a better time.
    """)

    # ── Token Generation ───────────────────────────────────────────────
    gr.Markdown("---\n### 🎟️ Step 1: Generate Your Token")
    with gr.Row():
        dept_dropdown = gr.Dropdown(
            choices=[d["name"] for d in DEPARTMENTS],
            label="Select Your Department",
            value="Cardiology",
        )
        get_token_btn = gr.Button("🎟️ Get Token", variant="primary", size="lg")

    with gr.Group(visible=False) as token_card_group:
        token_card_out = gr.Markdown("")

    # ── Department Load ────────────────────────────────────────────────
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

    # ── AI Recommendations ─────────────────────────────────────────────
    gr.Markdown("---\n### 🤖 Step 3: AI Recommendations")
    gr.Markdown("*If your department is overloaded, click **Update My Schedule** to reschedule your slot.*")

    MAX_RECS = 8
    groups, rec_mds, act_dicts, act_btns, act_msgs = [], [], [], [], []

    for i in range(MAX_RECS):
        with gr.Group(visible=False) as grp:
            with gr.Row():
                with gr.Column(scale=4):
                    rec_md = gr.Markdown("")
                with gr.Column(scale=1, min_width=180):
                    act_btn = gr.Button("📅 Update My Schedule", variant="primary", size="sm")
            act_msg = gr.Markdown("")
            act_dict_state = gr.State(None)
        groups.append(grp)
        rec_mds.append(rec_md)
        act_dicts.append(act_dict_state)
        act_btns.append(act_btn)
        act_msgs.append(act_msg)

    gr.Markdown("""
    ---
    ### 🧠 How This Works
    | Situation | What AI Does |
    |-----------|-------------|
    | Your dept > 60% full | Delays your slot by 15 min, notifies you with reason |
    | Your dept > 80% full | Delays your slot by 30 min, notifies you with reason |
    | Queue too long | Delays your slot by 20 min, notifies you with reason |
    | Your dept is fine | No change — your original time is kept ✅ |

    *You always stay with your same doctor & department. Only the time is adjusted.*

    *Built for the OpenEnv Hackathon · QTrack by XYZ Hospital Team*
    """)

    # ── Wire: Get Token ───────────────────────────────────────────────
    get_token_btn.click(
        fn=get_token,
        inputs=[dept_dropdown],
        outputs=[token_card_group, token_state, token_card_out],
    )

    # ── Wire: Run AI Engine ───────────────────────────────────────────
    def update_ui(card_p, gen_p, peds_p, ortho_p, derm_p, neuro_p, radio_p):
        load_summary, stats, recommendations = run_ai_engine(
            card_p, gen_p, peds_p, ortho_p, derm_p, neuro_p, radio_p
        )
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

    run_btn.click(fn=update_ui,
                  inputs=[card, gen, peds, ortho, derm, neuro, radio],
                  outputs=all_outputs)

    random_btn.click(fn=random_scenario, inputs=[],
                     outputs=[card, gen, peds, ortho, derm, neuro, radio])

    # ── Wire: Update My Schedule buttons ─────────────────────────────
    for i in range(MAX_RECS):
        act_btns[i].click(
            fn=apply_action,
            inputs=[act_dicts[i], token_state],
            outputs=[token_state, token_card_out, act_msgs[i]],
        )


demo.launch(server_name="0.0.0.0", server_port=7860)
