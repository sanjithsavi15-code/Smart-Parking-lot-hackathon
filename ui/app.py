import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from env.openenv_api import ParkingEnv
from env.models import Action, ActionType
from env.tasks import TASK_REGISTRY

# ─────────────────────────────────────────────────────────────────────────────
# 1. FastAPI Backend (For the OpenEnv Automated Checker)
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Smart Parking OpenEnv API")

# Global instance for the automated API checker
api_env = ParkingEnv()

class ResetRequest(BaseModel):
    task_id: str = "basic_park"

@app.post("/reset")
def api_reset(req: ResetRequest):
    try:
        return api_env.reset(task_id=req.task_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def api_step(action: Action):
    try:
        return api_env.step(action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def api_state():
    try:
        return api_env.state()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/grade")
def api_grade():
    try:
        return api_env.summary()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# 2. Gradio UI (For Human Interaction)
# ─────────────────────────────────────────────────────────────────────────────

def get_new_env():
    return ParkingEnv()

def render_grid(obs):
    if not obs:
        return "<i>Press Reset to start a task.</i>"
    
    html = "<div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 10px;'>"
    for slot in obs.available_slots:
        bg_color = "#2d3748" if not slot.is_occupied else "#742a2a"
        border_color = "#48bb78" if not slot.is_occupied else "#f56565"
        occ = f"🚗 Car {slot.occupant_id}" if slot.is_occupied else "✅ Empty"
        
        html += f"""
        <div style='background-color: {bg_color}; border: 2px solid {border_color}; border-radius: 8px; padding: 10px; text-align: center; color: white; font-family: sans-serif;'>
            <div style='font-weight: bold; font-size: 1.1em;'>{slot.id}</div>
            <div style='font-size: 0.8em; color: #a0aec0; margin-bottom: 5px;'>{slot.slot_type.value}</div>
            <div style='font-size: 0.9em;'>{occ}</div>
        </div>
        """
    html += "</div>"
    return html

def render_queue(obs):
    if not obs:
        return "<i>No cars in queue.</i>"
    if not obs.incoming_queue:
        return "<div style='color: #48bb78; font-weight: bold;'>🎉 Queue is empty!</div>"
        
    html = "<div style='display: flex; flex-direction: column; gap: 8px;'>"
    for car in obs.incoming_queue:
        icon = "⚡" if car.car_type.value == "EV" else "⭐" if car.car_type.value == "VIP" else "🚙"
        html += f"""
        <div style='background-color: #2b6cb0; padding: 8px 12px; border-radius: 6px; color: white;'>
            <b>{icon} Car {car.id}</b> &mdash; {car.car_type.value} 
            <span style='float: right; font-size: 0.8em; color: #e2e8f0;'>Arrived T={car.entry_time}</span>
        </div>
        """
    html += "</div>"
    return html

def update_dropdowns(obs):
    if not obs:
        return gr.update(choices=[], value=None), gr.update(choices=[], value=None)
    car_choices = [(f"Car {c.id} ({c.car_type.value})", c.id) for c in obs.incoming_queue]
    slot_choices = [(f"{s.id} ({s.slot_type.value})", s.id) for s in obs.available_slots if not s.is_occupied]
    return gr.update(choices=car_choices, value=None), gr.update(choices=slot_choices, value=None)

def reset_env(env, task_id):
    if not env:
        env = get_new_env()
    obs = env.reset(task_id=task_id)
    grid = render_grid(obs)
    queue = render_queue(obs)
    car_u, slot_u = update_dropdowns(obs)
    task_name = TASK_REGISTRY[task_id].display_name
    log = f"<div style='color: #63b3ed; margin-bottom: 5px;'><i>🔄 Environment reset to task: {task_name}. Time Step: {obs.current_time_step}, Revenue: ${obs.current_revenue:.2f}</i></div>"
    return env, grid, queue, car_u, slot_u, log

def toggle_inputs(action_type):
    if action_type == "ASSIGN":
        return gr.update(interactive=True), gr.update(interactive=True)
    elif action_type == "REJECT":
        return gr.update(interactive=True), gr.update(interactive=False, value=None)
    else: 
        return gr.update(interactive=False, value=None), gr.update(interactive=False, value=None)

def execute_action(env, log_history, action_type, car_id, slot_id):
    if env is None or env.task_id is None:
        err = "<div style='color: #f56565;'><b>Error:</b> Please Reset/Load a task first!</div>"
        return env, gr.update(), gr.update(), gr.update(), gr.update(), err + log_history
        
    if env.is_done:
        msg = "<div style='color: #fbd38d;'><i>Episode is already finished! Please reset to play again.</i></div>"
        return env, gr.update(), gr.update(), gr.update(), gr.update(), msg + log_history

    try:
        act_enum = ActionType(action_type)
        if act_enum == ActionType.WAIT:
            act = Action(action_type=act_enum, car_id=None, slot_id=None)
        elif act_enum == ActionType.REJECT:
            if car_id is None: raise ValueError("Must select a car to REJECT.")
            act = Action(action_type=act_enum, car_id=car_id, slot_id=None)
        else:
            if car_id is None or slot_id is None: raise ValueError("Must select both a car and a slot to ASSIGN.")
            act = Action(action_type=act_enum, car_id=car_id, slot_id=slot_id)
            
        res = env.step(act)
        obs = res.observation
        grid = render_grid(obs)
        queue = render_queue(obs)
        car_u, slot_u = update_dropdowns(obs)
        
        info_msg = res.info.get('message', str(res.info.get('error', '')))
        reward_val = res.reward.value
        reward_color = "#48bb78" if reward_val > 0 else "#f56565" if reward_val < 0 else "#a0aec0"
        
        entry = f"<div style='border-bottom: 1px solid #4a5568; padding: 5px 0;'>"
        entry += f"<b>T={obs.current_time_step}</b> | Action: <b>{action_type}</b> "
        if act_enum != ActionType.WAIT:
            entry += f"[Car {car_id}" + (f" → Slot {slot_id}]" if slot_id else "]")
        entry += f" | Reward: <span style='color: {reward_color}; font-weight: bold;'>{reward_val:.2f}</span>"
        entry += f"<br><span style='font-size: 0.9em; color: #cbd5e0;'>{info_msg}</span></div>"
        
        if res.done:
            summary = env.summary()
            entry = f"<div style='background-color: #276749; padding: 10px; border-radius: 5px; margin: 10px 0; color: white;'>" \
                    f"<b>🎉 Episode Finished!</b><br>Final Score: <b>{summary.final_score:.2f} / 1.0</b><br>" \
                    f"Total Revenue: ${summary.total_revenue:.2f}<br>" \
                    f"Cars Parked: {summary.cars_parked} | Rejected: {summary.cars_rejected} | Invalid Actions: {summary.invalid_actions}</div>" + entry
            
        return env, grid, queue, car_u, slot_u, entry + log_history
    except Exception as e:
        err = f"<div style='color: #f56565; padding: 5px 0;'><b>Error:</b> {str(e)}</div>"
        return env, gr.update(), gr.update(), gr.update(), gr.update(), err + log_history

def execute_wait(env, log_history):
    return execute_action(env, log_history, "WAIT", None, None)

def build_app():
    def get_task_label(task):
        emoji = "🟢" if task.difficulty.lower() == "easy" else "🟡" if task.difficulty.lower() == "medium" else "🔴"
        return f"{emoji} {task.difficulty.capitalize()} — {task.display_name}"

    with gr.Blocks(title="Smart Parking Lot", theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as interface:
        gr.Markdown("<div style='text-align: center; padding: 20px;'><h1>🅿️ Smart Parking Lot</h1><p style='color: gray; font-size: 1.1em;'>OpenEnv-compatible simulation · Agentic AI Hackathon Demo</p></div>")
        env_state = gr.State(get_new_env)
        with gr.Row():
            task_choices = [(get_task_label(v), k) for k, v in TASK_REGISTRY.items()]
            task_dropdown = gr.Dropdown(choices=task_choices, label="Select Task", value="basic_park", scale=3)
            reset_btn = gr.Button("🔄 Reset / Load Task", variant="primary", scale=1)
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### 🅿️ Parking Grid")
                grid_display = gr.HTML(value="<i>Press Reset to start a task.</i>")
            with gr.Column(scale=1):
                gr.Markdown("### 🚦 Incoming Queue")
                queue_display = gr.HTML(value="")
        gr.Markdown("### 🎮 Take an Action")
        with gr.Row():
            action_type_radio = gr.Radio(choices=["ASSIGN", "REJECT", "WAIT"], label="Action Type", value="ASSIGN", scale=1)
            car_dropdown = gr.Dropdown(label="Car (from queue)", choices=[], scale=2)
            slot_dropdown = gr.Dropdown(label="Slot (for ASSIGN)", choices=[], scale=2)
        with gr.Row():
            exec_btn = gr.Button("▶ Execute Action", variant="primary")
            wait_btn = gr.Button("⏸ WAIT", variant="secondary")
        gr.Markdown("### 📋 Action Log")
        log_display = gr.HTML(value="")

        reset_btn.click(fn=reset_env, inputs=[env_state, task_dropdown], outputs=[env_state, grid_display, queue_display, car_dropdown, slot_dropdown, log_display])
        action_type_radio.change(fn=toggle_inputs, inputs=action_type_radio, outputs=[car_dropdown, slot_dropdown])
        exec_btn.click(fn=execute_action, inputs=[env_state, log_display, action_type_radio, car_dropdown, slot_dropdown], outputs=[env_state, grid_display, queue_display, car_dropdown, slot_dropdown, log_display])
        wait_btn.click(fn=execute_wait, inputs=[env_state, log_display], outputs=[env_state, grid_display, queue_display, car_dropdown, slot_dropdown, log_display])
        
    return interface

# Generate the Gradio App
gradio_app = build_app()

# Mount Gradio onto the FastAPI router 
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    # Launch uvicorn to serve the API + Gradio simultaneously
    uvicorn.run(app, host="0.0.0.0", port=7860)