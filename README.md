# 🅿️ Smart Parking OS: An Agentic AI Benchmark

[![OpenEnv 1.0](https://img.shields.io/badge/OpenEnv-1.0--Compliant-green)](https://github.com/OpenEnv)
[![Gradio](https://img.shields.io/badge/UI-Gradio--4.20-orange)](https://gradio.app/)
[![Pydantic V2](https://img.shields.io/badge/Validation-Pydantic--V2-red)](https://docs.pydantic.dev/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)

**Smart Parking OS** is a high-fidelity, **OpenEnv-compliant** simulation environment designed to benchmark the spatial reasoning and long-term planning capabilities of LLM Agents. 

Developed for the **Agentic AI Hackathon**, this project transitions from a legacy Java logic system to a modern, Python-native **state machine**. It provides a deterministic playground where AI agents must manage complex logistics, prioritize specialized vehicle types (EV/VIP), and maximize revenue under pressure.

---

## 🏗️ Architectural Evolution

This project features a complete rewrite optimized for the **Agentic Loop**:
* **Zero-Hallucination Guardrails:** Utilizes **Pydantic V2** to strictly validate LLM outputs against domain models (`Car`, `Slot`, `Action`) before they reach the engine.
* **Reactive Visualization:** A custom **Gradio** dashboard that renders the parking grid using dynamic HTML/Flexbox for real-time monitoring.
* **Deterministic Grading:** Includes a robust evaluation suite that scores agents based on revenue efficiency and rule adherence.

---

## 🛠️ Tech Stack

* **Logic:** Python 3.10 (Core Engine)
* **Validation:** Pydantic V2 (Strict Type Safety)
* **Interface:** Gradio (Stateful Web UI)
* **Standard:** OpenEnv 1.0 (Standardized AI Environment Schema)
* **Deployment:** Docker (Optimized for Hugging Face Spaces)

---

## 📂 Project Structure

```text
smart-parking-env/
├── env/
│   ├── engine.py       # Core simulation logic & revenue calculation
│   ├── models.py       # Pydantic V2 domain models & schema
│   ├── openenv_api.py  # OpenEnv RL-style API (reset, step, state)
│   └── tasks.py        # Difficulty-scaled benchmark scenarios
├── ui/
│   └── app.py          # Gradio frontend & session management
├── Dockerfile          # HF Spaces-ready container configuration
├── openenv.yaml        # OpenEnv 1.0 Metadata
└── requirements.txt    # Dependency manifest
```

## ⚙️ Simulation Rules

The environment operates as a deterministic, step-based state machine. Agents must make real-time decisions based on a full observation of the parking lot grid and the incoming vehicle queue.

### 🎮 Agent Actions
On every time step, the agent must submit exactly one of the following actions:

* **`ASSIGN`**: Park a specific `car_id` in a target `slot_id`. This succeeds only if the slot is currently vacant.
* **`REJECT`**: Permanently remove a car from the FIFO queue. Use this strategically to save specialized slots for higher-priority arrivals.
* **`WAIT`**: Do nothing and allow the simulation clock to tick forward. This is essential for waiting for currently occupied slots to clear.

---

### 🏆 Reward Shaping
The environment uses a shaped reward system to train agents toward optimal long-term planning and rule adherence:

| Action | Reward | Context |
| :--- | :---: | :--- |
| **Base Success** | `+0.2` | Successfully parking any vehicle in a valid slot. |
| **Perfect Match** | `+0.5` | Bonus for matching an **EV** to a charger or **VIP** to a Premium bay. |
| **Inefficiency** | `-0.1` | Penalty for rejecting a car when a valid compatible slot was available. |
| **Illegal Move** | `-0.5` | Penalty for hallucinated moves (e.g., parking in an occupied slot). |

---

### 🎯 Benchmark Tasks
These scenarios are used to evaluate the agent's performance and generate a final score:

* **🟢 Basic Park (Easy)**
    * **Scenario:** 5 standard cars, empty lot.
    * **Objective:** Park all vehicles within 20 steps.
* **🟡 EV Sort (Medium)**
    * **Scenario:** Mixed queue (3 EV, 3 Standard) with limited EV chargers.
    * **Objective:** Maximize "Perfect Match" bonuses while keeping the queue moving.
* **🔴 Rush Hour (Hard)**
    * **Scenario:** 90% lot occupancy with a sudden surge of VIP and Standard cars.
    * **Objective:** Maximize total revenue within a strict 15-step limit.

---

## 🚀 Getting Started

### 📦 Local Installation
To set up the environment on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/smart-parking-env.git](https://github.com/yourusername/smart-parking-env.git)
   cd smart-parking-env
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
    ```
3. **Launch the simulation UI:**
    ```bash
     python ui/app.py
    ```
    *The Gradio dashboard will launch at http://localhost:7860.*
 ### 🚢 Deployment
 This project is pre-configured for Hugging Face Spaces or any Docker-compatible cloud provider.

 **Build and Run Locally:**
   ```bash
   docker build -t smart-parking-os .
   docker run -p 7860:7860 smart-parking-os
   ```

**Created for the Agentic AI Hackathon 2026**

*Optimizing infrastructure through Agentic Intelligence.*
