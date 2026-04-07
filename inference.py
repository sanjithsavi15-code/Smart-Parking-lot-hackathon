import os
import json
from openai import OpenAI
from env.openenv_api import ParkingEnv
from env.models import Action, ActionType

def run_inference():
    print("Initializing Smart Parking Environment...")
    env = ParkingEnv()
    
    # Initialize the Grok client using the OpenAI SDK
    # Replace 'grok-beta' with the exact model name provided by your hackathon
    client = OpenAI(
        api_key=os.environ.get("GROK_API_KEY", "your-grok-api-key-here"),
        base_url="https://api.x.ai/v1"
    )
    
    # Let's test it on the Medium difficulty task
    task_id = "ev_sort"
    print(f"\n--- Starting Task: {task_id} ---")
    obs = env.reset(task_id)
    
    while not env.is_done:
        print(f"\nTime Step: {obs.current_time_step}")
        
        # 1. Package the environment state into a prompt for Grok
        prompt = f"""
        You are an AI Smart Parking Manager.
        
        Current Environment State:
        {obs.model_dump_json(indent=2)}
        
        Rules:
        - EV cars must go to EV_CHARGING slots.
        - STANDARD cars must go to STANDARD slots.
        - VIP cars must go to PREMIUM slots.
        - If no matching slots are available, you can REJECT the car.
        - If you want to skip a turn, you can WAIT.
        
        Determine the best action for the first car in the incoming_queue.
        You MUST respond with a perfectly formatted JSON object matching this exact schema:
        {{"action_type": "ASSIGN" | "REJECT" | "WAIT", "car_id": <integer or null>, "slot_id": "<string or null>"}}
        """
        
        try:
            # 2. Call the Grok LLM
            response = client.chat.completions.create(
                model="grok-beta", 
                messages=[
                    {"role": "system", "content": "You are a JSON-only API. You only output raw, valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # 3. Parse Grok's JSON response into your Pydantic Action model
            raw_response = response.choices[0].message.content
            action_data = json.loads(raw_response)
            action = Action(**action_data)
            
            print(f"🤖 Grok decided: {action.action_type.value} "
                  f"(Car: {action.car_id}, Slot: {action.slot_id})")
            
            # 4. Execute the action in the environment
            step_result = env.step(action)
            obs = step_result.observation
            
            # Print any errors or messages from the environment engine
            if 'error' in step_result.info:
                print(f"⚠️  Engine Warning: {step_result.info['error']}")
            
        except Exception as e:
            print(f"❌ Error during LLM processing: {e}")
            print("Forcing a WAIT action to prevent the loop from crashing...")
            step_result = env.step(Action(action_type=ActionType.WAIT, car_id=None, slot_id=None))
            obs = step_result.observation

    # 5. The episode is finished. Print the final grade!
    summary = env.summary()
    print("\n" + "="*40)
    print("🎉 EPISODE COMPLETE 🎉")
    print(f"Final Score:      {summary.final_score:.2f} / 1.0")
    print(f"Total Revenue:    ${summary.total_revenue:.2f}")
    print(f"Cars Parked:      {summary.cars_parked}")
    print(f"Invalid Actions:  {summary.invalid_actions}")
    print("="*40)

if __name__ == "__main__":
    run_inference()
