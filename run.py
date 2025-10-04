import time 
import os
import importlib.util
import sys
from typing import Any

def load_agent(agent_path: str) -> Any:
    if not os.path.exists(agent_path):
        raise ValueError("agent.py not found in submission")
    
    spec = importlib.util.spec_from_file_location("agent", agent_path)
    if spec is None or spec.loader is None:
        raise ValueError("Could not load agent.py")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent"] = module
    spec.loader.exec_module(module)
    
    if not hasattr(module, "Agent"):
        raise ValueError("agent.py must define an Agent class")
    
    return module.Agent()

def evaluate_agent(agent: Any) -> None:
    start_time = time.time()
    try: 
        print("🚀 Starting Evaluator-Optimizer Workflow")
        print("=" * 50)
        
        agent.initialize()
        result = agent.process()
        
        total_time = time.time() - start_time
        
        print(f"⏱️  Total time: {total_time:.2f} seconds")

        
        print("\n🎉 Workflow completed successfully!")
        print("📖 Open 'output.html' in your browser to view the evaluated report.")
        
    except Exception as e:
        print(f"❌ Error running workflow: {e}")

def main():
    agent_path = "agent.py"
    agent = load_agent(agent_path)
    print("Agent loaded...")
    evaluate_agent(agent)

if __name__ == "__main__":
    main()


