#!/usr/bin/env python3
"""
Runner script for the LangGraph Agentic Workflow
This script demonstrates how to use the comprehensive data analysis workflow.
"""

import os
import sys
from langgraph_agent import LangGraphAgent

def main():
    """Main function to run the LangGraph workflow"""
    
    print("🚀 Starting LangGraph Agentic Workflow")
    print("=" * 50)
    
    # Initialize the agent
    agent = LangGraphAgent()
    
    # Multiple ways to specify input dataset:
    
    # Option 1: Use default dataset (dataset.csv)
    # agent.initialize()
    
    # Option 2: Specify just the filename (searches in current and parent directories)
    # agent.initialize(dataset_filename="my_data.csv")
    
    # Option 3: Specify full path to dataset
    # agent.initialize(dataset_path="/full/path/to/your/dataset.csv")
    
    # Option 4: Load from URL
    # agent.initialize(dataset_url="https://raw.githubusercontent.com/user/repo/main/data.csv")
    
    # Option 5: Change default filename when creating agent
    # agent = LangGraphAgent(default_dataset_filename="my_default_data.csv")
    # agent.initialize()
    
    # For this example, we'll use the default dataset
    try:
        agent.initialize()
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        print("\n💡 Try one of these options:")
        print("   - agent.initialize(dataset_filename='your_file.csv')")
        print("   - agent.initialize(dataset_path='/full/path/to/file.csv')")
        print("   - agent.initialize(dataset_url='https://example.com/data.csv')")
        return 1
    
    # Process the dataset through the workflow
    try:
        print("\n🔄 Processing dataset through workflow...")
        result = agent.process()
        
        print("\n🎉 Workflow completed successfully!")
        print(f"📊 Generated {len(result['generated_charts'])} charts")
        print(f"📝 Created {len(result['chart_narratives'])} narratives")
        print("📄 HTML report saved as 'output.html'")
        
        # Print summary of results
        print("\n📋 Workflow Summary:")
        print(f"  - Data profiling: ✅ Completed")
        print(f"  - Chart recommendations: {len(result['chart_recommendations'])} charts suggested")
        print(f"  - Chart generation: {len([c for c in result['generated_charts'] if c['generation_status'] == 'success'])} successful")
        print(f"  - Chart evaluation: ✅ Completed")
        print(f"  - Narrative writing: ✅ Completed")
        print(f"  - HTML report: ✅ Generated")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during workflow execution: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
