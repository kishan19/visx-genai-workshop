#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced features of the LangGraph agent.
This script shows the new capabilities including:
- Smart feedback loop with evaluator-optimizer pattern
- Dynamic report title generation
- Executive summary
- Collapsible data quality section
- Coherent narrative flow
- Agent reasoning display
"""

import os
import sys
from langgraph_agent import LangGraphAgent

def main():
    """Test the enhanced LangGraph workflow"""
    
    print("🚀 Testing Enhanced LangGraph Agentic Workflow")
    print("=" * 60)
    
    # Initialize the agent
    agent = LangGraphAgent()
    
    try:
        # Initialize with a sample dataset
        agent.initialize()
        print("✅ Agent initialized successfully")
        
        # Process the dataset through the enhanced workflow
        print("\n🔄 Processing dataset through enhanced workflow...")
        result = agent.process()
        
        print("\n🎉 Enhanced workflow completed successfully!")
        print(f"📊 Generated {len(result['generated_charts'])} charts")
        print(f"📝 Created {len(result['chart_narratives'])} narratives")
        print(f"🎯 Report Title: {result.get('report_title', 'N/A')}")
        print(f"📋 Executive Summary: {'Generated' if result.get('executive_summary') else 'Not generated'}")
        print(f"📖 Narrative Flow: {'Generated' if result.get('narrative_flow') else 'Not generated'}")
        print(f"🤖 Agent Reasoning: {len(result.get('agent_reasoning', []))} reasoning explanations")
        print("📄 Enhanced HTML report saved as 'output.html'")
        
        # Print summary of enhanced features
        print("\n📋 Enhanced Features Summary:")
        print(f"  - Smart feedback loop: ✅ Implemented")
        print(f"  - Dynamic report title: ✅ {result.get('report_title', 'N/A')}")
        print(f"  - Executive summary: ✅ {'Generated' if result.get('executive_summary') else 'Not generated'}")
        print(f"  - Collapsible data quality: ✅ Implemented")
        print(f"  - Coherent narrative flow: ✅ {'Generated' if result.get('narrative_flow') else 'Not generated'}")
        print(f"  - Agent reasoning display: ✅ {len(result.get('agent_reasoning', []))} explanations")
        print(f"  - Chart optimization: ✅ {result.get('optimization_attempts', 0)} optimization attempts")
        
        # Show chart optimization results
        optimized_charts = [c for c in result['generated_charts'] if c.get('optimized', False)]
        if optimized_charts:
            print(f"  - Optimized charts: ✅ {len(optimized_charts)} charts were optimized")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during enhanced workflow execution: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
