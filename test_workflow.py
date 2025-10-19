#!/usr/bin/env python3
"""
Test script for the LangGraph Agentic Workflow
This script provides a simple way to test the workflow with sample data.
"""

import pandas as pd
import numpy as np
from langgraph_agent import LangGraphAgent

def create_sample_dataset():
    """Create a sample dataset for testing"""
    np.random.seed(42)
    
    # Create sample data
    data = {
        'category': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'value': np.random.normal(100, 15, 100),
        'score': np.random.uniform(0, 100, 100),
        'date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'revenue': np.random.exponential(1000, 100)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values for testing
    df.loc[10:15, 'value'] = np.nan
    df.loc[20:25, 'score'] = np.nan
    
    return df

def test_workflow():
    """Test the LangGraph workflow with sample data"""
    
    print("🧪 Testing LangGraph Agentic Workflow")
    print("=" * 50)
    
    # Create sample dataset
    print("📊 Creating sample dataset...")
    sample_df = create_sample_dataset()
    print(f"✅ Created dataset with shape: {sample_df.shape}")
    print(f"   Columns: {list(sample_df.columns)}")
    
    # Save sample dataset
    sample_df.to_csv("test_dataset.csv", index=False)
    print("💾 Saved sample dataset as 'test_dataset.csv'")
    
    # Initialize agent with sample dataset
    print("\n🤖 Initializing LangGraph agent...")
    agent = LangGraphAgent()
    agent.initialize(dataset_path="test_dataset.csv")
    print("✅ Agent initialized successfully")
    
    # Process the workflow
    print("\n🔄 Processing workflow...")
    try:
        result = agent.process()
        
        print("\n🎉 Test completed successfully!")
        print(f"📊 Generated {len(result['generated_charts'])} charts")
        print(f"📝 Created {len(result['chart_narratives'])} narratives")
        print("📄 HTML report saved as 'output.html'")
        
        # Print detailed results
        print("\n📋 Detailed Results:")
        for i, chart in enumerate(result['generated_charts']):
            status = "✅" if chart['generation_status'] == 'success' else "❌"
            print(f"  Chart {i+1}: {status} {chart['title']}")
        
        print(f"\n📈 Data Profile Summary:")
        profile = result['data_profile']
        print(f"  - Categorical fields: {len(profile['field_classification']['categorical'])}")
        print(f"  - Numerical fields: {len(profile['field_classification']['numerical'])}")
        print(f"  - Missing values: {sum(profile['data_quality']['missing_values'].values())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_workflow()
    if success:
        print("\n🎯 Test completed successfully! Check 'output.html' for results.")
    else:
        print("\n💥 Test failed. Check the error messages above.")
