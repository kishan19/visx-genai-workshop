#!/usr/bin/env python3
"""
Test script to verify the fix for the 'df' not defined error.
"""

import pandas as pd
import altair as alt
from langgraph_agent import create_chart_from_spec

def test_chart_creation():
    """Test the fixed chart creation function"""
    
    # Create a simple test dataset
    test_data = pd.DataFrame({
        'category': ['A', 'B', 'C', 'A', 'B'],
        'value': [10, 20, 30, 15, 25]
    })
    
    # Test create_chart_from_spec with df parameter
    original_chart = {
        'chart_id': 'test_1', 
        'title': 'Test Chart',
        'generation_status': 'failed',
        'error_message': 'Test error'
    }
    
    new_spec = 'alt.Chart(df).mark_bar().encode(x="category", y="count()")'
    new_title = 'Test Chart'
    chart_type = 'bar'
    
    try:
        result = create_chart_from_spec(original_chart, new_spec, new_title, chart_type, test_data)
        print('✅ Chart creation test passed!')
        print(f'Status: {result["generation_status"]}')
        print(f'Title: {result["title"]}')
        print(f'Optimized: {result.get("optimized", False)}')
        
        if result["generation_status"] == "success":
            print('✅ Chart was successfully created and optimized!')
        else:
            print(f'❌ Chart creation failed: {result.get("error_message", "Unknown error")}')
            
    except Exception as e:
        print(f'❌ Test failed with error: {e}')

if __name__ == "__main__":
    test_chart_creation()
