#!/usr/bin/env python3
"""
Examples of different ways to specify input datasets for the LangGraph workflow
"""

from langgraph_agent import LangGraphAgent

def example_1_default_dataset():
    """Example 1: Use the default dataset (dataset.csv)"""
    print("📊 Example 1: Using default dataset")
    
    agent = LangGraphAgent()
    agent.initialize()  # Uses default "dataset.csv"
    result = agent.process()
    return result

def example_2_custom_filename():
    """Example 2: Specify just the filename (searches in current and parent directories)"""
    print("📊 Example 2: Using custom filename")
    
    agent = LangGraphAgent()
    agent.initialize(dataset_filename="my_custom_data.csv")
    result = agent.process()
    return result

def example_3_full_path():
    """Example 3: Specify full path to dataset"""
    print("📊 Example 3: Using full path")
    
    agent = LangGraphAgent()
    agent.initialize(dataset_path="/Users/username/Documents/my_data.csv")
    result = agent.process()
    return result

def example_4_url():
    """Example 4: Load dataset from URL"""
    print("📊 Example 4: Loading from URL")
    
    agent = LangGraphAgent()
    agent.initialize(dataset_url="https://raw.githubusercontent.com/user/repo/main/data.csv")
    result = agent.process()
    return result

def example_5_custom_default():
    """Example 5: Change default filename when creating agent"""
    print("📊 Example 5: Custom default filename")
    
    agent = LangGraphAgent(default_dataset_filename="my_default_data.csv")
    agent.initialize()  # Will use "my_default_data.csv" as default
    result = agent.process()
    return result

def example_6_relative_path():
    """Example 6: Using relative path"""
    print("📊 Example 6: Using relative path")
    
    agent = LangGraphAgent()
    agent.initialize(dataset_path="./data/my_dataset.csv")
    result = agent.process()
    return result

def example_7_parent_directory():
    """Example 7: File in parent directory"""
    print("📊 Example 7: File in parent directory")
    
    agent = LangGraphAgent()
    agent.initialize(dataset_path="../data.csv")
    result = agent.process()
    return result

def main():
    """Run all examples (uncomment the one you want to test)"""
    
    print("🚀 LangGraph Dataset Configuration Examples")
    print("=" * 60)
    
    # Uncomment the example you want to run:
    
    # example_1_default_dataset()
    # example_2_custom_filename()
    # example_3_full_path()
    # example_4_url()
    # example_5_custom_default()
    # example_6_relative_path()
    # example_7_parent_directory()
    
    print("\n💡 To use any of these examples:")
    print("1. Uncomment the example function you want to run")
    print("2. Make sure your dataset file exists in the specified location")
    print("3. Run: python dataset_config_examples.py")
    
    print("\n📋 Available options for dataset input:")
    print("   • dataset_filename: Just filename (searches multiple locations)")
    print("   • dataset_path: Full or relative path to file")
    print("   • dataset_url: URL to CSV file")
    print("   • default_dataset_filename: Change default in constructor")

if __name__ == "__main__":
    main()
