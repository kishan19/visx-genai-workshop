# ---------- Imports ----------
from __future__ import annotations
import os
import json
import math
import uuid
import operator
from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional, Dict, Any, Union
import ast
import random
import textwrap

import pandas as pd
import numpy as np
import altair as alt
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

# ---------- LLM Bootstrap ----------
llm = None
try:
    from helpers import get_llm  # provided by user
    llm = get_llm()
    print("LLM successfully loaded..")
except Exception as e:
    print(f"Error in loading LLM: {e}")
    print("Enhanced features requiring LLM will be disabled.")

# ---------- State Definition ----------
class AgentState(TypedDict):
    """State for the LangGraph workflow"""
    # Input data
    dataset: pd.DataFrame
    dataset_info: str
    dataset_sample: str
    
    # Data profiling results
    data_profile: Dict[str, Any]
    data_quality_report: str
    
    # Chart recommendations
    chart_recommendations: List[Dict[str, Any]]
    
    # Chart generation
    generated_charts: List[Dict[str, Any]]
    chart_evaluation_results: List[Dict[str, Any]]
    
    # Narrative and final output
    chart_narratives: List[Dict[str, Any]]
    final_html_report: str
    
    # Control flow
    current_iteration: int
    max_iterations: int
    should_continue: bool

# ---------- Data Loading Helpers ----------
def load_dataframe(local_path: str, filename: str, url: Optional[str] = None) -> pd.DataFrame:
    """Load a pandas DataFrame from either a URL or local file."""
    if url:
        return pd.read_csv(url)
    file_path = os.path.join(local_path, filename)
    ##take max 5000 rows for altair compatibility
    return pd.read_csv(file_path)

# ---------- Agent 1: Data Profiling Agent ----------
def data_profiling_agent(state: AgentState) -> AgentState:
    """
    Creates a comprehensive data quality report from the CSV, identifying fields 
    good for visualization tasks such as categorical, numerical, time-series and text data.
    """
    df = state["dataset"]
    
    # Sample data for analysis
    sample_data = df.head(10).to_dict(orient="records")
    
    data_profiling_prompt = textwrap.dedent(f"""
    You are a Data Profiling Agent. Analyze the following dataset sample and create a comprehensive data quality report.

    Dataset Sample (first 10 rows):
    {sample_data}

    Dataset Shape: {df.shape}
    Columns: {list(df.columns)}

    Please provide a detailed analysis including:

    1. **Data Quality Assessment:**
       - Missing values per column
       - Data types and their appropriateness
       - Duplicate rows
       - Outliers detection

    2. **Field Classification:**
       - Categorical fields (with unique value counts)
       - Numerical fields (with statistical summary)
       - Time-series fields (if any)
       - Text fields (with length analysis)

    3. **Visualization Readiness:**
       - Which fields are suitable for different chart types
       - Relationships between fields
       - Data distribution insights

    4. **Data Story Potential:**
       - Key patterns or trends visible
       - Interesting correlations
       - Business insights potential

    Return your analysis as a structured JSON with the following format:
    {{
        "data_quality": {{
            "missing_values": {{"column_name": count}},
            "data_types": {{"column_name": "type"}},
            "duplicates": count,
            "outliers": ["column_name1", "column_name2"]
        }},
        "field_classification": {{
            "categorical": ["field1", "field2"],
            "numerical": ["field1", "field2"],
            "time_series": ["field1"],
            "text": ["field1"]
        }},
        "visualization_readiness": {{
            "suitable_for_bar": ["field1", "field2"],
            "suitable_for_line": ["field1"],
            "suitable_for_scatter": ["field1", "field2"],
            "suitable_for_histogram": ["field1"]
        }},
        "data_story_potential": {{
            "key_patterns": ["pattern1", "pattern2"],
            "correlations": ["field1 vs field2"],
            "insights": ["insight1", "insight2"]
        }}
    }}
    """)

    if llm:
        response = llm.invoke([
            SystemMessage(content=data_profiling_prompt),
            HumanMessage(content="Analyze the dataset and provide the data profiling report.")
        ])
        
        try:
            data_profile = json.loads(response.content)
        except:
            # Fallback if JSON parsing fails
            data_profile = {
                "data_quality": {"missing_values": {}, "data_types": {}, "duplicates": 0, "outliers": []},
                "field_classification": {"categorical": [], "numerical": [], "time_series": [], "text": []},
                "visualization_readiness": {"suitable_for_bar": [], "suitable_for_line": [], "suitable_for_scatter": [], "suitable_for_histogram": []},
                "data_story_potential": {"key_patterns": [], "correlations": [], "insights": []}
            }
    else:
        # Fallback analysis without LLM
        data_profile = {
            "data_quality": {
                "missing_values": df.isnull().sum().to_dict(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "duplicates": df.duplicated().sum(),
                "outliers": []
            },
            "field_classification": {
                "categorical": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                "numerical": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                "time_series": [],
                "text": []
            },
            "visualization_readiness": {
                "suitable_for_bar": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                "suitable_for_line": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                "suitable_for_scatter": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
                "suitable_for_histogram": df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            },
            "data_story_potential": {
                "key_patterns": ["Data distribution patterns"],
                "correlations": ["Numerical field relationships"],
                "insights": ["Statistical insights from the dataset"]
            }
        }

    return {
        **state,
        "data_profile": data_profile,
        "data_quality_report": json.dumps(data_profile, indent=2)
    }

# ---------- Agent 2: Visualization Strategist Agent ----------
def visualization_strategist_agent(state: AgentState) -> AgentState:
    """
    Creates a list of top 5-6 charts that can be plotted using Altair library, 
    in order to create a compelling data led story narrative.
    """
    df = state["dataset"]
    data_profile = state["data_profile"]
    
    visualization_prompt = textwrap.dedent(f"""
    You are a Visualization Strategist Agent. Based on the data profiling results, recommend the top 5-6 most compelling visualizations using Altair library.

    Data Profile Summary:
    {json.dumps(data_profile, indent=2)}

    Dataset Columns: {list(df.columns)}
    Dataset Shape: {df.shape}

    Consider the following for each recommendation:

    1. **Chart Type**: Choose from: bar, line, scatter, histogram, boxplot, heatmap, area, pie
    2. **Fields Used**: Specify which columns to use (x, y, color, size, etc.)
    3. **Purpose**: What insight or story this chart reveals
    4. **Priority**: Rank from 1-6 (1 being highest priority)
    5. **Altair Specification**: Provide the Altair chart specification

    Return a JSON list of chart recommendations:
    [
        {{
            "chart_id": "chart_1",
            "chart_type": "bar",
            "title": "Chart Title",
            "fields_used": {{
                "x": "column1",
                "y": "column2",
                "color": "column3"
            }},
            "purpose": "This chart reveals...",
            "priority": 1,
            "altair_spec": "alt.Chart(df).mark_bar().encode(x='column1', y='column2', color='column3')",
            "insights_expected": ["insight1", "insight2"]
        }},
        ...
    ]

    Focus on creating a compelling narrative flow where each chart builds upon the previous one to tell a complete data story.
    """)

    if llm:
        response = llm.invoke([
            SystemMessage(content=visualization_prompt),
            HumanMessage(content="Generate chart recommendations for the dataset.")
        ])
        
        try:
            chart_recommendations = json.loads(response.content)
        except:
            # Fallback recommendations
            chart_recommendations = generate_fallback_recommendations(df)
    else:
        chart_recommendations = generate_fallback_recommendations(df)

    return {
        **state,
        "chart_recommendations": chart_recommendations
    }

def generate_fallback_recommendations(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generate fallback chart recommendations when LLM is not available"""
    recommendations = []
    
    # Get column types
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    chart_id = 1
    
    # Bar chart for categorical data
    if categorical_cols and numerical_cols:
        recommendations.append({
            "chart_id": f"chart_{chart_id}",
            "chart_type": "bar",
            "title": f"Distribution of {categorical_cols[0]}",
            "fields_used": {
                "x": categorical_cols[0],
                "y": numerical_cols[0]
            },
            "purpose": f"Shows the distribution of {numerical_cols[0]} across different {categorical_cols[0]} categories",
            "priority": chart_id,
            "altair_spec": f"alt.Chart(df).mark_bar().encode(x='{categorical_cols[0]}', y='{numerical_cols[0]}')",
            "insights_expected": ["Category distribution", "Value comparisons"]
        })
        chart_id += 1
    
    # Histogram for numerical data
    if numerical_cols:
        recommendations.append({
            "chart_id": f"chart_{chart_id}",
            "chart_type": "histogram",
            "title": f"Distribution of {numerical_cols[0]}",
            "fields_used": {
                "x": numerical_cols[0]
            },
            "purpose": f"Shows the distribution pattern of {numerical_cols[0]}",
            "priority": chart_id,
            "altair_spec": f"alt.Chart(df).mark_bar().encode(alt.X('{numerical_cols[0]}', bin=alt.Bin(maxbins=20)), y='count()')",
            "insights_expected": ["Data distribution", "Central tendency"]
        })
        chart_id += 1
    
    # Scatter plot if we have two numerical columns
    if len(numerical_cols) >= 2:
        recommendations.append({
            "chart_id": f"chart_{chart_id}",
            "chart_type": "scatter",
            "title": f"Relationship between {numerical_cols[0]} and {numerical_cols[1]}",
            "fields_used": {
                "x": numerical_cols[0],
                "y": numerical_cols[1]
            },
            "purpose": f"Explores the relationship between {numerical_cols[0]} and {numerical_cols[1]}",
            "priority": chart_id,
            "altair_spec": f"alt.Chart(df).mark_circle().encode(x='{numerical_cols[0]}', y='{numerical_cols[1]}')",
            "insights_expected": ["Correlation analysis", "Outlier detection"]
        })
        chart_id += 1
    
    return recommendations[:6]  # Limit to 6 charts

# ---------- Agent 3: Altair Chart Generator Agent ----------
def altair_chart_generator_agent(state: AgentState) -> AgentState:
    """
    Generates charts using Altair code based on the recommendations.
    """
    df = state["dataset"]
    chart_recommendations = state["chart_recommendations"]
    
    generated_charts = []
    
    for rec in chart_recommendations:
        try:
            # Generate the Altair chart
            chart_spec = rec["altair_spec"]
            
            # Execute the Altair specification
            chart = eval(chart_spec)
            
            # Add title and styling
            chart = chart.properties(
                title=rec["title"],
                width=600,
                height=400
            ).resolve_scale(
                color='independent'
            )
            
            # Convert to JSON for storage
            chart_json = chart.to_json()
            
            generated_charts.append({
                "chart_id": rec["chart_id"],
                "chart_type": rec["chart_type"],
                "title": rec["title"],
                "altair_spec": chart_spec,
                "chart_json": chart_json,
                "fields_used": rec["fields_used"],
                "purpose": rec["purpose"],
                "generation_status": "success",
                "error_message": None
            })
            
        except Exception as e:
            generated_charts.append({
                "chart_id": rec["chart_id"],
                "chart_type": rec["chart_type"],
                "title": rec["title"],
                "altair_spec": rec["altair_spec"],
                "chart_json": None,
                "fields_used": rec["fields_used"],
                "purpose": rec["purpose"],
                "generation_status": "failed",
                "error_message": str(e)
            })
    
    return {
        **state,
        "generated_charts": generated_charts
    }

# ---------- Agent 4: Chart Evaluator Agent ----------
def chart_evaluator_agent(state: AgentState) -> AgentState:
    """
    Evaluates charts generated for quality and completeness.
    Provides feedback for improvement if needed.
    """
    generated_charts = state["generated_charts"]
    df = state["dataset"]
    
    evaluation_results = []
    
    for chart in generated_charts:
        evaluation = {
            "chart_id": chart["chart_id"],
            "quality_score": 0,
            "completeness_score": 0,
            "overall_score": 0,
            "feedback": [],
            "recommendations": [],
            "approved": False
        }
        
        # Evaluate based on generation status
        if chart["generation_status"] == "success":
            evaluation["quality_score"] = 8
            evaluation["completeness_score"] = 9
            evaluation["approved"] = True
            evaluation["feedback"].append("Chart generated successfully")
        else:
            evaluation["quality_score"] = 2
            evaluation["completeness_score"] = 1
            evaluation["feedback"].append(f"Chart generation failed: {chart['error_message']}")
            evaluation["recommendations"].append("Fix the Altair specification or data field issues")
        
        # Additional quality checks
        if chart["generation_status"] == "success":
            # Check if chart has meaningful data
            fields_used = chart["fields_used"]
            for field, column in fields_used.items():
                if column in df.columns:
                    if df[column].isnull().all():
                        evaluation["quality_score"] -= 2
                        evaluation["feedback"].append(f"Field {column} has all null values")
                    elif df[column].nunique() == 1:
                        evaluation["quality_score"] -= 1
                        evaluation["feedback"].append(f"Field {column} has only one unique value")
                else:
                    evaluation["quality_score"] -= 3
                    evaluation["feedback"].append(f"Field {column} not found in dataset")
                    evaluation["approved"] = False
        
        evaluation["overall_score"] = (evaluation["quality_score"] + evaluation["completeness_score"]) / 2
        evaluation_results.append(evaluation)
    
    # Determine if we should continue (for feedback loop)
    failed_charts = [e for e in evaluation_results if not e["approved"]]
    should_continue = len(failed_charts) > 0 and state["current_iteration"] < state["max_iterations"]
    
    return {
        **state,
        "chart_evaluation_results": evaluation_results,
        "should_continue": should_continue,
        "current_iteration": state["current_iteration"] + 1
    }

# ---------- Agent 5: Narrative Writer Agent ----------
def narrative_writer_agent(state: AgentState) -> AgentState:
    """
    Writes a narrative on each chart generated, with key insights.
    """
    generated_charts = state["generated_charts"]
    chart_evaluation_results = state["chart_evaluation_results"]
    data_profile = state["data_profile"]
    
    chart_narratives = []
    
    for i, chart in enumerate(generated_charts):
        evaluation = chart_evaluation_results[i] if i < len(chart_evaluation_results) else {}
        
        if chart["generation_status"] == "success" and evaluation.get("approved", False):
            narrative_prompt = textwrap.dedent(f"""
            You are a Narrative Writer Agent. Write a compelling narrative for the following chart:

            Chart Details:
            - Title: {chart['title']}
            - Type: {chart['chart_type']}
            - Purpose: {chart['purpose']}
            - Fields Used: {chart['fields_used']}

            Data Context:
            {json.dumps(data_profile, indent=2)}

            Write a narrative that includes:
            1. **Chart Overview**: Brief description of what the chart shows
            2. **Key Insights**: 2-3 main insights from the data
            3. **Data Story**: How this chart contributes to the overall data story
            4. **Business Implications**: What this means for decision-making
            5. **Recommendations**: Actionable recommendations based on the insights

            Keep the narrative engaging, data-driven, and professional.
            """)

            if llm:
                response = llm.invoke([
                    SystemMessage(content=narrative_prompt),
                    HumanMessage(content="Write a narrative for this chart.")
                ])
                narrative = response.content
            else:
                # Fallback narrative
                narrative = f"""
                ## {chart['title']}
                
                This {chart['chart_type']} chart visualizes the relationship between {', '.join(chart['fields_used'].values())}.
                
                **Key Insights:**
                - The data shows interesting patterns in the {chart['chart_type']} visualization
                - There are notable trends that can inform decision-making
                - The visualization reveals important relationships in the dataset
                
                **Business Implications:**
                This chart provides valuable insights for understanding the data patterns and can guide strategic decisions.
                
                **Recommendations:**
                - Monitor the trends shown in this visualization
                - Consider the patterns when making data-driven decisions
                - Use this insight to inform future analysis
                """
        else:
            narrative = f"""
            ## {chart['title']} - Chart Generation Failed
            
            This chart could not be generated due to: {chart.get('error_message', 'Unknown error')}
            
            **Recommendation:** Please review the data fields and chart specification to resolve the issue.
            """
        
        chart_narratives.append({
            "chart_id": chart["chart_id"],
            "narrative": narrative,
            "chart_title": chart["title"],
            "chart_type": chart["chart_type"]
        })
    
    return {
        **state,
        "chart_narratives": chart_narratives
    }

# ---------- Final HTML Report Generator ----------
def generate_final_html_report(state: AgentState) -> AgentState:
    """
    Generates the final HTML report combining all insights from the dataset.
    """
    chart_narratives = state["chart_narratives"]
    generated_charts = state["generated_charts"]
    data_profile = state["data_profile"]
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Data-Driven Report</title>
        <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                border-left: 4px solid #3498db;
                padding-left: 15px;
            }}
            h3 {{
                color: #2c3e50;
            }}
            .chart-container {{
                margin: 30px 0;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 8px;
                background-color: #fafafa;
            }}
            .narrative {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            .data-profile {{
                background-color: #e8f4f8;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            .insight-box {{
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
            }}
            .error-box {{
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                padding: 15px;
                border-radius: 5px;
                margin: 10px 0;
                color: #721c24;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Data-Driven Analysis Report</h1>
            
            <div class="data-profile">
                <h2>📈 Dataset Overview</h2>
                <p><strong>Dataset Shape:</strong> {state['dataset'].shape[0]} rows × {state['dataset'].shape[1]} columns</p>
                <p><strong>Columns:</strong> {', '.join(state['dataset'].columns)}</p>
                <h3>Data Quality Summary:</h3>
                <ul>
                    <li><strong>Missing Values:</strong> {sum(state['dataset'].isnull().sum())} total missing values</li>
                    <li><strong>Duplicate Rows:</strong> {state['dataset'].duplicated().sum()} duplicates</li>
                    <li><strong>Data Types:</strong> {dict(state['dataset'].dtypes.astype(str))}</li>
                </ul>
            </div>
    """
    
    # Add each chart and narrative
    for i, (chart, narrative) in enumerate(zip(generated_charts, chart_narratives)):
        html_content += f"""
            <div class="chart-container">
                <h2>📊 {chart['title']}</h2>
        """
        
        if chart["generation_status"] == "success":
            html_content += f"""
                <div id="chart_{i}"></div>
                <script>
                    vegaEmbed('#chart_{i}', {chart['chart_json']});
                </script>
            """
        else:
            html_content += f"""
                <div class="error-box">
                    <strong>Chart Generation Failed:</strong> {chart.get('error_message', 'Unknown error')}
                </div>
            """
        
        html_content += f"""
                <div class="narrative">
                    {narrative['narrative'].replace('\\n', '<br>')}
                </div>
            </div>
        """
    
    # Add summary and conclusions
    html_content += f"""
            <div class="insight-box">
                <h2>🎯 Executive Summary</h2>
                <p>This report analyzed {len(generated_charts)} visualizations generated from the dataset. 
                The analysis reveals key patterns and insights that can inform data-driven decision making.</p>
                
                <h3>Key Findings:</h3>
                <ul>
                    <li>Generated {len([c for c in generated_charts if c['generation_status'] == 'success'])} successful visualizations</li>
                    <li>Identified {len([c for c in generated_charts if c['generation_status'] == 'failed'])} charts that need refinement</li>
                    <li>Provided comprehensive narratives for each successful visualization</li>
                </ul>
                
                <h3>Recommendations:</h3>
                <ul>
                    <li>Review failed chart generations and refine data field selections</li>
                    <li>Use successful visualizations to guide business decisions</li>
                    <li>Consider additional data collection for enhanced insights</li>
                </ul>
            </div>
            
            <div style="text-align: center; margin-top: 40px; color: #7f8c8d;">
                <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Generated by LangGraph Agentic Workflow</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML file
    with open("output.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return {
        **state,
        "final_html_report": html_content
    }

# ---------- Conditional Edge Functions ----------
def should_continue_evaluation(state: AgentState) -> str:
    """Determines whether to continue with chart evaluation feedback loop"""
    if state["should_continue"]:
        return "chart_generator"
    else:
        return "narrative_writer"

# ---------- Main Workflow Creation ----------
def create_langgraph_workflow() -> StateGraph:
    """Creates the complete LangGraph workflow with all agents"""
    
    # Create the workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes for each agent
    workflow.add_node("data_profiling", data_profiling_agent)
    workflow.add_node("visualization_strategist", visualization_strategist_agent)
    workflow.add_node("chart_generator", altair_chart_generator_agent)
    workflow.add_node("chart_evaluator", chart_evaluator_agent)
    workflow.add_node("narrative_writer", narrative_writer_agent)
    workflow.add_node("html_generator", generate_final_html_report)
    
    # Define the workflow edges
    workflow.add_edge(START, "data_profiling")
    workflow.add_edge("data_profiling", "visualization_strategist")
    workflow.add_edge("visualization_strategist", "chart_generator")
    workflow.add_edge("chart_generator", "chart_evaluator")
    
    # Conditional edge for feedback loop
    workflow.add_conditional_edges(
        "chart_evaluator",
        should_continue_evaluation,
        {
            "chart_generator": "chart_generator",
            "narrative_writer": "narrative_writer"
        }
    )
    
    workflow.add_edge("narrative_writer", "html_generator")
    workflow.add_edge("html_generator", END)
    
    return workflow.compile()

# ---------- Main Agent Class ----------
class LangGraphAgent:
    """Main agent class that orchestrates the entire workflow"""
    
    def __init__(self, default_dataset_filename: str = "dataset.csv"):
        """
        Initialize the LangGraph agent
        
        Args:
            default_dataset_filename: Default filename to use when no specific path is provided
        """
        self.workflow = None
        self.df = None
        self.default_dataset_filename = default_dataset_filename
    
    def initialize(self, dataset_path: str = None, dataset_url: str = None, dataset_filename: str = None):
        """
        Initialize the agent with a dataset
        
        Args:
            dataset_path: Full path to the dataset file (e.g., "/path/to/data.csv")
            dataset_url: URL to the dataset (e.g., "https://example.com/data.csv")
            dataset_filename: Just the filename (will look in current directory and parent directories)
        """
        self.workflow = create_langgraph_workflow()
        
        # Load dataset with multiple options
        if dataset_url:
            print(f"Loading dataset from URL: {dataset_url}")
            self.df = pd.read_csv(dataset_url)
        elif dataset_path:
            print(f"Loading dataset from path: {dataset_path}")
            self.df = pd.read_csv(dataset_path)
        elif dataset_filename:
            print(f"Loading dataset with filename: {dataset_filename}")
            # Try multiple locations for the file
            possible_paths = [
                dataset_filename,  # Current directory
                f"../{dataset_filename}",  # Parent directory
                f"../../{dataset_filename}",  # Grandparent directory
                f"./{dataset_filename}",  # Explicit current directory
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.df = pd.read_csv(path)
                    print(f"Found dataset at: {path}")
                    break
            else:
                raise FileNotFoundError(f"Dataset file '{dataset_filename}' not found in any of the searched locations: {possible_paths}")
        else:
            # Use default dataset filename
            print(f"Using default dataset filename: {self.default_dataset_filename}")
            script_path = "/".join(os.path.abspath(__file__).split("/")[:-2])
            self.df = load_dataframe(script_path, self.default_dataset_filename)
        
        print(f"Dataset loaded: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
    
    def process(self) -> Dict[str, Any]:
        """Process the dataset through the complete workflow"""
        if self.workflow is None:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        
        # Prepare initial state
        sample_data = self.df.head(10).to_dict(orient="records")
        
        initial_state = {
            "dataset": self.df,
            "dataset_info": f"Dataset with {self.df.shape[0]} rows and {self.df.shape[1]} columns",
            "dataset_sample": json.dumps(sample_data, indent=2),
            "data_profile": {},
            "data_quality_report": "",
            "chart_recommendations": [],
            "generated_charts": [],
            "chart_evaluation_results": [],
            "chart_narratives": [],
            "final_html_report": "",
            "current_iteration": 0,
            "max_iterations": 3,
            "should_continue": False
        }
        
        # Execute the workflow
        print("Starting LangGraph workflow execution...")
        final_state = self.workflow.invoke(initial_state)
        
        print("✅ Workflow completed successfully!")
        print(f"📊 Generated {len(final_state['generated_charts'])} charts")
        print(f"📝 Created {len(final_state['chart_narratives'])} narratives")
        print("📄 HTML report saved as 'output.html'")
        
        return final_state

# ---------- Main Execution ----------
if __name__ == "__main__":
    # Create and run the agent
    agent = LangGraphAgent()
    #agent.initialize(dataset_path="reviews_tests.csv")
    # agent.initialize(dataset_path="survey.csv")
    ##For challenge dataset
    agent.initialize()
    result = agent.process()
    
    print("\n🎉 LangGraph Agentic Workflow completed!")
    print("Check 'output.html' for the final report.")
