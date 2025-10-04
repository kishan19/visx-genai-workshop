# ---------- Imports ----------
from __future__ import annotations
import os
import json
import math
import uuid
import operator
from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional, Dict, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.utils import PlotlyJSONEncoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict
import numpy as np

# Try to import Altair as fallback
try:
    import altair as alt
    ALTAIR_AVAILABLE = True
    print("Altair library available for fallback chart generation")
except ImportError:
    ALTAIR_AVAILABLE = False
    print("Altair library not available, using Plotly only")

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
    # Fallback to OpenAI (or any LC chat model you prefer)
    # Note: keep this minimal to avoid import errors if OpenAI isn't configured
    print(f"Error in loading LLM: {e}")
    print("Enhanced features requiring LLM will be disabled.")

# ---------- Data Loading Helpers ----------
script_path = "/".join(os.path.abspath(__file__).split("/")[:-2])
print("Script being run at..", script_path)


def load_dataframe(local_path: str, filename: str, url: Optional[str] = None) -> pd.DataFrame:
    """Load a pandas DataFrame from either a URL or local file."""
    if url:
        return pd.read_csv(url)
    file_path = os.path.join(local_path, filename)
    return pd.read_csv(file_path)


# ---------- Dataset Configuration ----------
# User-configurable dataset source
file_url = "https://raw.githubusercontent.com/demoPlz/mini-template/main/studio/dataset.csv"

# For challenge main file
file_name = "dataset.csv"
df = load_dataframe(local_path=script_path, filename=file_name, url=file_url)

# # For another file on another domain
# file_name = "reviews_tests.csv"
# df = load_dataframe(local_path=script_path, filename=file_name)

print("Dataframe loaded...", df.shape)


# ---------- State and Models ----------
class DataQualityState(TypedDict):
    """State for the data quality analysis workflow."""
    dataset_info: Dict[str, Any]
    data_quality_report: Optional[Dict[str, Any]]
    chart_recommendations: Optional[Dict[str, Any]]
    final_report: Optional[Dict[str, Any]]
    generated_charts: Optional[Dict[str, Any]]
    narrative_insights: Optional[Dict[str, Any]]
    dynamic_title: Optional[str]
    executive_summary: Optional[str]
    data_quality_table: Optional[Dict[str, Any]]
    enhanced_chart_narrative: Optional[str]
    conclusion: Optional[str]
    final_html_path: Optional[str]


class DataQualityReport(BaseModel):
    """Schema for the data quality report."""
    dataset_overview: Dict[str, Any] = Field(description="Basic dataset information")
    column_analysis: Dict[str, Dict[str, Any]] = Field(description="Analysis of each column")
    data_quality_issues: List[str] = Field(description="List of identified data quality issues")
    missing_data_summary: Dict[str, Any] = Field(description="Summary of missing data")
    data_types_summary: Dict[str, Any] = Field(description="Summary of data types")


class ChartRecommendation(BaseModel):
    """Schema for individual chart recommendation."""
    rank: int = Field(description="Priority rank of the chart")
    chart_type: str = Field(description="Specific chart type name")
    chart_name: str = Field(description="Descriptive name for the chart")
    fields_required: List[str] = Field(description="Fields needed for this chart")
    data_preprocessing: str = Field(description="Required data transformations")
    why_this_chart_helps: str = Field(description="Combined purpose and key insights explanation")
    storytelling_impact: str = Field(description="Why this chart is compelling")
    data_quality_notes: str = Field(description="Quality considerations and mitigations")
    priority_reason: str = Field(description="Why this chart is ranked at this position")


class ChartRecommendations(BaseModel):
    """Schema for comprehensive chart recommendations."""
    chart_recommendations: List[ChartRecommendation] = Field(description="List of recommended charts")
    narrative_flow: str = Field(description="How charts work together to tell a complete story")
    data_quality_summary: str = Field(description="Overall assessment of data quality for visualization")
    implementation_notes: str = Field(description="Technical considerations for creating these charts")


# ---------- Chart Recommendation Prompt ----------
CHART_RECOMMENDATION_PROMPT = """You are an expert data visualization consultant specializing in creating compelling data-driven narratives. Your task is to analyze a comprehensive data quality report and recommend the top 5-6 most impactful chart types for data storytelling.

## Input: Data Quality Report
The data quality report contains the following key information:
- **Dataset Overview**: Total rows, columns, memory usage, missing data percentage
- **Column Analysis**: For each column - data type, missing values, unique values, sample values, quality score, potential issues
- **Data Types Summary**: Categorized columns (numeric, categorical, datetime, other)
- **Missing Data Summary**: Columns with missing values and patterns
- **Data Quality Issues**: Overall dataset problems and concerns
- **Value Ranges**: For numeric columns - min, max, mean, std, outliers
- **Category Cardinality**: For categorical columns - unique values, top values distribution
- **Type Consistency**: Data type validation results

## Your Task
Based on the data quality report, recommend **exactly 7-8 chart types** that would create the most compelling data storytelling narrative. For each chart recommendation, provide:

### 1. Chart Type & Name
- Specify the exact chart type (e.g., "Time Series Line Chart", "Correlation Heatmap", "Stacked Bar Chart")
- Give it a descriptive name that captures the story it tells

### 2. Fields Required
- List the specific columns/fields needed for this chart (maximum 3 fields)
- Use color/filter as the 3rd dimension for 2-D charts
- Indicate if any data preprocessing is required (e.g., aggregation, filtering, transformation)

### 3. Why do I think this chart helps?
- Explain the purpose of this chart AND the key insights it reveals
- Describe why this chart is compelling for the audience
- Combine both purpose and insights into a single comprehensive explanation

### 4. Data Quality Considerations
- Address any data quality issues that might affect this chart
- Suggest mitigation strategies if needed
- Note if the chart requires high-quality data or can work with some missing values

### 5. Chart Priority
- Rank the chart by storytelling impact (1 = highest impact)
- Explain why this chart should be prioritized

## Chart Selection Guidelines

### Prioritize Charts That:
1. **Tell a Complete Story**: Each chart should reveal a specific insight or trend that builds the narrative
2. **Leverage High-Quality Data**: Prefer fields with high quality scores and low missing values
3. **Show Relationships**: Focus on charts that reveal correlations, trends, or comparisons
4. **Support Narrative Flow**: Charts should build upon each other to create a cohesive story
5. **Engage the Audience**: Choose visually appealing and intuitive chart types

### Consider These Chart Categories:
- **Temporal Analysis**: Line charts, area charts, heatmaps for time-based trends
- **Distribution Analysis**: Histograms, box plots, violin plots for data distribution
- **Relationship Analysis**: Scatter plots, correlation matrices, bubble charts
- **Categorical Analysis**: Bar charts, pie charts, treemaps for category comparisons
- **Text Analysis**: Word clouds, sentiment analysis charts, topic modeling visualizations
- **Geographic Analysis**: Maps, choropleth charts (if location data available)
- **Hierarchical Analysis**: Treemaps, sunburst charts, dendrograms

### Data Quality Thresholds:
- **High Quality**: Quality score > 0.8, missing values < 5%
- **Medium Quality**: Quality score 0.5-0.8, missing values 5-20%
- **Low Quality**: Quality score < 0.5, missing values > 20% (avoid unless essential)

## Output Format

Provide your recommendations in this exact JSON format:

```json
{
  "chart_recommendations": [
    {
      "rank": 1,
      "chart_type": "Specific chart type name",
      "chart_name": "Descriptive name for the chart",
      "fields_required": ["field1", "field2", "field3"],
      "data_preprocessing": "Any required data transformations",
      "why_this_chart_helps": "Combined purpose and key insights explanation",
      "storytelling_impact": "Why this chart is compelling",
      "data_quality_notes": "Quality considerations and mitigations",
      "priority_reason": "Why this chart is ranked at this position"
    }
  ],
  "narrative_flow": "How these charts work together to tell a complete story",
  "data_quality_summary": "Overall assessment of data quality for visualization",
  "implementation_notes": "Technical considerations for creating these charts"
}
```

**Important**: 
- Ensure you recommend exactly 7-8 charts. Focus on comprehensive charts that help build a complete story end-to-end.
- **DO NOT include any meta-commentary, word counts, internal reasoning, or chain-of-thought text in your response.**
- **Provide ONLY the JSON output as specified above.**

## Remember
- Focus on **storytelling impact** over technical complexity
- Consider the **audience** and what insights would be most valuable to them
- Ensure charts **build upon each other** to create a complete narrative
- Address **data quality issues** proactively in your recommendations
- Prioritize **actionable insights** that drive decision-making

Now analyze the provided data quality report and provide your top 5-6 chart recommendations for compelling data storytelling."""


# ---------- Node Functions ----------
def analyze_dataset_and_quality(state: DataQualityState) -> DataQualityState:
    """Extract dataset information and generate comprehensive data quality report using Python logic."""
    
    # Basic dataset information
    dataset_info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "memory_usage": df.memory_usage(deep=True).sum(),
        "sample_data": df.head(3).to_dict('records')
    }
    
    # Calculate missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    # Initialize data quality report
    data_quality_report = {
        "dataset_overview": {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": round(dataset_info["memory_usage"] / (1024 * 1024), 2),
            "missing_values_total": int(missing_values.sum()),
            "missing_values_percentage": round((missing_values.sum() / (len(df) * len(df.columns))) * 100, 2)
        },
        "column_analysis": {},
        "data_quality_issues": [],
        "missing_data_summary": {
            "columns_with_missing": [],
            "missing_patterns": "Random missing values"
        },
        "data_types_summary": {
            "numeric_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "other_columns": []
        }
    }
    
    # Analyze each column
    for column in df.columns:
        col_data = df[column]
        col_dtype = str(col_data.dtype)
        
        # Basic column stats
        missing_count = int(col_data.isnull().sum())
        missing_pct = round((missing_count / len(df)) * 100, 2)
        unique_count = int(col_data.nunique())
        
        # Get sample values (non-null, unique, up to 5)
        sample_values = col_data.dropna().unique()[:5].tolist()
        
        # Determine potential issues
        potential_issues = []
        quality_score = 1.0
        
        # Check for missing data
        if missing_count > 0:
            potential_issues.append(f"Missing {missing_count} values ({missing_pct}%)")
            quality_score -= (missing_pct / 100) * 0.3
        
        # Check for low cardinality in categorical data
        if col_dtype == 'object' and unique_count < len(df) * 0.1:
            potential_issues.append(f"Low cardinality: {unique_count} unique values")
        
        # Check for high cardinality in categorical data
        if col_dtype == 'object' and unique_count > len(df) * 0.8:
            potential_issues.append(f"High cardinality: {unique_count} unique values")
        
        # Check for outliers in numeric data
        if pd.api.types.is_numeric_dtype(col_data):
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                potential_issues.append(f"Potential outliers: {outliers} values")
        
        # Check for data type inconsistencies
        if col_dtype == 'object':
            # Check if it might be numeric
            try:
                pd.to_numeric(col_data.dropna())
                potential_issues.append("Object type but contains numeric data")
            except:
                pass
        
        # Categorize columns by data type
        if pd.api.types.is_numeric_dtype(col_data):
            data_quality_report["data_types_summary"]["numeric_columns"].append(column)
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            data_quality_report["data_types_summary"]["datetime_columns"].append(column)
        elif col_dtype == 'object':
            data_quality_report["data_types_summary"]["categorical_columns"].append(column)
        else:
            data_quality_report["data_types_summary"]["other_columns"].append(column)
        
        # Add column analysis
        data_quality_report["column_analysis"][column] = {
            "data_type": col_dtype,
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "unique_values": unique_count,
            "sample_values": sample_values,
            "potential_issues": potential_issues,
            "quality_score": round(quality_score, 3)
        }
    
    # Identify columns with missing data
    columns_with_missing = missing_values[missing_values > 0].index.tolist()
    data_quality_report["missing_data_summary"]["columns_with_missing"] = columns_with_missing
    
    # Generate overall data quality issues
    overall_issues = []
    
    # Check for high missing data
    if data_quality_report["dataset_overview"]["missing_values_percentage"] > 10:
        overall_issues.append(f"High missing data: {data_quality_report['dataset_overview']['missing_values_percentage']}%")
    
    # Check for columns with very low quality scores
    low_quality_cols = [
        col for col, analysis in data_quality_report["column_analysis"].items()
        if analysis["quality_score"] < 0.5
    ]
    if low_quality_cols:
        overall_issues.append(f"Low quality columns: {', '.join(low_quality_cols)}")
    
    # Check for data type inconsistencies
    object_cols = data_quality_report["data_types_summary"]["categorical_columns"]
    for col in object_cols:
        if data_quality_report["column_analysis"][col]["potential_issues"]:
            if "Object type but contains numeric data" in data_quality_report["column_analysis"][col]["potential_issues"]:
                overall_issues.append(f"Data type inconsistency in column: {col}")
    
    data_quality_report["data_quality_issues"] = overall_issues
    
    # Update state
    state["dataset_info"] = dataset_info
    state["data_quality_report"] = data_quality_report
    
    print("Dataset analyzed and quality report generated")
    return state





def generate_chart_recommendations(state: DataQualityState) -> DataQualityState:
    """Generate comprehensive chart recommendations using LLM analysis."""
    
    # Get the data quality report
    quality_report = state.get("data_quality_report", {})
    
    if not quality_report:
        print("Warning: No data quality report available for chart recommendations")
        state["chart_recommendations"] = None
        return state
    
    if llm is None:
        print("LLM not available, using fallback chart recommendations")
        # Create basic chart recommendations based on data types
        numeric_cols = quality_report.get("data_types_summary", {}).get("numeric_columns", [])
        categorical_cols = quality_report.get("data_types_summary", {}).get("categorical_columns", [])
        
        fallback_recommendations = {
            "chart_recommendations": [
                {
                    "rank": 1,
                    "chart_type": "Bar Chart",
                    "chart_name": "Category Distribution Analysis",
                    "fields_required": categorical_cols[:2] if len(categorical_cols) >= 2 else ["field1", "field2"],
                    "data_preprocessing": "Count aggregation by categories",
                    "why_this_chart_helps": "Shows the distribution of categories across the dataset, revealing which categories are most common and identifying patterns in the data structure",
                    "storytelling_impact": "Establishes the foundational understanding of data distribution patterns",
                    "data_quality_notes": "Requires clean categorical data",
                    "priority_reason": "Provides essential overview of data landscape"
                },
                {
                    "rank": 2,
                    "chart_type": "Line Chart",
                    "chart_name": "Trend Analysis Over Time",
                    "fields_required": [categorical_cols[0] if categorical_cols else "field1", numeric_cols[0] if numeric_cols else "field2"],
                    "data_preprocessing": "Aggregate numeric values by time period",
                    "why_this_chart_helps": "Reveals how key metrics have evolved over time, showing trends and patterns in the data",
                    "storytelling_impact": "Demonstrates temporal patterns and changes in the dataset",
                    "data_quality_notes": "Requires time-based data and numeric values",
                    "priority_reason": "Shows temporal evolution of key metrics"
                },
                {
                    "rank": 3,
                    "chart_type": "Scatter Plot",
                    "chart_name": "Correlation Analysis",
                    "fields_required": numeric_cols[:2] if len(numeric_cols) >= 2 else ["field1", "field2"],
                    "data_preprocessing": "None required",
                    "why_this_chart_helps": "Examines the relationship between two numeric variables, revealing correlations and patterns in the data",
                    "storytelling_impact": "Identifies relationships and dependencies between variables",
                    "data_quality_notes": "Requires numeric data for both variables",
                    "priority_reason": "Reveals important relationships in the data"
                },
                {
                    "rank": 4,
                    "chart_type": "Box Plot",
                    "chart_name": "Distribution Comparison",
                    "fields_required": [categorical_cols[0] if categorical_cols else "field1", numeric_cols[0] if numeric_cols else "field2"],
                    "data_preprocessing": "None required",
                    "why_this_chart_helps": "Compares distributions across different categories, showing which groups have higher or lower values and the variability within each group",
                    "storytelling_impact": "Reveals group-level differences and data variability",
                    "data_quality_notes": "Requires categorical and numeric data",
                    "priority_reason": "Shows group-level analysis"
                },
                {
                    "rank": 5,
                    "chart_type": "Histogram",
                    "chart_name": "Data Distribution Overview",
                    "fields_required": [numeric_cols[0] if numeric_cols else "field1"],
                    "data_preprocessing": "None required",
                    "why_this_chart_helps": "Shows the overall distribution of numeric data, revealing the shape, spread, and central tendency of the dataset",
                    "storytelling_impact": "Provides understanding of data distribution and outliers",
                    "data_quality_notes": "Requires numeric data",
                    "priority_reason": "Establishes data distribution understanding"
                },
                {
                    "rank": 6,
                    "chart_type": "Scatter Plot",
                    "chart_name": "Correlation Analysis",
                    "fields_required": numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols + ["field1"],
                    "data_preprocessing": "None required",
                    "why_this_chart_helps": "Examines the relationship between numeric variables, revealing correlations and patterns in the data",
                    "storytelling_impact": "Identifies relationships and dependencies between variables",
                    "data_quality_notes": "Requires numeric data",
                    "priority_reason": "Relationship analysis"
                },
                {
                    "rank": 7,
                    "chart_type": "Violin Plot",
                    "chart_name": "Detailed Distribution Analysis",
                    "fields_required": [categorical_cols[0] if categorical_cols else "field1", numeric_cols[0] if numeric_cols else "field2"],
                    "data_preprocessing": "None required",
                    "why_this_chart_helps": "Shows the full distribution shape for each category, revealing multimodal patterns and density variations",
                    "storytelling_impact": "Provides deeper insights into distribution shapes and density patterns",
                    "data_quality_notes": "Requires categorical and numeric data",
                    "priority_reason": "Advanced distribution analysis"
                },
                {
                    "rank": 8,
                    "chart_type": "Bar Chart",
                    "chart_name": "Category Analysis",
                    "fields_required": categorical_cols[:2] + [numeric_cols[0] if numeric_cols else "field1"],
                    "data_preprocessing": "Group by categories",
                    "why_this_chart_helps": "Shows how categories relate to numeric values, revealing patterns and relationships",
                    "storytelling_impact": "Demonstrates categorical relationships with numeric data",
                    "data_quality_notes": "Requires categorical and numeric variables",
                    "priority_reason": "Category-numeric analysis"
                }
            ],
            "narrative_flow": "This comprehensive analysis begins with understanding the data distribution, then examines temporal trends, explores relationships between variables, compares group-level performance, provides distribution overview, analyzes correlations comprehensively, examines detailed distributions, and concludes with complex categorical interactions.",
            "data_quality_summary": "Data quality assessment completed for comprehensive analysis",
            "implementation_notes": "Standard chart generation approach for any dataset"
        }
        state["chart_recommendations"] = fallback_recommendations
        return state
    
    try:
        # Extract only relevant data quality information for chart recommendations
        relevant_data = extract_relevant_data_for_charts(quality_report)
        
        # Format the relevant data for the LLM prompt
        formatted_report = format_relevant_data_for_chart_prompt(relevant_data)
        
        # Create the complete prompt
        complete_prompt = f"""
{CHART_RECOMMENDATION_PROMPT}

## Data Quality Report to Analyze:

{formatted_report}

Please analyze this data quality report and provide your chart recommendations in the specified JSON format.
"""
        
        # Generate recommendations using LLM
        response = llm.invoke(complete_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse the JSON response
        import re
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without markdown wrapper
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("Could not extract JSON from LLM response")
        
        recommendations_data = json.loads(json_str)
        
        # Validate and structure the recommendations
        chart_recommendations = {
            "chart_recommendations": recommendations_data.get("chart_recommendations", []),
            "narrative_flow": recommendations_data.get("narrative_flow", ""),
            "data_quality_summary": recommendations_data.get("data_quality_summary", ""),
            "implementation_notes": recommendations_data.get("implementation_notes", ""),
            "generated_at": datetime.now().isoformat()
        }
        
        state["chart_recommendations"] = chart_recommendations
        print("Chart recommendations generated successfully")
        
    except Exception as e:
        print(f"Error generating chart recommendations: {e}")
        print(f"LLM Response: {response_text if 'response_text' in locals() else 'No response'}")
        state["chart_recommendations"] = None
    
    return state


def extract_relevant_data_for_charts(report: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the data quality information relevant for chart recommendations."""
    
    relevant_data = {
        "dataset_overview": {},
        "column_analysis": {},
        "data_types_summary": {},
        "data_quality_issues": []
    }
    
    # Extract basic dataset info
    if 'dataset_overview' in report:
        overview = report['dataset_overview']
        relevant_data["dataset_overview"] = {
            "total_rows": overview.get('total_rows'),
            "total_columns": overview.get('total_columns'),
            "missing_values_percentage": overview.get('missing_values_percentage')
        }
    
    # Extract column analysis with only chart-relevant fields
    if 'column_analysis' in report:
        for col, analysis in report['column_analysis'].items():
            relevant_data["column_analysis"][col] = {
                "data_type": analysis.get('data_type'),
                "missing_count": analysis.get('missing_count'),
                "missing_percentage": analysis.get('missing_percentage'),
                "unique_values": analysis.get('unique_values'),
                "quality_score": analysis.get('quality_score'),
                "potential_issues": analysis.get('potential_issues', [])
            }
    
    # Extract data types summary
    if 'data_types_summary' in report:
        types = report['data_types_summary']
        relevant_data["data_types_summary"] = {
            "numeric_columns": types.get('numeric_columns', []),
            "categorical_columns": types.get('categorical_columns', []),
            "datetime_columns": types.get('datetime_columns', []),
            "other_columns": types.get('other_columns', [])
        }
    
    # Extract data quality issues
    if 'data_quality_issues' in report:
        relevant_data["data_quality_issues"] = report['data_quality_issues']
    
    return relevant_data


def format_relevant_data_for_chart_prompt(relevant_data: Dict[str, Any]) -> str:
    """Format the relevant data quality information for the LLM prompt."""
    
    formatted_report = []
    
    # Dataset overview
    if relevant_data.get("dataset_overview"):
        overview = relevant_data["dataset_overview"]
        formatted_report.append(f"## Dataset Overview")
        formatted_report.append(f"- Total Rows: {overview.get('total_rows', 'N/A')}")
        formatted_report.append(f"- Total Columns: {overview.get('total_columns', 'N/A')}")
        formatted_report.append(f"- Missing Values: {overview.get('missing_values_percentage', 'N/A')}%")
    
    # Column analysis (only chart-relevant information)
    if relevant_data.get("column_analysis"):
        formatted_report.append(f"\n## Column Analysis")
        for col, analysis in relevant_data["column_analysis"].items():
            formatted_report.append(f"- {col}:")
            formatted_report.append(f"  - Data Type: {analysis.get('data_type', 'N/A')}")
            formatted_report.append(f"  - Missing: {analysis.get('missing_count', 0)} ({analysis.get('missing_percentage', 0):.1f}%)")
            formatted_report.append(f"  - Unique Values: {analysis.get('unique_values', 0)}")
            formatted_report.append(f"  - Quality Score: {analysis.get('quality_score', 0):.3f}")
            if analysis.get('potential_issues'):
                formatted_report.append(f"  - Issues: {', '.join(analysis['potential_issues'])}")
    
    # Data types summary
    if relevant_data.get("data_types_summary"):
        types = relevant_data["data_types_summary"]
        formatted_report.append(f"\n## Data Types Summary")
        formatted_report.append(f"- Numeric Columns: {', '.join(types.get('numeric_columns', []))}")
        formatted_report.append(f"- Categorical Columns: {', '.join(types.get('categorical_columns', []))}")
        formatted_report.append(f"- Datetime Columns: {', '.join(types.get('datetime_columns', []))}")
        formatted_report.append(f"- Other Columns: {', '.join(types.get('other_columns', []))}")
    
    # Data quality issues
    if relevant_data.get("data_quality_issues"):
        issues = relevant_data["data_quality_issues"]
        if issues:
            formatted_report.append(f"\n## Data Quality Issues")
            for issue in issues:
                formatted_report.append(f"- {issue}")
    
    return "\n".join(formatted_report)


def create_final_report(state: DataQualityState) -> DataQualityState:
    """Combine all analyses into a final comprehensive report."""
    
    final_report = {
        "timestamp": datetime.now().isoformat(),
        "dataset_info": state.get("dataset_info", {}),
        "data_quality_report": state.get("data_quality_report", {}),
        "chart_recommendations": state.get("chart_recommendations", {}),
        "summary": {
            "total_fields_analyzed": len(state.get("dataset_info", {}).get("columns", [])),
            "quality_score": "calculated_based_on_issues",
            "chart_recommendations_count": len(state.get("chart_recommendations", {}).get("chart_recommendations", []))
        }
    }
    
    state["final_report"] = final_report
    print("Final report generated with chart recommendations")
    return state


def generate_plotly_charts(state: DataQualityState) -> DataQualityState:
    """Generate Plotly charts from chart recommendations and save to PDF."""
    
    chart_recommendations = state.get("chart_recommendations", {})
    if not chart_recommendations or not chart_recommendations.get("chart_recommendations"):
        print("No chart recommendations available for chart generation")
        state["generated_charts"] = None
        return state
    
    generated_charts = {
        "charts": [],
        "pdf_path": None,
        "generated_at": datetime.now().isoformat()
    }
    
    try:

        charts_data = []
        
        for rec in chart_recommendations["chart_recommendations"]:
            chart_info = {
                "rank": rec.get("rank", 0),
                "chart_type": rec.get("chart_type", ""),
                "chart_name": rec.get("chart_name", ""),
                "fields_required": rec.get("fields_required", []),
                "why_this_chart_helps": rec.get("why_this_chart_helps", ""),
                "plotly_figure": None,
                "chart_data": None,
                "success": False,
                "error": None
            }
            
            try:
                # Generate the chart based on chart type
                fig, chart_data = create_plotly_chart(rec, df)
                
                # Validate that the chart has meaningful data
                if fig is not None and chart_data is not None:
                    # Check if the chart has actual data points
                    try:
                        has_data = validate_chart_has_data(fig, chart_data)
                        
                        if has_data:
                            chart_info["plotly_figure"] = fig
                            chart_info["chart_data"] = chart_data
                            chart_info["success"] = True
                            
                            # Save individual chart as HTML
                            chart_filename = f"chart_{rec.get('rank', 0)}_{rec.get('chart_name', 'chart').replace(' ', '_').lower()}.html"
                            chart_path = os.path.join(script_path, chart_filename)
                            # fig.write_html(chart_path)
                            chart_info["html_path"] = chart_path
                            
                            charts_data.append(chart_info)
                            print(f"Generated chart {rec.get('rank', 0)}: {rec.get('chart_name', '')}")
                        else:
                            chart_info["error"] = "Chart generated but contains no meaningful data"
                            charts_data.append(chart_info)
                            print(f"Chart {rec.get('rank', 0)} has no meaningful data: {rec.get('chart_name', '')}")
                    except Exception as validation_error:
                        chart_info["error"] = f"Chart validation error: {str(validation_error)}"
                        charts_data.append(chart_info)
                        print(f"Chart {rec.get('rank', 0)} validation failed: {validation_error}")
                else:
                    chart_info["error"] = "Failed to generate chart figure or data"
                    charts_data.append(chart_info)
                    print(f"Failed to generate chart {rec.get('rank', 0)}: {rec.get('chart_name', '')}")
            except Exception as e:
                chart_info["error"] = str(e)
                charts_data.append(chart_info)
                print(f"Error generating chart {rec.get('rank', 0)}: {e}")
        
        generated_charts["charts"] = charts_data
        state["generated_charts"] = generated_charts
        
    except Exception as e:
        print(f"Error in chart generation workflow: {e}")
        state["generated_charts"] = None
    
    return state


def generate_narrative_insights(state: DataQualityState) -> DataQualityState:
    """Generate AI-powered narrative insights and storytelling flow for the charts."""
    
    generated_charts = state.get("generated_charts", {})
    chart_recommendations = state.get("chart_recommendations", {})
    
    if not generated_charts or not generated_charts.get("charts"):
        print("No generated charts available for narrative insights")
        state["narrative_insights"] = None
        return state
    
    if llm is None:
        print("LLM not available, using fallback narrative insights")
        state["narrative_insights"] = {
            "chart_insights": [],
            "story_flow": "The data analysis reveals important patterns and trends that provide valuable insights for decision-making.",
            "executive_narrative": "This comprehensive analysis provides key insights into the dataset patterns and trends.",
            "key_takeaways": ["Data quality assessment completed", "Charts generated successfully", "Patterns identified in the data"],
            "generated_at": datetime.now().isoformat()
        }
        return state
    
    try:
        narrative_insights = {
            "chart_insights": [],
            "story_flow": "",
            "executive_narrative": "",
            "key_takeaways": [],
            "generated_at": datetime.now().isoformat()
        }
        
        # Generate insights for each successful chart
        successful_charts = [c for c in generated_charts["charts"] if c.get("success")]
        
        if not successful_charts:
            print("No successful charts found for narrative insights generation")
            narrative_insights["chart_insights"] = []
            narrative_insights["story_flow"] = "No charts available for story generation."
            narrative_insights["executive_narrative"] = "No charts available for executive summary."
            narrative_insights["key_takeaways"] = ["No charts available for analysis."]
            state["narrative_insights"] = narrative_insights
            return state
        
        print(f"Generating AI insights for {len(successful_charts)} charts...")
        
        for chart_info in successful_charts:
            try:
                chart_insight = generate_chart_insight(chart_info, chart_recommendations)
                narrative_insights["chart_insights"].append(chart_insight)
                print(f"✓ Generated insight for chart {chart_info.get('rank', 0)}: {chart_info.get('chart_name', 'Unknown')}")
            except Exception as e:
                print(f"✗ Error generating insight for chart {chart_info.get('rank', 0)}: {e}")
                # Add a fallback insight
                narrative_insights["chart_insights"].append({
                    "chart_rank": chart_info.get("rank", 0),
                    "chart_name": chart_info.get("chart_name", ""),
                    "ai_insight": f"Unable to generate AI insight for this chart due to an error: {str(e)}",
                    "data_summary": "Data summary unavailable due to error.",
                    "storytelling_hooks": []
                })
        
        # Generate overall story flow
        try:
            print("Generating story flow...")
            narrative_insights["story_flow"] = generate_story_flow(successful_charts, chart_recommendations)
            print("✓ Story flow generated successfully")
        except Exception as e:
            print(f"✗ Error generating story flow: {e}")
            narrative_insights["story_flow"] = "Unable to generate story flow due to an error."
        
        # Generate executive narrative
        try:
            print("Generating executive narrative...")
            narrative_insights["executive_narrative"] = generate_executive_narrative(successful_charts, chart_recommendations)
            print("✓ Executive narrative generated successfully")
        except Exception as e:
            print(f"✗ Error generating executive narrative: {e}")
            narrative_insights["executive_narrative"] = "Unable to generate executive narrative due to an error."
        
        # Generate key takeaways
        try:
            print("Generating key takeaways...")
            narrative_insights["key_takeaways"] = generate_key_takeaways(successful_charts, chart_recommendations)
            print("✓ Key takeaways generated successfully")
        except Exception as e:
            print(f"✗ Error generating key takeaways: {e}")
            narrative_insights["key_takeaways"] = ["Unable to generate key takeaways due to an error."]
        
        state["narrative_insights"] = narrative_insights
        print("Narrative insights generated successfully")
        
    except Exception as e:
        print(f"Error generating narrative insights: {e}")
        state["narrative_insights"] = None
    
    return state


def generate_chart_insight(chart_info: Dict[str, Any], chart_recommendations: Dict[str, Any]) -> Dict[str, Any]:
    """Generate AI insight for a specific chart based on actual data analysis."""
    
    # Get the actual chart data for analysis
    chart_data = chart_info.get("chart_data")
    
    # Create a more focused prompt that analyzes the actual data
    chart_prompt = f"""
You are a data analyst examining a specific chart. Analyze the actual data and provide insights based on what you observe.

## Chart Information:
- **Chart Name**: {chart_info.get('chart_name', 'N/A')}
- **Chart Type**: {chart_info.get('chart_type', 'N/A')}
- **Fields Used**: {', '.join(chart_info.get('fields_required', []))}

## Actual Chart Data:
{analyze_chart_data_for_insights(chart_data, chart_info.get('chart_type', ''))}

## Your Task:
Based on the actual data above, provide specific insights about:
1. **What patterns do you see?** (e.g., "Category A has the highest value at 2,178")
2. **What are the key findings?** (e.g., "The distribution shows a clear leader with 3x more activity than others")
3. **What does this mean?** (e.g., "This suggests Category A is the most significant factor")
4. **What actions should be taken?** (e.g., "Other categories should be analyzed for improvement opportunities")

## Important:
- Focus on specific numbers and patterns from the data
- Don't repeat the chart purpose - analyze what the data actually shows
- Be specific about which categories/factors are dominant, highest, lowest, etc.
- Use concrete observations from the data
- Keep your response to exactly 150 words maximum
- **DO NOT include any meta-commentary, word counts, internal reasoning, or chain-of-thought text**

Write a concise, insightful analysis in exactly 150 words that captures the most important insights from the actual data. Provide only the analysis, no additional commentary. Do not use markdown formatting like **bold** or *italic* text.
"""

    try:
        # Simple timeout mechanism
        import threading
        import time
        
        result = {"response": None, "error": None}
        
        def llm_call():
            try:
                response = llm.invoke(chart_prompt)
                result["response"] = response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                result["error"] = str(e)
        
        # Start LLM call in a separate thread
        thread = threading.Thread(target=llm_call)
        thread.daemon = True
        thread.start()
        
        # Wait for up to 30 seconds
        thread.join(timeout=60)
        
        if thread.is_alive():
            return {
                "chart_rank": chart_info.get("rank", 0),
                "chart_name": chart_info.get("chart_name", ""),
                "ai_insight": "Analysis timed out. Chart data shows patterns that require further investigation.",
                "data_summary": summarize_chart_data(chart_info),
                "storytelling_hooks": []
            }
        
        if result["error"]:
            raise Exception(result["error"])
        
        insight_text = result["response"]
        
        # Clean up the response
        insight_text = insight_text.strip()
        if insight_text.startswith("Based on the data"):
            insight_text = insight_text[19:]  # Remove "Based on the data"
        if insight_text.startswith("The data shows"):
            insight_text = insight_text[14:]  # Remove "The data shows"
        
        return {
            "chart_rank": chart_info.get("rank", 0),
            "chart_name": chart_info.get("chart_name", ""),
            "ai_insight": insight_text,
            "data_summary": summarize_chart_data(chart_info),
            "storytelling_hooks": extract_storytelling_hooks(insight_text)
        }
            
    except Exception as e:
        return {
            "chart_rank": chart_info.get("rank", 0),
            "chart_name": chart_info.get("chart_name", ""),
            "ai_insight": f"Unable to generate AI insight: {str(e)}",
            "data_summary": summarize_chart_data(chart_info),
            "storytelling_hooks": []
        }


def summarize_chart_data(chart_info: Dict[str, Any]) -> str:
    """Create a summary of the chart data for the LLM."""
    
    chart_data = chart_info.get("chart_data")
    if chart_data is None:
        return "No data available for analysis."
    
    try:
        if isinstance(chart_data, pd.DataFrame):
            if chart_data.empty:
                return "DataFrame is empty."
            summary = f"""
Data Shape: {chart_data.shape}
Columns: {list(chart_data.columns)}
Sample Data:
{chart_data.head(3).to_string()}
"""
        elif isinstance(chart_data, pd.Series):
            if chart_data.empty:
                return "Series is empty."
            summary = f"""
Data Type: Series
Length: {len(chart_data)}
Sample Values:
{chart_data.head(5).to_string()}
"""
        else:
            # Handle other data types
            if hasattr(chart_data, '__len__'):
                length = len(chart_data)
                if length == 0:
                    return "Data is empty."
                summary = f"Data Type: {type(chart_data).__name__}, Length: {length}"
            else:
                summary = f"Data Type: {type(chart_data).__name__}, Value: {str(chart_data)[:200]}"
        
        return summary
    except Exception as e:
        return f"Data summary unavailable. Error: {str(e)}"


def extract_storytelling_hooks(insight_text: str) -> List[str]:
    """Extract key storytelling hooks from the insight text."""
    
    hooks = []
    
    # Look for common storytelling patterns
    if "trend" in insight_text.lower():
        hooks.append("trend_analysis")
    if "pattern" in insight_text.lower():
        hooks.append("pattern_identification")
    if "surprising" in insight_text.lower() or "unexpected" in insight_text.lower():
        hooks.append("surprising_finding")
    if "opportunity" in insight_text.lower():
        hooks.append("opportunity_identification")
    if "risk" in insight_text.lower() or "concern" in insight_text.lower():
        hooks.append("risk_assessment")
    if "recommendation" in insight_text.lower() or "suggest" in insight_text.lower():
        hooks.append("actionable_recommendation")
    
    return hooks


def generate_story_flow(successful_charts: List[Dict], chart_recommendations: Dict[str, Any]) -> str:
    """Generate a narrative flow that connects all charts into a cohesive story."""
    
    story_prompt = f"""
You are a master data storyteller. Create a compelling narrative flow that connects these charts into a cohesive, engaging story.

## Charts in the Story:
{format_charts_for_story(successful_charts)}

## Original Narrative Flow:
{chart_recommendations.get('narrative_flow', 'N/A')}

## Your Task:
Write a 3-4 paragraph narrative that:
1. **Opens with a compelling hook** that draws readers in
2. **Flows smoothly from chart to chart** with natural transitions
3. **Builds tension and reveals insights progressively**
4. **Creates a sense of discovery** as each chart adds to the story
5. **Ends with a powerful conclusion** that ties everything together

## Style Guidelines:
- Use professional, analytical language
- Create smooth transitions between charts
- Build logical progression and reveal insights systematically
- Present data insights in a clear, structured manner
- Connect each chart to the broader analytical narrative
- Maintain formal tone appropriate for business reports
- Avoid casual language or colloquialisms
- **DO NOT include any meta-commentary, word counts, internal reasoning, or chain-of-thought text**

Write a compelling analytical narrative that guides readers through each chart systematically. Provide only the narrative, no additional commentary.
"""

    try:
        response = llm.invoke(story_prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"Unable to generate story flow: {str(e)}"


def format_charts_for_story(charts: List[Dict]) -> str:
    """Format charts for the story generation prompt."""
    
    formatted = []
    for chart in charts:
        formatted.append(f"""
Chart {chart.get('rank', 0)}: {chart.get('chart_name', 'N/A')}
- Type: {chart.get('chart_type', 'N/A')}
- Purpose: {chart.get('purpose', 'N/A')}
- Key Insight: {chart.get('key_insights', 'N/A')}
""")
    
    return "\n".join(formatted)


def generate_executive_narrative(successful_charts: List[Dict], chart_recommendations: Dict[str, Any]) -> str:
    """Generate an executive-level narrative summary."""
    
    executive_prompt = f"""
You are a senior data analyst presenting detailed insights from the data to a conference audience. 
Create a compelling executive summary based on these data visualizations.

## Charts Available:
{format_charts_for_executive(successful_charts)}

## Data Quality Context:
{chart_recommendations.get('data_quality_summary', 'N/A')}

## Your Task:
Write a 2-3 paragraph executive summary that:
1. **Captures the most critical insights** for strategic decision-making
2. **Highlights key trends and patterns** that matter to the business
3. **Identifies opportunities and risks** that require attention
4. **Provides clear, actionable recommendations** for next steps
5. **Uses executive-level language** - concise, strategic, and impactful

## Style Guidelines:
- Start with the most important finding
- Use clear, concise language
- Focus on business impact and strategic implications
- End with specific, actionable recommendations
- Keep it under 500 words

Write an executive summary that would help audience quickly understand the insights presented.
"""

    try:
        response = llm.invoke(executive_prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"Unable to generate executive narrative: {str(e)}"


def format_charts_for_executive(charts: List[Dict]) -> str:
    """Format charts for executive summary generation."""
    
    formatted = []
    for chart in charts:
        formatted.append(f"""
• {chart.get('chart_name', 'N/A')} - {chart.get('key_insights', 'N/A')}
""")
    
    return "\n".join(formatted)


def generate_key_takeaways(successful_charts: List[Dict], chart_recommendations: Dict[str, Any]) -> List[str]:
    """Generate key takeaways from all charts."""
    
    takeaways_prompt = f"""
You are a data analyst extracting the most important insights from a comprehensive data visualization report.

## Charts Analyzed:
{format_charts_for_takeaways(successful_charts)}

## Your Task:
Generate 5-7 key takeaways that:
1. **Capture the most important insights** across all charts
2. **Are specific and actionable** rather than generic
3. **Highlight surprising or unexpected findings**
4. **Identify clear opportunities or risks**
5. **Provide strategic value** for decision-making

## Format:
Return only a numbered list of key takeaways, each 1-2 sentences long. Focus on insights that would be most valuable to stakeholders.

Example:
1. [Specific insight with business impact]
2. [Another key finding with implications]
3. [Surprising pattern or trend]
...

Generate 5-7 compelling key takeaways. **DO NOT include any meta-commentary, word counts, internal reasoning, or chain-of-thought text.** Do not use markdown formatting like **bold** or *italic* text.
"""

    try:
        response = llm.invoke(takeaways_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse numbered list
        lines = response_text.strip().split('\n')
        takeaways = []
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-') or line.startswith('*')):
                # Remove numbering/bullets and clean up
                clean_line = line.lstrip('0123456789.•-* ').strip()
                if clean_line:
                    takeaways.append(clean_line)
        
        return takeaways[:7]  # Limit to 7 takeaways
        
    except Exception as e:
        return [f"Unable to generate key takeaways: {str(e)}"]


def format_charts_for_takeaways(charts: List[Dict]) -> str:
    """Format charts for key takeaways generation."""
    
    formatted = []
    for chart in charts:
        formatted.append(f"""
Chart {chart.get('rank', 0)}: {chart.get('chart_name', 'N/A')}
- Type: {chart.get('chart_type', 'N/A')}
- Key Insight: {chart.get('key_insights', 'N/A')}
""")
    
    return "\n".join(formatted)


def validate_chart_has_data(fig, chart_data) -> bool:
    """Enhanced validation that a chart has meaningful data."""
    
    try:
        # Check if the figure has any traces with data
        if not fig.data:
            print("Chart validation failed: No traces found")
            return False
        
        # Check each trace for meaningful data
        for trace in fig.data:
            # For different trace types, check if they have data
            if hasattr(trace, 'x') and hasattr(trace, 'y'):
                # Check if x and y have non-empty data
                if trace.x is not None and trace.y is not None:
                    if len(trace.x) > 0 and len(trace.y) > 0:
                        # Check if the data has any non-null values
                        if any(x is not None and x != '' for x in trace.x) and any(y is not None and y != '' for y in trace.y):
                            # Additional check: ensure we have meaningful variation
                            if len(set(trace.x)) > 1 and len(set(trace.y)) > 1:
                                return True
                            else:
                                print(f"Chart validation failed: No meaningful variation in data (x: {len(set(trace.x))}, y: {len(set(trace.y))})")
                        else:
                            print("Chart validation failed: All values are null or empty")
                    else:
                        print("Chart validation failed: Empty x or y arrays")
                else:
                    print("Chart validation failed: x or y is None")
            elif hasattr(trace, 'z') and trace.z is not None:
                # For heatmaps, check if z data exists
                if len(trace.z) > 0 and any(any(val is not None and val != '' for val in row) for row in trace.z):
                    return True
                else:
                    print("Chart validation failed: Heatmap z data is empty or all null")
            else:
                print("Chart validation failed: Trace missing x, y, or z attributes")
        
        # If we have chart_data, also validate it
        if chart_data is not None:
            if isinstance(chart_data, pd.DataFrame):
                if not chart_data.empty and len(chart_data) > 0:
                    # Check for meaningful data variation
                    for col in chart_data.columns:
                        if chart_data[col].nunique() > 1:
                            return True
                    print("Chart validation failed: DataFrame has no meaningful variation")
                else:
                    print("Chart validation failed: DataFrame is empty")
            elif isinstance(chart_data, pd.Series):
                if not chart_data.empty and len(chart_data) > 0:
                    if chart_data.nunique() > 1:
                        return True
                    else:
                        print("Chart validation failed: Series has no meaningful variation")
                else:
                    print("Chart validation failed: Series is empty")
            elif hasattr(chart_data, '__len__'):
                try:
                    if len(chart_data) > 0:
                        return True
                    else:
                        print("Chart validation failed: Chart data is empty")
                except (TypeError, ValueError):
                    # Handle cases where len() doesn't work or returns ambiguous results
                    print("Chart validation failed: Cannot determine chart data length")
            else:
                # For other data types, try to evaluate as boolean
                try:
                    if bool(chart_data):
                        return True
                    else:
                        print("Chart validation failed: Chart data evaluates to False")
                except (ValueError, TypeError):
                    print("Chart validation failed: Cannot evaluate chart data")
        
        print("Chart validation failed: No valid data found in any trace or chart_data")
        return False
        
    except Exception as e:
        print(f"Error validating chart data: {e}")
        return False



    
    # Time Series Line Chart
    if "time series" in chart_type or "line" in chart_type:
        if len(fields_required) >= 2:
            x_field = fields_required[0]  # Assume first field is time
            y_field = fields_required[1]  # Assume second field is value
            
            # Validate that we have enough data points for a line chart
            if len(chart_df) < 3:
                raise ValueError(f"Insufficient data points ({len(chart_df)}) for line chart visualization")
            
            # Aggregate data by time period for cleaner visualization
            if pd.api.types.is_numeric_dtype(chart_df[y_field]):
                chart_data = chart_df.groupby(x_field)[y_field].mean().reset_index()
            else:
                chart_data = chart_df.groupby(x_field).size().reset_index(name='Count')
                y_field = 'Count'
            
            # Validate aggregated data
            if len(chart_data) < 2:
                raise ValueError(f"Insufficient aggregated data points ({len(chart_data)}) for line chart")
            
            # If there's a third field, use it for color grouping
            if len(fields_required) >= 3:
                color_field = fields_required[2]
                # Aggregate with color grouping
                chart_data = chart_df.groupby([x_field, color_field])[y_field].mean().reset_index()
                
                # Validate that we have data for multiple colors
                if chart_data[color_field].nunique() < 2:
                    raise ValueError(f"Insufficient unique values in color field '{color_field}' for grouped line chart")
                
                fig = px.line(chart_data, x=x_field, y=y_field, color=color_field,
                            title=chart_name, labels={x_field: x_field, y_field: y_field},
                            color_discrete_sequence=px.colors.qualitative.Set3)
            else:
                fig = px.line(chart_data, x=x_field, y=y_field,
                            title=chart_name, labels={x_field: x_field, y_field: y_field},
                            color_discrete_sequence=['#3498db'])
    
    # Correlation Heatmap
    elif "correlation" in chart_type or "heatmap" in chart_type:
        # Calculate correlation matrix
        numeric_fields = [field for field in fields_required if pd.api.types.is_numeric_dtype(chart_df[field])]
        if len(numeric_fields) >= 2:
            corr_matrix = chart_df[numeric_fields].corr()
            fig = px.imshow(corr_matrix, 
                          title=chart_name,
                          color_continuous_scale='RdBu_r',
                          aspect='auto',
                          text_auto=True,
                          color_continuous_midpoint=0)
            chart_data = corr_matrix
    
    # Stacked Bar Chart
    elif "stacked bar" in chart_type:
        if len(fields_required) >= 2:
            x_field = fields_required[0]
            y_field = fields_required[1]
            color_field = fields_required[2] if len(fields_required) >= 3 else None
            
            if color_field:
                fig = px.bar(chart_df, x=x_field, y=y_field, color=color_field,
                           title=chart_name, barmode='stack',
                           color_discrete_sequence=px.colors.qualitative.Set3)
            else:
                fig = px.bar(chart_df, x=x_field, y=y_field,
                           title=chart_name,
                           color_discrete_sequence=['#3498db'])
            
            chart_data = chart_df.groupby([x_field, color_field])[y_field].sum().reset_index() if color_field else chart_df.groupby(x_field)[y_field].sum().reset_index()
    
    # Scatter Plot
    elif "scatter" in chart_type:
        if len(fields_required) >= 2:
            x_field = fields_required[0]
            y_field = fields_required[1]
            color_field = fields_required[2] if len(fields_required) >= 3 else None
            
            if color_field:
                fig = px.scatter(chart_df, x=x_field, y=y_field, color=color_field,
                               title=chart_name, trendline="ols",
                               color_discrete_sequence=px.colors.qualitative.Set3)
            else:
                fig = px.scatter(chart_df, x=x_field, y=y_field,
                               title=chart_name, trendline="ols",
                               color_discrete_sequence=['#3498db'])
            
            chart_data = chart_df
    
    # Box Plot
    elif "box" in chart_type:
        if len(fields_required) >= 2:
            x_field = fields_required[0]  # Categorical field
            y_field = fields_required[1]  # Numeric field
            
            fig = px.box(chart_df, x=x_field, y=y_field,
                        title=chart_name,
                        color_discrete_sequence=px.colors.qualitative.Set3)
            
            chart_data = chart_df.groupby(x_field)[y_field].describe()
    
    # Bar Chart (default for categorical data)
    elif "bar" in chart_type:
        if len(fields_required) >= 2:
            x_field = fields_required[0]
            y_field = fields_required[1]
            
            # Handle different aggregation types based on chart purpose
            if "distribution" in chart_name.lower() or "count" in chart_name.lower():
                # For distribution charts, count occurrences
                if len(fields_required) >= 2:
                    if len(fields_required) == 2:
                        # Simple count by x_field
                        chart_data = chart_df[x_field].value_counts().reset_index()
                        chart_data.columns = [x_field, 'Count']
                        
                        # Validate that we have meaningful counts
                        if len(chart_data) < 2 or chart_data['Count'].sum() < 2:
                            raise ValueError(f"Insufficient data for distribution chart: {len(chart_data)} categories with {chart_data['Count'].sum()} total items")
                        
                        fig = px.bar(chart_data, x=x_field, y='Count',
                                   title=chart_name,
                                   color_discrete_sequence=['#3498db'])
                    else:
                        # Count by multiple fields (e.g., Category and Type)
                        chart_data = chart_df.groupby(fields_required[:-1]).size().reset_index(name='Count')
                        
                        # Validate grouped data
                        if len(chart_data) < 2 or chart_data['Count'].sum() < 2:
                            raise ValueError(f"Insufficient data for grouped distribution chart: {len(chart_data)} groups with {chart_data['Count'].sum()} total items")
                        
                        fig = px.bar(chart_data, x=fields_required[0], y='Count', 
                                   color=fields_required[1] if len(fields_required) > 1 else None,
                                   title=chart_name,
                                   color_discrete_sequence=px.colors.qualitative.Set3)
            else:
                # For other bar charts, use sum/mean aggregation
                if pd.api.types.is_numeric_dtype(chart_df[y_field]):
                    chart_data = chart_df.groupby(x_field)[y_field].mean().reset_index()
                    
                    # Validate aggregated data
                    if len(chart_data) < 2:
                        raise ValueError(f"Insufficient data for bar chart: {len(chart_data)} categories")
                    
                    fig = px.bar(chart_df, x=x_field, y=y_field,
                               title=chart_name,
                               color_discrete_sequence=['#3498db'])
                else:
                    chart_data = chart_df.groupby(x_field).size().reset_index(name='Count')
                    
                    # Validate count data
                    if len(chart_data) < 2 or chart_data['Count'].sum() < 2:
                        raise ValueError(f"Insufficient data for count bar chart: {len(chart_data)} categories with {chart_data['Count'].sum()} total items")
                    
                    fig = px.bar(chart_data, x=x_field, y='Count',
                               title=chart_name,
                               color_discrete_sequence=['#3498db'])
    
    # Histogram
    elif "histogram" in chart_type:
        if len(fields_required) >= 1:
            field = fields_required[0]
            
            fig = px.histogram(chart_df, x=field,
                             title=chart_name,
                             color_discrete_sequence=['#e67e22'])
            
            chart_data = chart_df[field].value_counts().reset_index()
    
    # If no specific chart type matched, create a basic visualization
    else:
        if len(fields_required) >= 2:
            x_field = fields_required[0]
            y_field = fields_required[1]
            
                    # Try to determine the best chart type based on data types
        if pd.api.types.is_numeric_dtype(chart_df[x_field]) and pd.api.types.is_numeric_dtype(chart_df[y_field]):
            fig = px.scatter(chart_df, x=x_field, y=y_field, title=chart_name,
                           color_discrete_sequence=['#3498db'])
        elif pd.api.types.is_numeric_dtype(chart_df[y_field]):
            fig = px.bar(chart_df, x=x_field, y=y_field, title=chart_name,
                        color_discrete_sequence=['#3498db'])
        else:
            fig = px.histogram(chart_df, x=x_field, title=chart_name,
                             color_discrete_sequence=['#e67e22'])
        
        chart_data = chart_df
    
    if fig:
        # Apply professional theme with better colors and contrast
        fig.update_layout(
            title={
                'text': chart_name,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50', 'weight': 'bold'}
            },
            plot_bgcolor='white',
            font={'color': '#2c3e50', 'size': 13},
            height=600,
            width=900,
            showlegend=True,
            legend={
                'bgcolor': 'rgba(255,255,255,0.9)',
                'bordercolor': '#bdc3c7',
                'borderwidth': 1,
                'font': {'size': 12}
            },
            margin=dict(l=80, r=80, t=100, b=80),
            hovermode='closest',
            hoverlabel={
                'bgcolor': 'white',
                'font_size': 12,
                'font_family': 'Arial'
            }
        )
        
        # Update axes for better visibility
        fig.update_xaxes(
            gridcolor='#ecf0f1',
            gridwidth=1,
            zerolinecolor='#bdc3c7',
            zerolinewidth=1,
            showline=True,
            linecolor='#2c3e50',
            linewidth=1
        )
        
        fig.update_yaxes(
            gridcolor='#ecf0f1',
            gridwidth=1,
            zerolinecolor='#bdc3c7',
            zerolinewidth=1,
            showline=True,
            linecolor='#2c3e50',
            linewidth=1
        )
        
        # Apply better color schemes based on chart type
        if "line" in chart_type or "time series" in chart_type:
            fig.update_traces(
                line=dict(width=4),
                marker=dict(size=8, opacity=0.8)
            )
        elif "bar" in chart_type:
            fig.update_traces(
                marker_color='#3498db',
                marker_line_color='#2980b9',
                marker_line_width=2,
                opacity=0.85
            )
        elif "scatter" in chart_type:
            fig.update_traces(
                marker=dict(size=10, opacity=0.8),
                line=dict(width=3, color='#e74c3c')
            )
        elif "box" in chart_type:
            fig.update_traces(
                marker_color='#9b59b6',
                marker_line_color='#8e44ad',
                marker_line_width=2,
                opacity=0.8
            )
        elif "histogram" in chart_type:
            fig.update_traces(
                marker_color='#e67e22',
                marker_line_color='#d35400',
                marker_line_width=2,
                opacity=0.8
            )
    
    # If Plotly failed and Altair is available, try Altair as fallback
    if fig is None and ALTAIR_AVAILABLE:
        try:
            print(f"Plotly failed for chart '{chart_name}', trying Altair fallback...")
            fig, chart_data = create_altair_chart(recommendation, chart_df)
            if fig is not None:
                print(f"✅ Altair fallback successful for chart '{chart_name}'")
        except Exception as altair_error:
            print(f"❌ Altair fallback also failed: {altair_error}")
    
    return fig, chart_data


def generate_html_report(charts_data: List[Dict], chart_recommendations: Dict, script_path: str, state: DataQualityState) -> str:
    """Generate an interactive HTML report containing all charts embedded directly."""
    
    try:
        html_filename = "output.html"
        html_path = os.path.join(script_path, html_filename)
        
        # Extract narrative insights from state
        narrative_insights = state.get("narrative_insights", {})
        
        # Start building HTML content
        html_content = []
        
        # HTML header with enhanced CSS styling for narrative flow
        html_content.append("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Story: Interactive Visualization Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.7;
            color: #2c3e50;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        }
        .header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 50px 20px;
            border-radius: 15px;
            margin-bottom: 40px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            position: relative;
            overflow: hidden;
        }
        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
            pointer-events: none;
        }
        .header h1 {
            margin: 0;
            font-size: 3em;
            font-weight: 300;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        .header p {
            margin: 15px 0 0 0;
            font-size: 1.2em;
            opacity: 0.95;
            font-weight: 300;
        }
        .story-flow {
            background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
            padding: 40px;
            margin-bottom: 40px;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            border-left: 5px solid #3498db;
            position: relative;
        }
        .story-flow::before {
            content: '📖';
            position: absolute;
            top: -15px;
            left: 30px;
            background: white;
            padding: 10px;
            border-radius: 50%;
            font-size: 1.5em;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .story-flow h2 {
            color: #2c3e50;
            margin-top: 0;
            font-size: 1.8em;
            font-weight: 600;
        }
        .story-flow p {
            font-size: 1.1em;
            line-height: 1.8;
            color: #34495e;
            margin-bottom: 20px;
        }
        .section {
            background: white;
            padding: 40px;
            margin-bottom: 40px;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .section:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
        }
        .section h2 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
            font-size: 2em;
            font-weight: 600;
        }
        .section h3 {
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 1.5em;
            font-weight: 600;
        }
        .section h4 {
            color: #7f8c8d;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 1.1em;
            font-weight: 600;
        }
        .chart-container {
            background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
            padding: 30px;
            margin: 30px 0;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            border-left: 5px solid #3498db;
            transition: all 0.3s ease;
            position: relative;
        }
        .chart-container:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
        }
        .chart-container::before {
            content: '📊';
            position: absolute;
            top: -15px;
            left: 30px;
            background: white;
            padding: 10px;
            border-radius: 50%;
            font-size: 1.5em;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .chart-description {
            background: linear-gradient(135deg, #ecf0f1 0%, #e9ecef 100%);
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 25px;
            border-left: 4px solid #3498db;
        }
        .chart-description h4 {
            margin-top: 0;
            color: #2c3e50;
            font-size: 1.2em;
            font-weight: 600;
        }
        .chart-description p {
            margin: 10px 0;
            line-height: 1.6;
        }
        .ai-insight {
            background: linear-gradient(135deg, #e8f4fd 0%, #d1ecf1 100%);
            padding: 25px;
            border-radius: 10px;
            margin: 25px 0;
            border-left: 4px solid #17a2b8;
            position: relative;
        }
        .ai-insight::before {
            content: '🤖';
            position: absolute;
            top: -10px;
            left: 20px;
            background: white;
            padding: 8px;
            border-radius: 50%;
            font-size: 1.2em;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }
        .ai-insight h4 {
            margin-top: 0;
            color: #2c3e50;
            font-size: 1.2em;
            font-weight: 600;
        }
        .ai-insight p {
            margin: 10px 0;
            line-height: 1.7;
            color: #34495e;
        }
        .chart-plot {
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        .metadata {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            border-radius: 10px;
            font-size: 0.95em;
            color: #6c757d;
            margin-top: 20px;
            border-left: 4px solid #6c757d;
        }
        .success-badge {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            display: inline-block;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(39, 174, 96, 0.3);
        }
        .error-badge {
            background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            display: inline-block;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(231, 76, 60, 0.3);
        }
        .toc {
            background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            margin-bottom: 40px;
            border-left: 5px solid #3498db;
        }
        .toc h3 {
            margin-top: 0;
            color: #2c3e50;
            font-size: 1.5em;
            font-weight: 600;
        }
        .toc ul {
            list-style-type: none;
            padding-left: 0;
        }
        .toc li {
            padding: 8px 0;
            border-bottom: 1px solid #ecf0f1;
        }
        .toc li:last-child {
            border-bottom: none;
        }
        .toc a {
            color: #3498db;
            text-decoration: none;
            font-weight: 500;
            font-size: 1.1em;
            transition: color 0.3s ease;
        }
        .toc a:hover {
            color: #2980b9;
            text-decoration: underline;
        }
        .key-takeaways {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            padding: 30px;
            border-radius: 15px;
            margin: 30px 0;
            border-left: 5px solid #f39c12;
            position: relative;
        }
        .key-takeaways::before {
            content: '💡';
            position: absolute;
            top: -15px;
            left: 30px;
            background: white;
            padding: 10px;
            border-radius: 50%;
            font-size: 1.5em;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .key-takeaways h3 {
            margin-top: 0;
            color: #2c3e50;
            font-size: 1.5em;
            font-weight: 600;
        }
        .key-takeaways ul {
            padding-left: 20px;
        }
        .key-takeaways li {
            margin: 10px 0;
            line-height: 1.6;
            color: #34495e;
        }
        .executive-summary {
            background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
            padding: 30px;
            border-radius: 15px;
            margin: 30px 0;
            border-left: 5px solid #28a745;
            position: relative;
        }
        .executive-summary::before {
            content: '🎯';
            position: absolute;
            top: -15px;
            left: 30px;
            background: white;
            padding: 10px;
            border-radius: 50%;
            font-size: 1.5em;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .executive-summary h3 {
            margin-top: 0;
            color: #2c3e50;
            font-size: 1.5em;
            font-weight: 600;
        }
        .executive-summary p {
            line-height: 1.7;
            color: #34495e;
            font-size: 1.1em;
        }
        .scroll-indicator {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: #ecf0f1;
            z-index: 1000;
        }
        .scroll-progress {
            height: 100%;
            background: linear-gradient(90deg, #3498db, #9b59b6);
            width: 0%;
            transition: width 0.3s ease;
        }
        .data-quality-summary {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 25px;
            border-left: 4px solid #17a2b8;
        }
        .summary-stats {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }
        .stat-item {
            text-align: center;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            min-width: 120px;
        }
        .stat-number {
            display: block;
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
            margin-bottom: 5px;
        }
        .stat-label {
            font-size: 0.9em;
            color: #6c757d;
            font-weight: 500;
        }
        .data-quality-table-container {
            margin-top: 30px;
        }
        .table-wrapper {
            overflow-x: auto;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }
        .data-quality-table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            font-size: 0.9em;
        }
        .data-quality-table th {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            padding: 15px 10px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #2980b9;
        }
        .data-quality-table td {
            padding: 12px 10px;
            border-bottom: 1px solid #ecf0f1;
            vertical-align: top;
        }
        .data-quality-table tr:hover {
            background-color: #f8f9fa;
        }
        .high-quality {
            background-color: #d4edda;
        }
        .medium-quality {
            background-color: #fff3cd;
        }
        .low-quality {
            background-color: #f8d7da;
        }
        .quality-score {
            font-weight: bold;
            padding: 4px 8px;
            border-radius: 4px;
            background: #e9ecef;
        }
        .enhanced-narrative {
            background: linear-gradient(135deg, #e8f4fd 0%, #d1ecf1 100%);
            padding: 25px;
            border-radius: 10px;
            border-left: 4px solid #17a2b8;
            margin: 20px 0;
        }
        .conclusion {
            background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
            padding: 25px;
            border-radius: 10px;
            border-left: 4px solid #28a745;
            margin: 20px 0;
        }
        .narrative-transition {
            background: linear-gradient(135deg, #e8f4fd 0%, #d1ecf1 100%);
            padding: 30px;
            border-radius: 15px;
            margin: 40px 0 30px 0;
            border-left: 5px solid #17a2b8;
            position: relative;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }
        .narrative-transition::before {
            content: '📊';
            position: absolute;
            top: -15px;
            left: 30px;
            background: white;
            padding: 10px;
            border-radius: 50%;
            font-size: 1.5em;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        .narrative-transition p {
            margin: 0;
            line-height: 1.8;
            color: #2c3e50;
            font-size: 1.15em;
            font-weight: 400;
        }
        .executive-content, .story-content, .conclusion-content {
            line-height: 1.8;
            color: #34495e;
            font-size: 1.1em;
        }
        .executive-content p, .story-content p, .conclusion-content p {
            margin-bottom: 1.2em;
        }
        .executive-content p:last-child, .story-content p:last-child, .conclusion-content p:last-child {
            margin-bottom: 0;
        }
        .executive-content h3, .story-content h3, .conclusion-content h3 {
            color: #2c3e50;
            font-size: 1.3em;
            font-weight: 600;
            margin-top: 1.5em;
            margin-bottom: 0.8em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.3em;
        }
        .executive-content h4, .story-content h4, .conclusion-content h4 {
            color: #34495e;
            font-size: 1.1em;
            font-weight: 600;
            margin-top: 1.2em;
            margin-bottom: 0.6em;
        }
        .executive-content ul, .story-content ul, .conclusion-content ul {
            margin-left: 1.5em;
            margin-bottom: 1em;
        }
        .executive-content li, .story-content li, .conclusion-content li {
            margin-bottom: 0.5em;
        }
        .narrative-flow-content {
            line-height: 1.8;
            color: #34495e;
            font-size: 1.1em;
        }
        .narrative-flow-content p {
            margin-bottom: 1.2em;
        }
        .narrative-flow-content h3, .narrative-flow-content h4 {
            color: #2c3e50;
            font-size: 1.3em;
            font-weight: 600;
            margin-top: 1.5em;
            margin-bottom: 0.8em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.3em;
        }
        .key-takeaways-content {
            margin-top: 20px;
        }
        .takeaway-item {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 4px solid #f39c12;
        }
        .takeaway-item h4 {
            color: #2c3e50;
            font-size: 1.2em;
            font-weight: 600;
            margin-top: 0;
            margin-bottom: 15px;
        }
        .takeaway-content {
            line-height: 1.7;
            color: #34495e;
        }
        .takeaway-content p {
            margin-bottom: 1em;
        }
        .conclusion-summary {
            background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%);
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            border-left: 4px solid #28a745;
        }
        .conclusion-summary h4 {
            color: #2c3e50;
            font-size: 1.2em;
            font-weight: 600;
            margin-top: 0;
            margin-bottom: 15px;
        }
        .conclusion-summary p {
            line-height: 1.7;
            color: #34495e;
            margin: 0;
        }
        .report-footer {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px 20px;
            margin-top: 60px;
            text-align: center;
            border-radius: 15px 15px 0 0;
        }
        .footer-content p {
            margin: 8px 0;
            font-size: 0.95em;
            opacity: 0.9;
        }
        @media (max-width: 768px) {
            body {
                padding: 10px;
            }
            .header h1 {
                font-size: 2em;
            }
            .section, .chart-container {
                padding: 20px;
            }
        }
        
        /* Collapsible Section Styles */
        .collapsible-section {
            margin: 20px 0;
        }
        .collapsible-btn {
            width: 100%;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 2px solid #dee2e6;
            border-radius: 10px;
            padding: 15px 20px;
            text-align: left;
            cursor: pointer;
            font-size: 1em;
            font-weight: 500;
            color: #495057;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .collapsible-btn:hover {
            background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
            border-color: #adb5bd;
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        .collapsible-btn:active {
            transform: translateY(0);
        }
        .expand-icon {
            font-size: 0.8em;
            transition: transform 0.3s ease;
            color: #6c757d;
        }
        .collapsible-btn.expanded .expand-icon {
            transform: rotate(90deg);
        }
        .btn-text {
            flex: 1;
        }
        .collapsible-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
            background: white;
            border-radius: 0 0 10px 10px;
            border: 2px solid #dee2e6;
            border-top: none;
        }
        .collapsible-content.expanded {
            max-height: 2000px;
            transition: max-height 0.5s ease-in;
        }
        .collapsible-content > div {
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="scroll-indicator">
        <div class="scroll-progress" id="scrollProgress"></div>
    </div>
""")
        
        # Dynamic Header
        dynamic_title = state.get("dynamic_title", "Data Story: Interactive Visualization Report")
        html_content.append(f"""
    <div class="header">
        <h1>{dynamic_title}</h1>
        <p>An AI-powered narrative journey through your data insights</p>
    </div>
""")
        
        # Table of Contents
        successful_charts = [c for c in charts_data if c.get("success")]
        if successful_charts:
            html_content.append("""
    <div class="toc">
        <h3>Table of Contents</h3>
        <ul>
            <li><a href="#executive-summary">Executive Summary</a></li>
            <li><a href="#data-quality">Data Quality Assessment</a></li>
            <li><a href="#data-story">What the data illustrates</a>
                <ul>
""")
            
            # Add chart sub-headings to table of contents
            for i, chart_info in enumerate(successful_charts):
                chart_title = f"Chart {i+1}: {chart_info['chart_name']}"
                html_content.append(f"""
                    <li><a href="#chart-{chart_info['rank']}">{chart_title}</a></li>
""")
            
            html_content.append("""
                </ul>
            </li>
            <li><a href="#key-takeaways">Key Takeaways</a></li>
        </ul>
    </div>
""")
        
        # Executive Summary Section
        executive_summary = state.get("executive_summary")
        if executive_summary:
            # Clean up any markdown formatting that might be causing bold text
            clean_summary = executive_summary.replace('**', '').replace('*', '').strip()
            html_content.append(f"""
    <div class="executive-summary" id="executive-summary">
        <h3>Executive Summary</h3>
        <div class="executive-content">
            <p>{clean_summary}</p>
        </div>
    </div>
""")
        
        # Data Quality Assessment Section (Collapsible)
        data_quality_table = state.get("data_quality_table")
        if data_quality_table:
            html_content.append(f"""
    <div class="section" id="data-quality">
        <h2>Data Quality Assessment</h2>
                    <div class="data-quality-summary">
                <h4>Summary Statistics</h4>
            <div class="summary-stats">
                <div class="stat-item">
                    <span class="stat-number">{data_quality_table['summary_stats']['total_fields']}</span>
                    <span class="stat-label">Total Fields</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{data_quality_table['summary_stats']['high_quality_fields']}</span>
                    <span class="stat-label">High Quality</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{data_quality_table['summary_stats']['medium_quality_fields']}</span>
                    <span class="stat-label">Medium Quality</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{data_quality_table['summary_stats']['low_quality_fields']}</span>
                    <span class="stat-label">Low Quality</span>
                </div>
            </div>
        </div>
        
        <!-- Collapsible Data Quality Details -->
        <div class="collapsible-section">
            <button class="collapsible-btn" onclick="toggleCollapsible('data-quality-details')">
                <span class="expand-icon">▶</span>
                <span class="btn-text">Expand to see schema of data and data quality</span>
            </button>
            <div class="collapsible-content" id="data-quality-details">
                <div class="data-quality-table-container">
                    <h4>Detailed Field Analysis</h4>
                    <div class="table-wrapper">
                        <table class="data-quality-table">
                            <thead>
                                <tr>
                                    <th>Field</th>
                                    <th>Data Type</th>
                                    <th>Missing Values</th>
                                    <th>Unique Values</th>
                                    <th>Quality Score</th>
                                    <th>Issues</th>
                                </tr>
                            </thead>
                            <tbody>
""")
            
            for row in data_quality_table['table_data']:
                quality_class = "high-quality" if float(row["Quality Score"]) >= 0.8 else "medium-quality" if float(row["Quality Score"]) >= 0.5 else "low-quality"
                html_content.append(f"""
                                <tr class="{quality_class}">
                                    <td><strong>{row['Field']}</strong></td>
                                    <td>{row['Data Type']}</td>
                                    <td>{row['Missing Values']}</td>
                                    <td>{row['Unique Values']}</td>
                                    <td><span class="quality-score">{row['Quality Score']}</span></td>
                                    <td>{row['Issues']}</td>
                                </tr>
""")
            
            html_content.append("""
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
""")
        

        # Charts Section
        html_content.append("""
    <div class="section" id="data-story">
        <h2>What the data illustrates</h2>
""")
        
        # Filter only successful charts
        successful_charts = [c for c in charts_data if c.get("success")]
        
        # Generate narrative flow snippets for each chart
        narrative_snippets = generate_narrative_snippets_for_charts(successful_charts, chart_recommendations)
        
        for i, chart_info in enumerate(successful_charts):
            chart_id = f"chart-{chart_info['rank']}"
            
            # Find corresponding AI insight
            ai_insight = None
            if narrative_insights and narrative_insights.get("chart_insights"):
                for insight in narrative_insights["chart_insights"]:
                    if insight.get("chart_rank") == chart_info.get("rank"):
                        ai_insight = insight
                        break
            
            # Create dynamic title for each chart
            dynamic_chart_title = f"Chart {i+1}: {chart_info['chart_name']}"
            
            # Blend narrative snippet and AI insight
            narrative_text = ""
            if i < len(narrative_snippets):
                narrative_text = narrative_snippets[i]
            
            # Add AI insight to narrative if available
            if ai_insight and ai_insight.get("ai_insight"):
                if narrative_text:
                    narrative_text += " " + ai_insight['ai_insight']
                else:
                    narrative_text = ai_insight['ai_insight']
            
            html_content.append(f"""
        <div class="chart-container" id="{chart_id}">
            <h3>{dynamic_chart_title}</h3>
""")
            
            # Add blended narrative above chart
            if narrative_text:
                html_content.append(f"""
            <div class="narrative-transition">
                <p>{narrative_text}</p>
            </div>
""")
            
            # Add chart plot first
            html_content.append(f"""
            <div class="chart-plot" id="plot-{chart_info['rank']}">
                <!-- Chart will be embedded here -->
            </div>
            
            <div class="chart-description">
                <h4>Agent's reasoning for developing the visual</h4>
                <p>{chart_info.get('why_this_chart_helps', 'N/A')}</p>
                <h4>Chart Type</h4>
                <p>{chart_info.get('chart_type', 'N/A')}</p>
                <h4>Fields Used</h4>
                <p>{', '.join(chart_info.get('fields_required', []))}</p>
            </div>
        </div>
    """)
        
        html_content.append("""
    </div>
""")
        
        # Key Takeaways Section (blended with conclusion and AI insights)
        conclusion = state.get("conclusion")
        key_takeaways = narrative_insights.get("key_takeaways", []) if narrative_insights else []
        
        if conclusion or key_takeaways:
            # Generate blended summary using LLM
            blended_summary = generate_blended_key_takeaways(conclusion, key_takeaways)
            
            html_content.append(f"""
    <div class="section" id="key-takeaways">
        <h2>Key Takeaways</h2>
        <div class="key-takeaways-content">
            <div class="blended-summary">
                <p>{blended_summary}</p>
            </div>
        </div>
    </div>
""")
        
        # JavaScript to embed charts
        html_content.append("""
    <script>
        // Collapsible functionality
        function toggleCollapsible(elementId) {
            const content = document.getElementById(elementId);
            const button = content.previousElementSibling;
            const icon = button.querySelector('.expand-icon');
            
            if (content.classList.contains('expanded')) {
                // Collapse
                content.classList.remove('expanded');
                button.classList.remove('expanded');
                icon.textContent = '▶';
            } else {
                // Expand
                content.classList.add('expanded');
                button.classList.add('expanded');
                icon.textContent = '▼';
            }
        }
        
        // Scroll progress indicator
        window.addEventListener('scroll', function() {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
            const scrollPercent = (scrollTop / scrollHeight) * 100;
            document.getElementById('scrollProgress').style.width = scrollPercent + '%';
        });
        
        // Smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
        
        // Function to embed Plotly charts
        function embedCharts() {
""")
        
        for chart_info in successful_charts:
            if chart_info.get("plotly_figure"):
                try:
                    # Convert Plotly figure to JSON
                    fig_json = chart_info["plotly_figure"].to_json()
                    html_content.append(f"""
            // Embed chart {chart_info['rank']}
            try {{
                Plotly.newPlot('plot-{chart_info["rank"]}', {fig_json}.data, {fig_json}.layout, {{responsive: true, displayModeBar: true}});
            }} catch (error) {{
                console.error('Error embedding chart {chart_info["rank"]}:', error);
                document.getElementById('plot-{chart_info["rank"]}').innerHTML = '<p style="color: #e74c3c; text-align: center; padding: 20px;">Chart could not be loaded. Please refresh the page.</p>';
            }}
""")
                except Exception as e:
                    html_content.append(f"""
            // Chart {chart_info['rank']} failed to serialize
            document.getElementById('plot-{chart_info["rank"]}').innerHTML = '<p style="color: #e74c3c; text-align: center; padding: 20px;">Chart data could not be loaded.</p>';
""")
        
        html_content.append("""
        }
        
        // Load charts when page is ready
        document.addEventListener('DOMContentLoaded', function() {
            embedCharts();
        });
    </script>
    
    <!-- Footer -->
    <footer class="report-footer">
        <div class="footer-content">
            <p>📅 Report generated on """ + datetime.now().strftime('%B %d, %Y at %I:%M %p') + """</p>
            <p>🔧 Powered by AI-Enhanced Data Analysis</p>
        </div>
    </footer>
</body>
</html>
""")
        
        # Write HTML file
        print("writing html file...in",html_path)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(''.join(html_content))

        with open(os.path.join(script_path, "output.html"), 'w', encoding='utf-8') as f:
            f.write(''.join(html_content))

        
        return html_path
        
    except Exception as e:
        print(f"Error generating HTML report: {e}")
        return None


# ---------- Workflow Definition ----------
def create_data_quality_workflow() -> StateGraph:
    """Create the LangGraph workflow for data quality analysis."""
    
    workflow = StateGraph(DataQualityState)
    
    # Add nodes
    workflow.add_node("analyze_dataset_and_quality", analyze_dataset_and_quality)
    workflow.add_node("generate_chart_recommendations", generate_chart_recommendations)
    workflow.add_node("create_final_report", create_final_report)
    workflow.add_node("generate_plotly_charts", generate_plotly_charts)
    workflow.add_node("generate_narrative_insights", generate_narrative_insights)
    
    # Add new enhanced report nodes
    workflow.add_node("generate_dynamic_title", generate_dynamic_report_title)
    workflow.add_node("generate_executive_summary", generate_executive_summary_section)
    workflow.add_node("create_data_quality_table", create_data_quality_assessment_table)
    workflow.add_node("generate_enhanced_narrative", generate_enhanced_chart_narrative)
    workflow.add_node("generate_conclusion", generate_conclusion_section)
    workflow.add_node("generate_final_html", generate_final_html_report)
    
    # Add feedback loop node
    workflow.add_node("implement_chart_feedback_loop", implement_chart_feedback_loop)
    
    # Define the flow
    workflow.set_entry_point("analyze_dataset_and_quality")
    workflow.add_edge("analyze_dataset_and_quality", "generate_chart_recommendations")
    workflow.add_edge("generate_chart_recommendations", "create_final_report")
    workflow.add_edge("create_final_report", "generate_plotly_charts")
    workflow.add_edge("generate_plotly_charts", "implement_chart_feedback_loop")  # Add feedback loop after chart generation
    workflow.add_edge("implement_chart_feedback_loop", "generate_narrative_insights")
    workflow.add_edge("generate_narrative_insights", "generate_dynamic_title")
    workflow.add_edge("generate_dynamic_title", "create_data_quality_table")
    workflow.add_edge("create_data_quality_table", "generate_executive_summary")
    workflow.add_edge("generate_executive_summary", "generate_enhanced_narrative")
    workflow.add_edge("generate_enhanced_narrative", "generate_conclusion")
    workflow.add_edge("generate_conclusion", "generate_final_html")
    workflow.set_finish_point("generate_final_html")
    
    return workflow.compile()


# ---------- Agent Wrapper ----------
class Agent:
    def __init__(self):
        self.workflow = None

    def initialize(self):
        """Initialize the data quality analysis workflow."""
        self.workflow = create_data_quality_workflow()

    def process(self) -> Dict[str, Any]:
        """Process the dataset and return the complete analysis report."""
        if not self.workflow:
            raise ValueError("Workflow not initialized. Call initialize() first.")
        
        # Initialize state
        initial_state = {
            "dataset_info": {},
            "data_quality_report": None,
            "chart_recommendations": None,
            "final_report": None,
            "generated_charts": None,
            "narrative_insights": None,
            "dynamic_title": None,
            "executive_summary": None,
            "data_quality_table": None,
            "enhanced_chart_narrative": None,
            "conclusion": None,
            "final_html_path": None
        }
        
        # Run the workflow
        result = self.workflow.invoke(initial_state)

        print("Agent successfully has completed running report...")
        
        # Return the complete result
        return result


# ---------- Main ----------
def main():
    agent = Agent()
    agent.initialize()
    result = agent.process()


# ---------- Utility Functions ----------
def analyze_chart_data_for_insights(chart_data, chart_type: str) -> str:
    """Analyze chart data and extract specific insights for LLM analysis."""
    
    if chart_data is None:
        return "No data available for analysis."
    
    try:
        if isinstance(chart_data, pd.DataFrame):
            if chart_data.empty:
                return "DataFrame is empty."
            
            # Analyze based on chart type
            if "bar" in chart_type.lower():
                return analyze_bar_chart_data(chart_data)
            elif "line" in chart_type.lower() or "time series" in chart_type.lower():
                return analyze_line_chart_data(chart_data)
            elif "scatter" in chart_type.lower():
                return analyze_scatter_chart_data(chart_data)
            elif "correlation" in chart_type.lower() or "heatmap" in chart_type.lower():
                return analyze_correlation_data(chart_data)
            elif "box" in chart_type.lower():
                return analyze_box_chart_data(chart_data)
            elif "histogram" in chart_type.lower():
                return analyze_histogram_data(chart_data)
            else:
                return analyze_generic_data(chart_data)
                
        elif isinstance(chart_data, pd.Series):
            if chart_data.empty:
                return "Series is empty."
            return analyze_series_data(chart_data)
        else:
            return f"Data type: {type(chart_data).__name__}, Value: {str(chart_data)[:200]}"
            
    except Exception as e:
        return f"Error analyzing data: {str(e)}"


def analyze_bar_chart_data(df: pd.DataFrame) -> str:
    """Analyze bar chart data for insights."""
    try:
        if len(df.columns) >= 2:
            x_col = df.columns[0]
            y_col = df.columns[1]
            
            # Find top and bottom values
            top_values = df.nlargest(3, y_col)
            bottom_values = df.nsmallest(3, y_col)
            
            analysis = f"""
Bar Chart Analysis:
- Total categories: {len(df)}
- Highest value: {top_values.iloc[0][x_col]} = {top_values.iloc[0][y_col]:.2f}
- Second highest: {top_values.iloc[1][x_col]} = {top_values.iloc[1][y_col]:.2f}
- Third highest: {top_values.iloc[2][x_col]} = {top_values.iloc[2][y_col]:.2f}
- Lowest value: {bottom_values.iloc[0][x_col]} = {bottom_values.iloc[0][y_col]:.2f}
- Range: {df[y_col].max() - df[y_col].min():.2f}
- Average: {df[y_col].mean():.2f}
"""
            return analysis
    except Exception:
        pass
    
    return f"Bar chart with {len(df)} data points. Columns: {list(df.columns)}"


def analyze_line_chart_data(df: pd.DataFrame) -> str:
    """Analyze line chart data for insights."""
    try:
        if len(df.columns) >= 2:
            x_col = df.columns[0]
            y_col = df.columns[1]
            
            # Calculate trends
            df_sorted = df.sort_values(x_col)
            first_value = df_sorted.iloc[0][y_col]
            last_value = df_sorted.iloc[-1][y_col]
            change = last_value - first_value
            change_pct = (change / first_value * 100) if first_value != 0 else 0
            
            analysis = f"""
Line Chart Analysis:
- Time period: {df_sorted.iloc[0][x_col]} to {df_sorted.iloc[-1][x_col]}
- Starting value: {first_value:.2f}
- Ending value: {last_value:.2f}
- Total change: {change:.2f} ({change_pct:+.1f}%)
- Peak value: {df[y_col].max():.2f}
- Lowest value: {df[y_col].min():.2f}
- Trend: {'Increasing' if change > 0 else 'Decreasing' if change < 0 else 'Stable'}
"""
            return analysis
    except Exception:
        pass
    
    return f"Line chart with {len(df)} data points. Columns: {list(df.columns)}"


def analyze_scatter_chart_data(df: pd.DataFrame) -> str:
    """Analyze scatter plot data for insights."""
    try:
        if len(df.columns) >= 2:
            x_col = df.columns[0]
            y_col = df.columns[1]
            
            # Calculate correlation
            correlation = df[x_col].corr(df[y_col])
            
            analysis = f"""
Scatter Plot Analysis:
- Data points: {len(df)}
- X-axis range: {df[x_col].min():.2f} to {df[x_col].max():.2f}
- Y-axis range: {df[y_col].min():.2f} to {df[y_col].max():.2f}
- Correlation: {correlation:.3f} ({'Strong positive' if correlation > 0.7 else 'Moderate positive' if correlation > 0.3 else 'Weak positive' if correlation > 0 else 'Strong negative' if correlation < -0.7 else 'Moderate negative' if correlation < -0.3 else 'Weak negative' if correlation < 0 else 'No correlation'})
- X mean: {df[x_col].mean():.2f}
- Y mean: {df[y_col].mean():.2f}
"""
            return analysis
    except Exception:
        pass
    
    return f"Scatter plot with {len(df)} data points. Columns: {list(df.columns)}"


def analyze_correlation_data(df: pd.DataFrame) -> str:
    """Analyze correlation matrix data for insights."""
    try:
        analysis = f"""
Correlation Matrix Analysis:
- Matrix size: {df.shape[0]}x{df.shape[1]}
- Variables: {list(df.columns)}
- Strongest positive correlation: {find_strongest_correlation(df, positive=True)}
- Strongest negative correlation: {find_strongest_correlation(df, positive=False)}
- Average correlation: {df.values[np.triu_indices_from(df.values, k=1)].mean():.3f}
"""
        return analysis
    except Exception:
        pass
    
    return f"Correlation matrix with {df.shape[0]} variables"


def find_strongest_correlation(df: pd.DataFrame, positive: bool = True) -> str:
    """Find the strongest correlation in the matrix."""
    try:
        import numpy as np
        # Get upper triangle of correlation matrix
        upper_triangle = df.values[np.triu_indices_from(df.values, k=1)]
        indices = np.triu_indices_from(df.values, k=1)
        
        if positive:
            max_idx = np.argmax(upper_triangle)
        else:
            max_idx = np.argmin(upper_triangle)
        
        i, j = indices[0][max_idx], indices[1][max_idx]
        return f"{df.index[i]} vs {df.columns[j]} = {upper_triangle[max_idx]:.3f}"
    except Exception:
        return "Unable to calculate"


def analyze_box_chart_data(df: pd.DataFrame) -> str:
    """Analyze box plot data for insights."""
    try:
        if len(df.columns) >= 2:
            group_col = df.columns[0]
            value_col = df.columns[1]
            
            # Group statistics
            group_stats = df.groupby(group_col)[value_col].agg(['count', 'mean', 'std', 'min', 'max'])
            
            analysis = f"""
Box Plot Analysis:
- Groups: {len(group_stats)}
- Group with highest mean: {group_stats['mean'].idxmax()} ({group_stats['mean'].max():.2f})
- Group with lowest mean: {group_stats['mean'].idxmin()} ({group_stats['mean'].min():.2f})
- Group with most data: {group_stats['count'].idxmax()} ({group_stats['count'].max()} points)
- Overall range: {df[value_col].min():.2f} to {df[value_col].max():.2f}
- Overall mean: {df[value_col].mean():.2f}
"""
            return analysis
    except Exception:
        pass
    
    return f"Box plot with {len(df)} data points. Columns: {list(df.columns)}"


def analyze_histogram_data(df: pd.DataFrame) -> str:
    """Analyze histogram data for insights."""
    try:
        if len(df.columns) >= 1:
            value_col = df.columns[0]
            
            analysis = f"""
Histogram Analysis:
- Data points: {len(df)}
- Range: {df[value_col].min():.2f} to {df[value_col].max():.2f}
- Mean: {df[value_col].mean():.2f}
- Median: {df[value_col].median():.2f}
- Standard deviation: {df[value_col].std():.2f}
- Distribution shape: {'Skewed right' if df[value_col].skew() > 0.5 else 'Skewed left' if df[value_col].skew() < -0.5 else 'Normal'}
"""
            return analysis
    except Exception:
        pass
    
    return f"Histogram with {len(df)} data points. Columns: {list(df.columns)}"


def analyze_series_data(series: pd.Series) -> str:
    """Analyze series data for insights."""
    try:
        analysis = f"""
Series Analysis:
- Length: {len(series)}
- Range: {series.min():.2f} to {series.max():.2f}
- Mean: {series.mean():.2f}
- Median: {series.median():.2f}
- Most common value: {series.mode().iloc[0] if not series.mode().empty else 'N/A'}
- Unique values: {series.nunique()}
"""
        return analysis
    except Exception:
        pass
    
    return f"Series with {len(series)} values"


def analyze_generic_data(df: pd.DataFrame) -> str:
    """Analyze generic data for insights."""
    try:
        analysis = f"""
Generic Data Analysis:
- Shape: {df.shape}
- Columns: {list(df.columns)}
- Data types: {df.dtypes.to_dict()}
- Missing values: {df.isnull().sum().sum()}
- Sample data:
{df.head(3).to_string()}
"""
        return analysis
    except Exception:
        pass
    
    return f"Data with shape {df.shape}"


# ---------- Enhanced Report Structure Functions ----------

def generate_dynamic_report_title(state: DataQualityState) -> DataQualityState:
    """Generate a dynamic, compelling report title using LLM based on the content."""
    
    chart_recommendations = state.get("chart_recommendations", {})
    data_quality_report = state.get("data_quality_report", {})
    
    if not chart_recommendations or not chart_recommendations.get("chart_recommendations"):
        print("No chart recommendations available for title generation")
        state["dynamic_title"] = "Data Analysis Report"
        return state
    
    if llm is None:
        print("LLM not available, using fallback title")
        state["dynamic_title"] = "Comprehensive Data Analysis Report"
        return state
    
    try:
        title_prompt = f"""
You are an expert data storyteller and report writer. Create a compelling, dynamic title for a data analysis report based on the following information.

## Chart Recommendations:
{format_charts_for_title(chart_recommendations.get("chart_recommendations", []))}

## Dataset Overview:
- Total rows: {data_quality_report.get('dataset_overview', {}).get('total_rows', 'N/A')}
- Total columns: {data_quality_report.get('dataset_overview', {}).get('total_columns', 'N/A')}
- Data types: {', '.join(data_quality_report.get('data_types_summary', {}).get('numeric_columns', []) + data_quality_report.get('data_types_summary', {}).get('categorical_columns', []))}

## Your Task:
Create a compelling, professional title that:
1. **Captures the essence** of what the data reveals
2. **Is specific and descriptive** rather than generic
3. **Hints at the insights** that will be uncovered
4. **Uses engaging language** that draws readers in
5. **Is appropriate for business/analytics context**

## Title Guidelines:
- Keep it under 80 characters
- Use action words and descriptive terms
- Avoid generic terms like "Analysis" or "Report" unless necessary
- Make it specific to the data domain and insights
- Consider using a colon to separate main title from subtitle

## Examples of good titles:
- "Unveiling Hidden Patterns: A Deep Dive into Performance Metrics"
- "The Rise and Fall: Temporal Trends in User Engagement"
- "Beyond the Numbers: Strategic Insights from Customer Behavior"
- "Correlation Chronicles: Discovering Relationships in Complex Data"

Create a compelling title that would make someone want to read this report. **DO NOT include any meta-commentary, word counts, internal reasoning, or chain-of-thought text.**
"""

        response = llm.invoke(title_prompt)
        title = response.content if hasattr(response, 'content') else str(response)
        
        # Clean up the title
        title = title.strip()
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1]
        if title.startswith("Title: "):
            title = title[7:]
        
        state["dynamic_title"] = title
        print(f"Generated dynamic title: {title}")
        
    except Exception as e:
        print(f"Error generating dynamic title: {e}")
        state["dynamic_title"] = "Data Analysis Report"
    
    return state


def format_charts_for_title(charts: List[Dict]) -> str:
    """Format charts for title generation."""
    
    formatted = []
    for chart in charts[:3]:  # Use first 3 charts for title generation
        formatted.append(f"""
• {chart.get('chart_name', 'N/A')} - {chart.get('why_this_chart_helps', 'N/A')}
""")
    
    return "\n".join(formatted)


def generate_executive_summary_section(state: DataQualityState) -> DataQualityState:
    """Generate a comprehensive executive summary section using AI insights from each chart."""
    
    generated_charts = state.get("generated_charts", {})
    narrative_insights = state.get("narrative_insights", {})
    
    if not generated_charts or not generated_charts.get("charts"):
        print("No generated charts available for executive summary")
        state["executive_summary"] = "No data available for executive summary generation."
        return state
    
    # Get successful charts and their AI insights
    successful_charts = [c for c in generated_charts["charts"] if c.get("success")]
    chart_insights = narrative_insights.get("chart_insights", []) if narrative_insights else []
    
    if not successful_charts:
        print("No successful charts available for executive summary")
        state["executive_summary"] = "No charts were successfully generated for executive summary."
        return state
    
    if llm is None:
        print("LLM not available, using fallback executive summary")
        # Compile insights from AI insights
        if chart_insights:
            insights_summary = []
            for insight in chart_insights[:4]:  # Use first 4 insights for bullet points
                insights_summary.append(f"• {insight.get('ai_insight', 'Provides key insights')[:50]}...")
            
            state["executive_summary"] = f"""• This comprehensive data analysis reveals critical insights through {len(successful_charts)} carefully crafted visualizations examining key patterns and relationships in the data.

• {' '.join(insights_summary[:2])}

• Key findings demonstrate significant patterns and relationships that provide actionable insights for strategic decision-making and data understanding.

• The data quality assessment confirms the reliability of our analysis with comprehensive field validation and quality metrics."""
        else:
            state["executive_summary"] = """• This comprehensive data analysis provides valuable insights into the dataset through multiple visualization approaches.

• The analysis includes data quality assessment, chart generation, and narrative insights to help understand key patterns and trends.

• Key findings demonstrate significant relationships and patterns that provide actionable insights for data understanding and decision-making.

• The analysis framework ensures reliable and comprehensive examination of all data aspects."""
        return state
    
    try:
        # Format AI insights for the executive summary
        formatted_insights = format_ai_insights_for_executive(chart_insights)
        
        executive_prompt = f"""
You are a senior data analyst presenting detailed insights from the data to a conference audience. 
Create a concise executive summary based on these data visualizations.

## AI Insights from Generated Charts:
{formatted_insights}

## Chart Overview:
- Total charts analyzed: {len(successful_charts)}
- Charts with AI insights: {len(chart_insights)}

## Your Task:
Create a concise executive summary with 4-5 bullet points that:

1. **Captures the most critical insights** from the AI analysis
2. **Highlights key patterns and trends** that matter to the audience
3. **Identifies important implications** for the domain/field
4. **Provides clear, actionable insights** for understanding the data

## Requirements:
- **Format**: 4-5 bullet points, each 40-50 words
- **Content**: Focus on the most important findings from AI insights
- **Style**: Clear, professional language appropriate for a conference audience
- **Structure**: Each bullet should be a complete, standalone insight
- **Focus**: Domain-specific implications and patterns

## Style Guidelines:
- Use bullet point format (• or -)
- Each bullet should be 40-50 words maximum
- Focus on insights that would be most valuable to the audience
- Use clear, professional language
- Avoid technical jargon unless domain-specific
- **DO NOT include any meta-commentary, word counts, internal reasoning, or chain-of-thought text**

Write 4-5 concise bullet points that capture the most important insights from the data analysis. Provide only the bullet points, no additional commentary. Do not use markdown formatting like **bold** or *italic* text.
"""

        response = llm.invoke(executive_prompt)
        executive_summary = response.content if hasattr(response, 'content') else str(response)
        
        state["executive_summary"] = executive_summary
        print("Executive summary generated successfully from AI insights")
        
    except Exception as e:
        print(f"Error generating executive summary: {e}")
        state["executive_summary"] = "Unable to generate executive summary due to an error."
    
    return state


def generate_blended_key_takeaways(conclusion: str, key_takeaways: List[str]) -> str:
    """Generate a blended summary of conclusion and key takeaways using LLM."""
    
    if not conclusion and not key_takeaways:
        return "No key takeaways available."
    
    if llm is None:
        # Fallback: combine conclusion and takeaways
        combined_text = ""
        if conclusion:
            combined_text += conclusion + "\n\n"
        if key_takeaways:
            combined_text += "Key Takeaways:\n" + "\n".join([f"• {takeaway}" for takeaway in key_takeaways])
        return combined_text
    
    try:
        # Prepare input text
        input_text = ""
        if conclusion:
            input_text += f"Conclusion: {conclusion}\n\n"
        if key_takeaways:
            input_text += f"Key Takeaways:\n" + "\n".join([f"• {takeaway}" for takeaway in key_takeaways])
        
        blend_prompt = f"""
You are a data analyst creating a comprehensive summary that blends conclusion findings with key takeaways.

## Input Content:
{input_text}

## Your Task:
Create a cohesive, well-structured summary that:
1. **Synthesizes the most important insights** from both the conclusion and key takeaways
2. **Eliminates redundancy** and creates a unified narrative
3. **Maintains the most valuable insights** from both sources
4. **Creates a logical flow** that tells a complete story
5. **Uses clear, professional language** appropriate for a conference audience

## Requirements:
- Write 2-3 paragraphs (approximately 200-300 words)
- Focus on the most critical findings and implications
- Create a coherent narrative that flows naturally
- Remove any repetitive or conflicting information
- **DO NOT include any meta-commentary, word counts, internal reasoning, or chain-of-thought text**

Write a blended summary that effectively combines the conclusion and key takeaways into a unified, insightful narrative.
"""

        response = llm.invoke(blend_prompt)
        blended_text = response.content if hasattr(response, 'content') else str(response)
        
        return blended_text
        
    except Exception as e:
        print(f"Error generating blended key takeaways: {e}")
        # Fallback: return original content
        fallback_text = ""
        if conclusion:
            fallback_text += conclusion + "\n\n"
        if key_takeaways:
            fallback_text += "Key Takeaways:\n" + "\n".join([f"• {takeaway}" for takeaway in key_takeaways])
        return fallback_text


def format_ai_insights_for_executive(chart_insights: List[Dict]) -> str:
    """Format AI insights for executive summary generation."""
    
    if not chart_insights:
        return "No AI insights available."
    
    formatted = []
    for insight in chart_insights:
        chart_name = insight.get('chart_name', 'Unknown Chart')
        ai_insight = insight.get('ai_insight', 'No insight available')
        chart_rank = insight.get('chart_rank', 0)
        
        formatted.append(f"""
Chart {chart_rank}: {chart_name}
AI Insight: {ai_insight}
""")
    
    return "\n".join(formatted)


def format_charts_for_executive_summary(charts: List[Dict]) -> str:
    """Format charts for executive summary generation."""
    
    formatted = []
    for chart in charts:
        formatted.append(f"""
• {chart.get('chart_name', 'N/A')} - {chart.get('key_insights', 'N/A')}
""")
    
    return "\n".join(formatted)


def format_successful_charts_for_executive(charts: List[Dict]) -> str:
    """Format successful charts for executive summary."""
    
    if not charts:
        return "No charts were successfully generated."
    
    formatted = []
    for chart in charts:
        formatted.append(f"""
• {chart.get('chart_name', 'N/A')} - Successfully generated
""")
    
    return "\n".join(formatted)


def create_data_quality_assessment_table(state: DataQualityState) -> DataQualityState:
    """Create a data quality assessment table for the report."""
    
    data_quality_report = state.get("data_quality_report", {})
    
    if not data_quality_report:
        print("No data quality report available for assessment table")
        state["data_quality_table"] = None
        return state
    
    try:
        # Extract relevant information for the table
        column_analysis = data_quality_report.get("column_analysis", {})
        
        # Create table data
        table_data = []
        for col, analysis in column_analysis.items():
            table_data.append({
                "Field": col,
                "Data Type": analysis.get("data_type", "N/A"),
                "Missing Values": f"{analysis.get('missing_count', 0)} ({analysis.get('missing_percentage', 0):.1f}%)",
                "Unique Values": analysis.get("unique_values", 0),
                "Quality Score": f"{analysis.get('quality_score', 0):.3f}",
                "Issues": ", ".join(analysis.get("potential_issues", []))[:50] + "..." if len(", ".join(analysis.get("potential_issues", []))) > 50 else ", ".join(analysis.get("potential_issues", []))
            })
        
        # Sort by quality score (descending)
        table_data.sort(key=lambda x: float(x["Quality Score"]), reverse=True)
        
        state["data_quality_table"] = {
            "table_data": table_data,
            "summary_stats": {
                "total_fields": len(table_data),
                "high_quality_fields": len([row for row in table_data if float(row["Quality Score"]) >= 0.8]),
                "medium_quality_fields": len([row for row in table_data if 0.5 <= float(row["Quality Score"]) < 0.8]),
                "low_quality_fields": len([row for row in table_data if float(row["Quality Score"]) < 0.5]),
                "fields_with_missing_data": len([row for row in table_data if row["Missing Values"] != "0 (0.0%)"])
            }
        }
        
        print("Data quality assessment table created successfully")
        
    except Exception as e:
        print(f"Error creating data quality assessment table: {e}")
        state["data_quality_table"] = None
    
    return state


def generate_conclusion_section(state: DataQualityState) -> DataQualityState:
    """Generate a comprehensive conclusion section using LLM."""
    
    chart_recommendations = state.get("chart_recommendations", {})
    generated_charts = state.get("generated_charts", {})
    narrative_insights = state.get("narrative_insights", {})
    data_quality_report = state.get("data_quality_report", {})
    
    if not chart_recommendations or not chart_recommendations.get("chart_recommendations"):
        print("No chart recommendations available for conclusion")
        state["conclusion"] = "No data available for conclusion generation."
        return state
    
    if llm is None:
        print("LLM not available, using fallback conclusion")
        state["conclusion"] = "This analysis has provided comprehensive insights into the dataset. The data quality assessment, chart visualizations, and narrative analysis work together to reveal important patterns and trends. Future analysis should focus on exploring additional relationships and conducting deeper statistical analysis."
        return state
    
    try:
        # Get successful charts
        successful_charts = []
        if generated_charts and generated_charts.get("charts"):
            successful_charts = [c for c in generated_charts["charts"] if c.get("success")]
        
        conclusion_prompt = f"""
You are a senior data analyst writing the conclusion section of a comprehensive data analysis report. Create a compelling conclusion based on the analysis performed.

## Chart Recommendations:
{format_charts_for_conclusion(chart_recommendations.get("chart_recommendations", []))}

## Generated Charts:
{format_successful_charts_for_conclusion(successful_charts)}

## Data Quality Assessment:
- Total fields analyzed: {len(data_quality_report.get('column_analysis', {}))}
- Missing data percentage: {data_quality_report.get('dataset_overview', {}).get('missing_values_percentage', 0):.1f}%
- Data quality issues: {len(data_quality_report.get('data_quality_issues', []))}

## Your Task:
Write a comprehensive conclusion (3-4 paragraphs) that:

1. **Summarizes the key takeaways** from the entire analysis
2. **Highlights the most important insights** and their business implications
3. **Acknowledges limitations and shortcomings** of the analysis
4. **Suggests next steps and future analysis directions**
5. **Provides actionable recommendations** for stakeholders
6. **Ends with a compelling call to action**

## Structure:
- **Summary of Findings**: Recap the most important insights
- **Business Impact**: What these findings mean for the organization
- **Limitations**: Honest assessment of what we couldn't determine
- **Future Directions**: What should be analyzed next
- **Recommendations**: Specific actions to take

## Style Guidelines:
- Be honest about limitations and shortcomings
- Provide specific, actionable recommendations
- Suggest areas for future analysis and exploration
- Use confident but measured, professional language
- End with a strong call to action
- Maintain formal, academic tone appropriate for business reports
- Avoid casual language or colloquialisms
- **DO NOT include any meta-commentary, word counts, internal reasoning, or chain-of-thought text**

Write a conclusion that leaves readers with clear next steps and a desire to act on the insights. Provide only the conclusion, no additional commentary. Do not use markdown formatting like **bold** or *italic* text.
"""

        response = llm.invoke(conclusion_prompt)
        conclusion = response.content if hasattr(response, 'content') else str(response)
        
        state["conclusion"] = conclusion
        print("Conclusion section generated successfully")
        
    except Exception as e:
        print(f"Error generating conclusion: {e}")
        state["conclusion"] = "Unable to generate conclusion due to an error."
    
    return state


def format_charts_for_conclusion(charts: List[Dict]) -> str:
    """Format charts for conclusion generation."""
    
    formatted = []
    for chart in charts:
        formatted.append(f"""
• {chart.get('chart_name', 'N/A')} - {chart.get('key_insights', 'N/A')}
""")
    
    return "\n".join(formatted)


def format_successful_charts_for_conclusion(charts: List[Dict]) -> str:
    """Format successful charts for conclusion."""
    
    if not charts:
        return "No charts were successfully generated."
    
    formatted = []
    for chart in charts:
        formatted.append(f"""
• {chart.get('chart_name', 'N/A')} - Successfully generated
""")
    
    return "\n".join(formatted)


def generate_enhanced_chart_narrative(state: DataQualityState) -> DataQualityState:
    """Generate enhanced narrative flow between charts for better storytelling."""
    
    generated_charts = state.get("generated_charts", {})
    chart_recommendations = state.get("chart_recommendations", {})
    
    if not generated_charts or not generated_charts.get("charts"):
        print("No generated charts available for enhanced narrative")
        state["enhanced_chart_narrative"] = None
        return state
    
    if llm is None:
        print("LLM not available, using fallback enhanced narrative")
        state["enhanced_chart_narrative"] = "Our journey begins with an overview of the data patterns.\n\nBuilding on these initial insights, we dive deeper into specific relationships.\n\nAs we explore further, we discover how different factors interact.\n\nThis leads us to examine individual components in detail.\n\nFinally, we synthesize all these insights to understand the complete picture."
        return state
    
    try:
        successful_charts = [c for c in generated_charts["charts"] if c.get("success")]
        
        if not successful_charts:
            print("No successful charts found for enhanced narrative")
            state["enhanced_chart_narrative"] = None
            return state
        
        # Sort charts by rank for proper flow
        successful_charts.sort(key=lambda x: x.get("rank", 0))
        
        narrative_prompt = f"""
You are a master data storyteller. Create a compelling narrative flow that connects these charts into a cohesive, engaging story with smooth transitions.

## Charts in Sequence:
{format_charts_for_enhanced_narrative(successful_charts)}

## Your Task:
Write a narrative flow with 4-5 paragraphs that create smooth transitions between charts. Each paragraph should:

1. **Connect to the previous chart** and set up the next one
2. **Use engaging transitional phrases** that flow naturally
3. **Build suspense and curiosity** about what comes next
4. **Create logical story progression** from broad to specific or vice versa
5. **Make each chart feel like a natural next step** in the discovery process

## Structure:
- **Paragraph 1**: Transition from overview to first detailed insight
- **Paragraph 2**: Connect first chart to second chart's focus
- **Paragraph 3**: Bridge second chart to third chart's perspective
- **Paragraph 4**: Link third chart to fourth chart's analysis
- **Paragraph 5**: Connect to final chart and set up conclusion

## Transition Guidelines:
- Use professional phrases like "This analysis leads us to examine..." or "Building on these findings..."
- Create logical connections between insights
- Acknowledge when shifting focus with phrases like "Shifting our analytical perspective to..."
- Build systematic progression and reveal insights methodically
- Present data insights in a clear, structured manner

## Format:
Write 4-5 paragraphs separated by double line breaks (\\n\\n). Each paragraph should be 2-3 sentences that smoothly transition from one chart to the next.

Write a professional narrative flow that makes each chart feel like a natural progression in the analytical journey.
"""

        response = llm.invoke(narrative_prompt)
        enhanced_narrative = response.content if hasattr(response, 'content') else str(response)
        
        state["enhanced_chart_narrative"] = enhanced_narrative
        print("Enhanced chart narrative generated successfully")
        
    except Exception as e:
        print(f"Error generating enhanced chart narrative: {e}")
        state["enhanced_chart_narrative"] = None
    
    return state


def format_charts_for_enhanced_narrative(charts: List[Dict]) -> str:
    """Format charts for enhanced narrative generation."""
    
    formatted = []
    for i, chart in enumerate(charts, 1):
        formatted.append(f"""
Chart {i}: {chart.get('chart_name', 'N/A')}
- Type: {chart.get('chart_type', 'N/A')}
- Purpose: {chart.get('purpose', 'N/A')}
- Key Insight: {chart.get('key_insights', 'N/A')}
""")
    
    return "\n".join(formatted)


def generate_final_html_report(state: DataQualityState) -> DataQualityState:
    """Generate the final HTML report with all enhanced sections."""
    
    generated_charts = state.get("generated_charts", {})
    chart_recommendations = state.get("chart_recommendations", {})
    
    if not generated_charts or not generated_charts.get("charts"):
        print("No generated charts available for HTML report generation")
        state["final_html_path"] = None
        return state
    
    try:
        charts_data = generated_charts["charts"]
        
        # Generate HTML report with enhanced structure
        html_path = generate_html_report(charts_data, chart_recommendations, script_path, state)
        
        if html_path:
            state["final_html_path"] = html_path
            print(f"Final enhanced HTML report generated: {html_path}")
        else:
            state["final_html_path"] = None
            print("Failed to generate final HTML report")
        
    except Exception as e:
        print(f"Error generating final HTML report: {e}")
        state["final_html_path"] = None
    
    return state


def generate_narrative_snippets_for_charts(charts: List[Dict], chart_recommendations: Dict[str, Any]) -> List[str]:
    """Generate detailed narrative flow snippets for each chart to create smooth transitions."""
    
    if not charts:
        return []
    
    if llm is None:
        # Enhanced fallback snippets with more detail and context
        fallback_snippets = [
            "Our analysis begins with a comprehensive view of data distribution patterns across the dataset. This chart reveals the overall structure of the data, highlighting key categories and identifying patterns in the data landscape. Understanding these patterns sets the foundation for deeper analytical exploration.",
            
            "Building on our understanding of data distribution, we now examine how key metrics have evolved over time. This chart compares values across different time periods, revealing trends and patterns that help us understand how the data has changed and what factors may have influenced these changes.",
            
            "To complement our temporal analysis, we examine the relationships between different variables in the dataset. This chart shows how different factors correlate with each other, providing insights into dependencies and connections that may not be immediately apparent from individual variable analysis.",
            
            "Moving from relationships to distributions, we examine the spread and variability of our key metrics across different categories. This granular view reveals the full range of values and helps us understand what differentiates high-performing from low-performing groups within the data.",
            
            "Finally, we examine the overall distribution of our numeric data to understand the shape, central tendency, and variability of our key metrics. This comprehensive view provides insights into the data's statistical properties and helps identify any outliers or unusual patterns."
        ]
        return fallback_snippets[:len(charts)]
    
    try:
        narrative_prompt = f"""
You are a master data storyteller creating detailed narrative introductions for data analysis charts. Create comprehensive narrative snippets that introduce each chart with rich context and smooth transitions.

## Charts in Sequence:
{format_charts_for_narrative_snippets(charts)}

## Your Task:
Create exactly {len(charts)} detailed narrative snippets (one for each chart) that:
1. **Provide rich context** about what the chart will reveal
2. **Create smooth transitions** from the previous chart's findings
3. **Explain the analytical reasoning** behind examining this specific aspect
4. **Set up expectations** for the insights to come
5. **Use specific, detailed language** that matches the data analysis context

## Style Guidelines:
- Use professional, analytical language appropriate for data analysis
- Create logical progression that builds understanding step by step
- Reference specific data metrics and patterns
- Explain why each chart is important for understanding the dataset
- Make each snippet 2-3 sentences with substantial detail
- Connect each chart to the broader data analysis narrative

## Example Style:
"To complement our distribution analysis, we examine trends over time across the same categories, as temporal patterns can reveal changes and evolution in the data. This line chart shows how key metrics have evolved over time, providing insights into changing patterns and identifying significant trends."

## Format:
Return exactly {len(charts)} detailed snippets, one per line, separated by double line breaks (\\n\\n).

Write narrative snippets that provide rich context and make each chart feel like a natural progression in the data analysis journey. **DO NOT include any meta-commentary, word counts, internal reasoning, or chain-of-thought text.** Do not use markdown formatting like **bold** or *italic* text.
"""

        response = llm.invoke(narrative_prompt)
        snippets_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse snippets
        snippets = [s.strip() for s in snippets_text.split('\n\n') if s.strip()]
        
        # Ensure we have the right number of snippets
        if len(snippets) >= len(charts):
            return snippets[:len(charts)]
        else:
            # Enhanced fallback snippets
            fallback_snippets = [
                "Our analysis begins with a comprehensive view of data distribution patterns across the dataset. This visualization reveals the overall structure of the data, highlighting key categories and identifying patterns in the data landscape.",
                
                "Building on our understanding of data distribution, we now examine how key metrics have evolved over time. This chart compares values across different time periods, revealing trends and patterns that help us understand how the data has changed.",
                
                "To complement our temporal analysis, we examine the relationships between different variables in the dataset. This chart shows how different factors correlate with each other, providing insights into dependencies and connections.",
                
                "Moving from relationships to distributions, we examine the spread and variability of our key metrics across different categories. This granular view reveals the full range of values and helps us understand group differences.",
                
                "Finally, we examine the overall distribution of our numeric data to understand the shape, central tendency, and variability of our key metrics. This comprehensive view provides insights into the data's statistical properties."
            ]
            while len(snippets) < len(charts):
                snippets.append(fallback_snippets[len(snippets) % len(fallback_snippets)])
            return snippets[:len(charts)]
            
    except Exception as e:
        print(f"Error generating narrative snippets: {e}")
        # Return enhanced fallback snippets
        fallback_snippets = [
            "Our analysis begins with a comprehensive view of data distribution patterns across the dataset. This chart reveals the overall structure of the data, highlighting key categories and identifying patterns in the data landscape. Understanding these patterns sets the foundation for deeper analytical exploration.",
            
            "Building on our understanding of data distribution, we now examine how key metrics have evolved over time. This chart compares values across different time periods, revealing trends and patterns that help us understand how the data has changed and what factors may have influenced these changes.",
            
            "To complement our temporal analysis, we examine the relationships between different variables in the dataset. This chart shows how different factors correlate with each other, providing insights into dependencies and connections that may not be immediately apparent from individual variable analysis.",
            
            "Moving from relationships to distributions, we examine the spread and variability of our key metrics across different categories. This granular view reveals the full range of values and helps us understand what differentiates high-performing from low-performing groups within the data.",
            
            "Finally, we examine the overall distribution of our numeric data to understand the shape, central tendency, and variability of our key metrics. This comprehensive view provides insights into the data's statistical properties and helps identify any outliers or unusual patterns."
        ]
        return fallback_snippets[:len(charts)]


def format_charts_for_narrative_snippets(charts: List[Dict]) -> str:
    """Format charts for narrative snippet generation with detailed context."""
    
    formatted = []
    for i, chart in enumerate(charts, 1):
        formatted.append(f"""
Chart {i}: {chart.get('chart_name', 'N/A')}
- Chart Type: {chart.get('chart_type', 'N/A')}
- Fields Used: {', '.join(chart.get('fields_required', []))}
- Analytical Purpose: {chart.get('why_this_chart_helps', 'N/A')}
- Position in Analysis: Chart {i} of {len(charts)} in the data analysis sequence
""")
    
    return "\n".join(formatted)


def convert_markdown_to_html(text: str) -> str:
    """Convert markdown text to HTML for proper rendering."""
    if not text:
        return ""
    
    # Convert markdown headers
    text = text.replace('## ', '<h3>').replace(' ### ', '</h3><h4>').replace(' ###', '</h4>')
    text = text.replace('### ', '<h4>').replace(' #### ', '</h4><h5>').replace(' ####', '</h5>')
    
    # Convert remaining headers
    text = text.replace('## ', '<h3>').replace('\n## ', '</h3>\n<h3>')
    text = text.replace('### ', '<h4>').replace('\n### ', '</h4>\n<h4>')
    
    # Close any unclosed headers
    if text.count('<h3>') > text.count('</h3>'):
        text += '</h3>'
    if text.count('<h4>') > text.count('</h4>'):
        text += '</h4>'
    
    # Convert bullet points
    text = text.replace('\n• ', '\n<li>').replace('\n- ', '\n<li>')
    text = text.replace('\n* ', '\n<li>')
    
    # Wrap lists in ul tags
    lines = text.split('\n')
    in_list = False
    processed_lines = []
    
    for line in lines:
        if line.strip().startswith('<li>'):
            if not in_list:
                processed_lines.append('<ul>')
                in_list = True
            processed_lines.append(line)
        else:
            if in_list:
                processed_lines.append('</ul>')
                in_list = False
            processed_lines.append(line)
    
    if in_list:
        processed_lines.append('</ul>')
    
    text = '\n'.join(processed_lines)
    
    # Convert paragraphs
    paragraphs = text.split('\n\n')
    processed_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if para:
            if not para.startswith('<h') and not para.startswith('<ul>') and not para.startswith('<li>'):
                para = f'<p>{para}</p>'
            processed_paragraphs.append(para)
    
    return '\n\n'.join(processed_paragraphs)


# ---------- Feedback Loop System for Chart Generation ----------

def implement_chart_feedback_loop(state: DataQualityState) -> DataQualityState:
    """Implement a feedback loop to ensure we have at least 6-7 meaningful charts."""
    
    generated_charts = state.get("generated_charts", {})
    chart_recommendations = state.get("chart_recommendations", {})
    
    if not generated_charts or not generated_charts.get("charts"):
        print("No generated charts available for feedback loop")
        return state
    
    print("Starting chart feedback loop to ensure quality and quantity...")
    
    # Get current charts
    charts_data = generated_charts["charts"]
    successful_charts = [c for c in charts_data if c.get("success")]
    
    print(f"Initial state: {len(successful_charts)} successful charts out of {len(charts_data)} total")
    
    # Target: At least 6-7 meaningful charts
    target_charts = 7
    max_iterations = 3  # Prevent infinite loops
    
    iteration = 0
    while len(successful_charts) < target_charts and iteration < max_iterations:
        iteration += 1
        print(f"\n--- Feedback Loop Iteration {iteration} ---")
        print(f"Current successful charts: {len(successful_charts)}")
        print(f"Target: {target_charts}")
        
        # Analyze failed charts and identify issues
        failed_charts = [c for c in charts_data if not c.get("success")]
        print(f"Failed charts: {len(failed_charts)}")
        
        if not failed_charts:
            # All charts succeeded but we need more
            print("All charts succeeded but need more. Generating additional charts...")
            additional_charts = generate_additional_charts(state, target_charts - len(successful_charts))
            charts_data.extend(additional_charts)
        else:
            # Try to fix failed charts
            print("Attempting to fix failed charts...")
            fixed_charts = fix_failed_charts(failed_charts, state)
            
            # Replace failed charts with fixed ones
            for i, chart in enumerate(charts_data):
                if not chart.get("success"):
                    if fixed_charts:
                        charts_data[i] = fixed_charts.pop(0)
        
        # Re-validate all charts
        print("Re-validating all charts...")
        for chart in charts_data:
            plotly_figure = chart.get("plotly_figure")
            chart_data = chart.get("chart_data")
            
            # Check if we have both figure and data
            if plotly_figure is not None and chart_data is not None:
                try:
                    has_data = validate_chart_has_data(plotly_figure, chart_data)
                    if not has_data:
                        chart["success"] = False
                        chart["error"] = "Chart validated but contains no meaningful data"
                        print(f"Chart {chart.get('rank', 'Unknown')} failed validation: no meaningful data")
                except Exception as e:
                    print(f"Error validating chart {chart.get('rank', 'Unknown')}: {e}")
                    chart["success"] = False
                    chart["error"] = f"Validation error: {str(e)}"
            else:
                # If we don't have both figure and data, mark as failed
                chart["success"] = False
                chart["error"] = "Missing plotly figure or chart data"
                print(f"Chart {chart.get('rank', 'Unknown')} failed validation: missing figure or data")
        
        # Update successful charts count
        successful_charts = [c for c in charts_data if c.get("success")]
        print(f"After iteration {iteration}: {len(successful_charts)} successful charts")
        
        # If we still don't have enough, try alternative approaches
        if len(successful_charts) < target_charts:
            print("Still need more charts. Trying alternative approaches...")
            alternative_charts = generate_alternative_charts(state, target_charts - len(successful_charts))
            charts_data.extend(alternative_charts)
            
            # Re-validate again
            for chart in alternative_charts:
                plotly_figure = chart.get("plotly_figure")
                chart_data = chart.get("chart_data")
                
                if plotly_figure is not None and chart_data is not None:
                    try:
                        has_data = validate_chart_has_data(plotly_figure, chart_data)
                        chart["success"] = has_data
                        if not has_data:
                            chart["error"] = "Alternative chart failed validation"
                    except Exception as e:
                        print(f"Error validating alternative chart: {e}")
                        chart["success"] = False
                        chart["error"] = f"Validation error: {str(e)}"
                else:
                    chart["success"] = False
                    chart["error"] = "Alternative chart missing figure or data"
            
            successful_charts = [c for c in charts_data if c.get("success")]
    
    # Final validation and cleanup
    print(f"\n--- Final Feedback Loop Results ---")
    print(f"Total charts generated: {len(charts_data)}")
    print(f"Successful charts: {len(successful_charts)}")
    print(f"Failed charts: {len([c for c in charts_data if not c.get('success')])}")
    
    # Update state with improved charts
    generated_charts["charts"] = charts_data
    generated_charts["feedback_loop_completed"] = True
    generated_charts["final_successful_count"] = len(successful_charts)
    generated_charts["feedback_iterations"] = iteration
    
    state["generated_charts"] = generated_charts
    
    if len(successful_charts) >= target_charts:
        print(f"✅ Successfully achieved target of {target_charts} charts!")
    else:
        print(f"⚠️  Only achieved {len(successful_charts)} charts (target: {target_charts})")
    
    return state


def fix_failed_charts(failed_charts: List[Dict], state: DataQualityState) -> List[Dict]:
    """Attempt to fix failed charts using different approaches."""
    
    fixed_charts = []
    
    for failed_chart in failed_charts:
        print(f"Attempting to fix chart {failed_chart.get('rank', 'Unknown')}: {failed_chart.get('chart_name', 'Unknown')}")
        print(f"Original error: {failed_chart.get('error', 'Unknown error')}")
        
        # Try different approaches based on the error
        error = failed_chart.get('error', '').lower()
        
        if 'missing fields' in error or 'insufficient data' in error:
            # Try with different field combinations
            fixed_chart = try_alternative_fields(failed_chart, state)
        elif 'no meaningful data' in error or 'empty' in error:
            # Try different chart types or data preprocessing
            fixed_chart = try_alternative_chart_type(failed_chart, state)
        elif 'insufficient unique values' in error:
            # Try aggregation or different grouping
            fixed_chart = try_alternative_grouping(failed_chart, state)
        else:
            # Generic fix attempt
            fixed_chart = try_generic_fix(failed_chart, state)
        
        if fixed_chart and fixed_chart.get("success"):
            print(f"✅ Successfully fixed chart {failed_chart.get('rank', 'Unknown')}")
            fixed_charts.append(fixed_chart)
        else:
            print(f"❌ Failed to fix chart {failed_chart.get('rank', 'Unknown')}")
    
    return fixed_charts


def try_alternative_fields(failed_chart: Dict, state: DataQualityState) -> Dict:
    """Try alternative field combinations for a failed chart."""
    
    try:
        original_fields = failed_chart.get('fields_required', [])
        chart_type = failed_chart.get('chart_type', '')
        chart_name = failed_chart.get('chart_name', '')
        
        # Get available fields from dataset
        available_fields = list(df.columns)
        
        # Try different field combinations based on chart type
        if 'bar' in chart_type.lower() or 'distribution' in chart_name.lower():
            # For bar charts, try categorical + numeric combinations
            categorical_fields = [f for f in available_fields if df[f].dtype == 'object']
            numeric_fields = [f for f in available_fields if pd.api.types.is_numeric_dtype(df[f])]
            
            for cat_field in categorical_fields[:3]:  # Try first 3 categorical fields
                for num_field in numeric_fields[:3]:  # Try first 3 numeric fields
                    if cat_field != num_field:
                        new_fields = [cat_field, num_field]
                        try:
                            new_rec = {
                                "rank": failed_chart.get("rank", 0),
                                "chart_type": chart_type,
                                "chart_name": f"Distribution Analysis: {cat_field} vs {num_field}",
                                "fields_required": new_fields,
                                "why_this_chart_helps": f"Shows distribution of {num_field} across {cat_field} categories"
                            }
                            
                            fig, chart_data = create_plotly_chart(new_rec, df)
                            if fig and chart_data and validate_chart_has_data(fig, chart_data):
                                return {
                                    **failed_chart,
                                    "plotly_figure": fig,
                                    "chart_data": chart_data,
                                    "success": True,
                                    "fields_required": new_fields,
                                    "chart_name": new_rec["chart_name"],
                                    "why_this_chart_helps": new_rec["why_this_chart_helps"],
                                    "error": None
                                }
                        except Exception as e:
                            continue
        
        elif 'scatter' in chart_type.lower() or 'correlation' in chart_type.lower():
            # For scatter plots, try numeric field combinations
            numeric_fields = [f for f in available_fields if pd.api.types.is_numeric_dtype(df[f])]
            
            for i, field1 in enumerate(numeric_fields[:4]):
                for field2 in numeric_fields[i+1:5]:
                    if field1 != field2:
                        new_fields = [field1, field2]
                        try:
                            new_rec = {
                                "rank": failed_chart.get("rank", 0),
                                "chart_type": "Scatter Plot",
                                "chart_name": f"Correlation Analysis: {field1} vs {field2}",
                                "fields_required": new_fields,
                                "why_this_chart_helps": f"Examines relationship between {field1} and {field2}"
                            }
                            
                            fig, chart_data = create_plotly_chart(new_rec, df)
                            if fig and chart_data and validate_chart_has_data(fig, chart_data):
                                return {
                                    **failed_chart,
                                    "plotly_figure": fig,
                                    "chart_data": chart_data,
                                    "success": True,
                                    "fields_required": new_fields,
                                    "chart_name": new_rec["chart_name"],
                                    "why_this_chart_helps": new_rec["why_this_chart_helps"],
                                    "error": None
                                }
                        except Exception as e:
                            continue
        
        elif 'line' in chart_type.lower() or 'time' in chart_type.lower():
            # For line charts, try to find time-like fields
            potential_time_fields = []
            for field in available_fields:
                # Check if field might be time-related
                if any(time_word in field.lower() for time_word in ['date', 'time', 'year', 'month', 'day', 'period']):
                    potential_time_fields.append(field)
                elif df[field].dtype == 'object':
                    # Check if it might be a date string
                    try:
                        pd.to_datetime(df[field].dropna().head(10))
                        potential_time_fields.append(field)
                    except:
                        pass
            
            numeric_fields = [f for f in available_fields if pd.api.types.is_numeric_dtype(df[f])]
            
            for time_field in potential_time_fields[:2]:
                for num_field in numeric_fields[:3]:
                    if time_field != num_field:
                        new_fields = [time_field, num_field]
                        try:
                            new_rec = {
                                "rank": failed_chart.get("rank", 0),
                                "chart_type": "Line Chart",
                                "chart_name": f"Trend Analysis: {num_field} over {time_field}",
                                "fields_required": new_fields,
                                "why_this_chart_helps": f"Shows how {num_field} changes over {time_field}"
                            }
                            
                            fig, chart_data = create_plotly_chart(new_rec, df)
                            if fig and chart_data and validate_chart_has_data(fig, chart_data):
                                return {
                                    **failed_chart,
                                    "plotly_figure": fig,
                                    "chart_data": chart_data,
                                    "success": True,
                                    "fields_required": new_fields,
                                    "chart_name": new_rec["chart_name"],
                                    "why_this_chart_helps": new_rec["why_this_chart_helps"],
                                    "error": None
                                }
                        except Exception as e:
                            continue
    
    except Exception as e:
        print(f"Error in try_alternative_fields: {e}")
    
    return failed_chart


def try_alternative_chart_type(failed_chart: Dict, state: DataQualityState) -> Dict:
    """Try alternative chart types for a failed chart."""
    
    try:
        original_fields = failed_chart.get('fields_required', [])
        original_type = failed_chart.get('chart_type', '')
        
        # Get available fields
        available_fields = list(df.columns)
        
        # Try different chart types based on available data
        if len(original_fields) >= 2:
            field1, field2 = original_fields[0], original_fields[1]
            
            # Check data types
            is_numeric1 = pd.api.types.is_numeric_dtype(df[field1])
            is_numeric2 = pd.api.types.is_numeric_dtype(df[field2])
            
            # Try different chart types
            alternative_types = []
            
            if is_numeric1 and is_numeric2:
                alternative_types = [
                    ("Scatter Plot", f"Relationship Analysis: {field1} vs {field2}"),
                    ("Box Plot", f"Distribution Comparison: {field1} by {field2}"),
                    ("Histogram", f"Distribution Analysis: {field1}")
                ]
            elif not is_numeric1 and is_numeric2:
                alternative_types = [
                    ("Bar Chart", f"Category Analysis: {field2} by {field1}"),
                    ("Box Plot", f"Distribution by Category: {field2} across {field1}"),
                    ("Violin Plot", f"Distribution Comparison: {field2} by {field1}")
                ]
            elif is_numeric1 and not is_numeric2:
                alternative_types = [
                    ("Bar Chart", f"Category Analysis: {field1} by {field2}"),
                    ("Box Plot", f"Distribution by Category: {field1} across {field2}"),
                    ("Histogram", f"Distribution Analysis: {field1}")
                ]
            else:
                # Both categorical
                alternative_types = [
                    ("Bar Chart", f"Category Distribution: {field1} vs {field2}"),
                    ("Heatmap", f"Category Correlation: {field1} vs {field2}"),
                    ("Treemap", f"Hierarchical View: {field1} and {field2}")
                ]
            
            for chart_type, chart_name in alternative_types:
                try:
                    new_rec = {
                        "rank": failed_chart.get("rank", 0),
                        "chart_type": chart_type,
                        "chart_name": chart_name,
                        "fields_required": original_fields,
                        "why_this_chart_helps": f"Alternative visualization of {field1} and {field2} relationship"
                    }
                    
                    fig, chart_data = create_plotly_chart(new_rec, df)
                    if fig and chart_data and validate_chart_has_data(fig, chart_data):
                        return {
                            **failed_chart,
                            "plotly_figure": fig,
                            "chart_data": chart_data,
                            "success": True,
                            "chart_type": chart_type,
                            "chart_name": chart_name,
                            "why_this_chart_helps": new_rec["why_this_chart_helps"],
                            "error": None
                        }
                except Exception as e:
                    continue
    
    except Exception as e:
        print(f"Error in try_alternative_chart_type: {e}")
    
    return failed_chart


def try_alternative_grouping(failed_chart: Dict, state: DataQualityState) -> Dict:
    """Try alternative grouping strategies for a failed chart."""
    
    try:
        original_fields = failed_chart.get('fields_required', [])
        
        if len(original_fields) >= 2:
            # Try aggregating data differently
            field1, field2 = original_fields[0], original_fields[1]
            
            # Create aggregated data
            if pd.api.types.is_numeric_dtype(df[field2]):
                # Group by categorical field and aggregate numeric field
                aggregated_data = df.groupby(field1)[field2].agg(['count', 'mean', 'sum']).reset_index()
                
                # Try different aggregations
                for agg_col in ['count', 'mean', 'sum']:
                    if agg_col in aggregated_data.columns:
                        try:
                            new_rec = {
                                "rank": failed_chart.get("rank", 0),
                                "chart_type": "Bar Chart",
                                "chart_name": f"Aggregated Analysis: {agg_col} of {field2} by {field1}",
                                "fields_required": [field1, agg_col],
                                "why_this_chart_helps": f"Shows {agg_col} of {field2} grouped by {field1}"
                            }
                            
                            # Create temporary dataframe with aggregated data
                            temp_df = aggregated_data[[field1, agg_col]].copy()
                            temp_df.columns = [field1, agg_col]
                            
                            fig, chart_data = create_plotly_chart(new_rec, temp_df)
                            if fig and chart_data and validate_chart_has_data(fig, chart_data):
                                return {
                                    **failed_chart,
                                    "plotly_figure": fig,
                                    "chart_data": chart_data,
                                    "success": True,
                                    "chart_type": "Bar Chart",
                                    "chart_name": new_rec["chart_name"],
                                    "why_this_chart_helps": new_rec["why_this_chart_helps"],
                                    "fields_required": [field1, agg_col],
                                    "error": None
                                }
                        except Exception as e:
                            continue
    
    except Exception as e:
        print(f"Error in try_alternative_grouping: {e}")
    
    return failed_chart


def try_generic_fix(failed_chart: Dict, state: DataQualityState) -> Dict:
    """Try a generic fix approach for any failed chart."""
    
    try:
        # Try with the most basic chart type and available fields
        available_fields = list(df.columns)
        
        if len(available_fields) >= 2:
            # Try a simple bar chart with first two fields
            field1, field2 = available_fields[0], available_fields[1]
            
            try:
                new_rec = {
                    "rank": failed_chart.get("rank", 0),
                    "chart_type": "Bar Chart",
                    "chart_name": f"Basic Analysis: {field1} vs {field2}",
                    "fields_required": [field1, field2],
                    "why_this_chart_helps": f"Basic visualization of {field1} and {field2} relationship"
                }
                
                fig, chart_data = create_plotly_chart(new_rec, df)
                if fig and chart_data and validate_chart_has_data(fig, chart_data):
                    return {
                        **failed_chart,
                        "plotly_figure": fig,
                        "chart_data": chart_data,
                        "success": True,
                        "chart_type": "Bar Chart",
                        "chart_name": new_rec["chart_name"],
                        "why_this_chart_helps": new_rec["why_this_chart_helps"],
                        "fields_required": [field1, field2],
                        "error": None
                    }
            except Exception as e:
                pass
    
    except Exception as e:
        print(f"Error in try_generic_fix: {e}")
    
    return failed_chart


def generate_additional_charts(state: DataQualityState, count_needed: int) -> List[Dict]:
    """Generate additional charts to reach the target count."""
    
    additional_charts = []
    available_fields = list(df.columns)
    
    print(f"Generating {count_needed} additional charts...")
    
    # Create different chart types based on available data
    chart_ideas = []
    
    # Get data types
    numeric_fields = [f for f in available_fields if pd.api.types.is_numeric_dtype(df[f])]
    categorical_fields = [f for f in available_fields if df[f].dtype == 'object']
    
    # Generate chart ideas
    if len(numeric_fields) >= 2:
        chart_ideas.append({
            "chart_type": "Scatter Plot",
            "fields": numeric_fields[:2],
            "name": f"Correlation Analysis: {numeric_fields[0]} vs {numeric_fields[1]}"
        })
    
    if len(categorical_fields) >= 1 and len(numeric_fields) >= 1:
        chart_ideas.append({
            "chart_type": "Box Plot",
            "fields": [categorical_fields[0], numeric_fields[0]],
            "name": f"Distribution by Category: {numeric_fields[0]} across {categorical_fields[0]}"
        })
    
    if len(numeric_fields) >= 1:
        chart_ideas.append({
            "chart_type": "Histogram",
            "fields": [numeric_fields[0]],
            "name": f"Distribution Analysis: {numeric_fields[0]}"
        })
    
    if len(categorical_fields) >= 1:
        chart_ideas.append({
            "chart_type": "Bar Chart",
            "fields": [categorical_fields[0]],
            "name": f"Category Distribution: {categorical_fields[0]}"
        })
    
    # Add more generic ideas if needed
    for i in range(len(chart_ideas), count_needed):
        if len(available_fields) >= 2:
            chart_ideas.append({
                "chart_type": "Bar Chart",
                "fields": available_fields[:2],
                "name": f"Additional Analysis {i+1}: {available_fields[0]} vs {available_fields[1]}"
            })
    
    # Generate charts from ideas
    for i, idea in enumerate(chart_ideas[:count_needed]):
        try:
            chart_info = {
                "rank": 100 + i,  # Use high rank numbers for additional charts
                "chart_type": idea["chart_type"],
                "chart_name": idea["name"],
                "fields_required": idea["fields"],
                "why_this_chart_helps": f"Additional analysis to provide comprehensive data coverage",
                "plotly_figure": None,
                "chart_data": None,
                "success": False,
                "error": None
            }
            
            fig, chart_data = create_plotly_chart(chart_info, df)
            if fig and chart_data and validate_chart_has_data(fig, chart_data):
                chart_info["plotly_figure"] = fig
                chart_info["chart_data"] = chart_data
                chart_info["success"] = True
                chart_info["error"] = None
                print(f"✅ Generated additional chart: {idea['name']}")
            else:
                chart_info["error"] = "Failed to generate meaningful data"
                print(f"❌ Failed to generate additional chart: {idea['name']}")
            
            additional_charts.append(chart_info)
            
        except Exception as e:
            print(f"❌ Error generating additional chart {i+1}: {e}")
            additional_charts.append({
                "rank": 100 + i,
                "chart_type": idea["chart_type"],
                "chart_name": idea["name"],
                "fields_required": idea["fields"],
                "why_this_chart_helps": f"Additional analysis to provide comprehensive data coverage",
                "plotly_figure": None,
                "chart_data": None,
                "success": False,
                "error": str(e)
            })
    
    return additional_charts


def generate_alternative_charts(state: DataQualityState, count_needed: int) -> List[Dict]:
    """Generate alternative charts using different approaches."""
    
    alternative_charts = []
    available_fields = list(df.columns)
    
    print(f"Generating {count_needed} alternative charts...")
    
    # Try different analytical approaches
    approaches = [
        ("Statistical Summary", "Statistical Overview"),
        ("Data Quality Visualization", "Data Quality Analysis"),
        ("Outlier Detection", "Outlier Analysis"),
        ("Trend Analysis", "Trend Detection"),
        ("Pattern Recognition", "Pattern Analysis")
    ]
    
    for i, (approach, name) in enumerate(approaches[:count_needed]):
        try:
            # Create a simple but meaningful chart for each approach
            if len(available_fields) >= 1:
                field = available_fields[i % len(available_fields)]
                
                chart_info = {
                    "rank": 200 + i,  # Use even higher rank numbers
                    "chart_type": "Bar Chart" if df[field].dtype == 'object' else "Histogram",
                    "chart_name": f"{name}: {field}",
                    "fields_required": [field],
                    "why_this_chart_helps": f"Provides {approach.lower()} for {field}",
                    "plotly_figure": None,
                    "chart_data": None,
                    "success": False,
                    "error": None
                }
                
                fig, chart_data = create_plotly_chart(chart_info, df)
                if fig and chart_data and validate_chart_has_data(fig, chart_data):
                    chart_info["plotly_figure"] = fig
                    chart_info["chart_data"] = chart_data
                    chart_info["success"] = True
                    chart_info["error"] = None
                    print(f"✅ Generated alternative chart: {name}")
                else:
                    chart_info["error"] = "Failed to generate meaningful data"
                    print(f"❌ Failed to generate alternative chart: {name}")
                
                alternative_charts.append(chart_info)
            
        except Exception as e:
            print(f"❌ Error generating alternative chart {i+1}: {e}")
    
    return alternative_charts


# ---------- Enhanced Chart Validation ----------

def validate_chart_has_data(fig, chart_data) -> bool:
    """Enhanced validation that a chart has meaningful data."""
    
    try:
        # Check if the figure has any traces with data
        if not fig.data:
            return False
        
        # Check each trace for meaningful data
        for trace in fig.data:
            # For different trace types, check if they have data
            if hasattr(trace, 'x') and hasattr(trace, 'y'):
                # Check if x and y have non-empty data
                if trace.x is not None and trace.y is not None:
                    if len(trace.x) > 0 and len(trace.y) > 0:
                        # Check if the data has any non-null values
                        if any(x is not None and x != '' for x in trace.x) and any(y is not None and y != '' for y in trace.y):
                            # Additional check: ensure we have meaningful variation
                            if len(set(trace.x)) > 1 and len(set(trace.y)) > 1:
                                return True
            
            # For heatmaps, check if z data exists
            elif hasattr(trace, 'z') and trace.z is not None:
                if len(trace.z) > 0 and any(any(val is not None and val != '' for val in row) for row in trace.z):
                    return True
        
        # If we have chart_data, also validate it
        if chart_data is not None:
            if isinstance(chart_data, pd.DataFrame):
                if not chart_data.empty and len(chart_data) > 0:
                    # Check for meaningful data variation
                    for col in chart_data.columns:
                        if chart_data[col].nunique() > 1:
                            return True
            elif isinstance(chart_data, pd.Series):
                if not chart_data.empty and len(chart_data) > 0:
                    if chart_data.nunique() > 1:
                        return True
            elif hasattr(chart_data, '__len__'):
                if len(chart_data) > 0:
                    return True
        
        return False
        
    except Exception as e:
        print(f"Error validating chart data: {e}")
        return False


def create_altair_chart(recommendation: Dict[str, Any], chart_df: pd.DataFrame) -> tuple:
    """Create an Altair chart as fallback when Plotly fails."""
    
    if not ALTAIR_AVAILABLE:
        return None, None
    
    chart_type = recommendation.get("chart_type", "").lower()
    chart_name = recommendation.get("chart_name", "")
    fields_required = recommendation.get("fields_required", [])
    
    # Limit to 3 fields maximum
    if len(fields_required) > 3:
        fields_required = fields_required[:3]
    
    try:
        # Ensure we have at least 2 fields
        if len(fields_required) < 2:
            return None, None
        
        x_field = fields_required[0]
        y_field = fields_required[1]
        color_field = fields_required[2] if len(fields_required) > 2 else None
        
        # Create base chart
        if "bar" in chart_type:
            if color_field:
                chart = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X(x_field, title=x_field),
                    y=alt.Y(y_field, title=y_field),
                    color=alt.Color(color_field, title=color_field)
                )
            else:
                chart = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X(x_field, title=x_field),
                    y=alt.Y(y_field, title=y_field)
                )
        
        elif "scatter" in chart_type or "correlation" in chart_type:
            if color_field:
                chart = alt.Chart(chart_df).mark_circle(size=60).encode(
                    x=alt.X(x_field, title=x_field),
                    y=alt.Y(y_field, title=y_field),
                    color=alt.Color(color_field, title=color_field)
                )
            else:
                chart = alt.Chart(chart_df).mark_circle(size=60).encode(
                    x=alt.X(x_field, title=x_field),
                    y=alt.Y(y_field, title=y_field)
                )
        
        elif "line" in chart_type or "time series" in chart_type:
            if color_field:
                chart = alt.Chart(chart_df).mark_line().encode(
                    x=alt.X(x_field, title=x_field),
                    y=alt.Y(y_field, title=y_field),
                    color=alt.Color(color_field, title=color_field)
                )
            else:
                chart = alt.Chart(chart_df).mark_line().encode(
                    x=alt.X(x_field, title=x_field),
                    y=alt.Y(y_field, title=y_field)
                )
        
        elif "box" in chart_type:
            if color_field:
                chart = alt.Chart(chart_df).mark_boxplot().encode(
                    x=alt.X(color_field, title=color_field),
                    y=alt.Y(y_field, title=y_field)
                )
            else:
                chart = alt.Chart(chart_df).mark_boxplot().encode(
                    x=alt.X(x_field, title=x_field),
                    y=alt.Y(y_field, title=y_field)
                )
        
        elif "histogram" in chart_type:
            chart = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X(x_field, title=x_field, bin=True),
                y=alt.Y('count()', title='Count')
            )
        
        else:
            # Default to bar chart
            if color_field:
                chart = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X(x_field, title=x_field),
                    y=alt.Y(y_field, title=y_field),
                    color=alt.Color(color_field, title=color_field)
                )
            else:
                chart = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X(x_field, title=x_field),
                    y=alt.Y(y_field, title=y_field)
                )
        
        # Add title and styling
        chart = chart.properties(
            title=chart_name,
            width=600,
            height=400
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_title(
            fontSize=16,
            fontWeight='bold'
        )
        
        # Convert Altair chart to Plotly figure for compatibility
        try:
            # Use Altair's to_dict method to get the chart specification
            chart_dict = chart.to_dict()
            
            # Create a simple Plotly figure that represents the Altair chart
            fig = go.Figure()
            
            # Add a placeholder trace to represent the Altair chart
            fig.add_trace(go.Scatter(
                x=chart_df[x_field].tolist(),
                y=chart_df[y_field].tolist(),
                mode='markers',
                name=chart_name
            ))
            
            fig.update_layout(
                title=chart_name,
                xaxis_title=x_field,
                yaxis_title=y_field,
                plot_bgcolor='white',
                height=400,
                width=600
            )
            
            return fig, chart_df
            
        except Exception as conversion_error:
            print(f"Error converting Altair to Plotly: {conversion_error}")
            return None, None
            
    except Exception as e:
        print(f"Error creating Altair chart: {e}")
        return None, None


def create_plotly_chart(recommendation: Dict[str, Any], dataframe: pd.DataFrame) -> tuple:
    """Create a Plotly chart based on the recommendation with Altair fallback."""
    
    chart_type = recommendation.get("chart_type", "").lower()
    fields_required = recommendation.get("fields_required", [])
    chart_name = recommendation.get("chart_name", "")
    
    # Limit fields to maximum 3 for all charts
    if len(fields_required) > 3:
        fields_required = fields_required[:3]
        print(f"Limited fields to 3: {fields_required}")
    
    # Check if required fields exist in dataframe
    missing_fields = [field for field in fields_required if field not in dataframe.columns]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Filter dataframe to only include required fields and remove rows with missing values
    chart_df = dataframe[fields_required].dropna()
    
    if chart_df.empty:
        raise ValueError("No data available after removing missing values")
    
    # Additional validation: ensure we have enough data points for meaningful visualization
    if len(chart_df) < 2:
        raise ValueError(f"Insufficient data points ({len(chart_df)}) for meaningful visualization")
    
    # Check for sufficient unique values in categorical fields
    for field in fields_required:
        if field in chart_df.columns:
            unique_count = chart_df[field].nunique()
            if unique_count < 2:
                raise ValueError(f"Field '{field}' has insufficient unique values ({unique_count}) for visualization")
    
    fig = None
    chart_data = None
    
    # Time Series Line Chart
    if "time series" in chart_type or "line" in chart_type:
        if len(fields_required) >= 2:
            x_field = fields_required[0]  # Assume first field is time
            y_field = fields_required[1]  # Assume second field is value
            
            # Validate that we have enough data points for a line chart
            if len(chart_df) < 3:
                raise ValueError(f"Insufficient data points ({len(chart_df)}) for line chart visualization")
            
            # Aggregate data by time period for cleaner visualization
            if pd.api.types.is_numeric_dtype(chart_df[y_field]):
                chart_data = chart_df.groupby(x_field)[y_field].mean().reset_index()
            else:
                chart_data = chart_df.groupby(x_field).size().reset_index(name='Count')
                y_field = 'Count'
            
            # Validate aggregated data
            if len(chart_data) < 2:
                raise ValueError(f"Insufficient aggregated data points ({len(chart_data)}) for line chart")
            
            # If there's a third field, use it for color grouping
            if len(fields_required) >= 3:
                color_field = fields_required[2]
                # Aggregate with color grouping
                chart_data = chart_df.groupby([x_field, color_field])[y_field].mean().reset_index()
                
                # Validate that we have data for multiple colors
                if chart_data[color_field].nunique() < 2:
                    raise ValueError(f"Insufficient unique values in color field '{color_field}' for grouped line chart")
                
                fig = px.line(chart_data, x=x_field, y=y_field, color=color_field,
                            title=chart_name, labels={x_field: x_field, y_field: y_field},
                            color_discrete_sequence=px.colors.qualitative.Set3)
            else:
                fig = px.line(chart_data, x=x_field, y=y_field,
                            title=chart_name, labels={x_field: x_field, y_field: y_field},
                            color_discrete_sequence=['#3498db'])
    
    # Correlation Heatmap
    elif "correlation" in chart_type or "heatmap" in chart_type:
        # Calculate correlation matrix
        numeric_fields = [field for field in fields_required if pd.api.types.is_numeric_dtype(chart_df[field])]
        if len(numeric_fields) >= 2:
            corr_matrix = chart_df[numeric_fields].corr()
            fig = px.imshow(corr_matrix, 
                          title=chart_name,
                          color_continuous_scale='RdBu_r',
                          aspect='auto',
                          text_auto=True,
                          color_continuous_midpoint=0)
            chart_data = corr_matrix
    
    # Stacked Bar Chart
    elif "stacked bar" in chart_type:
        if len(fields_required) >= 2:
            x_field = fields_required[0]
            y_field = fields_required[1]
            color_field = fields_required[2] if len(fields_required) >= 3 else None
            
            if color_field:
                fig = px.bar(chart_df, x=x_field, y=y_field, color=color_field,
                           title=chart_name, barmode='stack',
                           color_discrete_sequence=px.colors.qualitative.Set3)
            else:
                fig = px.bar(chart_df, x=x_field, y=y_field,
                           title=chart_name,
                           color_discrete_sequence=['#3498db'])
            
            chart_data = chart_df.groupby([x_field, color_field])[y_field].sum().reset_index() if color_field else chart_df.groupby(x_field)[y_field].sum().reset_index()
    
    # Scatter Plot
    elif "scatter" in chart_type:
        if len(fields_required) >= 2:
            x_field = fields_required[0]
            y_field = fields_required[1]
            color_field = fields_required[2] if len(fields_required) >= 3 else None
            
            if color_field:
                fig = px.scatter(chart_df, x=x_field, y=y_field, color=color_field,
                               title=chart_name, trendline="ols",
                               color_discrete_sequence=px.colors.qualitative.Set3)
            else:
                fig = px.scatter(chart_df, x=x_field, y=y_field,
                               title=chart_name, trendline="ols",
                               color_discrete_sequence=['#3498db'])
            
            chart_data = chart_df
    
    # Box Plot
    elif "box" in chart_type:
        if len(fields_required) >= 2:
            x_field = fields_required[0]  # Categorical field
            y_field = fields_required[1]  # Numeric field
            
            fig = px.box(chart_df, x=x_field, y=y_field,
                        title=chart_name,
                        color_discrete_sequence=px.colors.qualitative.Set3)
            
            chart_data = chart_df.groupby(x_field)[y_field].describe()
    
    # Bar Chart (default for categorical data)
    elif "bar" in chart_type:
        if len(fields_required) >= 2:
            x_field = fields_required[0]
            y_field = fields_required[1]
            
            # Handle different aggregation types based on chart purpose
            if "distribution" in chart_name.lower() or "count" in chart_name.lower():
                # For distribution charts, count occurrences
                if len(fields_required) >= 2:
                    if len(fields_required) == 2:
                        # Simple count by x_field
                        chart_data = chart_df[x_field].value_counts().reset_index()
                        chart_data.columns = [x_field, 'Count']
                        
                        # Validate that we have meaningful counts
                        if len(chart_data) < 2 or chart_data['Count'].sum() < 2:
                            raise ValueError(f"Insufficient data for distribution chart: {len(chart_data)} categories with {chart_data['Count'].sum()} total items")
                        
                        fig = px.bar(chart_data, x=x_field, y='Count',
                                   title=chart_name,
                                   color_discrete_sequence=['#3498db'])
                    else:
                        # Count by multiple fields (e.g., Category and Type)
                        chart_data = chart_df.groupby(fields_required[:-1]).size().reset_index(name='Count')
                        
                        # Validate grouped data
                        if len(chart_data) < 2 or chart_data['Count'].sum() < 2:
                            raise ValueError(f"Insufficient data for grouped distribution chart: {len(chart_data)} groups with {chart_data['Count'].sum()} total items")
                        
                        fig = px.bar(chart_data, x=fields_required[0], y='Count', 
                                   color=fields_required[1] if len(fields_required) > 1 else None,
                                   title=chart_name,
                                   color_discrete_sequence=px.colors.qualitative.Set3)
            else:
                # For other bar charts, use sum/mean aggregation
                if pd.api.types.is_numeric_dtype(chart_df[y_field]):
                    chart_data = chart_df.groupby(x_field)[y_field].mean().reset_index()
                    
                    # Validate aggregated data
                    if len(chart_data) < 2:
                        raise ValueError(f"Insufficient data for bar chart: {len(chart_data)} categories")
                    
                    fig = px.bar(chart_df, x=x_field, y=y_field,
                               title=chart_name,
                               color_discrete_sequence=['#3498db'])
                else:
                    chart_data = chart_df.groupby(x_field).size().reset_index(name='Count')
                    
                    # Validate count data
                    if len(chart_data) < 2 or chart_data['Count'].sum() < 2:
                        raise ValueError(f"Insufficient data for count bar chart: {len(chart_data)} categories with {chart_data['Count'].sum()} total items")
                    
                    fig = px.bar(chart_data, x=x_field, y='Count',
                               title=chart_name,
                               color_discrete_sequence=['#3498db'])
    
    # Histogram
    elif "histogram" in chart_type:
        if len(fields_required) >= 1:
            field = fields_required[0]
            
            fig = px.histogram(chart_df, x=field,
                             title=chart_name,
                             color_discrete_sequence=['#e67e22'])
            
            chart_data = chart_df[field].value_counts().reset_index()
    
    # If no specific chart type matched, create a basic visualization
    else:
        if len(fields_required) >= 2:
            x_field = fields_required[0]
            y_field = fields_required[1]
            
                    # Try to determine the best chart type based on data types
        if pd.api.types.is_numeric_dtype(chart_df[x_field]) and pd.api.types.is_numeric_dtype(chart_df[y_field]):
            fig = px.scatter(chart_df, x=x_field, y=y_field, title=chart_name,
                           color_discrete_sequence=['#3498db'])
        elif pd.api.types.is_numeric_dtype(chart_df[y_field]):
            fig = px.bar(chart_df, x=x_field, y=y_field, title=chart_name,
                        color_discrete_sequence=['#3498db'])
        else:
            fig = px.histogram(chart_df, x=x_field, title=chart_name,
                             color_discrete_sequence=['#e67e22'])
        
        chart_data = chart_df
    
    if fig:
        # Apply professional theme with better colors and contrast
        fig.update_layout(
            title={
                'text': chart_name,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50', 'weight': 'bold'}
            },
            plot_bgcolor='white',
            font={'color': '#2c3e50', 'size': 13},
            height=600,
            width=900,
            showlegend=True,
            legend={
                'bgcolor': 'rgba(255,255,255,0.9)',
                'bordercolor': '#bdc3c7',
                'borderwidth': 1,
                'font': {'size': 12}
            },
            margin=dict(l=80, r=80, t=100, b=80),
            hovermode='closest',
            hoverlabel={
                'bgcolor': 'white',
                'font_size': 12,
                'font_family': 'Arial'
            }
        )
        
        # Update axes for better visibility
        fig.update_xaxes(
            gridcolor='#ecf0f1',
            gridwidth=1,
            zerolinecolor='#bdc3c7',
            zerolinewidth=1,
            showline=True,
            linecolor='#2c3e50',
            linewidth=1
        )
        
        fig.update_yaxes(
            gridcolor='#ecf0f1',
            gridwidth=1,
            zerolinecolor='#bdc3c7',
            zerolinewidth=1,
            showline=True,
            linecolor='#2c3e50',
            linewidth=1
        )
        
        # Apply better color schemes based on chart type
        if "line" in chart_type or "time series" in chart_type:
            fig.update_traces(
                line=dict(width=4),
                marker=dict(size=8, opacity=0.8)
            )
        elif "bar" in chart_type:
            fig.update_traces(
                marker_color='#3498db',
                marker_line_color='#2980b9',
                marker_line_width=2,
                opacity=0.85
            )
        elif "scatter" in chart_type:
            fig.update_traces(
                marker=dict(size=10, opacity=0.8),
                line=dict(width=3, color='#e74c3c')
            )
        elif "box" in chart_type:
            fig.update_traces(
                marker_color='#9b59b6',
                marker_line_color='#8e44ad',
                marker_line_width=2,
                opacity=0.8
            )
        elif "histogram" in chart_type:
            fig.update_traces(
                marker_color='#e67e22',
                marker_line_color='#d35400',
                marker_line_width=2,
                opacity=0.8
            )
    
    # If Plotly failed and Altair is available, try Altair as fallback
    if fig is None and ALTAIR_AVAILABLE:
        try:
            print(f"Plotly failed for chart '{chart_name}', trying Altair fallback...")
            fig, chart_data = create_altair_chart(recommendation, chart_df)
            if fig is not None:
                print(f"✅ Altair fallback successful for chart '{chart_name}'")
        except Exception as altair_error:
            print(f"❌ Altair fallback also failed: {altair_error}")
    
    return fig, chart_data


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Encountered an error:", e)
