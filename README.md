# LangGraph Agentic Workflow for Data Analysis

This implementation provides a comprehensive LangGraph-based evaluator orchestrator agentic workflow for generating data-driven reports from CSV datasets. The workflow follows the architecture depicted in the provided image and includes 5 specialized agents working together to create compelling visualizations and insights.

## 🏗️ Architecture Overview

The workflow consists of 5 specialized agents that work in sequence with feedback loops:

```
CSV Input → Data Profiling → Chart Recommendation → Chart Generator → Chart Evaluator
                                                                    ↓
HTML Output ← Narrative Writer ← (Feedback Loop) ← Chart Generator
```

## 🤖 Agents Description

### 1. **Data Profiling Agent**
- **Purpose**: Creates a comprehensive data quality report from the CSV
- **Functionality**: 
  - Identifies missing values, data types, duplicates, and outliers
  - Classifies fields as categorical, numerical, time-series, or text
  - Assesses visualization readiness for different chart types
  - Identifies data story potential and key patterns

### 2. **Visualization Strategist Agent**
- **Purpose**: Creates a list of top 5-6 charts for compelling data storytelling
- **Functionality**:
  - Recommends optimal chart types based on data profile
  - Prioritizes visualizations for narrative flow
  - Provides Altair specifications for each chart
  - Ensures charts build upon each other for complete story

### 3. **Altair Chart Generator Agent**
- **Purpose**: Generates charts using Altair library
- **Functionality**:
  - Executes Altair specifications from recommendations
  - Handles chart styling and formatting
  - Converts charts to JSON for HTML embedding
  - Manages error handling for failed generations

### 4. **Chart Evaluator Agent**
- **Purpose**: Evaluates charts for quality and completeness
- **Functionality**:
  - Scores chart quality and completeness
  - Provides feedback for improvements
  - Implements feedback loop for chart refinement
  - Approves or rejects charts based on criteria

### 5. **Narrative Writer Agent**
- **Purpose**: Writes compelling narratives for each chart with key insights
- **Functionality**:
  - Creates engaging narratives for successful charts
  - Provides business implications and recommendations
  - Explains how each chart contributes to the overall story
  - Handles error cases for failed chart generations

## 📁 File Structure

```
submission/
├── langgraph_agent.py          # Main workflow implementation
├── run_langgraph_workflow.py   # Runner script
├── requirements.txt            # Dependencies
├── README_LangGraph.md         # This documentation
├── helpers.py                  # LLM helper functions
└── dataset.csv                 # Sample dataset
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Workflow

```bash
python run_langgraph_workflow.py
```

### 3. View Results

The workflow generates an `output.html` file containing:
- Dataset overview and data quality summary
- Interactive Altair visualizations
- Compelling narratives for each chart
- Executive summary with key findings
- Business recommendations

## 🔧 Customization

### Using Custom Datasets

The workflow supports multiple ways to specify input datasets:

#### Option 1: Default Dataset
```python
from langgraph_agent import LangGraphAgent

agent = LangGraphAgent()
agent.initialize()  # Uses default "dataset.csv"
result = agent.process()
```

#### Option 2: Custom Filename (searches multiple locations)
```python
agent = LangGraphAgent()
agent.initialize(dataset_filename="my_data.csv")
result = agent.process()
```

#### Option 3: Full Path
```python
agent = LangGraphAgent()
agent.initialize(dataset_path="/full/path/to/your_dataset.csv")
result = agent.process()
```

#### Option 4: URL
```python
agent = LangGraphAgent()
agent.initialize(dataset_url="https://example.com/dataset.csv")
result = agent.process()
```

#### Option 5: Custom Default Filename
```python
agent = LangGraphAgent(default_dataset_filename="my_default_data.csv")
agent.initialize()  # Will use "my_default_data.csv" as default
result = agent.process()
```

#### Option 6: Relative Path
```python
agent = LangGraphAgent()
agent.initialize(dataset_path="./data/my_dataset.csv")
result = agent.process()
```

### Modifying Agent Behavior

Each agent can be customized by modifying the prompts in the respective functions:

- `data_profiling_agent()` - Customize data quality analysis
- `visualization_strategist_agent()` - Modify chart recommendation logic
- `altair_chart_generator_agent()` - Adjust chart generation parameters
- `chart_evaluator_agent()` - Change evaluation criteria
- `narrative_writer_agent()` - Customize narrative style

## 📊 Output Format

The workflow generates a comprehensive HTML report with:

1. **Dataset Overview**: Shape, columns, data quality metrics
2. **Interactive Charts**: Altair visualizations embedded with Vega-Lite
3. **Narratives**: Detailed insights and business implications
4. **Executive Summary**: Key findings and recommendations
5. **Error Handling**: Clear indication of any failed chart generations

## 🔄 Feedback Loop

The workflow includes an intelligent feedback loop:

- Chart Evaluator can send charts back to Chart Generator for refinement
- Maximum iterations are configurable (default: 3)
- Failed charts are clearly marked and explained
- Successful charts proceed to narrative writing

## 🛠️ Technical Details

### State Management
The workflow uses a comprehensive `AgentState` TypedDict to manage:
- Dataset and metadata
- Agent outputs and intermediate results
- Control flow variables
- Evaluation results

### Error Handling
- Graceful fallbacks when LLM is unavailable
- Comprehensive error reporting for failed chart generations
- Robust JSON parsing with fallback options
- Clear error messages in final HTML report

### Dependencies
- **LangGraph**: Workflow orchestration
- **Altair**: Chart generation
- **Pandas**: Data manipulation
- **LangChain**: LLM integration
- **Vega-Lite**: Chart rendering in HTML

## 🎯 Use Cases

This workflow is ideal for:

- **Business Intelligence**: Automated data analysis and reporting
- **Data Exploration**: Quick insights from new datasets
- **Report Generation**: Automated creation of data-driven reports
- **Data Storytelling**: Compelling narratives from raw data
- **Quality Assessment**: Comprehensive data profiling and validation

## 🔮 Future Enhancements

Potential improvements include:

- Support for additional chart libraries (Plotly, Matplotlib)
- Advanced statistical analysis integration
- Custom template support for HTML reports
- Real-time data source integration
- Multi-language narrative support
- Advanced feedback loop optimization

## 📝 Example Output

The generated HTML report includes:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Data-Driven Report</title>
    <!-- Vega-Lite CDN for interactive charts -->
</head>
<body>
    <h1>📊 Data-Driven Analysis Report</h1>
    
    <!-- Dataset Overview -->
    <div class="data-profile">
        <h2>📈 Dataset Overview</h2>
        <!-- Data quality metrics -->
    </div>
    
    <!-- Interactive Charts with Narratives -->
    <div class="chart-container">
        <h2>📊 Chart Title</h2>
        <div id="chart_0"></div>
        <div class="narrative">
            <!-- Compelling narrative with insights -->
        </div>
    </div>
    
    <!-- Executive Summary -->
    <div class="insight-box">
        <h2>🎯 Executive Summary</h2>
        <!-- Key findings and recommendations -->
    </div>
</body>
</html>
```

## 🤝 Contributing

To contribute to this workflow:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of the visxgenai template and follows the same licensing terms.
