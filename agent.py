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

