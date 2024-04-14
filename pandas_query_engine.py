import os
import logging
import sys

import pandas as pd
from IPython.display import Markdown, display
from llama_index.core.query_engine import PandasQueryEngine


# API_KEY = os.environ["api_key"]
# PROJECT_ID = os.environ["project_id"]

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

data = pd.read_csv('data.csv')

query_engine = PandasQueryEngine(df=data, verbose=True, synthesize_response = True)

query = "Which country has this highest number of channels?"

response =  query_engine.query(
    query,
)

display(Markdown(f"<b>{response}</b>"))


