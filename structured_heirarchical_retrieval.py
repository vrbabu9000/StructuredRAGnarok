"""Doing RAG well over multiple documents is hard. A general framework is given a user query, first select the relevant documents before selecting the content inside.

But selecting the documents can be tough - how can we dynamically select documents based on different properties depending on the user query?

In this notebook we show you our multi-document RAG architecture:

Represent each document as a concise metadata dictionary containing different properties: an extracted summary along with structured metadata.
Store this metadata dictionary as filters within a vector database.
Given a user query, first do auto-retrieval - infer the relevant semantic query and the set of filters to query this data (effectively combining text-to-SQL and semantic search)."""


import weaviate

# cloud
auth_config = weaviate.AuthApiKey(
    api_key="XRa15cDIkYRT7AkrpqT6jLfE4wropK1c1TGk"
)
client = weaviate.Client(
    "https://llama-index-test-v0oggsoz.weaviate.network",
    auth_client_secret=auth_config,
)

class_name = "LlamaIndex_docs"

from llama_index.core import SummaryIndex
from llama_index.core.async_utils import run_jobs
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import IndexNode
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)


async def aprocess_doc(doc, include_summary: bool = True):
    """Process doc."""
    metadata = doc.metadata

    date_tokens = metadata["created_at"].split("T")[0].split("-")
    year = int(date_tokens[0])
    month = int(date_tokens[1])
    day = int(date_tokens[2])

    assignee = (
        "" if "assignee" not in doc.metadata else doc.metadata["assignee"]
    )
    size = ""
    if len(doc.metadata["labels"]) > 0:
        size_arr = [l for l in doc.metadata["labels"] if "size:" in l]
        size = size_arr[0].split(":")[1] if len(size_arr) > 0 else ""
    new_metadata = {
        "state": metadata["state"],
        "year": year,
        "month": month,
        "day": day,
        "assignee": assignee,
        "size": size,
    }

    # now extract out summary
    summary_index = SummaryIndex.from_documents([doc])
    query_str = "Give a one-sentence concise summary of this issue."
    query_engine = summary_index.as_query_engine(
        llm=OpenAI(model="gpt-3.5-turbo")
    )
    summary_txt = await query_engine.aquery(query_str)
    summary_txt = str(summary_txt)

    index_id = doc.metadata["index_id"]
    # filter for the specific doc id
    filters = MetadataFilters(
        filters=[
            MetadataFilter(
                key="index_id", operator=FilterOperator.EQ, value=int(index_id)
            ),
        ]
    )

    # create an index node using the summary text
    index_node = IndexNode(
        text=summary_txt,
        metadata=new_metadata,
        obj=doc_index.as_retriever(filters=filters),
        index_id=doc.id_,
    )

    return index_node


async def aprocess_docs(docs):
    """Process metadata on docs."""

    index_nodes = []
    tasks = []
    for doc in docs:
        task = aprocess_doc(doc)
        tasks.append(task)

    index_nodes = await run_jobs(tasks, show_progress=True, workers=3)

    return index_nodes