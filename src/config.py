"""
Configuration file for the RAG system.
"""

from dataclasses import dataclass

@dataclass
class GlobalConfig:

    # Secrets
    project_scope_name: str = "langgraph_docs_rag"
    databricks_key_name: str = "databricks_rag"

    # Databricks PAT token placeholder
    secret: str = ""

    # Data storage and chunking dataframe setup
    catalog: str = "py_docs"
    schema: str = "default"
    volume_name: str = "raw_data_volume"
    source_file_name: str = "langgraph_docs.csv"
    text_table_name: str = "langgraph_docs_text"
    primary_key: str = "id"
    text_column: str = "text"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Vector search endpoint setup
    vector_search_endpoint_name: str = "doc_vector_endpoint"
    index_name: str = "langgraph_docs_idx"
    pipeline_type: str = "TRIGGERED"
    embedding_model_endpoint_name: str = "databricks-gte-large-en"

    # LLM setup
    chat_model: str = "databricks-llama-4-maverick"
    max_tokens: int = 500
    rag_model_name: str = "langgraph_docs_model"