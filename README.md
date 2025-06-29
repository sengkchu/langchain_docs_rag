# Langchain/Langgraph Documentation RAG

RAG Chatbot system built on Langchain/Langgraph documentations using Databricks.

Langchain/Langgraph documentation data scraped with Apify

Original code from the following guides: 

1. https://www.youtube.com/watch?v=p4qpIgj5Zjg
2. https://github.com/azbarbarian2020/RAG-Chatbot

#### src folder:

`0.databricks_secrets.ipynb`
Creates the secret scope and adds the PAT token.

`1.csv_to_table.ipynb`
Converts Langgraph/Langchain documentation CSV data to a table on Unity Catalog.

`2.vector_idx.ipynb`
Creates the vector search endpoint and the vector search index.

`3.rag.ipynb`
Creates the RAG chatbot.

`config.py` 
Configuration file for the notebooks above.