# Databricks notebook source
# DBTITLE 1,Install Dependencies
# MAGIC %pip install mlflow==2.10.1 langchain==0.1.5 databricks-vectorsearch==0.22 databricks-sdk==0.18.0 mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Secrets
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

#w.secrets.put_secret("langgraph_docs_rag","databricks_rag",string_value ="")

# COMMAND ----------

index_name="py_docs.default.langgraph_docs_idx"
VECTOR_SEARCH_ENDPOINT_NAME="doc_vector_endpoint"

# COMMAND ----------

# DBTITLE 1,Set needed parameters
import os

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(scope="langgraph_docs_rag", key="databricks_rag")

# COMMAND ----------

# DBTITLE 1,Build Retriever
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings

embedding_model = DatabricksEmbeddings(endpoint="databricks-gte-large-en")

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="text", embedding=embedding_model
    )
    return vectorstore.as_retriever()

# COMMAND ----------

# DBTITLE 1,Create the RAG Langchain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks

chat_model = ChatDatabricks(endpoint="databricks-llama-4-maverick", max_tokens = 200)

TEMPLATE = """You are an assistant for programmers using langchain and langgraph. You are answering questions about langgraph or langchain that you have data on. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. 

Use the following pieces of context to answer the question at the end:

{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# COMMAND ----------

question = {"query": "How can I create a PromptTemplate in langchain"}
answer = chain.run(question)
print(answer)

# COMMAND ----------

# DBTITLE 1,Test Langchain
question = {"query": "My gas burner won't light up?"}
answer = chain.run(question)
print(answer)

# COMMAND ----------

# DBTITLE 1,Register our Chain as a model to Unity Catalog
from mlflow.models import infer_signature
import mlflow
import langchain

mlflow.set_registry_uri("databricks-uc")
model_name = "py_docs.default.langgraph_docs_model"

with mlflow.start_run(run_name="appliance_chatbot_run") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )

# COMMAND ----------


