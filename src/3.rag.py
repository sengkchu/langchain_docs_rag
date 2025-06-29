# Databricks notebook source
# DBTITLE 1,Install Dependencies
# MAGIC %pip install mlflow==3.1.1 langchain==0.1.5 databricks-vectorsearch==0.56 databricks-sdk==0.55.0 mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports and setup
import os
import mlflow
import langchain
from mlflow.models import infer_signature
from mlflow.deployments import get_deploy_client
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks
from config import GlobalConfig

# Setup autolog
os.environ["PYSPARK_PIN_THREAD"] = "false"
mlflow.autolog()

# Set up notebook variables
cfg = GlobalConfig()
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
databricks_token = dbutils.secrets.get(scope=cfg.project_scope_name, key=cfg.databricks_key_name)

# COMMAND ----------

# DBTITLE 1,Build Retriever
embedding_model = DatabricksEmbeddings(endpoint=cfg.embedding_model_endpoint_name)

def get_retriever():

    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=databricks_token)

    vs_index = vsc.get_index(
        endpoint_name=cfg.vector_search_endpoint_name,
        index_name=f"{cfg.catalog}.{cfg.schema}.{cfg.index_name}"
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column=cfg.text_column, embedding=embedding_model
    )
    return vectorstore.as_retriever()

# COMMAND ----------

# DBTITLE 1,Create the RAG Langchain
chat_model = ChatDatabricks(endpoint=cfg.chat_model, max_tokens=cfg.max_tokens)

TEMPLATE = """You are an assistant for programmers using langchain and langgraph. You are answering questions about langgraph or langchain that you have data on. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. 

Use the following pieces of context to answer the question at the end:

{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# COMMAND ----------

# DBTITLE 1,Example Q/A #1
question_1 = {"query": "How can I create chunks for rag in langchain"}
answer = chain.run(question_1)
print(answer)

# COMMAND ----------

# DBTITLE 1,Example Q/A #2
question_2 = {"query": "My gas burner won't light up?"}
answer = chain.run(question_2)
print(answer)

# COMMAND ----------

# DBTITLE 1,Register our Chain as a model to Unity Catalog
mlflow.set_registry_uri("databricks-uc")
model_name = f"{cfg.catalog}.{cfg.schema}.{cfg.rag_model_name}"

with mlflow.start_run(run_name="langgraph_docs_rag") as run:
    signature = infer_signature(question_1, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch==0.56",
        ],
        input_example=question_1,
        signature=signature
    )

# COMMAND ----------

# DBTITLE 1,Create endpoint
client = get_deploy_client("databricks")
endpoint = client.create_endpoint(
    name=f"{cfg.rag_model_name}_endpoint",
    config={
        "served_entities": [
            {
                "name": f"lang-rag-endpoint",
                "entity_name": model_name,
                "entity_version": "5",
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }
        ]
    }
)

