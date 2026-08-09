import sys
import time
import os

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.data.data_preprocessing import get_policy_chunks
from src.data.data_cleaning import DataCleaning
from src.exception import CustomException
from src.logger import setup_logger
from src.utils.config import get_config
from src.vector_store.embeddings import get_embeddings


logger = setup_logger()

# Call get_config rather than assigning the function itself.
config = get_config()

# Load the embedding model once.
embedding = get_embeddings()

data_cleaning = DataCleaning()
cleaned_conversation_doc = data_cleaning.clean_conversation_data()


def get_pinecone_client() -> Pinecone:
    """
    Initialise and return the Pinecone client.
    """
    try:
        logger.info("Initialising Pinecone client...")

        client = Pinecone(api_key=config["PINECONE_API_KEY"])

        logger.info("Pinecone client initialised successfully.")
        return client

    except Exception as e:
        logger.error("Failed to initialise the Pinecone client.")
        raise CustomException(e,sys) from e

# Initialise the Pinecone client once.
pc = get_pinecone_client()

def wait_for_index_ready(index_name: str,timeout_seconds: int = 120) -> None:
    """
    Wait until a Pinecone index becomes ready.
    """
    try:
        logger.info(
            "Waiting for Pinecone index '%s' to become ready.", index_name)
        start_time = time.time()

        while True:
            index_description = pc.describe_index(index_name)
            status = index_description.status

            # Supports both dictionary-style and object-style Pinecone SDK responses.
            if isinstance(status, dict):
                ready = status.get("ready", False)
            else:
                ready = getattr(status,"ready",False)

            if ready:
                logger.info("Pinecone index '%s' is ready.",index_name)
                return

            elapsed_time = time.time() - start_time

            if elapsed_time >= timeout_seconds:
                raise TimeoutError(
                    f"Pinecone index '{index_name}' was not "
                    f"ready after {timeout_seconds} seconds.")
            time.sleep(2)

    except Exception as e:
        logger.error("An error occurred while waiting for index '%s'.",index_name)
        raise CustomException(e,sys) from e


def wait_for_index_deletion(index_name: str, timeout_seconds: int = 120) -> None:
    """
    Wait until a deleted Pinecone index no longer exists.
    """
    try:
        start_time = time.time()

        while index_name in pc.list_indexes().names():
            elapsed_time = time.time() - start_time

            if elapsed_time >= timeout_seconds:
                raise TimeoutError(
                    f"Pinecone index '{index_name}' was not " f"deleted after {timeout_seconds} seconds.")

            logger.info("Waiting for Pinecone index '%s' to be deleted.",index_name)
            time.sleep(2)

    except Exception as e:
        logger.error("An error occurred while waiting for index deletion.")
        raise CustomException(e,sys) from e


def create_index(index_name: str) -> None:
    """
    Create a Pinecone serverless index.
    """

    try:
        logger.info("Creating Pinecone index '%s'.",index_name)

        pc.create_index(name=index_name,
            dimension=config["PINECONE_DIMENSION"],
            metric=config["PINECONE_METRIC"],
            spec=ServerlessSpec(cloud=config["PINECONE_CLOUD"],region=config["PINECONE_REGION"]))
        wait_for_index_ready(index_name)
        logger.info("Created Pinecone index '%s' successfully.",index_name)

    except Exception as e:
        logger.error("Failed to create Pinecone index '%s'.", index_name)
        raise CustomException(e,sys) from e

def add_documents_to_index(index_name: str,chunks,embedding_model) -> PineconeVectorStore:
    """
    Add document chunks to a Pinecone index.
    """
    try:
        if not chunks:
            raise ValueError("No chunks were supplied for Pinecone indexing.")
        logger.info("Adding %s chunks to Pinecone index '%s'.", len(chunks),index_name)
        vector_store = (PineconeVectorStore.from_documents(
                documents=chunks,
                index_name=index_name,
                embedding=embedding_model))
        logger.info("Successfully added %s chunks to index '%s'.", len(chunks),index_name)
        return vector_store

    except Exception as e:
        logger.error("Failed to add documents to index '%s'.", index_name)
        raise CustomException(e,sys) from e


def get_total_vector_count(index_name: str) -> int:
    """
    Return the total number of vectors stored in an index.
    """

    try:
        index = pc.Index(index_name)

        stats = index.describe_index_stats()

        if isinstance(stats, dict):
            return int(stats.get("total_vector_count",0))

        return int(getattr(stats,"total_vector_count",0))

    except Exception as e:
        logger.error("Failed to retrieve statistics for index '%s'.",index_name)
        raise CustomException(e,sys) from e


def get_policy_vectorstore() -> PineconeVectorStore:
    """
    Create or retrieve the policy-document Pinecone vector store.

    If the index:
    - does not exist, create it and upload documents;
    - exists with the wrong dimension, recreate it;
    - exists but is empty, upload documents;
    - exists and contains vectors, reuse it.
    """

    try:
        logger.info("Initialising the policy vector store...")
        index_name = config["PINECONE_POLICY_INDEX"]

        expected_dimension = config["PINECONE_DIMENSION"]
        chunks = get_policy_chunks()
        existing_indexes = (pc.list_indexes().names())

        if index_name not in existing_indexes:
            logger.info("Index '%s' does not exist.",index_name)
            create_index(index_name)
            vector_store = add_documents_to_index(
                index_name=index_name,
                chunks=chunks,
                embedding_model=embedding)
            return vector_store
        logger.info("Index '%s' already exists.",index_name)
        index_description = pc.describe_index(index_name)

        if isinstance(index_description, dict):
            existing_dimension = (index_description.get("dimension"))
            existing_metric = (index_description.get("metric"))
        else:
            existing_dimension = getattr(index_description, "dimension",None)
            existing_metric = getattr(index_description, "metric", None)

        logger.info(
            "Existing index dimension: %s; metric: %s.", existing_dimension,existing_metric)

        if existing_dimension != expected_dimension:
            logger.warning(
                "Index '%s' has dimension %s, but dimension %s is required. Recreating the index.", index_name, existing_dimension, expected_dimension)

            pc.delete_index(index_name)

            wait_for_index_deletion(index_name)

            create_index(index_name)

            vector_store = add_documents_to_index(index_name=index_name, chunks=chunks, embedding_model=embedding)
            return vector_store
        total_vector_count = get_total_vector_count(index_name)

        if total_vector_count == 0:
            logger.info("Index '%s' exists but is empty.", index_name)

            vector_store = add_documents_to_index(
                index_name=index_name, chunks=chunks, embedding_model=embedding)
        else:
            logger.info(
                "Index '%s' already contains %s vectors. Skipping document upload.", index_name, total_vector_count)

            vector_store = (
                PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding))
        return vector_store

    except Exception as e:
        logger.error("Failed to initialise the policy vector store.")
        raise CustomException(e,sys) from e

    # storing conversational data to pinecone vector database
def get_conversation_vectorstore() -> PineconeVectorStore:
    try:
        logger.info("Initialising the conversation vector store...")
        index_name = config["PINECONE_CONVERSATION_INDEX"]

        expected_dimension = config["PINECONE_DIMENSION"]
        #chunks = get_policy_chunks()
        existing_indexes = (pc.list_indexes().names())

        if index_name not in existing_indexes:
            logger.info("Index '%s' does not exist.",index_name)
            create_index(index_name)
            vector_store = add_documents_to_index(
                index_name=index_name,
                cleaned_conversation_doc=cleaned_conversation_doc,
                embedding_model=embedding)
            return vector_store
        logger.info("Index '%s' already exists.",index_name)
        index_description = pc.describe_index(index_name)

        if isinstance(index_description, dict):
            existing_dimension = (index_description.get("dimension"))
            existing_metric = (index_description.get("metric"))
        else:
            existing_dimension = getattr(index_description, "dimension",None)
            existing_metric = getattr(index_description, "metric", None)

        logger.info(
            "Existing index dimension: %s; metric: %s.", existing_dimension,existing_metric)

        if existing_dimension != expected_dimension:
            logger.warning(
                "Index '%s' has dimension %s, but dimension %s is required. Recreating the index.", index_name, existing_dimension, expected_dimension)

            pc.delete_index(index_name)

            wait_for_index_deletion(index_name)

            create_index(index_name)

            vector_store = add_documents_to_index(index_name=index_name, cleaned_conversation_doc=cleaned_conversation_doc, embedding_model=embedding)
            return vector_store
        total_vector_count = get_total_vector_count(index_name)

        if total_vector_count == 0:
            logger.info("Index '%s' exists but is empty.", index_name)

            vector_store = add_documents_to_index(
                index_name=index_name, cleaned_conversation_doc=cleaned_conversation_doc, embedding_model=embedding)
        else:
            logger.info(
                "Index '%s' already contains %s vectors. Skipping document upload.", index_name, total_vector_count)

            vector_store = (
                PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding))
        return vector_store

    except Exception as e:
        logger.error("Failed to initialise the conversation vector store.")
        raise CustomException(e,sys) from e
    
    
if __name__ == "__main__":
    policy_vectorstore = get_policy_vectorstore()
    print("Policy vector store initialised successfully.")

    conversation_vectorstore = get_conversation_vectorstore()
    print("Conversation vector store initialised successfully.")