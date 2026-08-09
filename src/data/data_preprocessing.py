import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.data.data_cleaning import DataCleaning
from src.exception import CustomException
from src.logger import setup_logger

logger = setup_logger()

def get_policy_chunks():
    """
    Load, clean and split policy documents into chunks.
    """
    try:
        logger.info("Loading, cleaning and chunking policy documents...")
        data_cleaning = DataCleaning()
        cleaned_documents = (data_cleaning.clean_policy_documents())

        if not cleaned_documents:
            raise ValueError("No cleaned policy documents were available for chunking.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,separators=["\n\n","\n",". "," ",""])
        chunks = text_splitter.split_documents(cleaned_documents)

        if not chunks:
            raise ValueError("No chunks were created from the policy documents.")
        
        logger.info("Created %s chunks from the policy documents.", len(chunks))
        return chunks

    except Exception as e:
        logger.error("Failed to create policy-document chunks.")
        raise CustomException(e,sys) from e

#get_policy_chunks()
