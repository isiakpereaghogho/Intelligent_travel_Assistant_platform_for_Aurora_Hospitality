from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader
import sys
import os
import json
from src.utils.config import get_config
from src.exception import CustomException
from src.logger import setup_logger

logger = setup_logger()
config = get_config(validate_document_files=True)

class DataIngestion:
    def __init__(self):
        self.policy_path = Path(config['POLICY_DOC_DIR'])
        self.conversation_path = Path(config['CONVERSATION_JSON_PATH'])

    def load_policy_documents(self):
        """
        Load all PDF policy documents from the supplied directory.
        """
        try:
            logger.info("loading policy data...")
            if not self.policy_path.exists():
                raise FileNotFoundError(f"Policy directory was not found: {self.policy_path}")

            loader = PyPDFDirectoryLoader(str(self.policy_path))
            policy_document = loader.load()

            if not policy_document:
                raise ValueError("No policy documents were loaded.")

            logger.info(f"Loaded {len(policy_document)} policy document pages.")

            return policy_document
        except Exception as e:
            logger.error("error occured while loading policy documents")
            raise CustomException(e,sys)


    def load_conversation_data(self):
        """
        Load historical conversations from a JSON file.
        """
        try:
            logger.info("loading conversation data...")
            #file_path = Path(json_path)

            if not self.conversation_path.exists():
                raise FileNotFoundError(
                    f"Conversation file was not found: {self.conversation_path}")

            with self.conversation_path.open("r",encoding="utf-8") as file:
                conversation_document = json.load(file)

            if not isinstance(conversation_document, list):
                raise ValueError("The conversation JSON must contain a list of conversation records.")

            logger.info(f"Loaded {len(conversation_document)} conversation records.")

            return conversation_document
        except Exception as e:
            logger.error(f"error occured while loading conversation data: {e}")
            raise CustomException(e, sys)

# if __name__ == "__main__":

#     data_ingestion = DataIngestion()

#     policy_documents = data_ingestion.load_policy_documents()
#     print(f"Policy documents loaded: {len(policy_documents)}")

#     conversation_documents = data_ingestion.load_conversation_data()
#     print(f"Conversation records loaded: {len(conversation_documents)}")
    