import os
import sys
from dotenv import load_dotenv

from src.exception import CustomException
from src.logger import setup_logger


logger = setup_logger()

# Load values from .env locally.
load_dotenv(override=False)

# config.py is located at:
# project_root/src/utils/config.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOCUMENTS_DIR = os.path.join(BASE_DIR,"documents")
POLICY_DOC_DIR = os.path.join(DOCUMENTS_DIR,"policy_documents")
CONVERSATION_DOC_DIR = os.path.join(DOCUMENTS_DIR,"conversation_document")
CONVERSATION_JSON_PATH = os.path.join(CONVERSATION_DOC_DIR,"gsh_support_chats_jan_jun_2024.json")


def validate_documents(config: dict) -> None:
    """
    Validate that the required policy PDFs and conversation
    JSON file exist before running the ingestion pipeline.
    """

    policy_directory = config["POLICY_DOC_DIR"]
    conversation_json_path = config["CONVERSATION_JSON_PATH"]

    if not os.path.isdir(policy_directory):
        raise FileNotFoundError(
            "Policy document directory was not found: " f"{policy_directory}")

    policy_files = [file_name for file_name in os.listdir(policy_directory) if file_name.lower().endswith(".pdf")]

    if not policy_files:
        raise FileNotFoundError(
            "No PDF policy documents were found in: " f"{policy_directory}")

    if not os.path.isfile(conversation_json_path):
        raise FileNotFoundError(
            "Conversation JSON file was not found: " f"{conversation_json_path}")

    logger.info("Document validation completed successfully. " "%s policy PDF files and one conversation JSON file were found.", len(policy_files))


def get_config(validate_document_files: bool = False) -> dict:
    """
    Load and validate the Aurora chatbot configuration.

    Parameters
    ----------
    validate_document_files:
        Set to True when running the data-ingestion pipeline.
        This checks that the policy PDFs and JSON file exist.
    """

    try:
        logger.info("Loading Aurora chatbot configuration.")

        config = {
            # API credentials
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),

            # OpenAI settings
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL","gpt-5.4-mini"),
            "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL","text-embedding-3-small"),

            # Pinecone settings
            "PINECONE_POLICY_INDEX": os.getenv("PINECONE_POLICY_INDEX","aurora"),
            "PINECONE_CONVERSATION_INDEX": os.getenv("PINECONE_CONVERSATION_INDEX","aurora-conversation-data"),
            "PINECONE_DIMENSION": int(os.getenv("PINECONE_DIMENSION","1536")),
            "PINECONE_METRIC": os.getenv("PINECONE_METRIC","cosine"),
            "PINECONE_CLOUD": os.getenv("PINECONE_CLOUD","aws"),
            "PINECONE_REGION": os.getenv("PINECONE_REGION","us-east-1"),

            # Document paths
            "POLICY_DOC_DIR": os.getenv("POLICY_DOC_DIR",POLICY_DOC_DIR),
            "CONVERSATION_JSON_PATH": os.getenv("CONVERSATION_JSON_PATH",CONVERSATION_JSON_PATH),

            # SQLite database paths
            "MEMORY_DB_PATH": os.path.abspath(os.getenv("MEMORY_DB_PATH",os.path.join(BASE_DIR,"conversation_memory.db"))),
            "CACHE_DB_PATH": os.path.abspath(os.getenv("CACHE_DB_PATH",os.path.join(BASE_DIR,"aurora_cache.db"))),

            # Cache and confidence settings
            "MAX_CACHE_ENTRIES": int(os.getenv("MAX_CACHE_ENTRIES","100")),
            "CONFIDENCE_THRESHOLD": float(os.getenv("CONFIDENCE_THRESHOLD","0.7")),
            }

        # Validate required API keys.
        required_keys = ["OPENAI_API_KEY","PINECONE_API_KEY"]

        missing_keys = [key for key in required_keys if not config.get(key)]

        if missing_keys:
            raise ValueError("Missing required environment variables: " + ", ".join(missing_keys))

        # Validate basic configuration values.
        if config["PINECONE_DIMENSION"] <= 0:
            raise ValueError("PINECONE_DIMENSION must be greater than zero.")
        
        if config["MAX_CACHE_ENTRIES"] <= 0:
            raise ValueError("MAX_CACHE_ENTRIES must be greater than zero.")

        if not 0 <= config["CONFIDENCE_THRESHOLD"] <= 1:
            raise ValueError("CONFIDENCE_THRESHOLD must be between 0 and 1.")

        if validate_document_files:
            validate_documents(config)

        logger.info("Aurora chatbot configuration loaded successfully.")
        return config

    except Exception as error:
        logger.exception("Failed to load Aurora chatbot configuration.")

        raise CustomException(error,sys) from error