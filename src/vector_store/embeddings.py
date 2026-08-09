import sys
from langchain_openai import OpenAIEmbeddings
from src.utils.config import get_config

from src.exception import CustomException
from src.logger import setup_logger

logging = setup_logger()

def get_embeddings():
    try:
        logging.info("Initializing embeddings...")
        config = get_config()
        embedding = OpenAIEmbeddings(
            openai_api_key=config["OPENAI_API_KEY"]
        )
        test_vector = embedding.embed_query("Aurora Hospitality")

        print(f"Embedding dimension: {len(test_vector)}")
        logging.info("Embeddings initialized successfully")
        return embedding
    
    except Exception as e:
        logging.error(f"failed to initialize embeddings: {str(e)}")
        raise CustomException(str(e), sys)