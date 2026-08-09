import re
import unicodedata
from langchain_core.documents import Document
from src.exception import CustomException
from src.logger import setup_logger
import sys
from src.data.data_ingestion import DataIngestion
import json

logger = setup_logger()

class DataCleaning:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.policy_documents = (self.data_ingestion.load_policy_documents())
        self.conversation_documents = (self.data_ingestion.load_conversation_data())

    def clean_pdf_text(self, text):
        """
        Clean PDF-extracted text from a policy PDF while preserving useful document structure
        for chunking, retrieval, and RAG.
        """
        try:
            if not isinstance(text, str):
                return ""

            # Normalize Unicode representation
            text = unicodedata.normalize("NFKC", text)

            # Normalize line endings
            text = re.sub(r"\r\n?|\f", "\n", text)

            # Replace common PDF bullets and invisible characters
            text = re.sub(r"[\uf0a7\u2022\u25cf\u25aa\u25e6\u200b\u200c\u200d\ufeff]", " ", text)

            # Remove page labels such as "Page 2" or "Page 2 of 12"
            text = re.sub(r"\bPage\s+\d+(?:\s+of\s+\d+)?\b"," ", text, flags=re.IGNORECASE)

            # Remove common repeated headers and footers
            text = re.sub(r"\bConfidential\b", " ", text, flags=re.IGNORECASE)

            # Remove decorative separators and table borders
            text = re.sub(r"[_=-]{3,}", " ", text)
            text = re.sub(r"\|+", " ", text)

            # Reduce repeated dots, while retaining a normal ellipsis
            text = re.sub(r"\.{4,}", "...", text)

            # Remove URLs and email addresses
            text = re.sub(r"https?://\S+|www\.\S+", " ", text)
            text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " ", text)

            # Collapse spaces and tabs, but preserve newlines
            text = re.sub(r"[ \t]+", " ", text)

            # Remove spaces around newlines
            text = re.sub(r" *\n *", "\n", text)

            # Keep at most one blank line between paragraphs
            text = re.sub(r"\n{3,}", "\n\n", text)

            # Remove spaces before punctuation
            text = re.sub(r"\s+([.,!?;:])", r"\1", text)

            # Normalize spacing inside brackets
            text = re.sub(r"\(\s+", "(", text)
            text = re.sub(r"\s+\)", ")", text)
            logger.info("policy pdf text cleaned successfully...")
            return text.strip()
        except Exception as e:
            logger.error("error while cleaning policy pdf text")
            raise CustomException(e, sys)


    def clean_policy_documents(self):
        """
        Clean policy documents while preserving their metadata.
        """
        try:
            logger.info("Cleaning policy documents...")
            clean_docs = []
            for document in self.policy_documents:
                cleaned_text = self.clean_pdf_text(document.page_content)
                if cleaned_text:
                    clean_docs.append(Document(page_content=cleaned_text,metadata=document.metadata.copy()))
            logger.info("Cleaning policy documents successful...")
            return clean_docs
        except Exception as e:
            logger.error("error in cleaning policy documents")
            raise CustomException(e, sys)

    def clean_conversation_data(self):
        """
        Convert each JSON conversation record into a LangChain Document.
        """
        try:
            logger.info("Cleaning conversation data...")
            cleaned_conversation_docs = []
            for index, conversation in enumerate(self.conversation_documents):
                if not isinstance(conversation, dict):
                    logger.warning("Skipping record %s because it is not a dictionary.", index)
                    continue
                cleaned_record = {}
                for key, value in conversation.items():
                    if value is None:
                        cleaned_record[key] = ""
                    elif isinstance(value, (list, dict)):
                        cleaned_record[key] = json.dumps(value,ensure_ascii=False)
                    else:
                        cleaned_record[key] = " ".join( str(value).strip().split())
                page_content = "\n".join(f"{key}: {value}" for key, value in cleaned_record.items()if value)

                if not page_content:
                    logger.warning("Skipping empty conversation record at index %s.",index)
                    continue

                metadata = {"source": "historical_support_conversations","conversation_index": index}

                cleaned_doc = Document(page_content=page_content, metadata=metadata)

                cleaned_conversation_docs.append(cleaned_doc)

            if not cleaned_conversation_docs:
                raise ValueError("No conversation records remained after cleaning.")

            logger.info(
                "Successfully cleaned %s conversation records.", len(cleaned_conversation_docs))

            return cleaned_conversation_docs
        except Exception as e:
            logger.exception("Error in cleaning conversation data.")
            raise CustomException(e, sys) from e
             

if __name__ == "__main__":
    data_cleaning = DataCleaning()

    cleaned_policy_documents = (data_cleaning.clean_policy_documents())

    cleaned_conversation_documents = (data_cleaning.clean_conversation_data())

    print(f"Cleaned policy pages: {len(cleaned_policy_documents)}")

    print(f"Cleaned conversations: {len(cleaned_conversation_documents)}")

    if cleaned_policy_documents:
        print("\nFirst cleaned policy document:")
        print(cleaned_policy_documents[0].page_content[:500])

    if cleaned_conversation_documents:
        print("\nFirst cleaned conversation:")
        print(cleaned_conversation_documents[0].page_content[:500])