import sys
from langchain.tools import tool
from src.chains.agent_chains import create_policy_chain
from src.exception import CustomException
from src.logger import setup_logger
import json
from typing import Any

logger = setup_logger()

_policy_chain = None

def get_policy_agent():
    # Get or create policy agent chain
    global _policy_chain
    try:
        if _policy_chain is None:
            logger.info("Initializing policy agent...")
            _policy_chain = create_policy_chain()
            logger.info("Policy agent initialized successfully")
            return _policy_chain
    except Exception as e:
        logger.error(f"failed to get policy agent: {e}")
        raise CustomException(e, sys) from e


@tool
def policy_agent_tool(question: str, guest_type: str, loyalty: str, city: str, chat_history: list) -> dict[str, Any]:
    """
    Retrieve and interpret official Aurora Hospitality policy information
    using the current question, guest profile, city and recent conversation
    history.
    """
    try:
        if not question or not question.strip():
            return {
                "answer": "No question was provided.",
                "confidence": 0.0,
                "limitations": "The Policy Agent requires a valid question."
            }
        chain = get_policy_agent()
        result = chain.invoke(
            {
                "question": question.strip(),
                "chat_history": chat_history,
                "guest_type": guest_type,
                "loyalty": loyalty,
                "city": city
            }
        )

        raw_output = result.get("answer", "")       
        parsed_output = json.loads(raw_output)
        logger.info(f"Policy agent completed for question: {question[:50]}...")
        return {
                "answer": parsed_output.get("answer", ""),
                "confidence": float(parsed_output.get("confidence", 0.5)),
                "limitations": parsed_output.get("limitations", "")
            }

    except Exception as e:
        logger.error(f"Policy agent failed for question: {question[:50]}... Error: {e}")
        raise CustomException(e, sys) from e