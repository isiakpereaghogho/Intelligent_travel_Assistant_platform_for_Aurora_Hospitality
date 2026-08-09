import sys
from langchain.tools import tool
from src.chains.agent_chains import create_conversation_chain
from src.exception import CustomException
from src.logger import setup_logger
import json

logger = setup_logger()

_conversation_chain = None

def get_conversation_agent():
    # Get or create conversation agent chain
    global _conversation_chain
    try:
        if _conversation_chain is None:
            logger.info("Initializing conversation agent...")
            _conversation_chain = create_conversation_chain()
            logger.info("Conversation agent initialized successfully")
            return _conversation_chain
    except Exception as e:
        logger.error(f"failed to get conversation agent: {e}")
        raise CustomException(e, sys) from e

@tool
def conversation_agent_tool(conversation_query: str, chat_history_text: str, guest_type: str, loyalty: str, city: str):
    """
    Analyze historical support conversations and recommend how a similar
    guest enquiry should be handled.
    """

    if not conversation_query or not conversation_query.strip():
        return {
            "observed_patterns": "",
            "response_style": "",
            "conversation": "No valid guest question was provided.",
            "confidence": 0.0
        }
    try:
        combined_query = f"""

        GUEST PROFILE:
        Guest Type: {guest_type}
        Loyalty Tier: {loyalty}
        City: {city}

        RECENT CONVERSATION WITH THE CURRENT GUEST:
        {chat_history_text or "No previous conversation is available."}

        CURRENT GUEST QUESTION:
        {conversation_query}
        """
        chain = get_conversation_agent()
        result = chain.invoke({
            "query": combined_query
        })

        raw_output = result.get("result", "")
                
        parsed_output = json.loads(raw_output)

        logger.info(f"Conversation agent completed for question: {conversation_query[:50]}...")

        return {
                "observed_patterns": parsed_output.get("observed_patterns", ""),
                "response_style": parsed_output.get("response_style", ""),
                "conversation": parsed_output.get("conversation", ""),
                "confidence": float(parsed_output.get("confidence", 0.5))
            }           
    except Exception as e:
        logger.error(f"Conversation agent failed for question: {conversation_query[:50]}... Error: {e}")
        raise CustomException(e, sys) from e