import sys
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from src.utils.config import get_config
from src.vector_store.pinecone_client import get_policy_vectorstore, get_conversation_vectorstore
from src.exception import CustomException
from src.logger import setup_logger

logger = setup_logger()

def get_llm():
    try:
        logger.info("Initializing LLM...")
        config = get_config()
        llm = ChatOpenAI(model=config["OPENAI_MODEL"], temperature=0,
                         openai_api_key=config["OPENAI_API_KEY"])
        logger.info(f"LLM initialized successfully with model: {config['OPENAI_API_KEY']}")
        return llm
    except Exception as e:
        logger.error(f"failed to initialize LLM: (e)")
        raise CustomException(e, sys) from e

def create_policy_chain():
    # creating policy agent chain
    try:
        logger.info("creating policy chain...")
        llm = get_llm()
        policy_vectorstore = get_policy_vectorstore

        policy_prompt = PromptTemplate(input_variables=["context","question","guest_type","loyalty","city"],
        template="""
        You are a POLICY INTERPRETATION AGENT for Aurora Hospitality.

        ROLE
        - Interpret hotel policies strictly and accurately.
        - Use only the retrieved policy text.
        - Do not invent, infer, or soften policy information.
        - If some information is unavailable, state that clearly.
        - Refer the guest to customer support only for information that cannot be confirmed.
        - Do not mention retrieval systems, documents, or internal sources.

        GUEST CONTEXT
        Guest Type: {guest_type}
        Loyalty Tier: {loyalty}
        City: {city}

        POLICY DOCUMENTS
        {context}

        USER QUESTION
        {question}

        CONFIDENCE SCORING
        - 1.00: All parts of the answer are directly and clearly supported.
        - 0.80–0.99: Most parts are directly supported; only minor details are missing.
        - 0.60–0.79: The answer is partially supported, but important information is missing.
        - 0.30–0.59: Only a small portion of the question is supported.
        - 0.00–0.29: Little or no relevant policy information was found.

        Return ONLY a valid JSON object.

        Do not include

        - markdown
        - explanations
        - comments
        - code fences
        - ```json

        {{
            "answer": "Clear policy-based answer to the guest",
            "confidence": 0.0,
            "limitations": [],
            "policy_found":true
        }}
        """
        )

        policy_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=policy_vectorstore.as_retriever(),
        combine_docs_chain_kwargs={"prompt": policy_prompt},
        return_source_documents=False
        )
        logger.info("Policy chain created successfully...")
        return policy_chain
    
    except Exception as e:
        logger.error(f"error in creating policy chain")
        raise CustomException(e, sys) from e

def create_conversation_chain():
    # creating policy agent chain
    try:
        logger.info("Creating conversation chain...")
        llm = get_llm()
        conversation_vectorstore = get_conversation_vectorstore
        conversation_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are the CONVERSATION INTELLIGENCE AGENT for Aurora Hospitality.

        PURPOSE
        Analyse how similar guest enquiries were handled in previous support
        conversations and recommend an appropriate communication approach.

        ROLE
        - Identify useful patterns from similar historical conversations.
        - Focus on tone, uncertainty handling, escalation and resolution strategy.
        - Use historical conversations as behavioural guidance, not as authoritative policy.
        - Do not invent hotel rules, benefits, restrictions or property details.
        - If the historical context does not support part of the answer, clearly identify
        that limitation and recommend customer support only for the unsupported part.
        - Do not mention historical logs, retrieval systems, databases or internal sources.
        - Keep the proposed guest response polite, concise and helpful.

        HISTORICAL CONVERSATIONS
        {context}

        USER QUESTION
        {question}

        CONFIDENCE GUIDANCE
        - 1.00: Several closely matching successful conversations support the approach.
        - 0.80-0.99: Strongly similar conversations support most of the approach.
        - 0.60-0.79: Some relevant patterns exist, but important differences remain.
        - 0.30-0.59: Only limited or indirectly related examples were found.
        - 0.00-0.29: No useful historical pattern was found.

        Return only a valid JSON object.
        Do not use Markdown, code fences, comments or additional explanations.

        {{
            "observed_patterns": "Summary of relevant historical handling patterns",
            "response_style": "Recommended tone and communication style",
            "conversation": "Suggested guest-facing response",
            "confidence": 0.0
        }}
        """
        )
        
        conversation_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=conversation_vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": conversation_prompt},
            return_source_documents=False
        )
        logger.info("Conversation chain created successfully...")
        return conversation_chain
        
    except Exception as e:
        logger.error(f"error in creating conversation chain: {e}")
        raise CustomException(e, sys) from e

def create_aggregator_chain():
    # creating policy agent chain
    try:
        logger.info("Creating aggregator chain...")
        llm = get_llm()
      
        aggregator_prompt = PromptTemplate(input_variables=[
        "policy_agent_output",
        "conversation_agent_output",
        "chat_history_text",
        "question"],

        template="""
        You are the FINAL RESPONSE AGGREGATOR for Aurora Hospitality.

        Your task is to combine the authoritative policy interpretation with
        conversation-based communication guidance and produce one final response
        to the guest in English.

        DECISION RULES

        1. Treat the Policy Agent output as the authoritative source for hotel
        rules, eligibility, restrictions and permitted actions.

        2. Use the Conversation Agent output only to guide tone, empathy,
        explanation style, escalation and practical handling.

        3. If the outputs conflict, follow the Policy Agent output. The
        Conversation Agent may still guide how the answer is communicated.

        4. Use the confidence values internally to assess uncertainty, but never
        reveal confidence scores to the guest.

        5. If no relevant conversation pattern is available, answer using the
        Policy Agent output.

        6. If the policy information is incomplete or unclear, answer only the
        parts supported by the available information, include a brief
        disclaimer and recommend contacting Aurora Hospitality customer
        support for confirmation.

        7. If no relevant policy is available, do not treat historical
        conversations as official policy. Provide only safe general guidance
        and recommend customer support.

        INPUTS

        POLICY AGENT OUTPUT:
        {policy_agent_output}

        CONVERSATION AGENT OUTPUT:
        {conversation_agent_output}

        ORIGINAL USER QUESTION:
        {question}

        TASK

        Synthesize the inputs into one customer-facing response that:

        - directly answers the original question;
        - follows the official policy information;
        - is clear, practical and actionable;
        - sounds professional, natural and empathetic;
        - uses relevant conversation guidance without presenting it as policy;
        - includes a brief disclaimer only when necessary.

        FINAL ANSWER REQUIREMENTS

        - Write between 3 and 5 sentences.
        - Use clear, natural and customer-friendly English.
        - Do not mention agents, retrieval systems, documents, internal sources
        or confidence scores.
        - Do not expose internal reasoning.
        - Refer the guest to customer support only when clarification or further
        action is genuinely required.

        FINAL ANSWER:
        """
        )
        
        aggregator_chain = LLMChain(llm=llm, prompt=aggregator_prompt)
        logger.info("Aggregator chain created successfully...")
        return aggregator_chain
        
    except Exception as e:
        logger.error(f"error in creating aggregator chain: {e}")
        raise CustomException(e, sys) from e