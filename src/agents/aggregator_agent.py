import sys
import asyncio
import json
import time
from datetime import datetime, timezone
from src.agents.policy_agent import policy_agent_tool
from src.agents.conversation_agent import conversation_agent_tool 
from src.chains.agent_chains import create_aggregator_chain
from src.database.memory_db import ConversationMemory
from src.evaluators.agent_evaluator import RAGEvaluator
from src.chains.agent_chains import get_llm
from src.exception import CustomException
from src.logger import setup_logger
from src.utils.cache import(get_cached_response, get_cache_summary, get_cost_summary, cache_response, track_latency, is_cacheable_question, CONFIDENCE_THRESHOLD)


logger = setup_logger()

aggregator_chain = create_aggregator_chain()
memory_db = ConversationMemory()
evaluator = RAGEvaluator(llm=get_llm())
conversation_memory = ConversationMemory()

store_conversation = memory_db.store_memory

# COMBINED AGENT-CONFIDENCE CALCULATION

def normalise_confidence(value, default=0.5):
    """
    Convert a confidence value to a float and constrain it
    to the valid range of 0.0 to 1.0.

    If conversion fails, return the supplied default value.
    """

    try:
        confidence = float(value)
        return min(max(confidence, 0.0), 1.0)

    except Exception as e:
        logger.error("fails to convert confidence value...")
   
def parse_agent_output(output):
    """
    Convert an agent output into a Python dictionary.

    The output may already be a dictionary or may be returned
    as a JSON-formatted string.
    """
    try:
        if isinstance(output, dict):
            return output

        if isinstance(output, str):
            parsed_output = json.loads(output)

        if isinstance(parsed_output, dict):
            return parsed_output

        return {}
    
    except Exception as e:
         logger.error("failed to convert agent output")
            
def normalise_limitations(limitations):
    """
    Convert the Policy Agent's limitations into a clean list.

    Empty values and statements such as 'none' or
    'no limitations' are removed so that they do not
    incorrectly trigger a confidence penalty.
    """
    try:
        if limitations is None:
            return []

        if isinstance(limitations, str):
            limitations = [limitations]

        elif not isinstance(limitations, list):
            limitations = [str(limitations)]

        empty_limitation_values = {
            "",
            "none",
            "no limitation",
            "no limitations",
            "not applicable",
            "n/a"
        }

        cleaned_limitations = []

        for limitation in limitations:
            limitation_text = str(limitation).strip()

            if limitation_text.lower() not in empty_limitation_values:
                cleaned_limitations.append(limitation_text)

        return cleaned_limitations
    except Exception as e:
        logger.error("failed to convert limitations into a list")


def calculate_confidence_score(policy_output, conversation_output):
    """
    Calculate a combined confidence score from the Policy Agent
    and Conversation Agent outputs.

    The Policy Agent receives 60% of the total weight because
    official policy documents are more authoritative than
    historical conversation patterns.

    The Conversation Agent receives 40% of the total weight.

    A 20% reduction is applied when the Policy Agent reports
    meaningful limitations.
    """
    try:
        policy_data = parse_agent_output(policy_output)

        conversation_data = parse_agent_output(conversation_output)

        policy_confidence = normalise_confidence(policy_data.get("confidence"),default=0.5)

        conversation_confidence = normalise_confidence(conversation_data.get("confidence"), default=0.5)

        limitations = normalise_limitations(policy_data.get("limitations", []))

        # Give official policy knowledge greater weight.
        base_confidence = (
            policy_confidence * 0.60
            + conversation_confidence * 0.40
        )

        # Apply a 20% penalty when meaningful policy
        # limitations have been reported.
        limitation_penalty_applied = bool(limitations)

        if limitation_penalty_applied:
            base_confidence *= 0.80

        final_confidence = round(min(max(base_confidence, 0.0), 1.0), 2)

        logger.info(
        "Confidence calculation",
        f"Policy confidence: {policy_confidence:.2f}",
        "Conversation confidence: " f"{conversation_confidence:.2f}",
        f"Policy limitations: {limitations}",
        "Limitation penalty applied: " f"{limitation_penalty_applied}"
        f"Overall confidence: {final_confidence:.2f}"
        )
        return final_confidence
    except Exception as e:
        logger.error(f"failed to calculate confidence from agent..: {e}")

def check_escalation_needed(confidence_score):
    """
    Determine whether the generated response should be escalated for human review.
    """
    if confidence_score < CONFIDENCE_THRESHOLD:
        return (True, f"Low Confidence ({confidence_score:.2f})")

    return (False, f"OK ({confidence_score:.2f})")


def create_escalation_packet(question, confidence):
    """
    Create an escalation record for a low-confidence response.
    """
    return {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "question": question,
        "confidence_score": confidence,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "reason": "Confidence below threshold",
        "require_human_review": True
    }

def normalise_tool_result(result):
    """
    Convert a tool result into a Python dictionary where possible.

    Tool outputs may be dictionaries, JSON-formatted strings,
    or wrapper dictionaries containing an 'output' or 'result' field.
    """

    if isinstance(result, dict):
        # Some tool wrappers place the actual result inside
        # an 'output' field.
        if (
            "output" in result
            and len(result) == 1
        ):
            result = result["output"]

        # Only unwrap 'result' when it contains the complete
        # structured output rather than one field of it.
        elif (
            "result" in result
            and len(result) == 1
        ):
            result = result["result"]

        else:
            return result

    if isinstance(result, str):
        try:
            parsed_result = json.loads(result)

            if isinstance(parsed_result, dict):
                return parsed_result

        except json.JSONDecodeError:
            return {
                "raw_output": result
            }

    return {
        "raw_output": str(result)
    }

async def run_agents_parallel(
    question: str,
    guest_type: str,
    loyalty: str,
    city: str,
    chat_history_tuples,
    chat_history_text: str):
    """
        Run the Policy Agent and Conversation Agent concurrently.

        The Policy Agent Tool receives:
        - question
        - guest_type
        - loyalty
        - city
        - chat_history

        The Conversation Agent Tool receives:
        - conversation_query
        - guest_type
        - loyalty
        - city
        - chat_history_text
        """
    try:
        # Input expected by the Policy Agent Tool
        policy_input = {
            "question": question,
            "guest_type": guest_type,
            "loyalty": loyalty,
            "city": city,
            "chat_history": chat_history_tuples
        }

        # Input expected by the Conversation Agent Tool
        conversation_input = {
            "conversation_query": question,
            "guest_type": guest_type,
            "loyalty": loyalty,
            "city": city,
            "chat_history_text": (
                chat_history_text
                or "No previous conversation history."
            )
        }

        # Run both synchronous LangChain tools in worker threads
        # so they can execute concurrently.
        policy_task = asyncio.to_thread(
            policy_agent_tool.invoke,
            policy_input
        )

        conversation_task = asyncio.to_thread(
            conversation_agent_tool.invoke,
            conversation_input
        )

        policy_result, conversation_result = await asyncio.gather(policy_task, conversation_task)

        # Preserve the complete structured outputs, including
        # confidence scores and limitations.
        result = normalise_tool_result(
            policy_result, conversation_result
        )
        logger.info(f"parellel agents completed for question: {question[:50]}...")
        return result
    except Exception as e:
        logger.error(f"failed to run agents in parallel: {e}")
        raise CustomException(e, sys) from e

# Stores latency measurements across multiple requests
latency_metrics = {
    "parallel_agents": [],
    "reasoning_aggregator": [],
    "total_pipeline": []
}

async def agentic_rag_answer(question, guest_type, loyalty, city, session_id, use_cache=True, evaluate=True):
    """
    Run the complete Aurora Agentic RAG workflow.

    The function:

    1. Retrieves recent conversation memory.
    2. Determines whether the question is suitable for caching.
    3. Checks the persistent response cache.
    4. Runs the Policy and Conversation Agents in parallel.
    5. Calculates the combined confidence score.
    6. Checks whether human escalation is required.
    7. Calls the Reasoning Aggregator.
    8. Stores the interaction in conversation memory.
    9. Stores the generated response in the persistent cache.
    10. Tracks execution latency.
    """
    try:
        logger.info(f"Processing request - Session: {session_id}, Question: {question[:50]}...")
        # Start the total pipeline timer.
        total_start_time = time.time()

    
        # Retrieve recent conversation memory
        chat_history_tuples = memory_db.get_chat_history_tuples(session_id=session_id, limit=6)

        chat_history_text = memory_db.get_chat_history_text(session_id=session_id,limit=6)

    
        # Decide whether this question should use the cache
        should_use_cache = (use_cache and is_cacheable_question(question))

        # Check the persistent response cache
        if should_use_cache:
            cached_response = get_cached_response(
                question=question,
                guest_type=guest_type,
                loyalty=loyalty,
                city=city,
                session_id=session_id
            )

            if cached_response:
                cached_response["cache_hit"] = True

                # Measure the latency of this cache retrieval,
                # rather than returning the original generation time.
                cached_response["latency_ms"] = ( time.time() - total_start_time) * 1000

                answer_text = (
                    cached_response.get("text")
                    or cached_response.get("answer")
                    or cached_response.get("result"))

                if not answer_text:
                    raise ValueError(
                        "The cached response does not contain "
                        "a recognised answer field."
                    )

                # A cache-hit response is still part of the current
                # conversation and should be stored in memory.
                store_conversation(session_id, "user", question)

                store_conversation(session_id, "assistant", answer_text)

                print("Cache hit: previously stored response returned.")

                return cached_response

        # Run both specialist agents in parallel
        parallel_agents_start = time.time()

        policy_output, conversation_output = (
            await run_agents_parallel(
                question=question,
                guest_type=guest_type,
                loyalty=loyalty,
                city=city,
                chat_history_tuples=chat_history_tuples,
                chat_history_text=chat_history_text
            )
        )

        parallel_agents_latency = (
            time.time() - parallel_agents_start
        ) * 1000


        # Calculate the combined confidence score
        confidence = calculate_confidence_score(policy_output=policy_output, conversation_output=conversation_output)

        # Check whether human escalation is required
        needs_escalation, escalation_reason = (check_escalation_needed(confidence))

    
        # Run the Reasoning Aggregator
        reasoning_aggregator_start = time.time()

        final_answer = aggregator_chain.invoke(
            {
                "policy_agent_output": json.dumps(
                    policy_output,
                    ensure_ascii=False,
                    default=str
                ),
                "conversation_agent_output": json.dumps(
                    conversation_output,
                    ensure_ascii=False,
                    default=str
                ),
                "chat_history_text": chat_history_text,
                "question": question
            }
        )

        reasoning_aggregator_latency = (
            time.time() - reasoning_aggregator_start
        ) * 1000

    
        # Safely extract the final answer
        if isinstance(final_answer, dict):
            answer_text = (
                final_answer.get("text")
                or final_answer.get("answer")
                or final_answer.get("result")
            )
        else:
            answer_text = str(final_answer)

        if not answer_text:
            raise ValueError(
                "No recognised answer field was returned by "
                f"aggregator_chain: {final_answer}"
            )

    
        # Create an escalation packet where necessary
        escalation_packet = None

        if needs_escalation:
            print(
                f"Escalation needed: {escalation_reason}"
            )

            escalation_packet = create_escalation_packet(
                question=question,
                confidence=confidence)

        else:
            print(f"Confidence score: {confidence:.2f}")

        # Save the interaction to conversation memory
        store_conversation(session_id, "user", question)

        store_conversation(session_id, "assistant", answer_text)

        # Calculate and store latency measurements
        total_pipeline_latency = (time.time() - total_start_time) * 1000
        latency_metrics["parallel_agents"].append(parallel_agents_latency)
        latency_metrics["reasoning_aggregator"].append(reasoning_aggregator_latency)
        latency_metrics["total_pipeline"].append(total_pipeline_latency)

    
        # Create the complete response object
        response = {
            "text": answer_text,
            "confidence": confidence,
            "needs_escalation": needs_escalation,
            "escalation_reason": (
                escalation_reason
                if needs_escalation
                else None
            ),
            "escalation_packet": escalation_packet,
            "latency_ms": total_pipeline_latency,
            "parallel_agents_latency_ms": (
                parallel_agents_latency
            ),
            "reasoning_aggregator_latency_ms": (
                reasoning_aggregator_latency
            ),
            "policy_output": policy_output,
            "conversation_output": conversation_output,
            "cache_hit": False
        }

        # Store the newly generated response in the cache
        if should_use_cache:
            cache_response(
                question=question,
                guest_type=guest_type,
                loyalty=loyalty,
                city=city,
                response=response,
                session_id=session_id
            )
            print("Response successfully stored in the " "persistent cache.")

        elif use_cache:
            print("Question was identified as context-dependent, " "so the response was not cached.")

        return response
    except Exception as e:
        logger.error(f"error: {e}")
        
async def get_system_stats():
    # Get system statistics including cache and cost metrics
    cache_stats = get_cache_summary()
    cost_summary = get_cost_summary()

    #print_cache_dashboard()

    return {"cache": cache_stats, "cost": cost_summary}