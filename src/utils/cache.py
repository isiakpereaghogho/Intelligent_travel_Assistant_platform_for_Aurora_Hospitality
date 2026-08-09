import hashlib
import json
import sqlite3
import sys

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.database.cache_db import initialise_cache_database, get_cache_connection
from src.exception import CustomException
from src.logger import setup_logger
from src.utils.config import get_config


logger = setup_logger()
config = get_config()

# CONFIGURATION

CACHE_DB_PATH = Path(config.get("CACHE_DB_PATH", "aurora_cache.db"))
MAX_CACHE_ENTRIES = int(config.get("MAX_CACHE_ENTRIES", 1000))
CONFIDENCE_THRESHOLD = float(config.get("CONFIDENCE_THRESHOLD", 0.70))

INPUT_COST_PER_1K = float(config.get("INPUT_COST_PER_1K", 0.00015))

OUTPUT_COST_PER_1K = float(config.get("OUTPUT_COST_PER_1K", 0.00060))

# RUNTIME METRICS
cache_metrics: Dict[str, int] = {
    "hits": 0,
    "misses": 0
    }
latency_metrics: Dict[str, List[float]] = defaultdict(list)
token_usage_log: List[Dict[str, Any]] = []
retrieval_results: List[Dict[str, Any]] = []
generation_results: List[Dict[str, Any]] = []

def normalise_cache_value(value: Any) -> str:
    """
    Convert a value into a consistent string format for
    constructing cache keys.
    """

    if value is None:
        return ""

    return " ".join(str(value).strip().lower().split())


def get_utc_timestamp() -> str:
    """
    Return the current UTC timestamp in ISO 8601 format.
    """
    return datetime.now(timezone.utc).isoformat()

# CREATE CACHE KEY

def get_cache_key(question: Any,guest_type: Any, loyalty: Any, city: Any, session_id: Any) -> str:
    """
    Create a stable SHA-256 cache key from the user's question and profile information.
    """
    cache_input = {
        "question": normalise_cache_value(question),
        "guest_type": normalise_cache_value(guest_type),
        "loyalty": normalise_cache_value(loyalty),
        "city": normalise_cache_value(city),
        "session_id": normalise_cache_value(session_id)
    }

    cache_string = json.dumps(
        cache_input,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":")
    )

    cache_key = hashlib.sha256(cache_string.encode("utf-8")).hexdigest()
    return cache_key

# DESERIALISE CACHED RESPONSE

def _deserialise_cached_response(response_json: str) -> Dict[str, Any]:
    """
    Convert a stored JSON response back into a Python dictionary.
    """

    try:
        response = json.loads(response_json)

        if isinstance(response, dict):
            return response

        return { "text": str(response)}

    except json.JSONDecodeError as e:
        logger.error("Cached response contains invalid JSON.")
        raise CustomException(e, sys) from e

# DETERMINE WHETHER A QUESTION SHOULD BE CACHED
def is_cacheable_question(question):
    """
    Determine whether a question is suitable for caching.

    Clear standalone questions may use the cache. Short or
    context-dependent follow-up questions bypass the cache.
    """
    try: 
        logger.debug("Evaluating cacheability of question: %s", question)
        normalised_question = normalise_cache_value(question)

        if not normalised_question:
            return False

        follow_up_phrases = [
            "what about",
            "how about",
            "and what about",
            "and how about",
            "what time does it",
            "does it",
            "is it",
            "can it",
            "can they",
            "what about that",
            "how about that",
            "that one",
            "this one",
            "the same one"
        ]
        logger.info(" Cacheable question was successfully evaluated.")
        return not any(
            normalised_question.startswith(phrase)
            for phrase in follow_up_phrases
        )
    except Exception as e:
        logger.error("Failed to determine if question is cacheable.")
        raise CustomException(e, sys) from e
# RETRIEVE RESPONSE FROM CACHE

def get_cached_response(question: Any, guest_type: Any, loyalty: Any, city: Any, session_id: Any) -> Optional[Dict[str, Any]]:
    """
    Retrieve a previously cached response.
    A matching response produces a cache hit. No matching response produces a cache miss.
    """

    try:
        cache_key = get_cache_key(question=question, guest_type=guest_type, loyalty=loyalty, city=city, session_id=session_id)

        with get_cache_connection() as connection:
            row = connection.execute(
                """
                SELECT response_json
                FROM response_cache
                WHERE cache_key = ?
                """,
                (cache_key,)
            ).fetchone()
            # CACHE MISS
            if row is None:
                cache_metrics["misses"] += 1
                logger.info("Cache miss.")
                return None

       
            # CACHE HIT
            connection.execute(
                """
                UPDATE response_cache
                SET
                    last_accessed_at = ?,
                    access_count = access_count + 1
                WHERE cache_key = ?
                """,
                (get_utc_timestamp(), cache_key))
            connection.commit()
        cached_response = (_deserialise_cached_response(row["response_json"]))
        cached_response["cache_hit"] = True
        cache_metrics["hits"] += 1

        logger.info("Cache hit.")
        return cached_response

    except Exception as e:
        logger.error("Failed to retrieve a response from the cache.")
        raise CustomException(e, sys) from e



# STORE RESPONSE IN CACHE

def cache_response(question: Any, guest_type: Any, loyalty: Any, city: Any, response: Any, session_id: Any) -> bool:
    """
    Store a new response in the persistent cache. If the cache key already exists, update the existing response.
    """

    try:
        cache_key = get_cache_key(
            question=question,
            guest_type=guest_type,
            loyalty=loyalty,
            city=city,
            session_id=session_id
        )

        current_time = get_utc_timestamp()
        response_json = json.dumps(response, ensure_ascii=False, default=str)

        with get_cache_connection() as connection:
            connection.execute(
                """
                INSERT INTO response_cache (
                    cache_key,
                    question,
                    guest_type,
                    loyalty,
                    city,
                    session_id,
                    response_json,
                    created_at,
                    last_accessed_at,
                    access_count
                )

                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

                ON CONFLICT(cache_key)
                DO UPDATE SET

                    question = excluded.question,
                    guest_type = excluded.guest_type,
                    loyalty = excluded.loyalty,
                    city = excluded.city,
                    session_id = excluded.session_id,
                    response_json = excluded.response_json,
                    last_accessed_at = excluded.last_accessed_at
                """,
                (
                    cache_key,
                    str(question),
                    str(guest_type or ""),
                    str(loyalty or ""),
                    str(city or ""),
                    str(session_id or ""),
                    response_json,
                    current_time,
                    current_time, 
                    0))

            connection.commit()
        enforce_cache_limit()
        logger.info("Response stored in persistent cache.")
        return True

    except Exception as e:
        logger.error("Failed to store response in the cache.")
        raise CustomException(e, sys) from e

# ENFORCE CACHE SIZE LIMIT

def enforce_cache_limit(max_entries: Optional[int] = None) -> int:
    """
    Remove least-recently-used cache entries when the
    configured maximum number of entries is exceeded.
    """

    try:
        resolved_limit = (MAX_CACHE_ENTRIES if max_entries is None else int(max_entries))
        if resolved_limit < 1:
            raise ValueError("max_entries must be at least 1.")
        with get_cache_connection() as connection:
            cache_size = connection.execute(
                """
                SELECT COUNT(*)
                FROM response_cache
                """
            ).fetchone()[0]

            number_to_delete = max(cache_size - resolved_limit, 0)

            if number_to_delete > 0:

                connection.execute(
                    """
                    DELETE FROM response_cache

                    WHERE cache_key IN (

                        SELECT cache_key
                        FROM response_cache

                        ORDER BY last_accessed_at ASC

                        LIMIT ?
                    )
                    """,
                    (number_to_delete,)
                )

                connection.commit()

                logger.info("Removed %s least-recently-used cache entries.", number_to_delete)

            return number_to_delete

    except Exception as e:
        logger.error("Failed to enforce cache size limit.")
        raise CustomException(e, sys) from e

# CLEAR CACHE

def clear_response_cache() -> None:
    """
    Remove all stored responses and reset runtime cache metrics.
    """

    try:
        with get_cache_connection() as connection:
            connection.execute(
                """
                DELETE FROM response_cache
                """
            )
            connection.commit()
        cache_metrics.update({"hits": 0, "misses": 0})
        logger.info("Persistent response cache cleared.")

    except Exception as e:
        logger.error("Failed to clear the response cache.")
        raise CustomException(e, sys) from e


# CACHE SUMMARY

def get_cache_summary(display: bool = True) -> Dict[str, Any]:
    """
    Calculate and return cache performance metrics.
    """

    try:
        hits = cache_metrics["hits"]
        misses = cache_metrics["misses"]
        total_requests = (hits + misses)
        hit_rate = (hits / total_requests if total_requests > 0 else 0.0)

        with get_cache_connection() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS stored_entries,
                    COALESCE(
                        SUM(access_count),
                        0
                    ) AS total_cache_hits,
                    COALESCE(
                        MAX(last_accessed_at),
                        ''
                    ) AS last_accessed_at

                FROM response_cache
                """
            ).fetchone()

        summary = {
            "hits": hits,
            "misses": misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "stored_entries": row["stored_entries"],
            "total_persistent_cache_hits": row["total_cache_hits"],
            "last_accessed_at": row["last_accessed_at"] or None
        }

        if display:
            logger.info("Cache summary: %s hits, %s misses, %.2f%% hit rate.", hits, misses, hit_rate * 100)
        return summary

    except Exception as e:
        logger.error("Failed to calculate cache summary.")
        raise CustomException(e,sys) from e

# TRACK LATENCY
def track_latency(component: str,latency_ms: float) -> None:
    """
    Record latency for a system component.
    """
    latency_value = float(latency_ms)

    if latency_value < 0:
        raise ValueError("Latency cannot be negative.")

    latency_metrics[str(component)].append(latency_value)

# TRACK TOKEN USAGE AND COST

def track_token_usage(agent_name: str, input_tokens: int, output_tokens: int, latency_ms: float = 0) -> float:
    """
    Record token usage, estimated model cost and latency.
    """
    input_count = int(input_tokens)
    output_count = int(output_tokens)
    latency_value = float(latency_ms)

    if (input_count < 0 or output_count < 0):
        raise ValueError("Token counts cannot be negative.")
    if latency_value < 0:
        raise ValueError("Latency cannot be negative.")
    input_cost = (input_count / 1000) * INPUT_COST_PER_1K
    output_cost = (output_count / 1000) * OUTPUT_COST_PER_1K
    total_cost = (input_cost + output_cost)

    usage_entry = {
        "agent": str(agent_name),
        "timestamp": get_utc_timestamp(),
        "input_tokens": input_count,
        "output_tokens": output_count,
        "total_tokens": input_count + output_count,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "latency_ms": latency_value}

    token_usage_log.append(usage_entry)

    if latency_value > 0:
        track_latency(component=agent_name, latency_ms=latency_value)
    return total_cost

# COST SUMMARY

def get_cost_summary(display: bool = True) -> Dict[str, Any]:
    """
    Calculate accumulated token usage and estimated cost.
    """
    total_calls = len(token_usage_log)
    if total_calls == 0:
        summary = {
            "total_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "average_cost_per_call": 0.0,
            "average_latency_ms": 0.0
        }

        if display:
            logger.info("No token usage has been tracked yet.")
        return summary

    total_input_tokens = sum(item["input_tokens"] for item in token_usage_log)
    total_output_tokens = sum(item["output_tokens"] for item in token_usage_log)
    total_cost = sum(item["total_cost"] for item in token_usage_log)
    total_latency = sum(item["latency_ms"] for item in token_usage_log)

    summary = {"total_calls": total_calls,
        "input_tokens":total_input_tokens,
        "output_tokens":total_output_tokens,
        "total_tokens":total_input_tokens + total_output_tokens,
        "total_cost": total_cost,
        "average_cost_per_call": total_cost / total_calls,
        "average_latency_ms": total_latency / total_calls}

    if display:

        logger.info(
            "Cost summary: %s calls, %s tokens, "
            "$%.6f estimated cost.",
            total_calls,
            summary["total_tokens"],
            total_cost
        )
    return summary

# LATENCY SUMMARY

def get_latency_summary() -> Dict[str, Dict[str, float]]:
    """
    Return average, minimum and maximum latency
    for each component.
    """
    return {
        component: {
            "average_ms":sum(values)/ len(values),
            "minimum_ms": min(values),
            "maximum_ms": max(values),
            "requests":len(values)
        }
        for component, values
        in latency_metrics.items()
        if values
    }

# PERFORMANCE DASHBOARD
def show_metrics_dashboard() -> Dict[str, Any]:
    """
    Display and return operational metrics for
    the Aurora Agentic RAG system.
    """

    cache_summary = get_cache_summary(display=False)
    cost_summary = get_cost_summary(
        display=False    )

    latency_summary = (get_latency_summary())
    dashboard = {
        "cache": cache_summary,
        "latency": latency_summary,
        "cost": cost_summary,
        "configuration": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "maximum_cache_entries":MAX_CACHE_ENTRIES,
            "cache_database": str(CACHE_DB_PATH)}}

    print("=" * 60)
    print("Aurora Agentic RAG Performance Dashboard")
    print("=" * 60)

    # CACHE METRICS
  
    print("\nCache Metrics")
    print(f"Cached responses: {cache_summary['stored_entries']}")
    print(
        f"Cache requests: "
        f"{cache_summary['total_requests']}")
    print(
        f"Cache hit rate: "
        f"{cache_summary['hit_rate']:.1%} "
        f"({cache_summary['hits']} hits, "
        f"{cache_summary['misses']} misses)"
    )
    print(f"Persistent cache hits: {cache_summary['total_persistent_cache_hits']}")
    if cache_summary["last_accessed_at"]:
        print(f"Last cache access: {cache_summary['last_accessed_at']}")

    # LATENCY METRICS

    print("\nLatency Metrics")

    if latency_summary:
        for (component, metrics) in latency_summary.items():
            readable_name = (component.replace("_", " ").title())

            print(
                f"{readable_name}: "
                f"{metrics['average_ms']:.0f} ms average "
                f"({metrics['minimum_ms']:.0f}–"
                f"{metrics['maximum_ms']:.0f} ms, "
                f"{metrics['requests']} requests)"
            )

    else:
        print("No latency measurements recorded yet.")

    # COST METRICS

    print("\nCost Metrics")
    print(f"Tracked model calls: {cost_summary['total_calls']}")
    print(f"Total tokens: {cost_summary['total_tokens']}")
    print(f"Total estimated cost: ${cost_summary['total_cost']:.6f}")
    print(f"Average estimated cost per call: ${cost_summary['average_cost_per_call']:.6f}")

     # SYSTEM CONFIGURATION
  
    print("\nSystem Configuration")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.2f}")
    print(f"Maximum cache entries: {MAX_CACHE_ENTRIES}")
    print(f"Persistent cache database: {CACHE_DB_PATH}")

    print("=" * 60)
    return dashboard

# TEST / DIRECT EXECUTION

if __name__ == "__main__":
    show_metrics_dashboard()