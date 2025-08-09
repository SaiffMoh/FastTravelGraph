from langgraph.graph import StateGraph, END
from models import FlightSearchState, Message
from typing import List

from nodes import (
    analyze_conversation_node,
    normalize_info_node,
    format_body_node,
    get_access_token_node,
    get_flight_offers_node,
    display_results_node,
    summarize_node,
    generate_followup_node
)

def check_info_complete(state: FlightSearchState) -> str:
    """Check if all required information is collected and avoid infinite loops."""
    
    followup_count = state.get("followup_count", 0) + 1
    state["followup_count"] = followup_count

    # 1️⃣ If we have all info, skip to normalize
    if state.get("info_complete", False):
        return "normalize_info"

    # 2️⃣ Safety: stop looping after 3 follow-ups
    if followup_count > 3:
        state["needs_followup"] = False
        return "normalize_info"

    # 3️⃣ Default behavior: still missing info
    if state.get("needs_followup", True):
        return "generate_followup"

    return "normalize_info"



def check_api_success(state: FlightSearchState) -> str:
    """Check if API calls were successful"""
    if state.get("needs_followup", False):
        return "generate_followup"

    return "continue"

def create_flight_search_graph():
    """Create the LangGraph workflow for FastAPI"""
    workflow = StateGraph(FlightSearchState)

    # Add nodes
    workflow.add_node("analyze_conversation", analyze_conversation_node)
    workflow.add_node("generate_followup", generate_followup_node)
    workflow.add_node("normalize_info", normalize_info_node)
    workflow.add_node("format_body", format_body_node)
    workflow.add_node("get_auth", get_access_token_node)
    workflow.add_node("search_flights", get_flight_offers_node)
    workflow.add_node("display_results", display_results_node)
    workflow.add_node("summarize", summarize_node)

    # Add edges
    workflow.add_edge("analyze_conversation", "generate_followup")
    workflow.add_conditional_edges(
        "generate_followup",
        check_info_complete,
        {
            "generate_followup": "generate_followup",
            "normalize_info": "normalize_info"
        }
    )
    workflow.add_edge("normalize_info", "format_body")
    workflow.add_edge("format_body", "get_auth")
    workflow.add_edge("get_auth", "search_flights")
    workflow.add_edge("search_flights", "display_results")
    workflow.add_edge("display_results", "summarize")
    workflow.add_edge("summarize", END)

    # Set entry point
    workflow.set_entry_point("analyze_conversation")

    return workflow

from typing import Dict, Any
from models import FlightSearchState, Message


def initialize_state_from_request(message: str, conversation_history: List[Message]):
    """
    Initialize a valid FlightSearchState with safe defaults.
    """
    if not conversation_history:
        conversation_history = [
            {"role": "system", "content": (
                "You are a helpful AI travel assistant. "
                "Your goal is to help the user find flights, "
                "asking for any missing information politely."
            )}
        ]
    else:
        conversation_history = [
            msg if isinstance(msg, dict) else {"role": msg.role, "content": msg.content}
            for msg in conversation_history
        ]
    
    if message:
        conversation_history.append({"role": "user", "content": message})
    
    return {
        "conversation": conversation_history,
        "current_message": message or "",
        "needs_followup": True,
        "info_complete": False,
        "followup_question": None,
        "current_node": "analyze_conversation",
        "followup_count": 0  # ✅ ensure safe default
    }
