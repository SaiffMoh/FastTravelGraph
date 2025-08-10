from langgraph.graph import StateGraph, END
from models import FlightSearchState
from typing import Dict, Any

from nodes import (
    llm_conversation_node,
    analyze_conversation_node,
    normalize_info_node,
    format_body_node,
    get_access_token_node,
    get_flight_offers_node,
    display_results_node,
    summarize_node,
)


def check_info_complete(state: FlightSearchState) -> str:
    """Decide next step based on collected info without mutating state."""
    # If we have all info, proceed to search
    if state.get("info_complete", False):
        return "normalize_info"
    # Otherwise, end this turn and wait for more user input
    return "ask_followup"


def check_api_success(state: FlightSearchState) -> str:
    """Check if API calls were successful"""
    if state.get("needs_followup", False):
        return "generate_followup"
    return "continue"


def create_flight_search_graph():
    """Create the enhanced LangGraph workflow for intelligent flight search"""
    workflow = StateGraph(FlightSearchState)

    # Add nodes
    workflow.add_node("llm_conversation", llm_conversation_node)
    workflow.add_node("analyze_conversation", analyze_conversation_node)
    workflow.add_node("normalize_info", normalize_info_node)
    workflow.add_node("format_body", format_body_node)
    workflow.add_node("get_auth", get_access_token_node)
    workflow.add_node("search_flights", get_flight_offers_node)
    workflow.add_node("display_results", display_results_node)
    workflow.add_node("summarize", summarize_node)

    # Add edges
    # Start with LLM conversation to handle user input intelligently
    workflow.add_edge("llm_conversation", "analyze_conversation")
    
    # After analyze, if info complete proceed to search; else end this turn
    workflow.add_conditional_edges(
        "analyze_conversation",
        check_info_complete,
        {
            "normalize_info": "normalize_info",
            "ask_followup": END,
        }
    )
    
    # Sequential flow for flight search process
    workflow.add_edge("normalize_info", "format_body")
    workflow.add_edge("format_body", "get_auth")
    workflow.add_edge("get_auth", "search_flights")
    workflow.add_edge("search_flights", "display_results")
    workflow.add_edge("display_results", "summarize")
    workflow.add_edge("summarize", END)

    # Set entry point
    workflow.set_entry_point("llm_conversation")

    return workflow


def initialize_state_from_thread(thread_id: str, conversation_history: list, current_message: str) -> Dict[str, Any]:
    """
    Initialize FlightSearchState from thread-based conversation.
    """
    return {
        "thread_id": thread_id,
        "conversation": conversation_history,
        "current_message": current_message,
        "needs_followup": True,
        "info_complete": False,
        "followup_question": None,
        "current_node": "llm_conversation",
        "followup_count": 0,
        # Default to round trip (as per requirements)
        "trip_type": "round trip",
        # Debug trace for monitoring
        "node_trace": [],
        # Initialize empty fields for LLM to populate
        "departure_date": None,
        "origin": None,
        "destination": None,
        "cabin_class": None,
        "duration": None,
    }