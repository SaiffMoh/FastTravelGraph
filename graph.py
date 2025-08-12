# graph
from langgraph.graph import StateGraph, END
from models import FlightSearchState, HotelSearchState
from typing import Dict, Any
import re

from nodes import (
    llm_conversation_node,
    analyze_conversation_node,
    normalize_info_node,
    format_body_node,
    get_access_token_node,
    get_flight_offers_node,
    display_results_node,
    summarize_node,
    generate_followup_node,
    selection_nodes,
    get_city_IDs_node,
    get_hotel_offers_node,
    display_hotels_nodes,
    summarize_hotels_node
)
from nodes import _recent_offers_by_thread


def check_info_complete(state: FlightSearchState) -> str:
    """Decide next step based on collected info without mutating state.

    Special case: If flights were already displayed and the user message looks like a numeric
    selection, route to the selection node instead of re-running search.
    """
    try:
        # If user typed a number, and we either have results on state OR cached offers by thread → selection
        msg = str(state.get("current_message", ""))
        if re.search(r"\b\d+\b", msg):
            has_results_on_state = bool(state.get("formatted_results"))
            thread_id = state.get("thread_id", "") or "default"
            has_cached_offers = bool(_recent_offers_by_thread.get(thread_id))
            if has_results_on_state or has_cached_offers:
                return "selection_request"
    except Exception:
        pass

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


def check_selection_complete(state: HotelSearchState) -> str:
    """Check if flight selection is complete and ready for hotel search"""
    if state.get("city_code") and state.get("checkin_date") and state.get("checkout_date"):
        return "continue_hotel_search"
    return "ask_followup"


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
    workflow.add_node("generate_followup", generate_followup_node)
    workflow.add_node("selection", selection_nodes)
    workflow.add_node("get_city_IDs", get_city_IDs_node)
    workflow.add_node("get_hotel_offers", get_hotel_offers_node)
    workflow.add_node("display_hotels", display_hotels_nodes)
    workflow.add_node("summarize_hotels", summarize_hotels_node)

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
            "selection_request": "selection",
        }
    )
    
    # Sequential flow for flight search process
    workflow.add_edge("normalize_info", "format_body")
    workflow.add_edge("format_body", "get_auth")
    workflow.add_edge("get_auth", "search_flights")
    workflow.add_edge("search_flights", "display_results")
    workflow.add_edge("display_results", "summarize")
    # End after summarize so the user can see offers first; selection happens on next message
    # After selection, if we asked for user input, end the turn; else continue
    workflow.add_conditional_edges(
        "selection",
        check_selection_complete,
        {
            "continue_hotel_search": "get_city_IDs",
            "ask_followup": END,
        }
    )
    workflow.add_edge("get_city_IDs", "get_hotel_offers")
    workflow.add_edge("get_hotel_offers", "display_hotels")
    workflow.add_edge("display_hotels", "summarize_hotels")
    workflow.add_edge("summarize_hotels", END)

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