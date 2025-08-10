from langgraph.graph import StateGraph, END
from models import FlightSearchState, Message
from typing import List

from nodes import (
    llm_conversation_node,
    analyze_conversation_node,
    normalize_info_node,
    format_body_node,
    get_access_token_node,
    get_flight_offers_node,
    display_results_node,
    summarize_node,
    select_flight_offer_node,
    process_flight_selection_node,
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
    workflow.add_node("select_flight_offer", select_flight_offer_node)
    workflow.add_node("process_flight_selection", process_flight_selection_node)

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
    
    # After summarize, check if we need to wait for flight selection
    workflow.add_conditional_edges(
        "summarize",
        lambda state: "select_flight_offer" if state.get("formatted_results") else "end",
        {
            "select_flight_offer": "select_flight_offer",
            "end": END,
        }
    )
    
    # After select_flight_offer, check if user has made a selection
    workflow.add_conditional_edges(
        "select_flight_offer",
        lambda state: "process_flight_selection" if state.get("waiting_for_selection") else "end",
        {
            "process_flight_selection": "process_flight_selection",
            "end": END,
        }
    )
    
    # After processing selection, end the workflow
    workflow.add_edge("process_flight_selection", END)

    # Set entry point
    workflow.set_entry_point("llm_conversation")

    return workflow


def initialize_state_from_request(message: str, conversation_history: List[Message]):
    """
    Initialize a valid FlightSearchState with safe defaults for LLM-based processing.
    """
    if not conversation_history:
        conversation_history = [
            {"role": "system", "content": (
                "You are a helpful AI travel assistant specializing in flight bookings. "
                "Your goal is to help users find the best flights by gathering their preferences "
                "in a natural, conversational way. You can understand flexible date formats, "
                "casual location names, and abbreviated terms. Always be friendly and efficient."
            )}
        ]
    else:
        # Convert Message objects to dicts if needed
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
        # Initialize flight selection fields
        "all_offers": None,
        "selected_flight_offer_id": None,
        "selected_flight_offer": None,
        "selected_flight_date": None,
        "waiting_for_selection": False,
        "final_confirmation": None,
    }