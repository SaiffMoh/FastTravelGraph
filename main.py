from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError
from typing import List

from models import (
    ChatRequest, ChatResponse, ExtractedInfo, FlightResult,
    conversation_store, FlightSearchState
)
from graph import create_flight_search_graph

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Flight Search Chatbot API",
    description="AI-powered flight search assistant with thread-based conversations",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compile LangGraph workflow
graph = create_flight_search_graph().compile()


@app.get("/")
async def root():
    return {"message": "Flight Search Chatbot API v2.0 is running"}


@app.get("/health")
async def health():
    """Check if server and API keys are ready"""
    required_keys = ["OPENAI_API_KEY", "AMADEUS_CLIENT_ID", "AMADEUS_CLIENT_SECRET"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        return {
            "status": "warning",
            "message": f"Missing API keys: {', '.join(missing_keys)}",
            "missing_keys": missing_keys
        }

    return {"status": "healthy", "message": "All API keys configured"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handles the conversation for flight search using thread_id and user_msg.
    Chat history is managed by the backend.
    """
    try:
        # Validate inputs
        if not request.thread_id:
            raise HTTPException(status_code=400, detail="thread_id is required")
        
        user_message = request.user_msg.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="user_msg cannot be empty")

        # Validate API keys
        required_keys = ["OPENAI_API_KEY", "AMADEUS_CLIENT_ID", "AMADEUS_CLIENT_SECRET"]
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        if missing_keys:
            raise HTTPException(
                status_code=500,
                detail=f"Missing API keys: {', '.join(missing_keys)}"
            )

        # Get conversation history from backend store
        conversation_history = conversation_store.get_conversation(request.thread_id)
        
        # Add user message to conversation history
        conversation_store.add_message(request.thread_id, "user", user_message)
        updated_conversation = conversation_store.get_conversation(request.thread_id)

        # Initialize state for LangGraph
        state = FlightSearchState(
            thread_id=request.thread_id,
            conversation=updated_conversation,
            current_message=user_message,
            needs_followup=True,
            info_complete=False,
            trip_type="round trip",  # Always round trip
            node_trace=[]
        )

        # Run LangGraph workflow
        try:
            result = graph.invoke(state.dict())
        except GraphRecursionError:
            raise HTTPException(
                status_code=500,
                detail="Conversation loop limit reached — possible infinite loop in workflow"
            )

        # Build extracted info for response
        extracted_info = ExtractedInfo(
            departure_date=result.get("departure_date"),
            origin=result.get("origin"),
            destination=result.get("destination"),
            cabin_class=result.get("cabin_class"),
            trip_type=result.get("trip_type"),
            duration=result.get("duration")
        )

        # If still collecting information, return follow-up question
        if result.get("needs_followup", True):
            assistant_message = result.get("followup_question", "Could you provide more details about your flight?")
            
            # Add assistant response to conversation history
            conversation_store.add_message(request.thread_id, "assistant", assistant_message)
            
            return ChatResponse(
                response_type="question",
                message=assistant_message,
                extracted_info=extracted_info,
                thread_id=request.thread_id,
                debug_trace=result.get("node_trace")
            )

        # Build flight results if search completed
        flights = []
        if result.get("formatted_results"):
            flights = [
                FlightResult(
                    price=str(f.get("price", "N/A")),
                    currency=str(f.get("currency", "USD")),
                    search_date=str(f.get("search_date", "")) or None,
                    outbound={
                        "airline": str(f.get("outbound", {}).get("airline", "N/A")),
                        "flight_number": str(f.get("outbound", {}).get("flight_number", "N/A")),
                        "departure_airport": str(f.get("outbound", {}).get("departure_airport", "N/A")),
                        "arrival_airport": str(f.get("outbound", {}).get("arrival_airport", "N/A")),
                        "departure_time": str(f.get("outbound", {}).get("departure_time", "N/A")),
                        "arrival_time": str(f.get("outbound", {}).get("arrival_time", "N/A")),
                        "duration": str(f.get("outbound", {}).get("duration", "N/A")),
                        "stops": int(f.get("outbound", {}).get("stops", 0)) if f.get("outbound", {}).get("stops") is not None else None,
                        "layovers": [str(x) for x in (f.get("outbound", {}).get("layovers") or [])],
                    },
                    return_leg={
                        "airline": str(f.get("return_leg", {}).get("airline", "N/A")),
                        "flight_number": str(f.get("return_leg", {}).get("flight_number", "N/A")),
                        "departure_airport": str(f.get("return_leg", {}).get("departure_airport", "N/A")),
                        "arrival_airport": str(f.get("return_leg", {}).get("arrival_airport", "N/A")),
                        "departure_time": str(f.get("return_leg", {}).get("departure_time", "N/A")),
                        "arrival_time": str(f.get("return_leg", {}).get("arrival_time", "N/A")),
                        "duration": str(f.get("return_leg", {}).get("duration", "N/A")),
                        "stops": int(f.get("return_leg", {}).get("stops", 0)) if f.get("return_leg", {}).get("stops") is not None else None,
                        "layovers": [str(x) for x in (f.get("return_leg", {}).get("layovers") or [])],
                    } if f.get("return_leg") else None,
                )
                for f in result.get("formatted_results", [])
            ]

        # Prepare response message with summary
        assistant_message = result.get("summary", "Here are your flight options:")
        
        # Add assistant response to conversation history  
        conversation_store.add_message(request.thread_id, "assistant", assistant_message)

        return ChatResponse(
            response_type="results",
            message=assistant_message,
            extracted_info=extracted_info,
            flights=flights,
            summary=result.get("summary"),
            thread_id=request.thread_id,
            debug_trace=result.get("node_trace")
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing request"
        )


@app.post("/api/reset/{thread_id}")
async def reset_conversation(thread_id: str):
    """Reset conversation history for a specific thread"""
    conversation_store.clear_conversation(thread_id)
    return {"message": f"Conversation for thread {thread_id} has been reset"}


@app.get("/api/threads")
async def get_active_threads():
    """Get all active conversation threads"""
    threads = conversation_store.get_all_threads()
    return {"threads": threads, "count": len(threads)}


@app.get("/api/conversation/{thread_id}")
async def get_conversation_history(thread_id: str):
    """Get conversation history for a specific thread"""
    conversation = conversation_store.get_conversation(thread_id)
    return {
        "thread_id": thread_id,
        "conversation": conversation,
        "message_count": len(conversation)
    }


@app.post("/api/test/extract")
async def test_extraction(request: ChatRequest):
    """Test extracted data without full workflow"""
    try:
        # Get conversation history
        conversation_history = conversation_store.get_conversation(request.thread_id)
        conversation_store.add_message(request.thread_id, "user", request.user_msg)
        updated_conversation = conversation_store.get_conversation(request.thread_id)

        state = FlightSearchState(
            thread_id=request.thread_id,
            conversation=updated_conversation,
            current_message=request.user_msg,
            trip_type="round trip"
        )

        # Test LLM conversation parsing
        from nodes import llm_conversation_node
        result = llm_conversation_node(state.dict())

        return {
            "thread_id": request.thread_id,
            "extracted_info": {
                "departure_date": result.get("departure_date"),
                "origin": result.get("origin"),
                "destination": result.get("destination"),
                "cabin_class": result.get("cabin_class"),
                "trip_type": result.get("trip_type"),
                "duration": result.get("duration")
            },
            "needs_followup": result.get("needs_followup"),
            "followup_question": result.get("followup_question"),
            "info_complete": result.get("info_complete"),
            "conversation_length": len(updated_conversation)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Development endpoints
@app.get("/api/debug/conversations")
async def debug_all_conversations():
    """Debug endpoint to see all conversations (remove in production)"""
    all_conversations = {}
    for thread_id in conversation_store.get_all_threads():
        all_conversations[thread_id] = conversation_store.get_conversation(thread_id)
    return all_conversations


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)