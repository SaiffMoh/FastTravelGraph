from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError
from typing import List
import traceback
import logging
from utils.html_converter import question_to_html, html_table_converter
from graph import create_flight_search_graph
from models import ChatRequest, ExtractedInfo, FlightResult, conversation_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add print statements to see if imports work
load_dotenv()

# Checking environment variables
required_keys = ["OPENAI_API_KEY", "AMADEUS_CLIENT_ID", "AMADEUS_CLIENT_SECRET"]
for key in required_keys:
    value = os.getenv(key)
    if not value:
        print(f"{key}: MISSING")

# Initialize FastAPI app
app = FastAPI(
    title="Flight Search Chatbot API",
    description="AI-powered flight search assistant with thread-based conversations"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = create_flight_search_graph().compile()

@app.get("/")
async def root():
    return {"message": "Flight Search Chatbot API v2.0 is running"}

@app.get("/health")
async def health():
    """Check if server and API keys are ready"""
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        return {
            "status": "warning",
            "message": f"Missing API keys: {', '.join(missing_keys)}",
            "missing_keys": missing_keys
        }

    return {"status": "healthy", "message": "All API keys configured"}

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Handles the conversation for flight search using thread_id and user_msg."""
    
    # print(f"\n=== CHAT REQUEST RECEIVED ===")
    # print(f"Thread ID: {request.thread_id}")
    # print(f"User message: '{request.user_msg}'")
    # print(f"Request type: {type(request)}")
    
    try:
        # Validate inputs
        if not request.thread_id:
            print("ERROR: Missing thread_id")
            raise HTTPException(status_code=400, detail="thread_id is required")
        
        user_message = request.user_msg.strip()
        if not user_message:
            print("ERROR: Empty user message")
            raise HTTPException(status_code=400, detail="user_msg cannot be empty")


        # Validate API keys
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        if missing_keys:
            print(f"ERROR: Missing API keys: {missing_keys}")
            raise HTTPException(
                status_code=500,
                detail=f"Missing API keys: {', '.join(missing_keys)}"
            )


        # Get conversation history from backend store
        conversation_history = conversation_store.get_conversation(request.thread_id)
        print(f"✓ Got conversation history: {len(conversation_history)} messages")
        
        # Add user message to conversation history
        conversation_store.add_message(request.thread_id, "user", user_message)
        updated_conversation = conversation_store.get_conversation(request.thread_id)
        print(f"✓ Updated conversation: {len(updated_conversation)} messages")

        # Initialize state as dictionary for LangGraph
        state = {
            "thread_id": request.thread_id,
            "conversation": updated_conversation,
            "current_message": user_message,
            "needs_followup": True,
            "info_complete": False,
            "trip_type": "round trip",  # Always round trip
            "node_trace": [],
            # Initialize empty fields for LLM to populate
            "departure_date": None,
            "origin": None,
            "destination": None,
            "cabin_class": None,
            "duration": None,
            "followup_question": None,
            "current_node": "llm_conversation",
            "followup_count": 0
        }

        # Check if graph was compiled at startup
        if graph is None:
            print("ERROR: Graph was not compiled at startup")
            raise HTTPException(status_code=500, detail="Graph compilation failed")

        # Run LangGraph workflow
        result = graph.invoke(state)
        
        print("✓ LangGraph execution completed")
        # print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        # print(f"Node trace: {result.get('node_trace', [])}")

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
            
            # Format as HTML
            html_content = question_to_html(assistant_message, extracted_info)
            # print(f"Returning HTML response (length: {len(html_content)})")
            return html_content

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
            print(f"returned {len(flights)} flight results")

        # Prepare response message with summary
        assistant_message = result.get("summary", "Here are your flight options:")
        
        # Add assistant response to conversation history  
        conversation_store.add_message(request.thread_id, "assistant", assistant_message)
        
        html_content = html_table_converter(flights, assistant_message)
        print("✓ Formatted results HTML")
        print(f"Returning results HTML (length: {len(html_content)})")
        return html_content

    except HTTPException as he:
        print(f"HTTPException: {he.detail}")
        raise
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing request"
        )

# Keep your other endpoints...
@app.post("/api/reset/{thread_id}")
async def reset_conversation(thread_id: str):
    """Reset conversation history for a specific thread"""
    print(f"Resetting conversation for thread: {thread_id}")
    conversation_store.clear_conversation(thread_id)
    return {"message": f"Conversation for thread {thread_id} has been reset"}

@app.get("/api/threads")
async def get_active_threads():
    """Get all active conversation threads"""
    threads = conversation_store.get_all_threads()
    print(f"Getting active threads: {len(threads)} found")
    return {"threads": threads, "count": len(threads)}

# Add a simple test endpoint
@app.post("/test-simple")
async def test_simple(data: dict):
    """Simple test endpoint to verify requests are reaching the server"""
    print(f"=== TEST SIMPLE CALLED ===")
    print(f"Received data: {data}")
    return {"status": "received", "data": data, "message": "Server is working!"}

if __name__ == "__main__":
    print("Starting server...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)