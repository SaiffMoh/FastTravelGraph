from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError

from models import ChatRequest, ChatResponse, ExtractedInfo, FlightResult
from graph import create_flight_search_graph, initialize_state_from_request

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Flight Search Chatbot API",
    description="AI-powered flight search assistant that extracts information through conversation",
    version="1.0.0"
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
    return {"message": "Flight Search Chatbot API is running"}


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


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handles the conversation for flight search.
    """
    try:
        # Ensure message is present
        user_message = (request.message or "").strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Always use a list for history
        conversation_history = request.conversation_history or []
        if not isinstance(conversation_history, list):
            conversation_history = []

        # Validate API keys
        required_keys = ["OPENAI_API_KEY", "AMADEUS_CLIENT_ID", "AMADEUS_CLIENT_SECRET"]
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        if missing_keys:
            raise HTTPException(
                status_code=500,
                detail=f"Missing API keys: {', '.join(missing_keys)}"
            )

        # Initialize conversation state safely
        state = initialize_state_from_request(user_message, conversation_history)
        state.setdefault("conversation", conversation_history)
        state.setdefault("current_message", user_message)

        # Run LangGraph
        try:
            result = graph.invoke(state)
        except GraphRecursionError:
            raise HTTPException(
                status_code=500,
                detail="Conversation loop limit reached — possible infinite loop in workflow"
            )

        # Build extracted info
        extracted_info = ExtractedInfo(
            departure_date=result.get("departure_date"),
            origin=result.get("origin"),
            destination=result.get("destination"),
            cabin_class=result.get("cabin_class"),
            trip_type=result.get("trip_type"),
            duration=result.get("duration")
        )

        # Still collecting info
        if result.get("needs_followup", True):
            return ChatResponse(
                response_type="question",
                message=result.get("followup_question", "Could you provide more details about your flight?"),
                extracted_info=extracted_info
            )

        # Build flight results
        flights = [
            FlightResult(
                airline=str(f.get("airline", "N/A")),
                flight_number=str(f.get("flight_number", "N/A")),
                departure_airport=str(f.get("departure_airport", "N/A")),
                arrival_airport=str(f.get("arrival_airport", "N/A")),
                departure_time=str(f.get("departure_time", "N/A")),
                arrival_time=str(f.get("arrival_time", "N/A")),
                duration=str(f.get("duration", "N/A")),
                price=str(f.get("price", "N/A")),
                currency=str(f.get("currency", "USD"))
            )
            for f in result.get("formatted_results", [])
        ]

        return ChatResponse(
            response_type="results",
            message="Here are your flight options:",
            extracted_info=extracted_info,
            flights=flights,
            summary=result.get("summary")
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing request"
        )


@app.post("/reset")
async def reset_conversation():
    return {"message": "Conversation reset. You can start a new flight search."}


@app.post("/test/extract")
async def test_extraction(request: ChatRequest):
    """Test extracted data without full workflow"""
    try:
        state = initialize_state_from_request(request.message or "", request.conversation_history or [])
        from nodes import analyze_conversation_node
        result = analyze_conversation_node(state)

        return {
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
            "info_complete": result.get("info_complete")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
