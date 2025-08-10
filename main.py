from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError

from models import ChatRequest, ChatResponse, ExtractedInfo, FlightResult, DetailedOffer
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

        # Initialize conversation state safely (default round trip)
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
            # Check if we're waiting for flight selection
            if result.get("waiting_for_selection", False):
                # Format detailed flight offers for display
                detailed_offers = []
                all_offers = result.get("all_offers", [])
                
                for offer_data in all_offers:
                    details = offer_data.get("display_details", {})
                    detailed_offer = DetailedOffer(
                        offer_id=details.get("offer_id"),
                        day_type=offer_data.get("day_type", "unknown"),
                        price=details.get("price"),
                        search_date=details.get("search_date"),
                        outbound_details=details.get("outbound_details", {}),
                        return_details=details.get("return_details")
                    )
                    detailed_offers.append(detailed_offer)
                
                return ChatResponse(
                    response_type="selection",
                    message=result.get("followup_question", "Please select a flight offer to proceed."),
                    extracted_info=extracted_info,
                    debug_trace=result.get("node_trace"),
                    all_offers=detailed_offers,
                    waiting_for_selection=True
                )
            else:
                return ChatResponse(
                    response_type="question",
                    message=result.get("followup_question", "Could you provide more details about your flight?"),
                    extracted_info=extracted_info,
                    debug_trace=result.get("node_trace")
                )

        # Build flight results
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

        # Check if user has selected a flight offer
        if result.get("selected_flight_offer_id"):
            # Get the selected flight offer details
            selected_offer = result.get("selected_flight_offer", {})
            selected_offer_id = result.get("selected_flight_offer_id")
            
            # Create a detailed confirmation response
            confirmation_message = result.get("final_confirmation", "Your flight has been selected successfully!")
            
            return ChatResponse(
                response_type="confirmation",
                message=confirmation_message,
                extracted_info=extracted_info,
                flights=flights,
                summary=result.get("summary"),
                debug_trace=result.get("node_trace"),
                # Include selected flight details
                selected_flight_offer_id=selected_offer_id,
                selected_flight_offer=selected_offer
            )
        else:
            return ChatResponse(
                response_type="results",
                message="Here are your flight options:",
                extracted_info=extracted_info,
                flights=flights,
                summary=result.get("summary"),
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
