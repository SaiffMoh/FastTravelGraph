from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError
from typing import List, Optional, Any
from html import escape
import traceback
import logging

# Add this at the very top - CRITICAL for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add print statements to see if imports work
print("=== STARTING APPLICATION ===")
print("Loading environment...")
load_dotenv()

print("Checking environment variables...")
required_keys = ["OPENAI_API_KEY", "AMADEUS_CLIENT_ID", "AMADEUS_CLIENT_SECRET"]
for key in required_keys:
    value = os.getenv(key)
    if value:
        print(f"✓ {key}: Found (length: {len(value)})")
    else:
        print(f"✗ {key}: MISSING")

print("Importing modules...")
try:
    from graph import create_flight_search_graph
    print("✓ graph module imported")
except ImportError as e:
    print(f"✗ Failed to import graph: {e}")

try:
    from models import (
        ChatRequest, ChatResponse, ExtractedInfo, FlightResult,
        conversation_store, FlightSearchState
    )
    print("✓ models imported")
except ImportError as e:
    print(f"✗ Failed to import models: {e}")

print("Creating FastAPI app...")
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

def _get(obj: Any, attr: str, default: str = "N/A") -> str:
    """
    Helper to retrieve attribute from objects or keys from dicts.
    Returns default if not present or value is falsy (but not zero).
    """
    if obj is None:
        return default
    # If dict-like
    try:
        if isinstance(obj, dict):
            val = obj.get(attr, default)
            return val if val is not None else default
    except Exception:
        pass
    # If object with attribute
    try:
        val = getattr(obj, attr)
        return val if val is not None else default
    except Exception:
        return default

def format_extracted_info_html(extracted_info: ExtractedInfo) -> str:
    """Format extracted information as HTML"""
    html = "<div class='extracted-info'><h4>Current Information:</h4><ul>"
    
    if extracted_info.departure_date:
        html += f"<li><strong>Departure Date:</strong> {extracted_info.departure_date}</li>"
    if extracted_info.origin:
        html += f"<li><strong>From:</strong> {extracted_info.origin}</li>"
    if extracted_info.destination:
        html += f"<li><strong>To:</strong> {extracted_info.destination}</li>"
    if extracted_info.cabin_class:
        html += f"<li><strong>Cabin:</strong> {extracted_info.cabin_class.title()}</li>"
    if extracted_info.duration:
        html += f"<li><strong>Duration:</strong> {extracted_info.duration} days</li>"
    
    html += "</ul></div>"
    return html

def format_flights_html(flights: List[Any], summary: Optional[str] = None) -> str:
    """Build HTML string containing a table with flight rows then the summary text below."""
    if not flights:
        return "<div>No flights found for your criteria.</div>"

    style = (
        "<style>"
        "table.flight-table{width:100%;border-collapse:collapse;font-family:inherit;background:transparent;}"
        "table.flight-table th,table.flight-table td{border:1px solid;"
        "border-color:rgba(120,120,120,0.2);padding:6px 8px;text-align:left;}"
        "table.flight-table th{font-weight:bold;}"
        "@media (prefers-color-scheme:dark){"
        "table.flight-table th,table.flight-table td{border-color:rgba(180,180,180,0.2);color:#eee;}"
        "table.flight-table{color:#eee;}"
        "}"
        "@media (prefers-color-scheme:light){"
        "table.flight-table th,table.flight-table td{border-color:rgba(120,120,120,0.2);color:#222;}"
        "table.flight-table{color:#222;}"
        "}"
        ".summary-block{margin-top:12px;padding:10px;border:1px solid #eee;font-family:inherit;}"
        "</style>"
    )

    # Table header
    html = style + "<table class='flight-table'>"
    html += (
        "<thead>"
        "<tr>"
        "<th>#</th>"
        "<th>Outbound (airline / flight)</th>"
        "<th>Outbound route & times</th>"
        "<th>Outbound duration / stops</th>"
        "<th>Return (airline / flight)</th>"
        "<th>Return route & times</th>"
        "<th>Return duration / stops</th>"
        "<th>Price</th>"
        "<th>Search date</th>"
        "</tr>"
        "</thead><tbody>"
    )

    for i, flight in enumerate(flights, start=1):
        # Outbound fields
        out = _get(flight, "outbound", {})
        out_airline = _get(out, "airline")
        out_flight_no = _get(out, "flight_number")
        out_dep = _get(out, "departure_airport")
        out_arr = _get(out, "arrival_airport")
        out_dep_time = _get(out, "departure_time")
        out_arr_time = _get(out, "arrival_time")
        out_duration = _get(out, "duration")
        out_stops = _get(out, "stops", "0")
        out_layovers = _get(out, "layovers", [])
        if isinstance(out_layovers, (list, tuple)):
            out_layovers_display = ", ".join(map(str, out_layovers)) if out_layovers else ""
        else:
            out_layovers_display = str(out_layovers)

        # Return fields
        ret = _get(flight, "return_leg", {}) or _get(flight, "return", {})
        ret_airline = _get(ret, "airline")
        ret_flight_no = _get(ret, "flight_number")
        ret_dep = _get(ret, "departure_airport")
        ret_arr = _get(ret, "arrival_airport")
        ret_dep_time = _get(ret, "departure_time")
        ret_arr_time = _get(ret, "arrival_time")
        ret_duration = _get(ret, "duration")
        ret_stops = _get(ret, "stops", "0")
        ret_layovers = _get(ret, "layovers", [])
        if isinstance(ret_layovers, (list, tuple)):
            ret_layovers_display = ", ".join(map(str, ret_layovers)) if ret_layovers else ""
        else:
            ret_layovers_display = str(ret_layovers)

        # Price and search date
        price = _get(flight, "price", "N/A")
        currency = _get(flight, "currency", "")
        price_display = f"{escape(str(currency))} {escape(str(price))}" if price != "N/A" else "Price not available"
        search_date = _get(flight, "search_date", "")

        # Build HTML-safe strings
        out_header = f"{escape(str(out_airline))} {escape(str(out_flight_no))}"
        out_route = f"{escape(str(out_dep))} {escape(str(out_dep_time))} → {escape(str(out_arr))} {escape(str(out_arr_time))}"
        out_details = f"{escape(str(out_duration))}"
        if out_stops not in (None, "", "0"):
            out_details += f" / {escape(str(out_stops))} stop(s)"
            if out_layovers_display:
                out_details += f" ({escape(out_layovers_display)})"

        ret_header = f"{escape(str(ret_airline))} {escape(str(ret_flight_no))}" if ret_airline != "N/A" else "—"
        ret_route = f"{escape(str(ret_dep))} {escape(str(ret_dep_time))} → {escape(str(ret_arr))} {escape(str(ret_arr_time))}" if ret_airline != "N/A" else "—"
        ret_details = f"{escape(str(ret_duration))}"
        if ret_stops not in (None, "", "0"):
            ret_details += f" / {escape(str(ret_stops))} stop(s)"
            if ret_layovers_display:
                ret_details += f" ({escape(ret_layovers_display)})"

        html += (
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{out_header}</td>"
            f"<td>{out_route}</td>"
            f"<td>{out_details}</td>"
            f"<td>{ret_header}</td>"
            f"<td>{ret_route}</td>"
            f"<td>{ret_details}</td>"
            f"<td>{price_display}</td>"
            f"<td>{escape(str(search_date))}</td>"
            "</tr>"
        )

    html += "</tbody></table>"

    # Append summary text after the table
    if summary:
        html += f"<div class='summary-block'>{escape(str(summary))}</div>"

    return html

def format_question_html(question: str, extracted_info: ExtractedInfo) -> str:
    """Format follow-up question with current info as HTML"""
    html = f"<div class='question-response'>"
    html += f"<div class='question'>"
    html += f"<p>{question}</p>"
    html += f"</div>"
    
    # Show current progress
    info_items = []
    if extracted_info.departure_date:
        info_items.append(f"📅 {extracted_info.departure_date}")
    if extracted_info.origin:
        info_items.append(f"🛫 From {extracted_info.origin}")
    if extracted_info.destination:
        info_items.append(f"🛬 To {extracted_info.destination}")
    if extracted_info.cabin_class:
        info_items.append(f"💺 {extracted_info.cabin_class.title()}")
    if extracted_info.duration:
        info_items.append(f"📆 {extracted_info.duration} days")
    
    if info_items:
        html += f"<div class='progress'>"
        html += f"<p><strong>Information collected:</strong></p>"
        html += f"<p>{' • '.join(info_items)}</p>"
        html += f"</div>"
    
    html += f"</div>"
    return html

# Try to compile graph at startup to catch errors early
print("Compiling graph...")
try:
    graph = create_flight_search_graph().compile()
    print("✓ Graph compiled successfully at startup")
except Exception as e:
    print(f"✗ Failed to compile graph at startup: {e}")
    traceback.print_exc()
    graph = None

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
    
    # CRITICAL: Add logging to see if endpoint is even being called
    print(f"\n=== CHAT REQUEST RECEIVED ===")
    print(f"Thread ID: {request.thread_id}")
    print(f"User message: '{request.user_msg}'")
    print(f"Request type: {type(request)}")
    
    try:
        # Validate inputs
        if not request.thread_id:
            print("ERROR: Missing thread_id")
            raise HTTPException(status_code=400, detail="thread_id is required")
        
        user_message = request.user_msg.strip()
        if not user_message:
            print("ERROR: Empty user message")
            raise HTTPException(status_code=400, detail="user_msg cannot be empty")

        print("✓ Input validation passed")

        # Validate API keys
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        if missing_keys:
            print(f"ERROR: Missing API keys: {missing_keys}")
            raise HTTPException(
                status_code=500,
                detail=f"Missing API keys: {', '.join(missing_keys)}"
            )

        print("✓ API keys validated")

        # Get conversation history from backend store
        print("Getting conversation history...")
        try:
            conversation_history = conversation_store.get_conversation(request.thread_id)
            print(f"✓ Got conversation history: {len(conversation_history)} messages")
        except Exception as e:
            print(f"ERROR: Failed to get conversation history: {e}")
            raise HTTPException(status_code=500, detail=f"Conversation error: {e}")
        
        # Add user message to conversation history
        print("Adding user message to conversation...")
        try:
            conversation_store.add_message(request.thread_id, "user", user_message)
            updated_conversation = conversation_store.get_conversation(request.thread_id)
            print(f"✓ Updated conversation: {len(updated_conversation)} messages")
        except Exception as e:
            print(f"ERROR: Failed to add message: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to add message: {e}")

        # Initialize state as dictionary for LangGraph
        print("Initializing state...")
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
        print("✓ State initialized")

        # Check if graph was compiled at startup
        if graph is None:
            print("ERROR: Graph was not compiled at startup")
            raise HTTPException(status_code=500, detail="Graph compilation failed")

        # Run LangGraph workflow
        print("Starting LangGraph execution...")
        try:
            result = graph.invoke(state)
            print("✓ LangGraph execution completed")
            print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            print(f"Node trace: {result.get('node_trace', [])}")
        except GraphRecursionError as e:
            print(f"ERROR: Graph recursion error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Conversation loop limit reached — possible infinite loop in workflow"
            )
        except Exception as e:
            print(f"ERROR: LangGraph execution failed: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Graph execution error: {e}")

        # Build extracted info for response
        print("Building extracted info...")
        try:
            extracted_info = ExtractedInfo(
                departure_date=result.get("departure_date"),
                origin=result.get("origin"),
                destination=result.get("destination"),
                cabin_class=result.get("cabin_class"),
                trip_type=result.get("trip_type"),
                duration=result.get("duration")
            )
            print(f"✓ Extracted info: {extracted_info}")
        except Exception as e:
            print(f"ERROR: Failed to build extracted info: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to build extracted info: {e}")

        # If still collecting information, return follow-up question
        if result.get("needs_followup", True):
            print("Needs followup - preparing question response")
            assistant_message = result.get("followup_question", "Could you provide more details about your flight?")
            
            # Add assistant response to conversation history
            try:
                conversation_store.add_message(request.thread_id, "assistant", assistant_message)
                print("✓ Added assistant response to conversation")
            except Exception as e:
                print(f"WARNING: Failed to add assistant message to conversation: {e}")
            
            # Format as HTML
            try:
                html_content = format_question_html(assistant_message, extracted_info)
                print("✓ Formatted HTML response")
                print(f"Returning HTML response (length: {len(html_content)})")
                return html_content
            except Exception as e:
                print(f"ERROR: Failed to format HTML: {e}")
                return assistant_message  # Fallback to plain text

        # Build flight results if search completed
        print("Search completed - building flight results...")
        flights = []
        if result.get("formatted_results"):
            try:
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
                print(f"✓ Built {len(flights)} flight results")
            except Exception as e:
                print(f"ERROR: Failed to build flight results: {e}")
                flights = []

        # Prepare response message with summary
        assistant_message = result.get("summary", "Here are your flight options:")
        
        # Add assistant response to conversation history  
        try:
            conversation_store.add_message(request.thread_id, "assistant", assistant_message)
            print("✓ Added assistant results to conversation")
        except Exception as e:
            print(f"WARNING: Failed to add assistant results to conversation: {e}")
        
        # Format as HTML
        try:
            html_content = format_flights_html(flights, assistant_message)
            print("✓ Formatted results HTML")
            print(f"Returning results HTML (length: {len(html_content)})")
            return html_content
        except Exception as e:
            print(f"ERROR: Failed to format results HTML: {e}")
            return assistant_message  # Fallback to plain text

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