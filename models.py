from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, TypedDict
from datetime import datetime

class Message(BaseModel):
    role: str = Field(..., description="Role of the message sender (user, assistant, system)")
    content: str = Field(..., description="Content of the message")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="When the message was created")

class ChatRequest(BaseModel):
    thread_id: str = Field(..., description="Unique identifier for the conversation thread")
    user_msg: str = Field(..., description="The user's message")

class ExtractedInfo(BaseModel):
    departure_date: Optional[str] = Field(None, description="Departure date in YYYY-MM-DD format")
    origin: Optional[str] = Field(None, description="Origin city or airport")
    destination: Optional[str] = Field(None, description="Destination city or airport")
    cabin_class: Optional[str] = Field(None, description="Cabin class (economy, business, first class)")
    trip_type: Optional[str] = Field(None, description="Trip type (one way, round trip)")
    duration: Optional[int] = Field(None, description="Duration in days for round trip")

class FlightLeg(BaseModel):
    airline: str = Field(..., description="Airline code")
    flight_number: str = Field(..., description="Flight number")
    departure_airport: str = Field(..., description="Departure airport code")
    arrival_airport: str = Field(..., description="Arrival airport code")
    departure_time: str = Field(..., description="Departure time")
    arrival_time: str = Field(..., description="Arrival time")
    duration: str = Field(..., description="Flight duration")
    stops: Optional[int] = Field(None, description="Number of stops")
    layovers: List[str] = Field(default=[], description="Layover information")

class FlightResult(BaseModel):
    price: str = Field(..., description="Flight price")
    currency: str = Field(..., description="Price currency")
    search_date: Optional[str] = Field(None, description="Date this flight was searched for")
    outbound: FlightLeg = Field(..., description="Outbound flight details")
    return_leg: Optional[FlightLeg] = Field(None, description="Return flight details (for round trips)")

class ChatResponse(BaseModel):
    # response_type: str = Field(..., description="Type of response (question, results, error)")
    # message: str = Field(..., description="HTML-formatted response message to display")
    html_content: str = Field(..., description="Full HTML content for frontend display")
    # extracted_info: Optional[ExtractedInfo] = Field(None, description="Currently extracted flight information")
    # flights: Optional[List[FlightResult]] = Field(None, description="Flight search results")
    # summary: Optional[str] = Field(None, description="AI summary of results")
    # thread_id: str = Field(..., description="Thread ID for this conversation")
    # debug_trace: Optional[List[str]] = Field(None, description="Debug information about processing steps")

# Internal state model for LangGraph - using TypedDict for better compatibility
from typing import TypedDict


class FlightSearchState(TypedDict, total=False):
    # Thread management
    thread_id: str
    
    # Conversation tracking
    conversation: List[Dict[str, Any]]
    current_message: str
    
    # Extracted information
    departure_date: Optional[str]
    origin: Optional[str]
    destination: Optional[str]
    cabin_class: Optional[str]
    trip_type: str  # Default to round trip
    duration: Optional[int]
    
    # Normalized information for API calls
    origin_location_code: Optional[str]
    destination_location_code: Optional[str]
    normalized_departure_date: Optional[str]
    normalized_cabin: Optional[str]
    normalized_trip_type: Optional[str]
    
    # API request data
    body: Optional[Dict[str, Any]]
    access_token: Optional[str]
    
    # Results
    result: Optional[Dict[str, Any]]
    formatted_results: Optional[List[Dict[str, Any]]]
    summary: Optional[str]
    
    # Flow control
    needs_followup: bool
    info_complete: bool
    followup_question: Optional[str]
    current_node: Optional[str]
    followup_count: int
    
    # Debug information
    node_trace: List[str]


class HotelSearchState(TypedDict, total=False):
    thread_id: str
    user_message: str
    selected_flight: int
    city_code: str
    hotel_id: list[str]
    checkin_date: str
    checkout_date: str
    currency: str
    roomQuantty: int
    adult: int
    summary: str
    body: Optional[Dict[str, Any]]
    access_token: Optional[str]


# Conversation storage (in production, use Redis, PostgreSQL, etc.)
class ConversationStore:
    def __init__(self):
        self._conversations: Dict[str, List[Dict[str, Any]]] = {}
    
    def get_conversation(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a thread"""
        return self._conversations.get(thread_id, [])
    
    def add_message(self, thread_id: str, role: str, content: str) -> None:
        """Add a message to conversation history"""
        if thread_id not in self._conversations:
            self._conversations[thread_id] = [
                {
                    "role": "system", 
                    "content": (
                        "You are a helpful AI travel assistant specializing in flight bookings. "
                        "Your goal is to help users find the best flights by gathering their preferences "
                        "in a natural, conversational way. You can understand flexible date formats, "
                        "casual location names, and abbreviated terms. Always be friendly and efficient."
                    ),
                    "timestamp": datetime.now().isoformat()
                }
            ]
        
        self._conversations[thread_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def clear_conversation(self, thread_id: str) -> None:
        """Clear conversation history for a thread"""
        if thread_id in self._conversations:
            del self._conversations[thread_id]
    
    def get_all_threads(self) -> List[str]:
        """Get all active thread IDs"""
        return list(self._conversations.keys())

# Global conversation store instance
conversation_store = ConversationStore()