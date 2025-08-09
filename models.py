from typing import TypedDict, Optional, List, Dict, Any
from pydantic import BaseModel

# LangGraph State
class FlightSearchState(TypedDict):
    conversation: List[Dict[str, str]]  # Full conversation history
    current_message: str  # Latest user input
    
    # Extracted information
    departure_date: Optional[str]
    origin: Optional[str]
    destination: Optional[str]
    cabin_class: Optional[str]
    trip_type: Optional[str]
    duration: Optional[int]
    
    # Normalized data for API
    origin_location_code: Optional[str]
    destination_location_code: Optional[str]
    normalized_departure_date: Optional[str]
    normalized_cabin: Optional[str]
    normalized_trip_type: Optional[str]
    
    # API data
    access_token: Optional[str]
    body: Optional[Dict]
    result: Optional[Dict]
    
    # Response data
    formatted_results: Optional[str]
    summary: Optional[str]
    
    # Flow control
    info_complete: bool
    needs_followup: bool
    followup_question: Optional[str]
    current_node: str

# API Request/Response Models
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Message] = []

class ExtractedInfo(BaseModel):
    departure_date: Optional[str] = None
    origin: Optional[str] = None
    destination: Optional[str] = None
    cabin_class: Optional[str] = None
    trip_type: Optional[str] = None
    duration: Optional[int] = None

class FlightLeg(BaseModel):
    airline: str
    flight_number: str
    departure_airport: str
    arrival_airport: str
    departure_time: str
    arrival_time: str
    duration: str
    stops: Optional[int] = None
    layovers: Optional[List[str]] = None

class FlightResult(BaseModel):
    price: str
    currency: str
    search_date: Optional[str] = None
    outbound: FlightLeg
    return_leg: Optional[FlightLeg] = None

class ChatResponse(BaseModel):
    response_type: str  # "question" or "results" or "error"
    message: str
    extracted_info: ExtractedInfo
    flights: Optional[List[FlightResult]] = None
    summary: Optional[str] = None
    error_code: Optional[str] = None