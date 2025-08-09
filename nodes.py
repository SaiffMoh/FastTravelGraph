import json
import requests
import os
from datetime import datetime, timedelta
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from models import FlightSearchState
from validators import validate_extracted_info
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI client
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
)

def analyze_conversation_node(state: FlightSearchState) -> FlightSearchState:
    """Analyze full conversation to extract flight information"""
    
    # Format conversation for analysis
    conversation_text = ""
    for msg in state["conversation"]:
        conversation_text += f"{msg['role']}: {msg['content']}\n"
    conversation_text += f"user: {state['current_message']}\n"
    
    # Comprehensive extraction prompt
    extraction_prompt = f"""Analyze this ENTIRE conversation to extract flight booking information:

{conversation_text}

Look for:
1. All flight information mentioned throughout the conversation
2. Any corrections or updates the user made
3. What information is still missing

Extract ALL available information and return JSON:
{{
    "departure_date": "date string or null",
    "origin": "origin city/airport or null", 
    "destination": "destination city/airport or null",
    "cabin_class": "economy/business/first class or null",
    "trip_type": "one way/round trip or null",
    "duration": "number of days for round trip or null"
}}

Return only valid JSON. Use the LATEST information if there were corrections."""

    try:
        extraction_response = llm.invoke([HumanMessage(content=extraction_prompt)])
        extracted_info = json.loads(extraction_response.content)
        
        # Validate the extracted information
        validated_info, validation_errors = validate_extracted_info(extracted_info)
        
        # Update state with validated information
        for key, value in validated_info.items():
            if value is not None:
                state[key] = value
        
        # Check if information is complete
        required_fields = ['departure_date', 'origin', 'destination', 'cabin_class', 'trip_type']
        missing_fields = [field for field in required_fields if not state.get(field)]
        
        # Check duration for round trips
        if (state.get('trip_type', '').lower() in ['round trip', 'round_trip'] and 
            not state.get('duration')):
            missing_fields.append('duration')
        
        if missing_fields or validation_errors:
            # Generate follow-up question
            followup_prompt = f"""The user is booking a flight. Based on this conversation:

{conversation_text}

Current extracted info:
- Departure date: {state.get('departure_date', 'Missing')}
- Origin: {state.get('origin', 'Missing')}
- Destination: {state.get('destination', 'Missing')}
- Cabin class: {state.get('cabin_class', 'Missing')}
- Trip type: {state.get('trip_type', 'Missing')}
- Duration: {state.get('duration', 'Missing')} days

Missing fields: {missing_fields}
Validation errors: {validation_errors}

Generate a friendly, conversational question to get the missing information or fix validation errors.
Ask for only the MOST IMPORTANT missing piece of information.
Be natural and helpful."""

            followup_response = llm.invoke([HumanMessage(content=followup_prompt)])
            state["followup_question"] = followup_response.content
            state["needs_followup"] = True
            state["info_complete"] = False
        else:
            state["needs_followup"] = False
            state["info_complete"] = True
            
    except Exception as e:
        print(f"Error in conversation analysis: {e}")
        state["followup_question"] = "I had trouble understanding your message. Could you please tell me your flight details again?"
        state["needs_followup"] = True
        state["info_complete"] = False
    
    state["current_node"] = "analyze_conversation"
    return state

def normalize_info_node(state: FlightSearchState) -> FlightSearchState:
    """Normalize extracted information for Amadeus API format"""
    
    # Airport code mapping
    airport_mappings = {
        'new york': 'NYC', 'nyc': 'NYC', 'new york city': 'NYC',
        'los angeles': 'LAX', 'la': 'LAX',
        'chicago': 'CHI', 'london': 'LON', 'paris': 'PAR',
        'tokyo': 'TYO', 'dubai': 'DXB', 'amsterdam': 'AMS',
        'frankfurt': 'FRA', 'madrid': 'MAD', 'rome': 'ROM',
        'barcelona': 'BCN', 'milan': 'MIL', 'zurich': 'ZUR',
        'vienna': 'VIE', 'munich': 'MUC', 'berlin': 'BER',
        'istanbul': 'IST', 'cairo': 'CAI', 'doha': 'DOH',
        'singapore': 'SIN', 'hong kong': 'HKG', 'bangkok': 'BKK',
        'sydney': 'SYD', 'melbourne': 'MEL', 'toronto': 'YYZ',
        'vancouver': 'YVR', 'montreal': 'YUL', 'mexico city': 'MEX',
        'sao paulo': 'SAO', 'rio de janeiro': 'RIO', 'buenos aires': 'BUE'
    }
    
    def normalize_location_to_airport_code(location: str) -> str:
        """Convert city name to airport code using LLM if not in mapping"""
        location_lower = location.lower().strip()
        
        # Check direct mapping first
        if location_lower in airport_mappings:
            return airport_mappings[location_lower]
        
        # If already looks like airport code (3 letters), return as is
        if len(location.strip()) == 3 and location.isalpha():
            return location.upper()
        
        # Use LLM to find airport code
        airport_prompt = f"""Convert this city or location to its primary airport code: "{location}"

Return only the 3-letter IATA airport code (like LAX, JFK, LHR). 
If the location has multiple airports, return the main international airport code.
If you're unsure, return your best guess for the main airport code.

Examples:
- New York -> JFK
- Los Angeles -> LAX  
- London -> LHR
- Paris -> CDG
- Dubai -> DXB

Just return the 3-letter code, nothing else."""

        try:
            airport_response = llm.invoke([HumanMessage(content=airport_prompt)])
            airport_code = airport_response.content.strip().upper()
            
            # Validate it's 3 letters
            if len(airport_code) == 3 and airport_code.isalpha():
                return airport_code
            else:
                # Extract 3-letter code if response contains more text
                codes = re.findall(r'\b[A-Z]{3}\b', airport_code)
                if codes:
                    return codes[0]
                else:
                    # Fallback: return first 3 letters
                    return location[:3].upper()
        except Exception:
            # Fallback: return first 3 letters
            return location[:3].upper()
    
    def normalize_cabin_class(cabin: str) -> str:
        """Normalize cabin class to Amadeus format"""
        cabin_lower = cabin.lower()
        if 'economy' in cabin_lower or 'eco' in cabin_lower or 'coach' in cabin_lower:
            return 'ECONOMY'
        elif 'business' in cabin_lower or 'biz' in cabin_lower:
            return 'BUSINESS'
        elif 'first' in cabin_lower:
            return 'FIRST_CLASS'
        else:
            return 'ECONOMY'  # Default
    
    def normalize_trip_type(trip_type: str) -> str:
        """Normalize trip type to Amadeus format"""
        trip_lower = trip_type.lower()
        if 'round' in trip_lower or 'return' in trip_lower:
            return 'round_trip'
        else:
            return 'one_way'
    
    # Normalize each field
    try:
        # Origin and destination airport codes
        if state.get('origin'):
            state['origin_location_code'] = normalize_location_to_airport_code(state['origin'])
        
        if state.get('destination'):
            state['destination_location_code'] = normalize_location_to_airport_code(state['destination'])
        
        # Departure date (already in YYYY-MM-DD format from validation)
        if state.get('departure_date'):
            state['normalized_departure_date'] = state['departure_date']
        
        # Cabin class
        if state.get('cabin_class'):
            state['normalized_cabin'] = normalize_cabin_class(state['cabin_class'])
        
        # Trip type
        if state.get('trip_type'):
            state['normalized_trip_type'] = normalize_trip_type(state['trip_type'])
        
        state['current_node'] = 'normalize_info'
        
    except Exception as e:
        print(f"Error in normalization: {e}")
        state["followup_question"] = "Sorry, I had trouble processing your flight information. Please try again."
        state["needs_followup"] = True
    
    return state

def format_body_node(state: FlightSearchState) -> FlightSearchState:
    """Format the request body for Amadeus API"""
    
    def format_flight_offers_body(
        origin_location_code,
        destination_location_code,
        departure_date,
        cabin="ECONOMY",
        trip_type="one_way",
        duration=None
    ):
        origin_destinations = [
            {
                "id": "1",
                "originLocationCode": origin_location_code,
                "destinationLocationCode": destination_location_code,
                "departureDateTimeRange": {
                    "date": departure_date,
                    "time": "10:00:00"
                }
            }
        ]
        
        if trip_type == "round_trip" and duration is not None:
            # Calculate return date
            dep_date = datetime.strptime(departure_date, "%Y-%m-%d")
            return_date = (dep_date + timedelta(days=int(duration))).strftime("%Y-%m-%d")
            origin_destinations.append({
                "id": "2",
                "originLocationCode": destination_location_code,
                "destinationLocationCode": origin_location_code,
                "departureDateTimeRange": {
                    "date": return_date,
                    "time": "10:00:00"
                }
            })
        
        return {
            "currencyCode": "EGP",
            "originDestinations": origin_destinations,
            "travelers": [
                {
                    "id": "1",
                    "travelerType": "ADULT"
                }
            ],
            "sources": ["GDS"],
            "searchCriteria": {
                "maxFlightOffers": 5,  # Get more options
                "flightFilters": {
                    "cabinRestrictions": [
                        {
                            "cabin": cabin,
                            "coverage": "MOST_SEGMENTS",
                            "originDestinationIds": [od["id"] for od in origin_destinations]
                        }
                    ]
                }
            }
        }
    
    # Create the API request body
    state["body"] = format_flight_offers_body(
        origin_location_code=state.get("origin_location_code"),
        destination_location_code=state.get("destination_location_code"),
        departure_date=state.get("normalized_departure_date"),
        cabin=state.get("normalized_cabin", "ECONOMY"),
        trip_type=state.get("normalized_trip_type", "one_way"),
        duration=state.get("duration")
    )
    
    state["current_node"] = "format_body"
    return state

def get_access_token_node(state: FlightSearchState) -> FlightSearchState:
    """Get access token from Amadeus API"""
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": os.getenv("AMADEUS_CLIENT_ID"),
        "client_secret": os.getenv("AMADEUS_CLIENT_SECRET")
    }
    
    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        state["access_token"] = response.json()["access_token"]
        state["current_node"] = "get_auth"
    except Exception as e:
        print(f"Error getting access token: {e}")
        state["followup_question"] = "Sorry, I had trouble connecting to the flight search service. Please try again later."
        state["needs_followup"] = True
    
    return state

def get_flight_offers_node(state: FlightSearchState) -> FlightSearchState:
    """Get flight offers from Amadeus API"""
    url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    headers = {
        "Authorization": f"Bearer {state['access_token']}",
        "Content-Type": "application/json",
        "X-HTTP-Method-Override": "GET"
    }
    
    try:
        response = requests.post(url, headers=headers, json=state["body"])
        response.raise_for_status()
        state["result"] = response.json()
        state["current_node"] = "search_flights"
    except Exception as e:
        print(f"Error getting flight offers: {e}")
        state["followup_question"] = "Sorry, I had trouble finding flights for your search. Please check your destinations and dates."
        state["needs_followup"] = True
    
    return state

def display_results_node(state: FlightSearchState) -> FlightSearchState:
    """Format flight results for display"""
    
    def format_duration(duration_str):
        """Convert PT2H30M to '2h 30m'"""
        if not duration_str or not duration_str.startswith('PT'):
            return duration_str
        
        duration_str = duration_str[2:]  # Remove 'PT'
        hours = 0
        minutes = 0
        
        if 'H' in duration_str:
            hours_part = duration_str.split('H')[0]
            hours = int(hours_part) if hours_part.isdigit() else 0
            duration_str = duration_str.split('H')[1] if 'H' in duration_str else duration_str
        
        if 'M' in duration_str:
            minutes_part = duration_str.split('M')[0]
            minutes = int(minutes_part) if minutes_part.isdigit() else 0
        
        if hours > 0 and minutes > 0:
            return f"{hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h"
        elif minutes > 0:
            return f"{minutes}m"
        else:
            return "N/A"
    
    def format_time(datetime_str):
        """Format datetime to time"""
        if not datetime_str:
            return "N/A"
        try:
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            return dt.strftime('%H:%M')
        except:
            return datetime_str
    
    try:
        if not state.get("result") or "data" not in state["result"]:
            state["formatted_results"] = []
            state["needs_followup"] = True
            state["followup_question"] = "No flights found for your search criteria. Would you like to try different dates or destinations?"
            return state
        
        flights = state["result"]["data"]
        
        if not flights:
            state["formatted_results"] = []
            state["needs_followup"] = True
            state["followup_question"] = "No flights found for your search criteria. Would you like to try different dates or destinations?"
            return state
        
        # Format results as structured data for API response
        formatted_flights = []
        
        for flight in flights[:5]:  # Limit to 5 results
            price = flight.get("price", {})
            currency = price.get("currency", "USD")
            total_price = price.get("total", "N/A")
            
            # Get main itinerary (outbound)
            itineraries = flight.get("itineraries", [])
            if itineraries:
                main_itinerary = itineraries[0]
                segments = main_itinerary.get("segments", [])
                
                if segments:
                    first_segment = segments[0]
                    last_segment = segments[-1]
                    
                    departure = first_segment.get("departure", {})
                    arrival = last_segment.get("arrival", {})
                    
                    flight_info = {
                        "airline": first_segment.get("carrierCode", "N/A"),
                        "flight_number": first_segment.get("number", "N/A"),
                        "departure_airport": departure.get("iataCode", "N/A"),
                        "arrival_airport": arrival.get("iataCode", "N/A"),
                        "departure_time": format_time(departure.get("at", "")),
                        "arrival_time": format_time(arrival.get("at", "")),
                        "duration": format_duration(main_itinerary.get("duration", "")),
                        "price": total_price,
                        "currency": currency,
                        "stops": len(segments) - 1,
                        "full_details": flight  # Include full flight data
                    }
                    
                    formatted_flights.append(flight_info)
        
        state["formatted_results"] = formatted_flights
        state["current_node"] = "display_results"
        
    except Exception as e:
        print(f"Error formatting results: {e}")
        state["formatted_results"] = []
        state["followup_question"] = "Sorry, I had trouble formatting the flight results."
        state["needs_followup"] = True
    
    return state

def summarize_node(state: FlightSearchState) -> FlightSearchState:
    """Generate LLM summary and recommendation"""
    
    summary_prompt = f"""You are a helpful travel assistant. Based on the flight search results, provide a concise summary and recommendation.

Search Details:
- From: {state.get('origin', 'N/A')} ({state.get('origin_location_code', 'N/A')})
- To: {state.get('destination', 'N/A')} ({state.get('destination_location_code', 'N/A')})
- Date: {state.get('departure_date', 'N/A')}
- Cabin: {state.get('cabin_class', 'N/A')}
- Trip Type: {state.get('trip_type', 'N/A')}
- Duration: {state.get('duration', 'N/A')} days

Found {len(state.get('formatted_results', []))} flight options.

Flight Results Summary:
{json.dumps(state.get('formatted_results', []), indent=2)}

Please provide:
1. A brief summary of the search results
2. Your recommendation on the best option(s)
3. Any helpful travel tips or considerations
4. Mention if there are any potential issues (long layovers, very early/late flights, etc.)

Keep your response conversational and helpful, limit to 2-3 paragraphs."""

    try:
        if state.get("formatted_results") and len(state.get("formatted_results", [])) > 0:
            summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
            state["summary"] = summary_response.content
        else:
            state["summary"] = """I wasn't able to find any flights for your search criteria. This could be due to no available flights on your chosen date, route not being available, or other factors. Try adjusting your departure date by a few days or consider nearby airports."""

        state["current_node"] = "summarize"
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        state["summary"] = "I found your flights but had trouble generating the summary."
    
    return state

def generate_followup_node(state: FlightSearchState) -> FlightSearchState:
    """Generate follow-up question when information is incomplete"""
    # This node is mainly for flow control
    # The actual question generation happens in analyze_conversation_node
    state["current_node"] = "generate_followup"
    return state