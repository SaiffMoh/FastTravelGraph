import json
import requests
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from concurrent.futures import ThreadPoolExecutor, as_completed
from models import HotelSearchState
from models import FlightSearchState
from dotenv import load_dotenv
import json
import os
import requests
load_dotenv()

# Debug flag
DEBUG = str(os.getenv("DEBUG", "")).lower() in {"1", "true", "yes"}

def _debug_print(label: str, payload: Any = None):
    if DEBUG:
        try:
            if isinstance(payload, (dict, list)):
                print(f"[DEBUG] {label}:\n" + json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                print(f"[DEBUG] {label}: {payload}")
        except Exception:
            print(f"[DEBUG] {label} (unprintable payload)")

# Lazy LLM initialization to avoid import-time key errors
_llm = None

def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("LLM unavailable: OPENAI_API_KEY not set")
        _llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=api_key
        )
    return _llm


def llm_conversation_node(state: FlightSearchState) -> FlightSearchState:
    """LLM-driven conversational node that intelligently handles all user input parsing and follow-up questions."""
    try:
        (state.setdefault("node_trace", [])).append("llm_conversation")
    except Exception:
        pass

    conversation_text = "".join(f"{m['role']}: {m['content']}\n" for m in state.get("conversation", []))
    user_text = state.get("current_message", "")
    
    # Get current date for smart date parsing
    current_date = datetime.now()
    current_date_str = current_date.strftime("%Y-%m-%d")
    current_month = current_date.month
    current_day = current_date.day
    current_year = current_date.year

    try:
        if not os.getenv("OPENAI_API_KEY"):
            # Fallback if no LLM available
            state["followup_question"] = "I need an OpenAI API key to help you with flight bookings."
            state["needs_followup"] = True
            state["current_node"] = "llm_conversation"
            return state

        # Comprehensive LLM prompt for parsing and conversation management
        llm_prompt = f"""
            You are an expert travel assistant helping users book flights. Today's date is {current_date_str}.

            CONVERSATION SO FAR:
            {conversation_text}

            USER'S LATEST MESSAGE: "{user_text}"

            YOUR TASKS:
            1. Extract/update flight information from the entire conversation
            2. Intelligently parse dates and locations 
            3. Ask for ONE missing piece of information OR indicate completion

            DATE PARSING RULES (CRITICAL):
            - If user says "august 20th" or "Aug 20" → convert to "2025-08-20" 
            - If year omitted: use {current_year}, UNLESS month is before {current_month}, then use {current_year + 1}
            - If month and year omitted: use current month/year, UNLESS day is before {current_day}, then next month
            - If next month would be January, increment year too
            - Always output dates as YYYY-MM-DD

            LOCATION PARSING:
            - Convert casual names: "NYC" → "New York", "LA" → "Los Angeles"
            - Accept abbreviations and full names

            CABIN CLASS PARSING:
            - "eco" → "economy", "biz" → "business", "first" → "first class"

            REQUIRED INFORMATION:
            1. departure_date (YYYY-MM-DD format)
            2. origin (city name)
            3. destination (city name) 
            4. cabin_class (economy/business/first class)
            5. duration (number of days for round trip)

            CURRENT STATE:
            - departure_date: {state.get('departure_date', 'Not provided')}
            - origin: {state.get('origin', 'Not provided')}
            - destination: {state.get('destination', 'Not provided')}
            - cabin_class: {state.get('cabin_class', 'Not provided')}
            - duration: {state.get('duration', 'Not provided')}
            - trip_type: {state.get('trip_type', 'round trip')} (always round trip)

            RESPONSE FORMAT (JSON):
            {{
            "departure_date": "YYYY-MM-DD or null",
            "origin": "City Name or null", 
            "destination": "City Name or null",
            "cabin_class": "economy/business/first class or null",
            "duration": number_or_null,
            "followup_question": "Ask for ONE missing piece OR null if complete",
            "needs_followup": true_or_false,
            "info_complete": true_or_false
            }}

            EXAMPLES:
            - User: "I want to fly to Paris on august 20th" → {{"departure_date": "2025-08-20", "destination": "Paris", "followup_question": "Which city are you flying from?"}}
            - User: "from NYC, eco class" → {{"origin": "New York", "cabin_class": "economy", "followup_question": "Which city would you like to fly to?"}}
            - User: "5 days" → {{"duration": 5, "followup_question": "What date would you like to depart?"}}

            BE SMART: If user provides multiple pieces of info at once, extract all of them. Ask natural, conversational questions.
        """

        response = get_llm().invoke([HumanMessage(content=llm_prompt)])
        
        try:
            # Parse LLM response
            llm_result = json.loads(response.content)
            
            # Update state with extracted information
            if llm_result.get("departure_date"):
                state["departure_date"] = llm_result["departure_date"]
            if llm_result.get("origin"):
                state["origin"] = llm_result["origin"]
            if llm_result.get("destination"):
                state["destination"] = llm_result["destination"]
            if llm_result.get("cabin_class"):
                state["cabin_class"] = llm_result["cabin_class"]
            if llm_result.get("duration"):
                state["duration"] = llm_result["duration"]
                
            # Set conversation state
            state["followup_question"] = llm_result.get("followup_question")
            state["needs_followup"] = llm_result.get("needs_followup", True)
            state["info_complete"] = llm_result.get("info_complete", False)
            
            _debug_print("LLM extraction result", llm_result)
            
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return valid JSON
            print(f"LLM response parsing error. Raw response: {response.content}")
            state["followup_question"] = "I had trouble understanding. Could you please tell me your departure city, destination, and preferred travel date?"
            state["needs_followup"] = True
            state["info_complete"] = False

    except Exception as e:
        print(f"Error in LLM conversation node: {e}")
        # Fallback error handling
        state["followup_question"] = "I'm having technical difficulties. Please try again with your flight details."
        state["needs_followup"] = True
        state["info_complete"] = False

    state["current_node"] = "llm_conversation"
    return state


def analyze_conversation_node(state: FlightSearchState) -> FlightSearchState:
    """Validate the information extracted by the LLM conversation node."""
    try:
        (state.setdefault("node_trace", [])).append("analyze_conversation")
    except Exception:
        pass

    # Check completeness - all required fields must be present
    required_fields = ["departure_date", "origin", "destination", "cabin_class", "duration"]
    missing_fields = []
    
    for field in required_fields:
        if not state.get(field):
            missing_fields.append(field)
    
    # Validate departure date format
    departure_date = state.get("departure_date")
    if departure_date:
        try:
            # Validate date format and ensure it's not in the past
            parsed_date = datetime.strptime(departure_date, "%Y-%m-%d").date()
            if parsed_date < datetime.now().date():
                missing_fields.append("departure_date")
                state["departure_date"] = None
        except ValueError:
            missing_fields.append("departure_date")
            state["departure_date"] = None
    
    # Set completion status
    if missing_fields:
        state["info_complete"] = False
        state["needs_followup"] = True
        # The LLM should have already set an appropriate followup question
        if not state.get("followup_question"):
            state["followup_question"] = "I still need some information to search for flights. Could you help me with the missing details?"
    else:
        state["info_complete"] = True
        state["needs_followup"] = False
        state["followup_question"] = None
    
    _debug_print("Info completeness check", {
        "missing_fields": missing_fields,
        "info_complete": state["info_complete"],
        "current_state": {k: state.get(k) for k in required_fields}
    })
    
    state["current_node"] = "analyze_conversation"
    
    return state


def normalize_info_node(state: FlightSearchState) -> FlightSearchState:
    """Normalize extracted information for Amadeus API format using LLM for intelligent mapping."""
    try:
        (state.setdefault("node_trace", [])).append("normalize_info")
    except Exception:
        pass
    
    def normalize_location_to_airport_code(location: str) -> str:
        """Convert city name to airport code using LLM for intelligent mapping."""
        if not location:
            return ""
            
        # If already looks like airport code (3 letters), return as is
        if len(location.strip()) == 3 and location.isalpha():
            return location.upper()
        
        try:
            if os.getenv("OPENAI_API_KEY"):
                airport_prompt = f"""Convert this city or location to its primary IATA airport code: "{location}"

Rules:
- Return ONLY the 3-letter IATA airport code
- For cities with multiple airports, return the main international airport
- Examples: "New York" → "JFK", "Los Angeles" → "LAX", "London" → "LHR", "Paris" → "CDG"

Airport code:"""
                
                airport_response = get_llm().invoke([HumanMessage(content=airport_prompt)])
                airport_code = airport_response.content.strip().upper()
                
                # Extract 3-letter code from response
                codes = re.findall(r'\b[A-Z]{3}\b', airport_code)
                if codes:
                    return codes[0]
                elif len(airport_code) == 3 and airport_code.isalpha():
                    return airport_code
        except Exception as e:
            print(f"Error getting airport code for {location}: {e}")
        
        # Fallback mappings for common cities
        airport_mappings = {
            'new york': 'JFK', 'nyc': 'JFK', 'new york city': 'JFK',
            'los angeles': 'LAX', 'la': 'LAX', 'los angeles california': 'LAX',
            'chicago': 'ORD', 'london': 'LHR', 'paris': 'CDG',
            'tokyo': 'NRT', 'dubai': 'DXB', 'amsterdam': 'AMS',
            'frankfurt': 'FRA', 'madrid': 'MAD', 'rome': 'FCO',
            'barcelona': 'BCN', 'milan': 'MXP', 'zurich': 'ZUR',
        }
        
        location_lower = location.lower().strip()
        if location_lower in airport_mappings:
            return airport_mappings[location_lower]
        
        # Final fallback: first 3 letters
        return location[:3].upper()
    
    def normalize_cabin_class(cabin: str) -> str:
        """Normalize cabin class to Amadeus format"""
        if not cabin:
            return 'ECONOMY'
            
        cabin_lower = cabin.lower()
        if 'economy' in cabin_lower or 'eco' in cabin_lower or 'coach' in cabin_lower:
            return 'ECONOMY'
        elif 'business' in cabin_lower or 'biz' in cabin_lower:
            return 'BUSINESS'
        elif 'first' in cabin_lower:
            return 'FIRST_CLASS'
        else:
            return 'ECONOMY'  # Default
    
    try:
        # Normalize airport codes
        if state.get('origin'):
            state['origin_location_code'] = normalize_location_to_airport_code(state['origin'])
            _debug_print(f"Origin normalization", f"{state['origin']} → {state['origin_location_code']}")
        
        if state.get('destination'):
            state['destination_location_code'] = normalize_location_to_airport_code(state['destination'])
            _debug_print(f"Destination normalization", f"{state['destination']} → {state['destination_location_code']}")
        
        # Normalize other fields
        if state.get('departure_date'):
            state['normalized_departure_date'] = state['departure_date']
        
        if state.get('cabin_class'):
            state['normalized_cabin'] = normalize_cabin_class(state['cabin_class'])
            
        # Always round trip
        state['normalized_trip_type'] = 'round_trip'
        
        state['current_node'] = 'normalize_info'
        
    except Exception as e:
        print(f"Error in normalization: {e}")
        state["followup_question"] = "Sorry, I had trouble processing your flight information. Please try again."
        state["needs_followup"] = True
    
    return state


def format_body_node(state: FlightSearchState) -> FlightSearchState:
    """Format the request body for Amadeus API"""
    try:
        (state.setdefault("node_trace", [])).append("format_body")
    except Exception:
        pass
    
    def format_flight_offers_body(
        origin_location_code,
        destination_location_code,
        departure_date,
        cabin="ECONOMY",
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
        
        # Always add return leg for round trip
        if duration is not None:
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
                "maxFlightOffers": 5,
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
        duration=state.get("duration")
    )

    _debug_print("Amadeus request body", state.get("body"))
    
    state["current_node"] = "format_body"
    return state


def get_access_token_node(state: FlightSearchState) -> FlightSearchState:
    """Get access token from Amadeus API"""
    try:
        (state.setdefault("node_trace", [])).append("get_auth")
    except Exception:
        pass
        
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": os.getenv("AMADEUS_CLIENT_ID"),
        "client_secret": os.getenv("AMADEUS_CLIENT_SECRET")
    }

    if DEBUG:
        print("[DEBUG] Amadeus token: connecting…")
    try:
        response = requests.post(url, headers=headers, data=data, timeout=20)
        response.raise_for_status()
        token_json = response.json()
        state["access_token"] = token_json.get("access_token")
        state["current_node"] = "get_auth"
        if DEBUG:
            print("[DEBUG] Amadeus token: connected ✔")
    except Exception as e:
        print(f"Error getting access token: {e}")
        state["followup_question"] = "Sorry, I had trouble connecting to the flight search service. Please try again later."
        state["needs_followup"] = True
    
    return state


def get_flight_offers_node(state: FlightSearchState) -> FlightSearchState:
    """Get flight offers from Amadeus API for a 3-day window in parallel."""
    try:
        (state.setdefault("node_trace", [])).append("search_flights")
    except Exception:
        pass
        
    base_url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    headers = {
        "Authorization": f"Bearer {state['access_token']}",
        "Content-Type": "application/json"
    }

    start_date_str = state.get("normalized_departure_date")
    if not start_date_str:
        state["needs_followup"] = True
        state["followup_question"] = "What date would you like to depart?"
        return state

    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    except Exception:
        state["needs_followup"] = True
        state["followup_question"] = "Please provide a valid departure date."
        return state

    if DEBUG:
        print("[DEBUG] Amadeus flight-offers: connecting…")

    # Search 3-day window: departure date + 2 days
    bodies = []
    for day_offset in range(0, 3):
        query_date = (start_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
        body = dict(state["body"]) if state.get("body") else {}
        
        if body.get("originDestinations"):
            # Update departure date
            body["originDestinations"][0]["departureDateTimeRange"]["date"] = query_date
            
            # Update return date if round trip
            if len(body["originDestinations"]) > 1 and state.get("duration"):
                dep_date_dt = datetime.strptime(query_date, "%Y-%m-%d")
                return_date = (dep_date_dt + timedelta(days=int(state.get("duration", 0)))).strftime("%Y-%m-%d")
                body["originDestinations"][1]["departureDateTimeRange"]["date"] = return_date
        
        body.setdefault("searchCriteria", {}).setdefault("maxFlightOffers", 3)
        bodies.append((query_date, body))

    all_results = []

    def fetch_for_day(day_body_tuple):
        day, body = day_body_tuple
        try:
            resp = requests.post(base_url, headers=headers, json=body, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            flights = data.get("data", []) or []
            for f in flights[:5]:
                f["_search_date"] = day
            return flights[:5]
        except Exception as exc:
            print(f"Error getting flight offers for {day}: {exc}")
            return []

    # Parallel search across 3 days
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(fetch_for_day, b) for b in bodies]
        for fut in as_completed(futures):
            all_results.extend(fut.result())

    state["result"] = {"data": all_results}
    state["current_node"] = "search_flights"
    if DEBUG:
        print(f"[DEBUG] Amadeus flight-offers: found {len(all_results)} flights ✔")
    return state


def display_results_node(state: FlightSearchState) -> FlightSearchState:
    """Format flight results for display with outbound and return legs."""
    try:
        (state.setdefault("node_trace", [])).append("display_results")
    except Exception:
        pass

    def format_duration(duration_str):
        if not duration_str or not duration_str.startswith('PT'):
            return duration_str
        duration_str = duration_str[2:]
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
        if not datetime_str:
            return "N/A"
        try:
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            return dt.strftime('%H:%M')
        except:
            return datetime_str

    def build_leg(itinerary) -> Dict[str, Any]:
        segments = itinerary.get("segments", [])
        if not segments:
            return None
            
        layovers = []
        for i in range(len(segments) - 1):
            arr = segments[i].get("arrival", {})
            dep = segments[i+1].get("departure", {})
            layovers.append(f"{arr.get('iataCode','N/A')} {format_time(arr.get('at',''))} → {format_time(dep.get('at',''))}")
            
        first_segment = segments[0]
        last_segment = segments[-1]
        departure = first_segment.get("departure", {})
        arrival = last_segment.get("arrival", {})
        
        return {
            "airline": first_segment.get("carrierCode", "N/A"),
            "flight_number": first_segment.get("number", "N/A"),
            "departure_airport": departure.get("iataCode", "N/A"),
            "arrival_airport": arrival.get("iataCode", "N/A"),
            "departure_time": format_time(departure.get("at", "")),
            "arrival_time": format_time(arrival.get("at", "")),
            "duration": format_duration(itinerary.get("duration", "")),
            "stops": max(0, len(segments) - 1),
            "layovers": layovers,
        }

    try:
        flights = state.get("result", {}).get("data", [])
        if not flights:
            state["formatted_results"] = []
            state["needs_followup"] = True
            state["followup_question"] = "No flights found for your search criteria. Would you like to try different dates or destinations?"
            return state

        formatted: List[Dict[str, Any]] = []
        for flight in flights:
            itineraries = flight.get("itineraries", [])
            if not itineraries:
                continue
                
            outbound_leg = build_leg(itineraries[0])
            return_leg = build_leg(itineraries[1]) if len(itineraries) > 1 else None
            price = flight.get("price", {})
            
            formatted.append({
                "price": price.get("total", "N/A"),
                "currency": price.get("currency", "USD"),
                "search_date": flight.get("_search_date"),
                "outbound": outbound_leg,
                "return_leg": return_leg,
            })

        # Sort by price
        formatted.sort(key=lambda x: float(x["price"]) if x["price"] != "N/A" else float('inf'))
        state["formatted_results"] = formatted
        state["current_node"] = "display_results"

    except Exception as e:
        print(f"Error formatting results: {e}")
        state["formatted_results"] = []
        state["followup_question"] = "Sorry, I had trouble formatting the flight results."
        state["needs_followup"] = True
    
    return state


def summarize_node(state: FlightSearchState) -> FlightSearchState:
    """Generate LLM summary and recommendation."""
    try:
        (state.setdefault("node_trace", [])).append("summarize")
    except Exception:
        pass
    
    try:
        if not state.get("formatted_results") or not os.getenv("OPENAI_API_KEY"):
            state["summary"] = "Here are your flight options:"
            state["current_node"] = "summarize"
            return state

        summary_prompt = f"""You are a helpful travel assistant. 
        Based on the flight search results, provide a concise, friendly summary and recommendation. 
        Make it brief as possible. make it as strings only not markdown and don't add emojis.

Search Details:
- From: {state.get('origin', 'N/A')} ({state.get('origin_location_code', 'N/A')})
- To: {state.get('destination', 'N/A')} ({state.get('destination_location_code', 'N/A')})
- Date: {state.get('departure_date', 'N/A')}
- Cabin: {state.get('cabin_class', 'N/A')}
- Duration: {state.get('duration', 'N/A')} days

Found {len(state.get('formatted_results', []))} flight options across 3 days.

Flight Results (sorted by price):
{json.dumps(state.get('formatted_results', [])[:3], indent=2)}

Please provide:
1. A brief, enthusiastic summary of the search results
2. Your recommendation for the best option(s) considering price, timing, and convenience
3. Any helpful travel tips or considerations
4. Mention any concerns (long layovers, very early/late flights, etc.)

Keep it conversational, and helpful. Start with something like "Great! I found several flight options for your trip..."
"""

        summary_response = get_llm().invoke([HumanMessage(content=summary_prompt)])
        state["summary"] = summary_response.content
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        state["summary"] = "Great! I found your flight options. Here are the details:"
    
    state["current_node"] = "summarize"
    return state

    
# Legacy nodes for backward compatibility
def analyze_conversation_node_legacy(state: FlightSearchState) -> FlightSearchState:
    """Legacy analyze conversation node - now just calls the new llm_conversation logic"""
    return analyze_conversation_node(state)

def generate_followup_node(state: FlightSearchState) -> FlightSearchState:
    """Generate follow-up question - mostly handled by LLM conversation node now"""
    try:
        (state.setdefault("node_trace", [])).append("generate_followup")
    except Exception:
        pass
    state["current_node"] = "generate_followup"
    return state

# Aya
def selection_nodes(state: FlightSearchState) -> HotelSearchState:
    ...
# Rodaina & Saif
def get_city_IDs_node(state: HotelSearchState) -> HotelSearchState:
    """Get city IDs using Amadeus API for hotel search based on flight results."""
    url = "https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city"
    headers = {
        "Authorization": f"Bearer {state.get('access_token', '')}",
        "Content-Type": "application/json"
    }
    params = {
        "cityCode": state.get("city_code", "")  # fallback to CAI if not set
    }
    if DEBUG:
        print("[DEBUG] Getting hotels by city…")
    
    try: 
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        hotels_data = data.get("data", [])
        
        hotel_ids = []
        for hotel in hotels_data:
            hotel_id = hotel.get("hotelId")
            if hotel_id:
                hotel_ids.append(hotel_id)

        hotel_ids = hotel_ids[:20]  # limit to first 20
        state["hotel_id"] = hotel_ids

        if DEBUG:
            print(f"[DEBUG] Found {len(hotel_ids)} hotels in {params['cityCode']}.")
            print(f"[DEBUG] hotels in {params['cityCode']} are {hotel_ids}")

    except Exception as e:
        print(f"Error getting hotel IDs: {e}")
        state["followup_question"] = "Sorry, I had trouble finding hotels in your city. Please try again later."
        state["hotel_id"] = []
        
    return state


def get_hotel_offers_node(state: HotelSearchState) -> HotelSearchState:
    """Get hotel offers for a list of hotel IDs using Amadeus API."""
    url = "https://test.api.amadeus.com/v3/shopping/hotel-offers"
    headers = {
        "Authorization": f"Bearer {state.get('access_token', '')}",
        "Content-Type": "application/json"
    }

    hotel_ids = state.get("hotel_id", [])
    if not hotel_ids:
        state["followup_question"] = "No hotel IDs found to fetch offers."
        return state

    params = {
        "hotelIds": ",".join(hotel_ids),
        "checkInDate": state.get("check_in_date", datetime.now().strftime("%Y-%m-%d")),
        "checkOutDate": state.get("check_out_date", (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")),
        "currencyCode": "EGP"
    }

    if DEBUG:
        print(f"[DEBUG] Getting hotel offers for {len(hotel_ids)} hotels…")

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        hotels_offers = data.get("data", [])

        state["hotels_offers"] = hotels_offers
        if DEBUG:
            print(f"[DEBUG] Retrieved {len(hotels_offers)} hotel offers for {params['hotelIds']}.")
            print(f"[DEBUG] Retrieved {len(state['hotels_offers'])} hotel offers.")
            for hotel in hotels_offers:
                name = hotel['hotel']['name']
                check_in = hotel['offers'][0]['checkInDate']
                check_out = hotel['offers'][0]['checkOutDate']
                total_price = hotel['offers'][0]['price']['total']
                currency = hotel['offers'][0]['price']['currency']
                
                print(f"[DEBUG] {name} | {check_in} → {check_out} | {total_price} {currency}")

        
    except Exception as e:
        print(f"Error getting hotel offers: {e}")
        state["followup_question"] = (
            "Sorry, I had trouble retrieving hotel offers. Please try again later."
        )
        state["hotels_offers"] = []

    return state
# Ali
def display_hotels_nodes(state: HotelSearchState) -> HotelSearchState:
    ...
    
def summarize_hotels_node(state: HotelSearchState) -> HotelSearchState:
    ...