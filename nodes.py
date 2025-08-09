import json
import requests
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from models import FlightSearchState
from validators import validate_extracted_info
from dotenv import load_dotenv
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


def _extract_info_rule_based(text: str) -> Dict[str, Any]:
    """Lightweight heuristic extractor for flight info from free-form text."""
    result: Dict[str, Any] = {}
    lower = text.lower()

    # Origin and destination patterns
    # 1) from X to Y
    m = re.search(r"\bfrom\s+([a-zA-Z\s]+?)\s+to\s+([a-zA-Z\s]+)\b", lower)
    if not m:
        # 2) X to Y (avoid capturing questions like 'your origin and destination')
        m = re.search(r"\b([a-zA-Z][a-zA-Z\s]+?)\s+to\s+([a-zA-Z][a-zA-Z\s]+)\b", lower)
        if m and ("origin" in lower or "destination" in lower):
            # likely the assistant question, not user answer; ignore
            m = None
    if not m:
        # 3) origin X and destination Y
        m = re.search(r"origin(?:\s+city)?\s*[:=]?\s*([a-zA-Z\s]+?)[,;]?\s*(?:and\s*)?destination(?:\s+city)?\s*[:=]?\s*([a-zA-Z\s]+)\b", lower)
    if m:
        result["origin"] = m.group(1).strip().title()
        result["destination"] = m.group(2).strip().title()
    else:
        # 4) capture single-sided mentions
        m_from = re.search(r"\bfrom\s+([a-zA-Z\s]+)\b", lower)
        m_to = re.search(r"\bto\s+([a-zA-Z\s]+)\b", lower)
        if m_from:
            result["origin"] = m_from.group(1).strip().title()
        if m_to:
            result["destination"] = m_to.group(1).strip().title()

    # Trip type
    if re.search(r"\bround\b|\breturn\b|\bround\s*trip\b|\broundtrip\b", lower):
        result["trip_type"] = "round trip"
    elif re.search(r"one\s*way|oneway|single", lower):
        result["trip_type"] = "one way"

    # Cabin class
    if re.search(r"\beconomy|coach|eco\b", lower):
        result["cabin_class"] = "economy"
    elif re.search(r"business|biz|premium", lower):
        result["cabin_class"] = "business"
    elif re.search(r"first\s*class|\bfirst\b", lower):
        result["cabin_class"] = "first class"

    # Duration (for round trip), e.g., "for 5 days" or "5 days"
    dur = re.search(r"(for\s+)?(\d{1,3})\s*(days|day)\b", lower)
    if dur:
        try:
            result["duration"] = int(dur.group(2))
        except Exception:
            pass

    # Date patterns; validators will normalize
    date_patterns = [
        r"\b\d{4}-\d{2}-\d{2}\b",            # 2025-12-25
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",      # 12/25/2025 or 12/25/25
        r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",      # 12-25-2025
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(,\s*\d{4})?\b",
    ]
    for pat in date_patterns:
        dm = re.search(pat, lower)
        if dm:
            result["departure_date"] = dm.group(0)
            break

    return result


def _compose_followup(missing_fields):
    priority = [
        "departure_date",
        "trip_type",
        "duration",
        "origin",
        "destination",
        "cabin_class",
    ]
    for field in priority:
        if field in missing_fields:
            if field == "departure_date":
                return "What date would you like to depart? Please use a format like 2025-12-25."
            if field == "trip_type":
                return "Is this a one way or a round trip?"
            if field == "duration":
                return "How many days will you stay before returning?"
            if field == "origin":
                return "Which city are you flying from?"
            if field == "destination":
                return "Which city are you flying to?"
            if field == "cabin_class":
                return "Which cabin do you prefer: economy, business, or first class?"
    return "Could you share a bit more about your flight details?"


def analyze_conversation_node(state: FlightSearchState) -> FlightSearchState:
    """Analyze conversation to extract flight information using heuristics, optionally refined by LLM."""
    # Build conversation text and gather user-only history
    conversation_text = "".join(f"{m['role']}: {m['content']}\n" for m in state["conversation"])
    user_text = state.get("current_message", "")
    user_history = "\n".join(m["content"] for m in state["conversation"] if m.get("role") == "user")

    try:
        # Heuristic extraction from current message and user history
        extracted: Dict[str, Any] = {}
        for src in (user_text, user_history):
            info = _extract_info_rule_based(src or "")
            extracted.update({k: v for k, v in info.items() if v})

        # Force default to round trip
        state["trip_type"] = "round trip"

        # Optionally refine with LLM if available
        use_llm = bool(os.getenv("OPENAI_API_KEY"))
        if use_llm:
            extraction_prompt = f"""Analyze this ENTIRE conversation to extract flight booking information and output ONLY JSON keys shown:

{conversation_text}user: {user_text}

Keys: departure_date, origin, destination, cabin_class, trip_type, duration
Return only valid JSON.
"""
            try:
                extraction_response = get_llm().invoke([HumanMessage(content=extraction_prompt)])
                model_info = json.loads(extraction_response.content)
                if isinstance(model_info, dict):
                    extracted.update({k: v for k, v in model_info.items() if v})
            except Exception:
                pass

        # Validate and update state
        validated_info, validation_errors = validate_extracted_info(extracted)
        for key, value in validated_info.items():
            if value is not None:
                state[key] = value

        # Determine completeness
        required = ["departure_date", "origin", "destination", "cabin_class", "trip_type"]
        missing_fields = [f for f in required if not state.get(f)]
        # Always require duration for round trips
        if not state.get('duration'):
            missing_fields.append('duration')

        if missing_fields or validation_errors:
            # Follow-up question (LLM if available, else deterministic)
            if bool(os.getenv("OPENAI_API_KEY")):
                followup_prompt = f"""The user is booking a flight. Based on the conversation and current info below, ask ONE concise question to get the MOST IMPORTANT missing piece.

Missing fields: {missing_fields}
Validation errors: {validation_errors}

Current info: {json.dumps({k: state.get(k) for k in ['departure_date','origin','destination','cabin_class','trip_type','duration']}, ensure_ascii=False)}
"""
                try:
                    followup_response = get_llm().invoke([HumanMessage(content=followup_prompt)])
                    state["followup_question"] = followup_response.content
                except Exception:
                    state["followup_question"] = _compose_followup(missing_fields)
            else:
                state["followup_question"] = _compose_followup(missing_fields)

            state["needs_followup"] = True
            state["info_complete"] = False
        else:
            state["needs_followup"] = False
            state["info_complete"] = True

    except Exception as e:
        print(f"Error in conversation analysis: {e}")
        state["followup_question"] = _compose_followup(["departure_date"])  # sensible default
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
        """Convert city name to airport code using mapping, optionally LLM if available."""
        location_lower = location.lower().strip()
        
        # Check direct mapping first
        if location_lower in airport_mappings:
            return airport_mappings[location_lower]
        
        # If already looks like airport code (3 letters), return as is
        if len(location.strip()) == 3 and location.isalpha():
            return location.upper()
        
        # Optionally try LLM if configured
        if os.getenv("OPENAI_API_KEY"):
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
                airport_response = get_llm().invoke([HumanMessage(content=airport_prompt)])
                airport_code = airport_response.content.strip().upper()
                if len(airport_code) == 3 and airport_code.isalpha():
                    return airport_code
                codes = re.findall(r'\b[A-Z]{3}\b', airport_code)
                if codes:
                    return codes[0]
            except Exception:
                pass
        
        # Fallback: first 3 letters
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

    _debug_print("Amadeus request body (flight-offers)", state.get("body"))
    
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

    if DEBUG:
        print("[DEBUG] Amadeus token: connecting…")
    try:
        response = requests.post(url, headers=headers, data=data)
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
    """Get flight offers from Amadeus API for a 7-day window (input date + 6 days)."""
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
        state["followup_question"] = "Please provide a valid departure date (e.g., 2025-12-25)."
        return state

    if DEBUG:
        print("[DEBUG] Amadeus flight-offers: connecting…")

    all_results = []
    for day_offset in range(0, 7):
        query_date = (start_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
        body = dict(state["body"]) if state.get("body") else {}
        # Update departure date and round-trip return automatically based on duration
        if body.get("originDestinations"):
            body["originDestinations"][0]["departureDateTimeRange"]["date"] = query_date
            if len(body["originDestinations"]) > 1 and state.get("duration") is not None:
                # recompute return date from the new query_date
                dep_date_dt = datetime.strptime(query_date, "%Y-%m-%d")
                return_date = (dep_date_dt + timedelta(days=int(state.get("duration", 0)))).strftime("%Y-%m-%d")
                body["originDestinations"][1]["departureDateTimeRange"]["date"] = return_date
        # Ensure 5 offers
        body.setdefault("searchCriteria", {}).setdefault("maxFlightOffers", 5)

        try:
            response = requests.post(base_url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
            flights = data.get("data", []) or []
            # Annotate each with search_date and keep at most 5
            for f in flights[:5]:
                f["_search_date"] = query_date
            all_results.extend(flights[:5])
        except Exception as e:
            print(f"Error getting flight offers for {query_date}: {e}")
            continue

    state["result"] = {"data": all_results}
    state["current_node"] = "search_flights"
    if DEBUG:
        print("[DEBUG] Amadeus flight-offers: connected ✔")
    return state

def display_results_node(state: FlightSearchState) -> FlightSearchState:
    """Format flight results for display with layover details and search date."""

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

    try:
        flights = state.get("result", {}).get("data", [])
        if not flights:
            state["formatted_results"] = []
            state["needs_followup"] = True
            state["followup_question"] = "No flights found for your search criteria. Would you like to try different dates or destinations?"
            return state

        formatted_flights = []
        for flight in flights[:35]:  # 7 days * 5 offers
            price = flight.get("price", {})
            currency = price.get("currency", "USD")
            total_price = price.get("total", "N/A")
            search_date = flight.get("_search_date")

            itineraries = flight.get("itineraries", [])
            if not itineraries:
                continue

            # Use the outbound itinerary to display main info
            main_itinerary = itineraries[0]
            segments = main_itinerary.get("segments", [])
            layovers = []
            if segments:
                for i in range(len(segments) - 1):
                    arr = segments[i].get("arrival", {})
                    dep = segments[i+1].get("departure", {})
                    # Layover airport and time window
                    layovers.append(
                        f"{arr.get('iataCode','N/A')} {format_time(arr.get('at',''))} → {format_time(dep.get('at',''))}"
                    )

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
                    "stops": max(0, len(segments) - 1),
                    "layovers": layovers,
                    "search_date": search_date,
                    "full_details": flight
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
    """Generate LLM summary and recommendation (optional if LLM configured)."""
    
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
        if state.get("formatted_results") and len(state.get("formatted_results", [])) > 0 and os.getenv("OPENAI_API_KEY"):
            summary_response = get_llm().invoke([HumanMessage(content=summary_prompt)])
            state["summary"] = summary_response.content
        else:
            state["summary"] = state.get("summary") or "Here are your flight options."

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