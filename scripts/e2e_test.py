import os
import sys
import time
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Any
from bs4 import BeautifulSoup

import requests

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
CHAT_URL = f"{BASE_URL}/api/chat"
HEALTH_URL = f"{BASE_URL}/health"
RESET_URL = f"{BASE_URL}/reset"
CLIENT_TIMEOUT = float(os.getenv("CLIENT_TIMEOUT", "120"))

def pick_reply(question: str) -> str:
    """Generate appropriate replies based on the assistant's questions"""
    q = (question or "").lower()

    # Check for HTML content and extract text if needed
    if "<div" in q and "</div>" in q:
        soup = BeautifulSoup(q, 'html.parser')
        question_text = soup.get_text()
        q = question_text.lower()

    # Cities first to avoid matching "depart" in "departure city"
    if "departure city" in q or "destination city" in q or ("origin" in q and "destination" in q):
        return "Cairo to Dubai"
    if "which city are you flying from" in q:
        return "Cairo"
    if "which city are you flying to" in q:
        return "Dubai"

    if "one way" in q or ("round" in q and "trip" in q):
        return "round trip"
    if "cabin" in q or "economy" in q or "business" in q or "first" in q:
        return "economy"
    if "how many days" in q or "stay" in q or "duration" in q:
        return "5 days"
    if "date" in q or "depart" in q:
        return "2025-12-20"

    return "Cairo to Dubai, 2025-12-20, round trip, 5 days, economy."

def parse_html_response(html_content: str) -> Dict[str, Any]:
    """Parse HTML content to extract flight information and summary"""
    soup = BeautifulSoup(html_content, 'html.parser')

    # Check if this is a question response or results
    if soup.find('div', class_='question-response'):
        question_div = soup.find('div', class_='question')
        question = question_div.get_text(strip=True) if question_div else "No question found"

        progress_div = soup.find('div', class_='progress')
        progress_items = []
        if progress_div:
            progress_items = [li.get_text(strip=True) for li in progress_div.find_all('li')]

        return {
            "type": "question",
            "question": question,
            "progress": progress_items
        }
    elif soup.find('table', class_='flight-table'):
        flights = []
        table = soup.find('table', class_='flight-table')

        for row in table.find_all('tr')[1:]:  # Skip header row
            cols = row.find_all('td')
            if len(cols) >= 9:
                flight = {
                    "outbound": {
                        "airline_flight": cols[1].get_text(strip=True),
                        "route_times": cols[2].get_text(strip=True),
                        "duration_stops": cols[3].get_text(strip=True)
                    },
                    "return": {
                        "airline_flight": cols[4].get_text(strip=True),
                        "route_times": cols[5].get_text(strip=True),
                        "duration_stops": cols[6].get_text(strip=True)
                    },
                    "price": cols[7].get_text(strip=True),
                    "search_date": cols[8].get_text(strip=True)
                }
                flights.append(flight)

        summary_div = soup.find('div', class_='summary-block')
        summary = summary_div.get_text(strip=True) if summary_div else None

        return {
            "type": "results",
            "flights": flights,
            "summary": summary
        }

    return {"type": "unknown", "content": html_content}

def print_flight_table(flights: List[Dict[str, Any]]):
    """Print flights in a formatted table"""
    if not flights:
        print("No flights returned.")
        return

    # Determine column widths
    col_widths = {
        "price": 10,
        "out_airline": 15,
        "out_route": 30,
        "out_details": 20,
        "ret_airline": 15,
        "ret_route": 30,
        "ret_details": 20,
        "date": 12
    }

    # Print header
    print(tab_row([
        "Price",
        "Outbound Flight",
        "Outbound Route & Times",
        "Outbound Details",
        "Return Flight",
        "Return Route & Times",
        "Return Details",
        "Search Date"
    ]))

    print("-" * 150)

    # Print each flight
    for i, flight in enumerate(flights, 1):
        print(tab_row([
            flight["price"],
            flight["outbound"]["airline_flight"],
            flight["outbound"]["route_times"],
            flight["outbound"]["duration_stops"],
            flight["return"]["airline_flight"],
            flight["return"]["route_times"],
            flight["return"]["duration_stops"],
            flight["search_date"]
        ]))

def tab_row(cols: List[str]) -> str:
    """Format a row for table display with consistent column widths"""
    widths = {
        0: 10,  # Price
        1: 15,  # Outbound Flight
        2: 30,  # Outbound Route & Times
        3: 20,  # Outbound Details
        4: 15,  # Return Flight
        5: 30,  # Return Route & Times
        6: 20,  # Return Details
        7: 12   # Search Date
    }

    # Ensure we have enough width definitions
    for i in range(len(cols)):
        if i not in widths:
            widths[i] = 15  # Default width

    # Format each column with appropriate width
    formatted_cols = []
    for i, col in enumerate(cols):
        width = widths.get(i, 15)
        formatted_cols.append(col.ljust(width))

    return " | ".join(formatted_cols)

def run_auto():
    """Run automated test sequence"""
    # Optional: health check
    try:
        r = requests.get(HEALTH_URL, timeout=min(CLIENT_TIMEOUT, 10))
        print("Health:", r.json())
    except Exception as e:
        print("Warning: health check failed:", e)

    conversation_history = []
    user_message = "i want to travel from cairo to dubai"
    thread_id = "test_thread_1"  # Using a fixed thread ID for testing

    for step in range(1, 12):
        payload = {
            "thread_id": thread_id,
            "user_msg": user_message,
            "conversation_history": conversation_history,
        }

        try:
            resp = requests.post(CHAT_URL, json=payload, timeout=CLIENT_TIMEOUT)
            if resp.status_code != 200:
                print(f"Request failed: {resp.status_code} - {resp.text}")
                sys.exit(1)

            # Parse the HTML response
            parsed = parse_html_response(resp.text)

            if parsed["type"] == "question":
                print(f"\nStep {step} - Question received")
                print("Assistant:", parsed["question"])
                if parsed["progress"]:
                    print("Progress:", ", ".join(parsed["progress"]))

                # Add to conversation history
                conversation_history.append({"role": "user", "content": user_message})
                conversation_history.append({"role": "assistant", "content": parsed["question"]})

                # Generate reply
                user_message = pick_reply(parsed["question"])
                print("User:", user_message)
                time.sleep(0.3)

            elif parsed["type"] == "results":
                print(f"\nStep {step} - Results received")
                print_flight_table(parsed["flights"])

                if parsed["summary"]:
                    print("\nSummary:", parsed["summary"])

                print("\nDone.")
                return

            else:
                print(f"Unexpected response type: {parsed['type']}")
                print("Content:", parsed.get("content", "No content"))
                return

        except Exception as e:
            print(f"Error in step {step}: {str(e)}")
            return
        elif rtype == "selection":
            # Handle flight selection
            all_offers = data.get("all_offers", [])
            if all_offers:
                print_flight_offers_table(all_offers)
                # Auto-select the first offer for testing
                user_message = all_offers[0]["offer_id"]
                print(f"\nAuto-selecting: {user_message}")
            else:
                print("No offers available for selection")
                return
        elif rtype == "confirmation":
            print("Flight selection confirmed!")
            return

    print("Reached step limit without results. Check API keys and server logs.")

def run_interactive():
    """Run interactive test session"""
    print("Flight Search CLI (type /quit to exit, /reset to reset conversation)")

    # Health check
    try:
        r = requests.get(HEALTH_URL, timeout=min(CLIENT_TIMEOUT, 10))
        print("Health:", r.json())
    except Exception as e:
        print("Warning: health check failed:", e)

    conversation_history = []
    thread_id = "test_thread_1"  # Using a fixed thread ID for testing

    try:
        while True:
            user_message = input("You: ").strip()
            if not user_message:
                continue

            if user_message.lower() in {"/quit", "/exit"}:
                print("Bye!")
                return

            if user_message.lower() == "/reset":
                try:
                    requests.post(f"{RESET_URL}/{thread_id}", timeout=min(CLIENT_TIMEOUT, 10))
                except Exception as e:
                    print(f"Reset failed: {e}")
                conversation_history.clear()
                print("Conversation reset.")
                continue

            payload = {
                "thread_id": thread_id,
                "user_msg": user_message,
                "conversation_history": conversation_history,
            }

            try:
                resp = requests.post(CHAT_URL, json=payload, timeout=CLIENT_TIMEOUT)
                if resp.status_code != 200:
                    print(f"Request failed: {resp.status_code} - {resp.text}")
                    continue

                # Parse the HTML response
                parsed = parse_html_response(resp.text)

                if parsed["type"] == "question":
                    print("Assistant:", parsed["question"])
                    if parsed["progress"]:
                        print("Progress:", ", ".join(parsed["progress"]))

                    # Add to conversation history
                    conversation_history.append({"role": "user", "content": user_message})
                    conversation_history.append({"role": "assistant", "content": parsed["question"]})

                elif parsed["type"] == "results":
                    print_flight_table(parsed["flights"])
                    if parsed["summary"]:
                        print("\nSummary:", parsed["summary"])
                    print("(New search? continue chatting or type /reset)")

                else:
                    print(f"Unexpected response type: {parsed['type']}")
                    print("Content:", parsed.get("content", "No content"))

            except Exception as e:
                print(f"Error: {str(e)}")

    except KeyboardInterrupt:
        print("\nBye!")

def main():
    parser = argparse.ArgumentParser(description="Flight Search Chat CLI")
    parser.add_argument("--auto", action="store_true", help="Run in auto mode (no interactive input)")
    args = parser.parse_args()

    if args.auto:
        run_auto()
    else:
        run_interactive()

if __name__ == "__main__":
    main()