import os
import sys
import time
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Any

import requests

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
CHAT_URL = f"{BASE_URL}/chat"
HEALTH_URL = f"{BASE_URL}/health"
RESET_URL = f"{BASE_URL}/reset"
CLIENT_TIMEOUT = float(os.getenv("CLIENT_TIMEOUT", "120"))


def pick_reply(question: str) -> str:
    q = (question or "").lower()
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


def tab_row(cols: List[str]) -> str:
    widths = [max(len(c), 10) for c in cols]
    return " | ".join(c.ljust(w) for c, w in zip(cols, widths))


def print_grouped_tables(flights: List[Dict[str, Any]]):
    if not flights:
        print("No flights returned.")
        return
    by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in flights:
        by_day[f.get('search_date') or 'Unknown'].append(f)

    for day in sorted(by_day.keys()):
        print(f"\n=== Offers for {day} ===")
        header = [
            "Price", "Curr", "Leg", "Airline", "Number",
            "From", "To", "Dep", "Arr", "Dur", "Stops", "Layovers",
        ]
        print(tab_row(header))
        print("-" * 120)
        for f in by_day[day]:
            for leg_key, leg_name in (("outbound", "Out"), ("return_leg", "Ret")):
                leg = f.get(leg_key)
                if not leg:
                    continue
                layovers = "; ".join(leg.get('layovers') or []) or "non-stop"
                row = [
                    str(f.get("price", "N/A")),
                    str(f.get("currency", "USD")),
                    leg_name,
                    str(leg.get("airline", "N/A")),
                    str(leg.get("flight_number", "N/A")),
                    str(leg.get("departure_airport", "N/A")),
                    str(leg.get("arrival_airport", "N/A")),
                    str(leg.get("departure_time", "N/A")),
                    str(leg.get("arrival_time", "N/A")),
                    str(leg.get("duration", "N/A")),
                    str(leg.get("stops", 0)),
                    layovers,
                ]
                print(tab_row(row))


def print_flight_offers_table(all_offers: List[Dict[str, Any]]):
    """Display flight offers in a formatted table"""
    if not all_offers:
        print("No offers available for selection")
        return
    
    print("\n=== Flight Offers Available ===")
    
    # Create table headers
    headers = ["Offer ID", "Day Type", "Price", "Date", "Outbound", "Return", "Duration", "Stops"]
    col_widths = [12, 15, 15, 12, 25, 25, 15, 10]
    
    # Print header
    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * len(header_row))
    
    # Print each offer as a row
    for offer in all_offers:
        offer_id = offer.get("offer_id", "N/A")
        day_type = offer.get("day_type", "unknown")
        day_label = "🌟 SELECTED" if day_type == "selected" else f"📅 Alt {offer.get('date', 'N/A')}"
        price = offer.get("price", "N/A")
        search_date = offer.get("search_date", "N/A")
        
        # Format outbound details
        outbound = offer.get("outbound_details", {})
        outbound_str = f"{outbound.get('airline', 'N/A')} {outbound.get('flight_number', 'N/A')}"
        if outbound.get('route'):
            outbound_str += f"\n{outbound['route']}"
        if outbound.get('times'):
            outbound_str += f"\n{outbound['times']}"
        
        # Format return details
        return_details = offer.get("return_details")
        return_str = "N/A"
        if return_details:
            return_str = f"{return_details.get('airline', 'N/A')} {return_details.get('flight_number', 'N/A')}"
            if return_details.get('route'):
                return_str += f"\n{return_details['route']}"
            if return_details.get('times'):
                return_str += f"\n{return_details['times']}"
        
        # Format duration and stops
        outbound_duration = outbound.get('duration', 'N/A') if outbound else 'N/A'
        return_duration = return_details.get('duration', 'N/A') if return_details else 'N/A'
        duration_str = f"Out: {outbound_duration}\nRet: {return_duration}"
        
        outbound_stops = outbound.get('stops', 'N/A') if outbound else 'N/A'
        return_stops = return_details.get('stops', 'N/A') if return_details else 'N/A'
        stops_str = f"Out: {outbound_stops}\nRet: {return_stops}"
        
        # Create the row
        row = [
            offer_id.ljust(col_widths[0]),
            day_label.ljust(col_widths[1]),
            str(price).ljust(col_widths[2]),
            str(search_date).ljust(col_widths[3]),
            outbound_str.ljust(col_widths[4]),
            return_str.ljust(col_widths[5]),
            duration_str.ljust(col_widths[6]),
            stops_str.ljust(col_widths[7])
        ]
        
        print(" | ".join(row))
        print("-" * len(header_row))


def run_auto():
    # Optional: health check
    try:
        r = requests.get(HEALTH_URL, timeout=min(CLIENT_TIMEOUT, 10))
        print("Health:", r.json())
    except Exception as e:
        print("Warning: health check failed:", e)

    conversation_history: List[Dict[str, str]] = []
    user_message = "i want to travel from cairo to dubai"

    for step in range(1, 12):
        payload = {
            "message": user_message,
            "conversation_history": conversation_history,
        }
        resp = requests.post(CHAT_URL, json=payload, timeout=CLIENT_TIMEOUT)
        if resp.status_code != 200:
            print("Request failed", resp.status_code, resp.text)
            sys.exit(1)
        data = resp.json()

        rtype = data.get("response_type")
        trace = data.get("debug_trace") or []
        print(f"\nStep {step} -> response_type={rtype}  nodes={trace}")

        assistant_message = data.get("message", "")
        print("Assistant:", assistant_message)

        # Persist the turn so server is stateful across requests
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": assistant_message})

        if rtype == "results":
            flights = data.get("flights", [])
            print_grouped_tables([f for f in flights if f])
            summary = data.get("summary")
            if summary:
                print("\nSummary:\n", summary)
            print("\nDone.")
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

        # Otherwise, answer the follow-up
        user_message = pick_reply(assistant_message)
        print("User:", user_message)
        time.sleep(0.3)

    print("Reached step limit without results. Check API keys and server logs.")


def run_interactive():
    print("Flight Search CLI (type /quit to exit, /reset to reset conversation)")
    # Health
    try:
        r = requests.get(HEALTH_URL, timeout=min(CLIENT_TIMEOUT, 10))
        print("Health:", r.json())
    except Exception as e:
        print("Warning: health check failed:", e)

    conversation_history: List[Dict[str, str]] = []
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
                    requests.post(RESET_URL, timeout=min(CLIENT_TIMEOUT, 10))
                except Exception:
                    pass
                conversation_history.clear()
                print("Conversation reset.")
                continue

            payload = {
                "message": user_message,
                "conversation_history": conversation_history,
            }
            resp = requests.post(CHAT_URL, json=payload, timeout=CLIENT_TIMEOUT)
            if resp.status_code != 200:
                print("Request failed", resp.status_code, resp.text)
                continue
            data = resp.json()

            # Persist the turn
            conversation_history.append({"role": "user", "content": user_message})
            assistant_message = data.get("message", "")
            conversation_history.append({"role": "assistant", "content": assistant_message})

            rtype = data.get("response_type")
            trace = data.get("debug_trace") or []
            print("Assistant:", assistant_message)
            if trace:
                print("Nodes:", trace)

            if rtype == "results":
                flights = data.get("flights", [])
                print_grouped_tables([f for f in flights if f])
                summary = data.get("summary")
                if summary:
                    print("\nSummary:\n", summary)
                print("(New search? continue chatting or type /reset)")
            elif rtype == "selection":
                # Handle flight selection
                all_offers = data.get("all_offers", [])
                if all_offers:
                    print_flight_offers_table(all_offers)
                    print("\nPlease enter the Offer ID you want to select:")
                else:
                    print("No offers available for selection")
            elif rtype == "confirmation":
                print("Flight selection confirmed!")
                print("(New search? continue chatting or type /reset)")
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