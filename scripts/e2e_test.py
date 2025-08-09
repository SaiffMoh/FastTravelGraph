import os
import sys
import time
import json
from typing import List, Dict, Any

import requests

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
CHAT_URL = f"{BASE_URL}/chat"
HEALTH_URL = f"{BASE_URL}/health"


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


def print_flights(flights: List[Dict[str, Any]]):
    if not flights:
        print("No flights returned.")
        return
    for i, f in enumerate(flights, 1):
        layovers = f.get('layovers') or []
        layover_str = "; ".join(layovers) if layovers else "non-stop"
        date_tag = f" [{f.get('search_date')}]" if f.get('search_date') else ""
        print(f"#{i}{date_tag} {f.get('airline')} {f.get('flight_number')} | {f.get('departure_airport')} -> {f.get('arrival_airport')} | {f.get('departure_time')} - {f.get('arrival_time')} | {f.get('duration')} | {f.get('price')} {f.get('currency')} | stops: {f.get('stops')} | layovers: {layover_str}")


def main():
    # Optional: health check
    try:
        r = requests.get(HEALTH_URL, timeout=5)
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
        resp = requests.post(CHAT_URL, json=payload, timeout=60)
        if resp.status_code != 200:
            print("Request failed", resp.status_code, resp.text)
            sys.exit(1)
        data = resp.json()

        rtype = data.get("response_type")
        print(f"\nStep {step} -> response_type={rtype}")

        assistant_message = data.get("message", "")
        print("Assistant:", assistant_message)

        # Persist the turn so server is stateful across requests
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": assistant_message})

        if rtype == "results":
            flights = data.get("flights", [])
            print_flights(flights)
            summary = data.get("summary")
            if summary:
                print("\nSummary:\n", summary)
            print("\nDone.")
            return

        # Otherwise, answer the follow-up
        user_message = pick_reply(assistant_message)
        print("User:", user_message)
        time.sleep(0.3)

    print("Reached step limit without results. Check API keys and server logs.")


if __name__ == "__main__":
    main()