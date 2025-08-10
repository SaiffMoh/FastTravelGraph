# FastTravelGraph

An AI-powered flight search assistant built with FastAPI + LangGraph that:
- Extracts flight details through conversation
- Defaults to round-trip and asks for trip duration
- Queries Amadeus for up to 5 offers for the chosen day plus 2 extra days
- Returns grouped results by day with outbound and return legs, layovers, and stops
- Supports an interactive CLI to chat from your terminal

## Requirements
- Python 3.11+
- Amadeus test credentials
- Optional: OpenAI API key for nicer follow-ups/summaries

## Setup
1) Install deps
- Windows (inside venv):
  ```powershell
  pip install -r requirements.txt
  ```

2) Environment variables (put in .env or export in shell)
- AMADEUS_CLIENT_ID
- AMADEUS_CLIENT_SECRET
- OPENAI_API_KEY (optional)
- DEBUG=1 (optional; prints simple connection checkpoints)

## Run the API server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Health check:
```bash
curl http://localhost:8000/health
```

## Chat endpoint
POST /chat
- Request body:
```json
{
  "message": "i want to travel from cairo to dubai",
  "conversation_history": []
}
```
- Response types:
  - question: a follow-up message to collect a missing field
  - selection: request to select a flight offer from displayed results
  - results: grouped flight offers by day
  - confirmation: confirmation of selected flight offer

## Conversation flow rules
- Default trip type: round trip
- Duration is always required
- The assistant asks only for missing information in order of importance: date → duration → origin → destination → cabin

## Multi-day search
- The server fetches up to 5 offers for the chosen date and for the next 2 days (3 days total)
- Offers are grouped and labeled by search_date
- Each offer contains two legs when available: outbound and return

## Flight offer selection
- After displaying flight results, the system prompts the user to select a specific offer
- Each offer is assigned a unique ID (e.g., OFFER_001, OFFER_002)
- Users can select an offer by entering the corresponding Offer ID
- The system validates the selection and provides confirmation

## CLI usage (interactive chat)
From project root:
```bash
python scripts/e2e_test.py           # interactive mode
python scripts/e2e_test.py --auto    # auto mode (for quick e2e testing)
```
Commands in interactive mode:
- Type your message and press Enter
- /reset to clear conversation
- /quit to exit

You can change the server location or timeouts via env vars:
- BASE_URL (default http://localhost:8000)
- CLIENT_TIMEOUT (default 120)

## Output format (CLI)
- Results are printed as separate tables for each day
- Each offer prints two rows: Out (outbound) and Ret (return) with:
  - Price, currency, airline, flight number
  - From/To, departure/arrival times
  - Duration, stops, layover windows

## Debugging
- To confirm remote connectivity without printing payloads, set:
```bash
export DEBUG=1   # or $env:DEBUG="1" on PowerShell
```
You’ll see simple checkpoints like:
- [DEBUG] Amadeus token: connecting… / connected ✔
- [DEBUG] Amadeus flight-offers: connecting… / connected ✔

If requests time out:
- Increase CLIENT_TIMEOUT for the CLI
- Ensure your Amadeus credentials are correct
- Reduce `max_workers` or window size in `get_flight_offers_node` if needed

## Notes
- With OPENAI_API_KEY set, the bot generates more natural follow-ups and summaries
- Without it, deterministic follow-ups are used and summaries are simplified