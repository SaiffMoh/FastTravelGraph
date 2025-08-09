# FastTravelGraph

Run the server:

- Export required keys:
  - `OPENAI_API_KEY` (optional, enables LLM-based follow-ups and summaries)
  - `AMADEUS_CLIENT_ID`
  - `AMADEUS_CLIENT_SECRET`
- Install requirements: `python3 -m pip install --break-system-packages -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port 8000`

Sample request:

```bash
curl -X POST 'http://localhost:8000/chat' \
 -H 'accept: application/json' \
 -H 'Content-Type: application/json' \
 -d '{
  "message": "i want to travel from cairo to dubai",
  "conversation_history": []
}'
```

Expected response (no OPENAI_API_KEY):
- response_type: "question"
- message: targeted follow-up like "What date would you like to depart? Please use a format like 2025-12-25."
- extracted_info: includes origin="Cairo", destination="Dubai"