from typing import List, Optional, Any
from html import escape
from models import ExtractedInfo

def question_to_html(question: str, extracted_info: ExtractedInfo) -> str:
    """Format follow-up question with current info as HTML"""
    html = f"<div class='question-response'>"
    html += f"<div class='question'>"
    html += f"<p>{question}</p>"
    html += f"</div>"
    
    # Show current progress
    info_items = []
    if extracted_info.departure_date:
        info_items.append(f"📅 {extracted_info.departure_date}")
    if extracted_info.origin:
        info_items.append(f"🛫 From {extracted_info.origin}")
    if extracted_info.destination:
        info_items.append(f"🛬 To {extracted_info.destination}")
    if extracted_info.cabin_class:
        info_items.append(f"💺 {extracted_info.cabin_class.title()}")
    if extracted_info.duration:
        info_items.append(f"📆 {extracted_info.duration} days")
    
    if info_items:
        html += f"<div class='progress'>"
        html += f"<p><strong>Information collected:</strong></p>"
        html += f"<p>{' • '.join(info_items)}</p>"
        html += f"</div>"
    
    html += f"</div>"
    return html

def _get(obj: Any, attr: str, default: str = "N/A") -> str:
    """
    Helper to retrieve attribute from objects or keys from dicts.
    Returns default if not present or value is falsy (but not zero).
    """
    if obj is None:
        return default
    # If dict-like
    try:
        if isinstance(obj, dict):
            val = obj.get(attr, default)
            return val if val is not None else default
    except Exception:
        pass
    # If object with attribute
    try:
        val = getattr(obj, attr)
        return val if val is not None else default
    except Exception:
        return default

def format_extracted_info_html(extracted_info: ExtractedInfo) -> str:
    """Format extracted information as HTML"""
    html = "<div class='extracted-info'><h4>Current Information:</h4><ul>"
    
    if extracted_info.departure_date:
        html += f"<li><strong>Departure Date:</strong> {extracted_info.departure_date}</li>"
    if extracted_info.origin:
        html += f"<li><strong>From:</strong> {extracted_info.origin}</li>"
    if extracted_info.destination:
        html += f"<li><strong>To:</strong> {extracted_info.destination}</li>"
    if extracted_info.cabin_class:
        html += f"<li><strong>Cabin:</strong> {extracted_info.cabin_class.title()}</li>"
    if extracted_info.duration:
        html += f"<li><strong>Duration:</strong> {extracted_info.duration} days</li>"
    
    html += "</ul></div>"
    return html

def html_table_converter(flights: List[Any], summary: Optional[str] = None) -> str:
    """Build HTML string containing a table with flight rows then the summary text below."""
    if not flights:
        return "<div>No flights found for your criteria.</div>"

    style = (
        "<style>"
        "table.flight-table{width:100%;border-collapse:collapse;font-family:inherit;background:transparent;}"
        "table.flight-table th,table.flight-table td{border:1px solid;"
        "border-color:rgba(120,120,120,0.2);padding:6px 8px;text-align:left;}"
        "table.flight-table th{font-weight:bold;}"
        "@media (prefers-color-scheme:dark){"
        "table.flight-table th,table.flight-table td{border-color:rgba(180,180,180,0.2);color:#eee;}"
        "table.flight-table{color:#eee;}"
        "}"
        "@media (prefers-color-scheme:light){"
        "table.flight-table th,table.flight-table td{border-color:rgba(120,120,120,0.2);color:#222;}"
        "table.flight-table{color:#222;}"
        "}"
        ".summary-block{margin-top:12px;padding:10px;border:1px solid #eee;font-family:inherit;}"
        "</style>"
    )

    # Table header
    html = style + "<table class='flight-table'>"
    html += (
        "<thead>"
        "<tr>"
        "<th>#</th>"
        "<th>Outbound (airline / flight)</th>"
        "<th>Outbound route & times</th>"
        "<th>Outbound duration / stops</th>"
        "<th>Return (airline / flight)</th>"
        "<th>Return route & times</th>"
        "<th>Return duration / stops</th>"
        "<th>Price</th>"
        "<th>Search date</th>"
        "</tr>"
        "</thead><tbody>"
    )

    for i, flight in enumerate(flights, start=1):
        # Outbound fields
        out = _get(flight, "outbound", {})
        out_airline = _get(out, "airline")
        out_flight_no = _get(out, "flight_number")
        out_dep = _get(out, "departure_airport")
        out_arr = _get(out, "arrival_airport")
        out_dep_time = _get(out, "departure_time")
        out_arr_time = _get(out, "arrival_time")
        out_duration = _get(out, "duration")
        out_stops = _get(out, "stops", "0")
        out_layovers = _get(out, "layovers", [])
        if isinstance(out_layovers, (list, tuple)):
            out_layovers_display = ", ".join(map(str, out_layovers)) if out_layovers else ""
        else:
            out_layovers_display = str(out_layovers)

        # Return fields
        ret = _get(flight, "return_leg", {}) or _get(flight, "return", {})
        ret_airline = _get(ret, "airline")
        ret_flight_no = _get(ret, "flight_number")
        ret_dep = _get(ret, "departure_airport")
        ret_arr = _get(ret, "arrival_airport")
        ret_dep_time = _get(ret, "departure_time")
        ret_arr_time = _get(ret, "arrival_time")
        ret_duration = _get(ret, "duration")
        ret_stops = _get(ret, "stops", "0")
        ret_layovers = _get(ret, "layovers", [])
        if isinstance(ret_layovers, (list, tuple)):
            ret_layovers_display = ", ".join(map(str, ret_layovers)) if ret_layovers else ""
        else:
            ret_layovers_display = str(ret_layovers)

        # Price and search date
        price = _get(flight, "price", "N/A")
        currency = _get(flight, "currency", "")
        price_display = f"{escape(str(currency))} {escape(str(price))}" if price != "N/A" else "Price not available"
        search_date = _get(flight, "search_date", "")

        # Build HTML-safe strings
        out_header = f"{escape(str(out_airline))} {escape(str(out_flight_no))}"
        out_route = f"{escape(str(out_dep))} {escape(str(out_dep_time))} → {escape(str(out_arr))} {escape(str(out_arr_time))}"
        out_details = f"{escape(str(out_duration))}"
        if out_stops not in (None, "", "0"):
            out_details += f" / {escape(str(out_stops))} stop(s)"
            if out_layovers_display:
                out_details += f" ({escape(out_layovers_display)})"

        ret_header = f"{escape(str(ret_airline))} {escape(str(ret_flight_no))}" if ret_airline != "N/A" else "—"
        ret_route = f"{escape(str(ret_dep))} {escape(str(ret_dep_time))} → {escape(str(ret_arr))} {escape(str(ret_arr_time))}" if ret_airline != "N/A" else "—"
        ret_details = f"{escape(str(ret_duration))}"
        if ret_stops not in (None, "", "0"):
            ret_details += f" / {escape(str(ret_stops))} stop(s)"
            if ret_layovers_display:
                ret_details += f" ({escape(ret_layovers_display)})"

        html += (
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{out_header}</td>"
            f"<td>{out_route}</td>"
            f"<td>{out_details}</td>"
            f"<td>{ret_header}</td>"
            f"<td>{ret_route}</td>"
            f"<td>{ret_details}</td>"
            f"<td>{price_display}</td>"
            f"<td>{escape(str(search_date))}</td>"
            "</tr>"
        )

    html += "</tbody></table>"

    # Append summary text after the table
    if summary:
        html += f"<div class='summary-block'>{escape(str(summary))}</div>"

    return html
