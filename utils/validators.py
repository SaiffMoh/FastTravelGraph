import re
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional

def validate_extracted_info(extracted_info: dict) -> Tuple[dict, List[str]]:
    """Validate extracted information and return cleaned data + validation errors"""
    validated_info = {}
    errors = []
    
    # Validate departure date
    if extracted_info.get('departure_date'):
        date_validation = validate_date(extracted_info['departure_date'])
        if date_validation['valid']:
            validated_info['departure_date'] = date_validation['formatted_date']
        else:
            errors.append(f"Date issue: {date_validation['error']}")
    
    # Validate origin and destination
    for location_field in ['origin', 'destination']:
        if extracted_info.get(location_field):
            location = extracted_info[location_field].strip()
            if len(location) >= 2:  # Basic validation
                validated_info[location_field] = location.title()
            else:
                errors.append(f"Invalid {location_field}: too short")
    
    # Validate cabin class
    if extracted_info.get('cabin_class'):
        cabin = validate_cabin_class(extracted_info['cabin_class'])
        if cabin:
            validated_info['cabin_class'] = cabin
        else:
            errors.append("Invalid cabin class. Must be economy, business, or first class")
    
    # Validate trip type
    if extracted_info.get('trip_type'):
        trip_type = validate_trip_type(extracted_info['trip_type'])
        if trip_type:
            validated_info['trip_type'] = trip_type
        else:
            errors.append("Invalid trip type. Must be 'one way' or 'round trip'")
    
    # Validate duration
    if extracted_info.get('duration'):
        duration = validate_duration(extracted_info['duration'])
        if duration:
            validated_info['duration'] = duration
        else:
            errors.append("Invalid duration. Must be a positive number of days")
    
    return validated_info, errors

def validate_date(date_str: str) -> Dict[str, any]:
    """Validate and parse date string"""
    date_formats = [
        "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", 
        "%B %d, %Y", "%B %d", "%b %d", "%m/%d", "%d/%m"
    ]
    
    current_year = datetime.now().year
    today = datetime.now().date()
    
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_str.strip(), fmt).date()
            
            # If no year provided, assume current year
            if parsed_date.year == 1900:  # Default year for formats without year
                parsed_date = parsed_date.replace(year=current_year)
            
            # If date is in the past, try next year
            if parsed_date < today:
                if parsed_date.month < today.month or (parsed_date.month == today.month and parsed_date.day < today.day):
                    parsed_date = parsed_date.replace(year=current_year + 1)
            
            # Check if date is too far in the future (more than 2 years)
            if parsed_date > today + timedelta(days=730):
                return {"valid": False, "error": "Date is too far in the future (max 2 years)"}
            
            return {"valid": True, "formatted_date": parsed_date.strftime("%Y-%m-%d")}
            
        except ValueError:
            continue
    
    return {"valid": False, "error": f"Could not parse date '{date_str}'. Try formats like '2025-12-25' or 'December 25, 2025'"}

def validate_cabin_class(cabin_str: str) -> Optional[str]:
    """Validate and normalize cabin class"""
    cabin_lower = cabin_str.lower().strip()
    cabin_mappings = {
        'economy': 'economy',
        'eco': 'economy',
        'coach': 'economy',
        'business': 'business',
        'biz': 'business',
        'first': 'first class',
        'first class': 'first class',
        'premium': 'business'  # Treat premium as business
    }
    return cabin_mappings.get(cabin_lower)

def validate_trip_type(trip_str: str) -> Optional[str]:
    """Validate and normalize trip type"""
    trip_lower = trip_str.lower().strip()
    if any(word in trip_lower for word in ['round', 'return', 'two way', 'roundtrip']):
        return 'round trip'
    elif any(word in trip_lower for word in ['one way', 'oneway', 'single']):
        return 'one way'
    return None

def validate_duration(duration_str: str) -> Optional[int]:
    """Validate and extract duration in days"""
    # Extract numbers from string
    numbers = re.findall(r'\d+', str(duration_str))
    if numbers:
        duration = int(numbers[0])
        if 1 <= duration <= 365:  # Reasonable range
            return duration
    return None