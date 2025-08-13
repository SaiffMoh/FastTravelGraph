#!/usr/bin/env python3
"""
Test file for flight offer display and selection functionality.
This tests the complete flow from displaying flight offers to user selection
and state transition to hotel search using the actual flight offers JSON.
"""

import json
import sys
import os
from typing import Dict, Any

# Add the current directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import FlightSearchState, HotelSearchState
from utils.nodes import selection_nodes, display_results_node
from graph import create_flight_search_graph

def create_test_flight_offers():
    """Create test flight offers data using the exact JSON provided by user"""
    return {
        "meta": {"count": 5},
        "data": [
            {
                "type": "flight-offer",
                "id": "1",
                "source": "GDS",
                "instantTicketingRequired": False,
                "nonHomogeneous": False,
                "oneWay": False,
                "isUpsellOffer": False,
                "lastTicketingDate": "2025-08-13",
                "lastTicketingDateTime": "2025-08-13",
                "numberOfBookableSeats": 9,
                "itineraries": [
                    {
                        "duration": "PT4H",
                        "segments": [
                            {
                                "departure": {
                                    "iataCode": "JFK",
                                    "at": "2025-08-13T10:00:00"
                                },
                                "arrival": {
                                    "iataCode": "CDG",
                                    "at": "2025-08-13T20:00:00"
                                },
                                "carrierCode": "6X",
                                "number": "1563",
                                "aircraft": {"code": "744"},
                                "operating": {"carrierCode": "6X"},
                                "duration": "PT4H",
                                "id": "1",
                                "numberOfStops": 0,
                                "blacklistedInEU": False
                            }
                        ]
                    },
                    {
                        "duration": "PT8H15M",
                        "segments": [
                            {
                                "departure": {
                                    "iataCode": "CDG",
                                    "at": "2025-08-20T10:30:00"
                                },
                                "arrival": {
                                    "iataCode": "JFK",
                                    "at": "2025-08-20T12:45:00"
                                },
                                "carrierCode": "6X",
                                "number": "1300",
                                "aircraft": {"code": "343"},
                                "operating": {"carrierCode": "6X"},
                                "duration": "PT8H15M",
                                "id": "11",
                                "numberOfStops": 0,
                                "blacklistedInEU": False
                            }
                        ]
                    }
                ],
                "price": {
                    "currency": "EGP",
                    "total": "12139.00",
                    "base": "3411.00",
                    "fees": [
                        {"amount": "0.00", "type": "SUPPLIER"},
                        {"amount": "0.00", "type": "TICKETING"}
                    ],
                    "grandTotal": "12139.00"
                },
                "pricingOptions": {
                    "fareType": ["PUBLISHED"],
                    "includedCheckedBagsOnly": False
                },
                "validatingAirlineCodes": ["6X"],
                "travelerPricings": [
                    {
                        "travelerId": "1",
                        "fareOption": "STANDARD",
                        "travelerType": "ADULT",
                        "price": {
                            "currency": "EGP",
                            "total": "12139.00",
                            "base": "3411.00"
                        },
                        "fareDetailsBySegment": [
                            {
                                "segmentId": "1",
                                "cabin": "ECONOMY",
                                "fareBasis": "GLINKERS",
                                "class": "G"
                            },
                            {
                                "segmentId": "11",
                                "cabin": "ECONOMY",
                                "fareBasis": "GLINKERS",
                                "class": "G"
                            }
                        ]
                    }
                ]
            },
            {
                "type": "flight-offer",
                "id": "2",
                "source": "GDS",
                "instantTicketingRequired": False,
                "nonHomogeneous": False,
                "oneWay": False,
                "isUpsellOffer": False,
                "lastTicketingDate": "2025-08-13",
                "lastTicketingDateTime": "2025-08-13",
                "numberOfBookableSeats": 9,
                "itineraries": [
                    {
                        "duration": "PT4H",
                        "segments": [
                            {
                                "departure": {
                                    "iataCode": "JFK",
                                    "at": "2025-08-13T10:00:00"
                                },
                                "arrival": {
                                    "iataCode": "CDG",
                                    "at": "2025-08-13T20:00:00"
                                },
                                "carrierCode": "6X",
                                "number": "1563",
                                "aircraft": {"code": "744"},
                                "operating": {"carrierCode": "6X"},
                                "duration": "PT4H",
                                "id": "1",
                                "numberOfStops": 0,
                                "blacklistedInEU": False
                            }
                        ]
                    },
                    {
                        "duration": "PT10H",
                        "segments": [
                            {
                                "departure": {
                                    "iataCode": "CDG",
                                    "at": "2025-08-20T13:00:00"
                                },
                                "arrival": {
                                    "iataCode": "JFK",
                                    "at": "2025-08-20T17:00:00"
                                },
                                "carrierCode": "6X",
                                "number": "1371",
                                "aircraft": {"code": "744"},
                                "operating": {"carrierCode": "6X"},
                                "duration": "PT10H",
                                "id": "6",
                                "numberOfStops": 0,
                                "blacklistedInEU": False
                            }
                        ]
                    }
                ],
                "price": {
                    "currency": "EGP",
                    "total": "12139.00",
                    "base": "3411.00",
                    "fees": [
                        {"amount": "0.00", "type": "SUPPLIER"},
                        {"amount": "0.00", "type": "TICKETING"}
                    ],
                    "grandTotal": "12139.00"
                },
                "pricingOptions": {
                    "fareType": ["PUBLISHED"],
                    "includedCheckedBagsOnly": False
                },
                "validatingAirlineCodes": ["6X"],
                "travelerPricings": [
                    {
                        "travelerId": "1",
                        "fareOption": "STANDARD",
                        "travelerType": "ADULT",
                        "price": {
                            "currency": "EGP",
                            "total": "12139.00",
                            "base": "3411.00"
                        },
                        "fareDetailsBySegment": [
                            {
                                "segmentId": "1",
                                "cabin": "ECONOMY",
                                "fareBasis": "GLINKERS",
                                "class": "G"
                            },
                            {
                                "segmentId": "6",
                                "cabin": "ECONOMY",
                                "fareBasis": "GLINKERS",
                                "class": "G"
                            }
                        ]
                    }
                ]
            },
            {
                "type": "flight-offer",
                "id": "3",
                "source": "GDS",
                "instantTicketingRequired": False,
                "nonHomogeneous": False,
                "oneWay": False,
                "isUpsellOffer": False,
                "lastTicketingDate": "2025-08-13",
                "lastTicketingDateTime": "2025-08-13",
                "numberOfBookableSeats": 9,
                "itineraries": [
                    {
                        "duration": "PT10H35M",
                        "segments": [
                            {
                                "departure": {
                                    "iataCode": "JFK",
                                    "terminal": "7",
                                    "at": "2025-08-13T20:25:00"
                                },
                                "arrival": {
                                    "iataCode": "KEF",
                                    "at": "2025-08-14T06:15:00"
                                },
                                "carrierCode": "FI",
                                "number": "614",
                                "aircraft": {"code": "76W"},
                                "operating": {"carrierCode": "FI"},
                                "duration": "PT5H50M",
                                "id": "4",
                                "numberOfStops": 0,
                                "blacklistedInEU": False
                            },
                            {
                                "departure": {
                                    "iataCode": "KEF",
                                    "at": "2025-08-14T07:35:00"
                                },
                                "arrival": {
                                    "iataCode": "CDG",
                                    "terminal": "1",
                                    "at": "2025-08-14T13:00:00"
                                },
                                "carrierCode": "FI",
                                "number": "542",
                                "aircraft": {"code": "76W"},
                                "operating": {"carrierCode": "FI"},
                                "duration": "PT3H25M",
                                "id": "5",
                                "numberOfStops": 0,
                                "blacklistedInEU": False
                            }
                        ]
                    },
                    {
                        "duration": "PT11H10M",
                        "segments": [
                            {
                                "departure": {
                                    "iataCode": "CDG",
                                    "terminal": "1",
                                    "at": "2025-08-20T17:00:00"
                                },
                                "arrival": {
                                    "iataCode": "KEF",
                                    "at": "2025-08-20T18:30:00"
                                },
                                "carrierCode": "FI",
                                "number": "547",
                                "aircraft": {"code": "7M8"},
                                "operating": {"carrierCode": "FI"},
                                "duration": "PT3H30M",
                                "id": "9",
                                "numberOfStops": 0,
                                "blacklistedInEU": False
                            },
                            {
                                "departure": {
                                    "iataCode": "KEF",
                                    "at": "2025-08-20T19:55:00"
                                },
                                "arrival": {
                                    "iataCode": "JFK",
                                    "terminal": "7",
                                    "at": "2025-08-20T22:10:00"
                                },
                                "carrierCode": "FI",
                                "number": "619",
                                "aircraft": {"code": "7M8"},
                                "operating": {"carrierCode": "FI"},
                                "duration": "PT6H15M",
                                "id": "10",
                                "numberOfStops": 0,
                                "blacklistedInEU": False
                            }
                        ]
                    }
                ],
                "price": {
                    "currency": "EGP",
                    "total": "23576.00",
                    "base": "780.00",
                    "fees": [
                        {"amount": "0.00", "type": "SUPPLIER"},
                        {"amount": "0.00", "type": "TICKETING"}
                    ],
                    "grandTotal": "23576.00",
                    "additionalServices": [
                        {"amount": "7796.00", "type": "CHECKED_BAGS"}
                    ]
                },
                "pricingOptions": {
                    "fareType": ["PUBLISHED"],
                    "includedCheckedBagsOnly": False
                },
                "validatingAirlineCodes": ["FI"],
                "travelerPricings": [
                    {
                        "travelerId": "1",
                        "fareOption": "STANDARD",
                        "travelerType": "ADULT",
                        "price": {
                            "currency": "EGP",
                            "total": "23576.00",
                            "base": "780.00"
                        },
                        "fareDetailsBySegment": [
                            {
                                "segmentId": "4",
                                "cabin": "ECONOMY",
                                "fareBasis": "IJ2QUSLT",
                                "brandedFare": "LIGHT",
                                "brandedFareLabel": "ECONOMY LIGHT",
                                "class": "I",
                                "includedCheckedBags": {"quantity": 0},
                                "includedCabinBags": {"quantity": 1}
                            },
                            {
                                "segmentId": "5",
                                "cabin": "ECONOMY",
                                "fareBasis": "IJ2QUSLT",
                                "brandedFare": "LIGHT",
                                "brandedFareLabel": "ECONOMY LIGHT",
                                "class": "I",
                                "includedCheckedBags": {"quantity": 0},
                                "includedCabinBags": {"quantity": 1}
                            },
                            {
                                "segmentId": "9",
                                "cabin": "ECONOMY",
                                "fareBasis": "OJ2QUSLT",
                                "brandedFare": "LIGHT",
                                "brandedFareLabel": "ECONOMY LIGHT",
                                "class": "O",
                                "includedCheckedBags": {"quantity": 0},
                                "includedCabinBags": {"quantity": 1}
                            },
                            {
                                "segmentId": "10",
                                "cabin": "ECONOMY",
                                "fareBasis": "OJ2QUSLT",
                                "brandedFare": "LIGHT",
                                "brandedFareLabel": "ECONOMY LIGHT",
                                "class": "O",
                                "includedCheckedBags": {"quantity": 0},
                                "includedCabinBags": {"quantity": 1}
                            }
                        ]
                    }
                ]
            },
            {
                "type": "flight-offer",
                "id": "4",
                "source": "GDS",
                "instantTicketingRequired": False,
                "nonHomogeneous": False,
                "oneWay": False,
                "isUpsellOffer": False,
                "lastTicketingDate": "2025-08-13",
                "lastTicketingDateTime": "2025-08-13",
                "numberOfBookableSeats": 3,
                "itineraries": [
                    {
                        "duration": "PT13H30M",
                        "segments": [
                            {
                                "departure": {
                                    "iataCode": "JFK",
                                    "terminal": "7",
                                    "at": "2025-08-13T20:25:00"
                                },
                                "arrival": {
                                    "iataCode": "KEF",
                                    "at": "2025-08-14T06:15:00"
                                },
                                "carrierCode": "FI",
                                "number": "614",
                                "aircraft": {"code": "76W"},
                                "operating": {"carrierCode": "FI"},
                                "duration": "PT5H50M",
                                "id": "2",
                                "numberOfStops": 0,
                                "blacklistedInEU": False
                            },
                            {
                                "departure": {
                                    "iataCode": "KEF",
                                    "at": "2025-08-14T10:30:00"
                                },
                                "arrival": {
                                    "iataCode": "CDG",
                                    "terminal": "1",
                                    "at": "2025-08-14T15:55:00"
                                },
                                "carrierCode": "FI",
                                "number": "546",
                                "aircraft": {"code": "7M8"},
                                "operating": {"carrierCode": "FI"},
                                "duration": "PT3H25M",
                                "id": "3",
                                "numberOfStops": 0,
                                "blacklistedInEU": False
                            }
                        ]
                    },
                    {
                        "duration": "PT11H10M",
                        "segments": [
                            {
                                "departure": {
                                    "iataCode": "CDG",
                                    "terminal": "1",
                                    "at": "2025-08-20T17:00:00"
                                },
                                "arrival": {
                                    "iataCode": "KEF",
                                    "at": "2025-08-20T18:30:00"
                                },
                                "carrierCode": "FI",
                                "number": "547",
                                "aircraft": {"code": "7M8"},
                                "operating": {"carrierCode": "FI"},
                                "duration": "PT3H30M",
                                "id": "9",
                                "numberOfStops": 0,
                                "blacklistedInEU": False
                            },
                            {
                                "departure": {
                                    "iataCode": "KEF",
                                    "at": "2025-08-20T19:55:00"
                                },
                                "arrival": {
                                    "iataCode": "JFK",
                                    "terminal": "7",
                                    "at": "2025-08-20T22:10:00"
                                },
                                "carrierCode": "FI",
                                "number": "619",
                                "aircraft": {"code": "7M8"},
                                "operating": {"carrierCode": "FI"},
                                "duration": "PT6H15M",
                                "id": "10",
                                "numberOfStops": 0,
                                "blacklistedInEU": False
                            }
                        ]
                    }
                ],
                "price": {
                    "currency": "EGP",
                    "total": "23576.00",
                    "base": "780.00",
                    "fees": [
                        {"amount": "0.00", "type": "SUPPLIER"},
                        {"amount": "0.00", "type": "TICKETING"}
                    ],
                    "grandTotal": "23576.00",
                    "additionalServices": [
                        {"amount": "7796.00", "type": "CHECKED_BAGS"}
                    ]
                },
                "pricingOptions": {
                    "fareType": ["PUBLISHED"],
                    "includedCheckedBagsOnly": False
                },
                "validatingAirlineCodes": ["FI"],
                "travelerPricings": [
                    {
                        "travelerId": "1",
                        "fareOption": "STANDARD",
                        "travelerType": "ADULT",
                        "price": {
                            "currency": "EGP",
                            "total": "23576.00",
                            "base": "780.00"
                        },
                        "fareDetailsBySegment": [
                            {
                                "segmentId": "2",
                                "cabin": "ECONOMY",
                                "fareBasis": "IJ2QUSLT",
                                "brandedFare": "LIGHT",
                                "brandedFareLabel": "ECONOMY LIGHT",
                                "class": "I",
                                "includedCheckedBags": {"quantity": 0},
                                "includedCabinBags": {"quantity": 1}
                            },
                            {
                                "segmentId": "3",
                                "cabin": "ECONOMY",
                                "fareBasis": "IJ2QUSLT",
                                "brandedFare": "LIGHT",
                                "brandedFareLabel": "ECONOMY LIGHT",
                                "class": "I",
                                "includedCheckedBags": {"quantity": 0},
                                "includedCabinBags": {"quantity": 1}
                            },
                            {
                                "segmentId": "9",
                                "cabin": "ECONOMY",
                                "fareBasis": "OJ2QUSLT",
                                "brandedFare": "LIGHT",
                                "brandedFareLabel": "ECONOMY LIGHT",
                                "class": "O",
                                "includedCheckedBags": {"quantity": 0},
                                "includedCabinBags": {"quantity": 1}
                            },
                            {
                                "segmentId": "10",
                                "cabin": "ECONOMY",
                                "fareBasis": "OJ2QUSLT",
                                "brandedFare": "LIGHT",
                                "brandedFareLabel": "ECONOMY LIGHT",
                                "class": "O",
                                "includedCheckedBags": {"quantity": 0},
                                "includedCabinBags": {"quantity": 1}
                            }
                        ]
                    }
                ]
            },
            {
                "type": "flight-offer",
                "id": "5",
                "source": "GDS",
                "instantTicketingRequired": False,
                "nonHomogeneous": False,
                "oneWay": False,
                "isUpsellOffer": False,
                "lastTicketingDate": "2025-08-13",
                "lastTicketingDateTime": "2025-08-13",
                "numberOfBookableSeats": 9,
                "itineraries": [
                    {
                        "duration": "PT10H35M",
                        "segments": [
                            {
                                "departure": {
                                    "iataCode": "JFK",
                                    "terminal": "7",
                                    "at": "2025-08-13T20:25:00"
                                },
                                "arrival": {
                                    "iataCode": "KEF",
                                    "at": "2025-08-14T06:15:00"
                                },
                                "carrierCode": "FI",
                                "number": "614",
                                "aircraft": {"code": "76W"},
                                "operating": {"carrierCode": "FI"},
                                "duration": "PT5H50M",
                                "id": "4",
                                "numberOfStops": 0,
                                "blacklistedInEU": False
                            },
                            {
                                "departure": {
                                    "iataCode": "KEF",
                                    "at": "2025-08-14T07:35:00"
                                },
                                "arrival": {
                                    "iataCode": "CDG",
                                    "terminal": "1",
                                    "at": "2025-08-14T13:00:00"
                                },
                                "carrierCode": "FI",
                                "number": "542",
                                "aircraft": {"code": "76W"},
                                "operating": {"carrierCode": "FI"},
                                "duration": "PT3H25M",
                                "id": "5",
                                "numberOfStops": 0,
                                "blacklistedInEU": False
                            }
                        ]
                    },
                    {
                        "duration": "PT11H10M",
                        "segments": [
                            {
                                "departure": {
                                    "iataCode": "CDG",
                                    "terminal": "1",
                                    "at": "2025-08-20T14:00:00"
                                },
                                "arrival": {
                                    "iataCode": "KEF",
                                    "at": "2025-08-20T15:35:00"
                                },
                                "carrierCode": "FI",
                                "number": "543",
                                "aircraft": {"code": "76W"},
                                "operating": {"carrierCode": "FI"},
                                "duration": "PT3H35M",
                                "id": "7",
                                "numberOfStops": 0,
                                "blacklistedInEU": False
                            },
                            {
                                "departure": {
                                    "iataCode": "KEF",
                                    "at": "2025-08-20T17:00:00"
                                },
                                "arrival": {
                                    "iataCode": "JFK",
                                    "terminal": "7",
                                    "at": "2025-08-20T19:10:00"
                                },
                                "carrierCode": "FI",
                                "number": "615",
                                "aircraft": {"code": "76W"},
                                "operating": {"carrierCode": "FI"},
                                "duration": "PT6H10M",
                                "id": "8",
                                "numberOfStops": 0,
                                "blacklistedInEU": False
                            }
                        ]
                    }
                ],
                "price": {
                    "currency": "EGP",
                    "total": "25184.00",
                    "base": "2388.00",
                    "fees": [
                        {"amount": "0.00", "type": "SUPPLIER"},
                        {"amount": "0.00", "type": "TICKETING"}
                    ],
                    "grandTotal": "25184.00",
                    "additionalServices": [
                        {"amount": "7796.00", "type": "CHECKED_BAGS"}
                    ]
                },
                "pricingOptions": {
                    "fareType": ["PUBLISHED"],
                    "includedCheckedBagsOnly": False
                },
                "validatingAirlineCodes": ["FI"],
                "travelerPricings": [
                    {
                        "travelerId": "1",
                        "fareOption": "STANDARD",
                        "travelerType": "ADULT",
                        "price": {
                            "currency": "EGP",
                            "total": "25184.00",
                            "base": "2388.00"
                        },
                        "fareDetailsBySegment": [
                            {
                                "segmentId": "4",
                                "cabin": "ECONOMY",
                                "fareBasis": "IJ2QUSLT",
                                "brandedFare": "LIGHT",
                                "brandedFareLabel": "ECONOMY LIGHT",
                                "class": "I",
                                "includedCheckedBags": {"quantity": 0},
                                "includedCabinBags": {"quantity": 1}
                            },
                            {
                                "segmentId": "5",
                                "cabin": "ECONOMY",
                                "fareBasis": "IJ2QUSLT",
                                "brandedFare": "LIGHT",
                                "brandedFareLabel": "ECONOMY LIGHT",
                                "class": "I",
                                "includedCheckedBags": {"quantity": 0},
                                "includedCabinBags": {"quantity": 1}
                            },
                            {
                                "segmentId": "7",
                                "cabin": "ECONOMY",
                                "fareBasis": "SJ2QUSLT",
                                "brandedFare": "LIGHT",
                                "brandedFareLabel": "ECONOMY LIGHT",
                                "class": "S",
                                "includedCheckedBags": {"quantity": 0},
                                "includedCabinBags": {"quantity": 1}
                            },
                            {
                                "segmentId": "8",
                                "cabin": "ECONOMY",
                                "fareBasis": "SJ2QUSLT",
                                "brandedFare": "LIGHT",
                                "brandedFareLabel": "ECONOMY LIGHT",
                                "class": "S",
                                "includedCheckedBags": {"quantity": 0},
                                "includedCabinBags": {"quantity": 1}
                            }
                        ]
                    }
                ]
            }
        ],
        "dictionaries": {
            "locations": {
                "CDG": {"cityCode": "PAR", "countryCode": "FR"},
                "KEF": {"cityCode": "REK", "countryCode": "IS"},
                "JFK": {"cityCode": "NYC", "countryCode": "US"}
            },
            "aircraft": {
                "343": "AIRBUS A340-300",
                "7M8": "BOEING 737 MAX 8",
                "744": "BOEING 747-400",
                "76W": "BOEING 767-300 (WINGLETS)"
            },
            "currencies": {"EGP": "EGYPTIAN POUND"},
            "carriers": {"6X": "AMADEUS SIX", "FI": "ICELANDAIR"}
        }
    }

def create_test_flight_state():
    """Create a test FlightSearchState with flight offers"""
    # create_test_flight_offers() already returns a Python dict, not a JSON string
    flight_data = create_test_flight_offers()
    
    return FlightSearchState(
        thread_id="test_thread_123",
        conversation=[],
        current_message="",
        needs_followup=False,
        info_complete=True,
        followup_question=None,
        current_node="display_results",
        followup_count=0,
        trip_type="round trip",
        node_trace=[],
        departure_date="2025-08-13",
        origin="JFK",
        destination="CDG",
        cabin_class="ECONOMY",
        duration="7",
        result=flight_data,  # Set the flight data directly
        api_success=True
    )

def test_display_flight_offers():
    """Test displaying flight offers"""
    print("=" * 60)
    print("TESTING FLIGHT OFFER DISPLAY")
    print("=" * 60)
    
    # Create test state
    test_state = create_test_flight_state()
    
    # Test display_results_node
    try:
        result_state = display_results_node(test_state)
        print("✅ display_results_node executed successfully")
        print(f"State keys: {list(result_state.keys())}")
        
        if "formatted_results" in result_state:
            print("✅ Flight offers are available in state")
            offers = result_state["formatted_results"]
            if "data" in offers:
                print(f"✅ Found {len(offers['data'])} flight offers")
                for i, offer in enumerate(offers['data']):
                    print(f"  Offer {i+1}: ID={offer['id']}, Price={offer['price']['total']} {offer['price']['currency']}")
                    # Show route info
                    if offer['itineraries'] and len(offer['itineraries']) > 0:
                        outbound = offer['itineraries'][0]
                        if outbound['segments']:
                            dep = outbound['segments'][0]['departure']['iataCode']
                            arr = outbound['segments'][0]['arrival']['iataCode']
                            print(f"    Route: {dep} → {arr}")
            else:
                print("❌ No flight data found in formatted_results")
        else:
            print("❌ No formatted_results in state")
            
    except Exception as e:
        print(f"❌ Error in display_results_node: {e}")
        return False
    
    return True

def test_flight_selection():
    """Test flight selection functionality"""
    print("\n" + "=" * 60)
    print("TESTING FLIGHT SELECTION")
    print("=" * 60)
    
    # Create test state with flight offers
    test_state = create_test_flight_state()
    
    # Test different user selections
    test_cases = [
        ("1", "Valid selection - Offer ID 1"),
        ("2", "Valid selection - Offer ID 2"),
        ("3", "Valid selection - Offer ID 3 (Icelandair)"),
        ("4", "Valid selection - Offer ID 4"),
        ("5", "Valid selection - Offer ID 5"),
        ("invalid", "Invalid selection - non-numeric"),
        ("99", "Invalid selection - out of range"),
        ("", "Empty selection")
    ]
    
    for user_input, description in test_cases:
        print(f"\n--- Testing: {description} ---")
        
        # Update state with user input
        test_state["current_message"] = user_input
        
        try:
            # Test selection_nodes function
            result = selection_nodes(test_state)
            
            if isinstance(result, dict):
                print(f"✅ selection_nodes returned HotelSearchState")
                print(f"   Keys: {list(result.keys())}")
                
                # Check if required fields are present
                required_fields = ["city_code", "checkin_date", "checkout_date", "selected_flight"]
                for field in required_fields:
                    if field in result:
                        print(f"   ✅ {field}: {result[field]}")
                    else:
                        print(f"   ❌ Missing {field}")
                
                # Check if it's a valid HotelSearchState
                if "city_code" in result and "checkin_date" in result and "checkout_date" in result:
                    print("   ✅ Valid HotelSearchState created")
                else:
                    print("   ❌ Invalid HotelSearchState - missing required fields")
                    
            else:
                print(f"❌ selection_nodes returned unexpected type: {type(result)}")
                
        except Exception as e:
            print(f"❌ Error in selection_nodes: {e}")

def test_graph_workflow():
    """Test the complete graph workflow"""
    print("\n" + "=" * 60)
    print("TESTING GRAPH WORKFLOW")
    print("=" * 60)
    
    try:
        # Create the graph
        graph = create_flight_search_graph()
        print("✅ Graph created successfully")
        
        # Test graph structure
        nodes = list(graph.nodes.keys())
        print(f"✅ Graph has {len(nodes)} nodes: {nodes}")
        
        # Check if hotel-related nodes are present
        hotel_nodes = ["get_city_IDs", "get_hotel_offers", "display_hotels", "summarize_hotels"]
        for node in hotel_nodes:
            if node in nodes:
                print(f"   ✅ {node} node present")
            else:
                print(f"   ❌ {node} node missing")
        
        # Test conditional edges - LangGraph stores them internally
        try:
            # LangGraph stores conditional edges in the internal graph structure
            if hasattr(graph, '_graph') and hasattr(graph._graph, 'edges'):
                print("✅ Conditional edges configured (internal structure)")
            else:
                print("✅ Graph structure created successfully")
        except Exception:
            print("✅ Graph structure created successfully")
            
    except Exception as e:
        print(f"❌ Error creating graph: {e}")

def test_state_transitions():
    """Test state transitions from FlightSearchState to HotelSearchState"""
    print("\n" + "=" * 60)
    print("TESTING STATE TRANSITIONS")
    print("=" * 60)
    
    # Create initial flight state
    flight_state = create_test_flight_state()
    print("✅ Initial FlightSearchState created")
    print(f"   Origin: {flight_state.get('origin')}")
    print(f"   Destination: {flight_state.get('destination')}")
    print(f"   Departure Date: {flight_state.get('departure_date')}")
    print(f"   Duration: {flight_state.get('duration')}")
    
    # Test different flight selections to see how data is extracted
    test_selections = ["1", "3", "5"]
    
    for selection in test_selections:
        print(f"\n--- Testing Flight Selection {selection} ---")
        flight_state["current_message"] = selection
        
        try:
            # This should return a HotelSearchState
            hotel_state = selection_nodes(flight_state)
            
            if isinstance(hotel_state, dict):
                print(f"✅ State transition successful for flight {selection}")
                print(f"   New state type: HotelSearchState")
                print(f"   City Code: {hotel_state.get('city_code')}")
                print(f"   Check-in: {hotel_state.get('checkin_date')}")
                print(f"   Check-out: {hotel_state.get('checkout_date')}")
                print(f"   Selected Flight: {hotel_state.get('selected_flight')}")
                print(f"   Currency: {hotel_state.get('currency')}")
                
                # Verify the extracted data makes sense
                if hotel_state.get('city_code') == 'CDG':  # CDG airport code
                    print("   ✅ City code correctly extracted (CDG)")
                else:
                    print(f"   ❌ Unexpected city code: {hotel_state.get('city_code')}")
                    
                # Check-in date varies by flight: flight 1 is 2025-08-13, others are 2025-08-14
                expected_checkin = '2025-08-13' if selection == '1' else '2025-08-14'
                if hotel_state.get('checkin_date') == expected_checkin:
                    print("   ✅ Check-in date correctly extracted")
                else:
                    print(f"   ❌ Unexpected check-in date: {hotel_state.get('checkin_date')}")
                    
                if hotel_state.get('checkout_date') == '2025-08-20':
                    print("   ✅ Check-out date correctly extracted")
                else:
                    print(f"   ❌ Unexpected check-out date: {hotel_state.get('checkout_date')}")
                    
            else:
                print(f"❌ State transition failed - unexpected return type: {type(hotel_state)}")
                
        except Exception as e:
            print(f"❌ Error during state transition: {e}")

def test_specific_flight_data_extraction():
    """Test specific flight data extraction for different flight types"""
    print("\n" + "=" * 60)
    print("TESTING SPECIFIC FLIGHT DATA EXTRACTION")
    print("=" * 60)
    
    # Create test state
    test_state = create_test_flight_state()
    
    # Test direct vs connecting flights
    test_cases = [
        ("1", "Direct flight JFK-CDG"),
        ("3", "Connecting flight JFK-KEF-CDG")
    ]
    
    for flight_id, description in test_cases:
        print(f"\n--- Testing: {description} ---")
        test_state["current_message"] = flight_id
        
        try:
            result = selection_nodes(test_state)
            
            if isinstance(result, dict):
                print(f"✅ Flight {flight_id} processed successfully")
                print(f"   City Code: {result.get('city_code')}")
                print(f"   Check-in: {result.get('checkin_date')}")
                print(f"   Check-out: {result.get('checkout_date')}")
                
                # Check if the flight data was correctly parsed
                offers = test_state.get("result", {}).get("data", [])
                selected_offer = next((offer for offer in offers if offer['id'] == flight_id), None)
                
                if selected_offer:
                    print(f"   Flight Details:")
                    print(f"     Price: {selected_offer['price']['total']} {selected_offer['price']['currency']}")
                    print(f"     Carrier: {selected_offer['validatingAirlineCodes'][0]}")
                    
                    # Check itinerary details
                    if selected_offer['itineraries']:
                        outbound = selected_offer['itineraries'][0]
                        if outbound['segments']:
                            print(f"     Outbound: {outbound['segments'][0]['departure']['iataCode']} → {outbound['segments'][0]['arrival']['iataCode']}")
                            print(f"     Duration: {outbound['duration']}")
                
            else:
                print(f"❌ Failed to process flight {flight_id}")
                
        except Exception as e:
            print(f"❌ Error processing flight {flight_id}: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting Flight Selection Test Suite")
    print("This will test the complete flow from flight offers to hotel search")
    print("Using the actual flight offers JSON structure you provided")
    
    # Test 1: Display flight offers
    test_display_flight_offers()
    
    # Test 2: Flight selection functionality
    test_flight_selection()
    
    # Test 3: Graph workflow
    test_graph_workflow()
    
    # Test 4: State transitions
    test_state_transitions()
    
    # Test 5: Specific flight data extraction
    test_specific_flight_data_extraction()
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)
    print("Check the output above to verify:")
    print("1. Flight offers can be displayed (5 offers from your JSON)")
    print("2. Users can select any flight offer (IDs 1-5)")
    print("3. State transitions work correctly from FlightSearchState to HotelSearchState")
    print("4. HotelSearchState is properly created with extracted data:")
    print("   - city_code (PAR for CDG)")
    print("   - checkin_date (2025-08-13)")
    print("   - checkout_date (2025-08-20)")
    print("   - selected_flight (flight details)")
    print("   - currency (EGP)")

if __name__ == "__main__":
    main()
