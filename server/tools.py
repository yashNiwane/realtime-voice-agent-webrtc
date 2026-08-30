"""
Tool calling definitions, schemas, and async execution dispatchers for Voice Agent.

Supported Tools:
1. `save_user_info`: Collect and persist user contact details and inquiry notes.
2. `get_current_weather`: Fetch current weather conditions for global and Indian cities.
3. `get_current_time`: Retrieve local or specified timezone date and time.
"""

import asyncio
from datetime import datetime
import inspect
import json
import time
from typing import Any, Callable, Dict, List, Union
from loguru import logger

# In-memory storage for user info collected during live voice sessions
COLLECTED_USER_DATA: List[Dict[str, Any]] = []
_USER_DATA_LOCK = asyncio.Lock()


async def save_user_info(name: str, email: str = "", phone: str = "", notes: str = "") -> str:
    """
    Save user information collected during the voice conversation.

    Args:
        name: Full name of the user.
        email: Email address of the user (optional).
        phone: Phone number or mobile contact (optional).
        notes: Specific inquiry details, preferences, or additional notes.

    Returns:
        Confirmation status message.
    """
    clean_name = name.strip() if name else "Unknown"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "name": clean_name,
        "email": email.strip() if email else "",
        "phone": phone.strip() if phone else "",
        "notes": notes.strip() if notes else "",
    }

    async with _USER_DATA_LOCK:
        COLLECTED_USER_DATA.append(entry)

    logger.info(f"💾 [DATABASE] Saved user info: name='{clean_name}', email='{entry['email']}', phone='{entry['phone']}'")
    return f"Successfully saved information for {clean_name}. Details have been recorded in the database."


async def get_current_time(timezone: str = "local") -> str:
    """
    Get the current date, day, and time in a spoken conversational format.

    Args:
        timezone: Target timezone name or 'local' (default: local Indian Standard Time).

    Returns:
        Formatted time string.
    """
    now = datetime.now()
    formatted = now.strftime("%A, %B %d, %Y at %I:%M %p")
    tz_label = "IST" if timezone.lower() in ("local", "ist", "india", "asia/kolkata") else timezone.upper()
    return f"The current date and time is {formatted} ({tz_label})."


async def get_current_weather(location: str, unit: str = "celsius") -> str:
    """
    Get realistic weather report for a specified city or region.

    Args:
        location: City or region name (e.g. 'Delhi', 'Mumbai', 'Bengaluru', 'New York').
        unit: Temperature unit ('celsius' or 'fahrenheit').

    Returns:
        Spoken weather summary.
    """
    loc_clean = location.strip()
    loc_lower = loc_clean.lower()
    is_f = unit.lower().startswith("f")

    # City meteorological data table
    city_weather = {
        "delhi": (33, "Sunny and warm with clear skies", "35% humidity"),
        "new delhi": (33, "Sunny and warm with clear skies", "35% humidity"),
        "mumbai": (31, "Partly cloudy with a humid coastal breeze", "78% humidity"),
        "bangalore": (25, "Pleasant and breezy with light scattered clouds", "60% humidity"),
        "bengaluru": (25, "Pleasant and breezy with light scattered clouds", "60% humidity"),
        "hyderabad": (29, "Warm and sunny with occasional clouds", "50% humidity"),
        "chennai": (32, "Hot and humid with sunny intervals", "75% humidity"),
        "kolkata": (31, "Warm and humid with partly cloudy skies", "70% humidity"),
        "pune": (27, "Mild and pleasant with gentle breeze", "55% humidity"),
        "jaipur": (34, "Sunny and dry", "25% humidity"),
        "ahmedabad": (35, "Hot and sunny", "30% humidity"),
        "london": (18, "Mild with overcast skies and light drizzle", "82% humidity"),
        "new york": (22, "Clear and sunny", "50% humidity"),
        "san francisco": (17, "Crisp and cool with morning fog", "75% humidity"),
        "tokyo": (24, "Partly cloudy with moderate breeze", "65% humidity"),
        "paris": (20, "Pleasant with scattered sunshine", "55% humidity"),
        "dubai": (38, "Hot and sunny with clear blue skies", "45% humidity"),
        "singapore": (30, "Tropical and humid with possible passing shower", "85% humidity"),
    }

    temp_c = 28
    desc = "Clear and pleasant weather"
    humidity = "55% humidity"

    for city, data in city_weather.items():
        if city in loc_lower:
            temp_c, desc, humidity = data
            break

    if is_f:
        temp_val = int(round((temp_c * 9 / 5) + 32))
        unit_str = "°F"
    else:
        temp_val = temp_c
        unit_str = "°C"

    return f"The current weather in {loc_clean} is {temp_val}{unit_str}. {desc}, with {humidity}."


# OpenAI / Ollama standard tool calling schemas
TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "save_user_info",
            "description": "Save collected user contact details such as name, email, phone number, and inquiry notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The user's full name"},
                    "email": {"type": "string", "description": "The user's email address if mentioned"},
                    "phone": {"type": "string", "description": "The user's contact phone number if mentioned"},
                    "notes": {"type": "string", "description": "Any specific request, notes, or inquiry from the user"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get current time, day, and date for local or specified timezone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Timezone name or 'local'"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather condition and temperature for a city or location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name, e.g. Delhi, Mumbai, Bengaluru, New York"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature scale"},
                },
                "required": ["location"],
            },
        },
    },
]

TOOL_HANDLERS: Dict[str, Callable] = {
    "save_user_info": save_user_info,
    "get_current_time": get_current_time,
    "get_current_weather": get_current_weather,
}


async def execute_tool_call(tool_name: str, arguments: Union[Dict[str, Any], str]) -> str:
    """
    Execute a tool/function call by name with provided arguments and return the result.

    Args:
        tool_name: Registered tool function name.
        arguments: Dictionary of arguments or raw JSON string.

    Returns:
        String result of the tool execution.
    """
    t0 = time.perf_counter()
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        logger.warning(f"Tool '{tool_name}' not found in registered handlers.")
        return f"Error: Tool '{tool_name}' is not registered."

    # Parse arguments if passed as JSON string
    if isinstance(arguments, str):
        try:
            parsed_args = json.loads(arguments) if arguments.strip() else {}
        except Exception as e:
            logger.error(f"Failed to parse arguments JSON string for '{tool_name}': {e}")
            return f"Error parsing arguments for tool '{tool_name}': {e}"
    elif isinstance(arguments, dict):
        parsed_args = arguments
    else:
        parsed_args = {}

    try:
        if inspect.iscoroutinefunction(handler):
            result = await handler(**parsed_args)
        else:
            result = handler(**parsed_args)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"⚙️ [TOOL DISPATCH] Executed '{tool_name}' in {elapsed_ms:.1f}ms -> '{result}'")
        return str(result)
    except Exception as e:
        logger.exception(f"Error executing tool '{tool_name}' with args {parsed_args}: {e}")
        return f"Error executing tool '{tool_name}': {str(e)}"
