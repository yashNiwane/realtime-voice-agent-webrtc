"""
Tool definitions and handlers for information collection and action execution with Gemma 4.
"""

import asyncio
from datetime import datetime
import inspect
from typing import Any, Dict
from loguru import logger

# In-memory storage for collected user information
COLLECTED_USER_DATA: list[Dict[str, Any]] = []


async def save_user_info(name: str, email: str = "", phone: str = "", notes: str = "") -> str:
    """Save user information collected during the conversation."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "name": name,
        "email": email,
        "phone": phone,
        "notes": notes,
    }
    COLLECTED_USER_DATA.append(entry)
    logger.info(f"💾 [INFO COLLECTED] Saved user info: {entry}")
    return f"Successfully saved information for {name}."


async def get_current_time(timezone: str = "local") -> str:
    """Get the current time and date."""
    now = datetime.now()
    return f"Current date and time is {now.strftime('%A, %B %d, %Y at %I:%M %p')}."


async def get_current_weather(location: str, unit: str = "celsius") -> str:
    """Get weather conditions for a city."""
    location_lower = location.lower()
    if "delhi" in location_lower:
        temp = "32°C, Sunny and warm"
    elif "mumbai" in location_lower:
        temp = "30°C, Humid with light sea breeze"
    elif "bangalore" in location_lower or "bengaluru" in location_lower:
        temp = "24°C, Pleasant and cloudy"
    else:
        temp = "28°C, Clear skies"
    return f"The current weather in {location} is {temp}."


# OpenAI / Ollama compatible function schemas
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "save_user_info",
            "description": "Save collected user details such as name, email, phone number, or notes/inquiry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The user's full name"},
                    "email": {"type": "string", "description": "The user's email address if provided"},
                    "phone": {"type": "string", "description": "The user's phone number if provided"},
                    "notes": {"type": "string", "description": "Additional notes, request details, or inquiry"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get current time and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "Timezone or 'local'"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather condition for a city or region.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name, e.g. Delhi, Mumbai, New York"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
]

TOOL_HANDLERS = {
    "save_user_info": save_user_info,
    "get_current_time": get_current_time,
    "get_current_weather": get_current_weather,
}


async def execute_tool_call(tool_name: str, arguments: dict) -> str:
    """Execute a function call by name with provided arguments."""
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return f"Error: Tool '{tool_name}' not found."
    try:
        if inspect.iscoroutinefunction(handler):
            return await handler(**arguments)
        return handler(**arguments)
    except Exception as e:
        return f"Error executing '{tool_name}': {str(e)}"
