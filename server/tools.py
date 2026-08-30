"""
Tool calling definitions, schemas, and async execution dispatchers for Voice Agent.

Supported Credit Card Sales Tools:
1. `check_card_eligibility`: Instant eligibility check & credit limit calculation.
2. `apply_credit_card`: Instant credit card application booking with Reference ID.
3. `get_card_benefits`: Fetch key benefits, cashback rates, lounge access, and rewards.
4. `save_user_info`: Save customer contact details and notes.
5. `get_current_weather`: Fetch weather report.
6. `get_current_time`: Retrieve local time.
"""

import asyncio
from datetime import datetime
import inspect
import json
import random
import time
from typing import Any, Callable, Dict, List, Union
from loguru import logger

# In-memory storage for customer leads and applications
COLLECTED_USER_DATA: List[Dict[str, Any]] = []
CREDIT_CARD_APPLICATIONS: List[Dict[str, Any]] = []
_LOCK = asyncio.Lock()


async def check_card_eligibility(
    monthly_income: int,
    employment_type: str = "salaried",
    city: str = "Metro",
) -> str:
    """
    Check customer credit card eligibility and estimated pre-approved limit.
    """
    min_sal = 25000
    if monthly_income >= min_sal:
        est_limit = min(monthly_income * 3, 500000)
        return (
            f"Congratulations! You are pre-approved for the Lifetime-Free Apex Platinum Card with "
            f"an estimated credit limit of ₹{est_limit:,}. Zero joining fee, 5% unlimited cashback, "
            f"and 8 complimentary airport lounge visits."
        )
    else:
        return (
            f"You are eligible for the Apex Secured Platinum Credit Card against a fixed deposit, "
            f"which also offers 100% approval and 4% cashback on all spends!"
        )


async def apply_credit_card(
    full_name: str,
    phone_number: str,
    card_variant: str = "Apex Platinum Lifetime Free",
    monthly_income: int = 50000,
) -> str:
    """
    Book a new credit card application for the customer and generate reference number.
    """
    ref_id = f"APEX-CC-{random.randint(10000, 99999)}"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "application_id": ref_id,
        "name": full_name.strip(),
        "phone": phone_number.strip(),
        "card_variant": card_variant,
        "monthly_income": monthly_income,
        "status": "PRE_APPROVED_SUCCESS",
    }
    async with _LOCK:
        CREDIT_CARD_APPLICATIONS.append(entry)
        COLLECTED_USER_DATA.append(entry)

    logger.info(f"💳 [CARD APPLICATION SUCCESS] Ref: {ref_id}, Customer: {full_name}, Phone: {phone_number}")
    return (
        f"Application successful! Your Reference ID is {ref_id}. "
        f"Your Lifetime Free {card_variant} card is pre-approved. Our verification team will activate your digital card within 24 hours."
    )


async def get_card_benefits(card_variant: str = "Apex Platinum") -> str:
    """
    Get the top USP benefits, rewards, and lounge access details of Apex credit cards.
    """
    return (
        "Apex Platinum Credit Card Top Benefits: "
        "1. 100% Lifetime Free (Zero joining and Zero annual fees forever). "
        "2. 5% Unlimited Cashback on dining, Swiggy, Zomato, Amazon, and Flipkart. "
        "3. 8 Complimentary Domestic & International Airport Lounge visits per year. "
        "4. 10,000 Welcome Bonus Reward Points on card activation. "
        "5. 1% Fuel Surcharge Waiver across all Indian petrol pumps."
    )


async def save_user_info(name: str, email: str = "", phone: str = "", notes: str = "") -> str:
    """Save customer contact details."""
    clean_name = name.strip() if name else "Customer"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "name": clean_name,
        "email": email.strip(),
        "phone": phone.strip(),
        "notes": notes.strip(),
    }
    async with _LOCK:
        COLLECTED_USER_DATA.append(entry)
    logger.info(f"💾 [SAVED LEAD] {clean_name} - Phone: {phone}")
    return f"Details recorded successfully for {clean_name}."


async def get_current_time(timezone: str = "local") -> str:
    now = datetime.now()
    return f"The current date and time is {now.strftime('%A, %B %d, %Y at %I:%M %p')} (IST)."


async def get_current_weather(location: str, unit: str = "celsius") -> str:
    return f"The weather in {location} is currently 28°C with pleasant skies."


# OpenAI / Llama.cpp standard tool calling schemas
TOOLS_SCHEMA: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "check_card_eligibility",
            "description": "Check if customer is eligible for credit card and calculate their pre-approved credit limit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "monthly_income": {"type": "integer", "description": "Customer's monthly net salary or income in INR"},
                    "employment_type": {"type": "string", "enum": ["salaried", "self-employed", "business"], "description": "Employment type"},
                    "city": {"type": "string", "description": "Customer's resident city"},
                },
                "required": ["monthly_income"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_credit_card",
            "description": "Book a new credit card application for the customer with their name and phone number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "full_name": {"type": "string", "description": "Customer's full legal name"},
                    "phone_number": {"type": "string", "description": "Customer's 10-digit mobile number"},
                    "card_variant": {"type": "string", "description": "Card variant (e.g. Apex Platinum Lifetime Free)"},
                    "monthly_income": {"type": "integer", "description": "Monthly income in INR"},
                },
                "required": ["full_name", "phone_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_card_benefits",
            "description": "Get detailed rewards, cashback percentage, airport lounge access, and fee details of the credit card.",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_variant": {"type": "string", "description": "Credit card variant"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_user_info",
            "description": "Save customer inquiry or contact details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Customer name"},
                    "email": {"type": "string", "description": "Email address"},
                    "phone": {"type": "string", "description": "Phone number"},
                    "notes": {"type": "string", "description": "Notes or preferences"},
                },
                "required": ["name"],
            },
        },
    },
]

TOOL_HANDLERS: Dict[str, Callable] = {
    "check_card_eligibility": check_card_eligibility,
    "apply_credit_card": apply_credit_card,
    "get_card_benefits": get_card_benefits,
    "save_user_info": save_user_info,
    "get_current_time": get_current_time,
    "get_current_weather": get_current_weather,
}


async def execute_tool_call(tool_name: str, arguments: Union[Dict[str, Any], str]) -> str:
    t0 = time.perf_counter()
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        logger.warning(f"Tool '{tool_name}' not found.")
        return f"Error: Tool '{tool_name}' is not registered."

    if isinstance(arguments, str):
        try:
            parsed_args = json.loads(arguments) if arguments.strip() else {}
        except Exception as e:
            return f"Error parsing args: {e}"
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
        logger.info(f"⚙️ [SALES TOOL] {tool_name}({parsed_args}) in {elapsed_ms:.1f}ms -> '{result}'")
        return str(result)
    except Exception as e:
        logger.exception(f"Error executing tool '{tool_name}': {e}")
        return f"Error executing tool '{tool_name}': {str(e)}"
