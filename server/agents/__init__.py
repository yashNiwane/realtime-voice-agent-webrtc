"""
Agent Package: Exports TelecallerAgent, LeadAnalystAgent, SalesCoachAgent, and prompt templates.
"""

from server.agents.analyst import (
    LeadAnalystAgent,
    SalesCoachAgent,
    analyze_lead_async,
    get_sales_coaching_async,
)
from server.agents.prompts import (
    ANALYST_SYSTEM_PROMPT,
    COACH_SYSTEM_PROMPT,
    TELECALLER_SYSTEM_PROMPT,
)
from server.agents.telecaller import TelecallerAgent, VoiceTelecallerSession

__all__ = [
    "TelecallerAgent",
    "VoiceTelecallerSession",
    "LeadAnalystAgent",
    "SalesCoachAgent",
    "analyze_lead_async",
    "get_sales_coaching_async",
    "TELECALLER_SYSTEM_PROMPT",
    "ANALYST_SYSTEM_PROMPT",
    "COACH_SYSTEM_PROMPT",
]
