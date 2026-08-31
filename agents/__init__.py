"""
Top-level Agent Package: Exports TelecallerAgent, LeadAnalystAgent, SalesCoachAgent.
"""

from server.agents import (
    ANALYST_SYSTEM_PROMPT,
    COACH_SYSTEM_PROMPT,
    TELECALLER_SYSTEM_PROMPT,
    LeadAnalystAgent,
    SalesCoachAgent,
    TelecallerAgent,
    VoiceTelecallerSession,
    analyze_lead_async,
    get_sales_coaching_async,
)

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
