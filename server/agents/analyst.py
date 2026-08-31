"""
Background Sidecar Agents: Lead Analyst & Sales Coach.
"""

import json
from typing import Any, Dict, List, Optional
from loguru import logger

from server.agents.prompts import ANALYST_SYSTEM_PROMPT, COACH_SYSTEM_PROMPT


async def analyze_lead_async(
    transcript: List[Dict[str, str]],
    lead_data: Dict[str, Any],
    llm_engine: Any = None,
) -> Dict[str, Any]:
    """
    Direct asynchronous analysis helper used by WebRTC voice pipeline and tools.
    """
    payload = {
        "transcript": transcript[-20:] if transcript else [],
        "captured": lead_data or {},
    }
    user_prompt = f"Analyze this live sales call transcript and return ONLY structured JSON:\n{json.dumps(payload, ensure_ascii=False)}"

    if llm_engine is not None and hasattr(llm_engine, "generate_completion"):
        try:
            raw_res = await llm_engine.generate_completion(
                prompt=user_prompt,
                system_prompt=ANALYST_SYSTEM_PROMPT,
                max_tokens=350,
            )
            # Clean JSON fences if any
            clean = raw_res.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(clean)
        except Exception as e:
            logger.warning(f"Analyst LLM JSON parse warning: {e}")

    # Fallback heuristic analysis if LLM is offline or parsing fails
    user_msgs = [t["content"] for t in transcript if t.get("role") == "user"]
    user_text = " ".join(user_msgs).lower()
    interest = "high" if any(w in user_text for w in ["yes", "haan", "apply", "chahiye", "interested", "card", "benefits"]) else "medium"
    
    return {
        "interest_level": interest,
        "sentiment": "positive" if "haan" in user_text or "yes" in user_text else "neutral",
        "captured": lead_data,
        "missing_fields": [k for k in ["name", "city", "monthly_income", "phone"] if k not in lead_data],
        "objections": [],
        "recommended_next_action": "Pitch 5% unlimited cashback and 8 airport lounge visits, then ask for monthly salary.",
        "summary_hindi": f"Customer credit card offer mein ruchi dikha rahe hain. Captured details: {len(lead_data)} fields.",
    }


async def get_sales_coaching_async(
    situation: str,
    recent_transcript: List[Dict[str, str]],
    lead_data: Dict[str, Any],
    llm_engine: Any = None,
) -> str:
    """
    Direct asynchronous sales coaching helper used by WebRTC voice pipeline and tools.
    """
    payload = {
        "situation": situation,
        "recent_transcript": recent_transcript[-8:] if recent_transcript else [],
        "lead": lead_data or {},
    }
    user_prompt = f"Give immediate tactical whisper advice for this situation:\n{json.dumps(payload, ensure_ascii=False)}"

    if llm_engine is not None and hasattr(llm_engine, "generate_completion"):
        try:
            advice = await llm_engine.generate_completion(
                prompt=user_prompt,
                system_prompt=COACH_SYSTEM_PROMPT,
                max_tokens=150,
            )
            return advice.strip()
        except Exception as e:
            logger.warning(f"Coach LLM execution warning: {e}")

    # Fallback tactical advice
    sit_lower = situation.lower()
    if "fee" in sit_lower or "charge" in sit_lower or "cost" in sit_lower:
        return "1) Customer fears hidden charges. 2) Say: 'Sir, yeh card 100% Lifetime Free hai, zero joining aur zero annual maintenance charges!' 3) Avoid mentioning interest rates unprompted."
    elif "busy" in sit_lower or "time" in sit_lower:
        return "1) Customer is occupied. 2) Say: 'Sir, main bas 30 seconds mein pre-approved limit bata deti hoon, ya kab call back karoon?' 3) Don't push full sales pitch."
    else:
        return "1) Customer is hesitating. 2) Say: 'Sir, kyunki yeh 100% free hai, aapko free airport lounge access aur 5% cashback milta hai bina kisi annual fee ke!' 3) Avoid speaking too fast."


# Pipecat sidecar worker definitions (if pipecat is available)
try:
    from pipecat.bus.messages import BusJobRequestMessage
    from pipecat.frames.frames import LLMMessagesAppendFrame
    from pipecat.workers.llm import LLMContextWorker

    class JobResponderAgent(LLMContextWorker):
        """Base for passive sidecar agents that answer job requests with their own LLM."""

        def __init__(self, name: str, *, llm, response_key: str):
            super().__init__(name, llm=llm)
            self._response_key = response_key
            self._job_id: Optional[str] = None

            @self.assistant_aggregator.event_handler("on_assistant_turn_stopped")
            async def _on_stopped(aggregator, message=None):
                if self._job_id is not None:
                    job_id, self._job_id = self._job_id, None
                    content = getattr(message, "content", "") or ""
                    await self.send_job_response(job_id, {self._response_key: content})

        async def on_job_request(self, message: BusJobRequestMessage) -> None:
            await super().on_job_request(message)
            logger.info(f"{self} received background job {message.job_id}")
            self._job_id = message.job_id
            prompt = self.build_prompt(message.payload or {})
            await self.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": prompt}],
                    run_llm=True,
                )
            )

        def build_prompt(self, payload: dict) -> str:
            return json.dumps(payload, ensure_ascii=False)

    class LeadAnalystAgent(JobResponderAgent):
        """Background analyst: turns the raw transcript into structured lead assessment."""

        def __init__(self, name: str, *, llm):
            super().__init__(name, llm=llm, response_key="assessment")

    class SalesCoachAgent(JobResponderAgent):
        """Background coach: returns tactical objection-handling whisper advice."""

        def __init__(self, name: str, *, llm):
            super().__init__(name, llm=llm, response_key="advice")

except ImportError:
    class LeadAnalystAgent:
        pass

    class SalesCoachAgent:
        pass
