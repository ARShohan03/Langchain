"""Reasoning module for generating scholarship alignment analysis."""

from typing import Any
from llm_client import MistralClient

PROMPT_TEMPLATE = """You are an expert academic scholarship advisor. Analyze the alignment between the candidate's profile and the scholarship.

User profile:
{profile}

Scholarship description:
{scholarship}

TASKS — Complete ALL four sections:

1. ELIGIBILITY: State one of: "Eligible", "Partially Eligible", or "Not Eligible"
2. SCORE: Provide a number from 0 to 100 (integer only, no %) representing how well the candidate fits this SPECIFIC scholarship. Be HONEST:
   - 90-100: Near-perfect match (field, GPA, research, language all exceed requirements)
   - 70-89: Strong match with minor gaps
   - 50-69: Moderate match — some requirements met, some not
   - 30-49: Weak match — significant gaps exist
   - 0-29: Poor fit — most requirements not met
3. ANALYSIS: 2-3 sentences explaining WHY the candidate does or does not align with this program. Reference specific fields, GPA thresholds, and requirements.
4. GAPS: One specific, actionable tip for this application.

RESPOND EXACTLY in this format (do NOT deviate):
[ELIGIBILITY]
<one of: Eligible / Partially Eligible / Not Eligible>
[SCORE]
<number 0-100>
[ANALYSIS]
<2-3 sentences>
[GAPS]
<1 sentence of advice>

Be honest and factual. Do NOT inflate scores."""


def build_reasoning_chain(llm: MistralClient):
    """
    Wrap the lightweight LLM client in a chain-like interface.
    """
    class ReasoningChain:
        def __init__(self, llm):
            self.llm = llm
            
        def invoke(self, inputs: dict) -> str:
            prompt = PROMPT_TEMPLATE.format(
                profile=inputs["profile"],
                scholarship=inputs["scholarship"]
            )
            response_text = self.llm.invoke(prompt)
            # Return a simple object with a .content attribute to match legacy expectations
            class Response:
                def __init__(self, content):
                    self.content = content
            return Response(response_text)
            
    return ReasoningChain(llm)
