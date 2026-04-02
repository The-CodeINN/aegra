"""Long-term memory schemas and constants for the agent.

Typed Pydantic schemas are shared between the hot-path tools
(create_manage_memory_tool / create_search_memory_tool) and the background
extraction node (create_memory_store_manager) so all three write to the same
structured store.
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_NAMESPACE = "memories"


class CareerGoal(BaseModel):
    """A career goal or target role the student is actively working toward."""

    role: str = Field(description="The target job title or career role.")
    industry: str | None = Field(
        default=None,
        description="Industry or domain (e.g. 'fintech', 'healthcare AI').",
    )
    timeline: str | None = Field(
        default=None,
        description="Student's stated timeline or urgency (e.g. '6 months', 'as soon as possible').",
    )
    priority: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Relative priority of this goal compared to other goals.",
    )


class StudentContext(BaseModel):
    """A persistent fact about the student's background, preferences, or constraints."""

    fact: str = Field(description="The specific fact to remember, written concisely.")
    category: Literal[
        "experience",
        "education",
        "skill",
        "preference",
        "constraint",
        "location",
        "motivation",
    ] = Field(description="The type of information this fact represents.")
    confidence: Literal["stated", "inferred"] = Field(
        default="stated",
        description="Whether the student stated this directly or the agent inferred it.",
    )


# Shared schema list consumed by hot-path tools and background extractor.
MEMORY_SCHEMAS: list[type[BaseModel]] = [CareerGoal, StudentContext]
