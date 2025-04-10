"""
Prompt templates for the Deep Research system.
This module contains all the prompt templates used by the system.
"""

from .scope_prompts import (
    SCOPE_DEFINITION_PROMPT,
    SUBQUERY_GENERATION_PROMPT
)

from .research_prompts import (
    RESEARCH_PLAN_PROMPT,
    DEPTH_ANALYSIS_PROMPT
)

from .evaluation_prompts import (
    SOURCE_EVALUATION_PROMPT,
    CREDIBILITY_ASSESSMENT_PROMPT
)

from .synthesis_prompts import (
    INFORMATION_SYNTHESIS_PROMPT,
    CROSS_VALIDATION_PROMPT
)

from .processing_prompts import (
    SUBQUERY_GENERATION_PROMPT
)

__all__ = [
    'SCOPE_DEFINITION_PROMPT',
    'SUBQUERY_GENERATION_PROMPT',
    'RESEARCH_PLAN_PROMPT',
    'DEPTH_ANALYSIS_PROMPT',
    'SOURCE_EVALUATION_PROMPT',
    'CREDIBILITY_ASSESSMENT_PROMPT',
    'INFORMATION_SYNTHESIS_PROMPT',
    'CROSS_VALIDATION_PROMPT'
] 