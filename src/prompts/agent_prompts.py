"""
Prompt templates for agent interactions and tool usage.
"""
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any

AGENT_SYSTEM_PROMPT = PromptTemplate.from_template(
    """You are a research assistant specialized in deep research and analysis. Return ONLY valid JSON objects.
    
    Your capabilities include:
    1. Generating and optimizing research plans
    2. Evaluating source credibility
    3. Fact-checking information
    4. Synthesizing findings
    5. Calibrating confidence levels
    
    Always maintain:
    - High standards of accuracy
    - Critical thinking
    - Source verification
    - Clear communication
    - Ethical research practices
    
    Available Tools:
    {tools}
    
    Tool Names:
    {tool_names}
    
    Return ONLY a JSON object with this exact structure:
    {{
        "action": "action_name",
        "parameters": {{
            "param1": "value1",
            "param2": "value2"
        }},
        "reasoning": "explanation of the decision",
        "confidence": 0.8
    }}
    """
)

AGENT_TOOL_PROMPT = PromptTemplate.from_template(
    """Select and use the most appropriate tool for this task. Return ONLY a valid JSON object.

    Task: {task}
    Context: {context}
    Available Tools: {tools}
    
    Return ONLY a JSON object with this exact structure:
    {{
        "selected_tool": "tool_name",
        "parameters": {{
            "param1": "value1",
            "param2": "value2"
        }},
        "rationale": "explanation of tool selection",
        "error_handling": {{
            "fallback_tool": "alternative_tool",
            "retry_strategy": "strategy description"
        }}
    }}"""
)

AGENT_PLANNING_PROMPT = PromptTemplate.from_template(
    """Plan the next steps for this research task. Return ONLY a valid JSON object.

    Current Task: {task}
    Progress: {progress}
    Available Tools: {tools}
    
    Return ONLY a JSON object with this exact structure:
    {{
        "steps": [
            {{
                "description": "step description",
                "tool": "tool_name",
                "priority": "high/medium/low",
                "dependencies": ["step1", "step2"]
            }}
        ],
        "critical_path": ["step1", "step3"],
        "contingencies": [
            {{
                "trigger": "condition description",
                "action": "action to take"
            }}
        ],
        "estimated_completion": "duration in minutes"
    }}"""
)

AGENT_EVALUATION_PROMPT = PromptTemplate.from_template(
    """Evaluate the results of this research step. Return ONLY a valid JSON object.

    Step: {step}
    Results: {results}
    Expected Outcomes: {expected}
    
    Return ONLY a JSON object with this exact structure:
    {{
        "completeness": {{
            "score": 0.8,
            "missing_elements": ["element1", "element2"]
        }},
        "quality": {{
            "score": 0.9,
            "strengths": ["strength1", "strength2"],
            "weaknesses": ["weakness1", "weakness2"]
        }},
        "gaps": [
            {{
                "description": "gap description",
                "severity": "high/medium/low",
                "mitigation": "how to address"
            }}
        ],
        "improvements": [
            {{
                "suggestion": "improvement description",
                "priority": "high/medium/low",
                "effort": "estimated effort"
            }}
        ]
    }}"""
)

AGENT_SYNTHESIS_PROMPT = PromptTemplate.from_template(
    """Synthesize the findings from multiple research steps. Return ONLY a valid JSON object.

    Steps: {steps}
    Findings: {findings}
    Context: {context}
    
    Return ONLY a JSON object with this exact structure:
    {{
        "patterns": [
            {{
                "description": "pattern description",
                "frequency": "occurrence frequency",
                "significance": "high/medium/low"
            }}
        ],
        "connections": [
            {{
                "source": "finding1",
                "target": "finding2",
                "relationship": "relationship description",
                "strength": 0.8
            }}
        ],
        "insights": [
            {{
                "description": "insight description",
                "confidence": 0.9,
                "supporting_evidence": ["evidence1", "evidence2"]
            }}
        ],
        "contradictions": [
            {{
                "finding1": "first finding",
                "finding2": "contradicting finding",
                "severity": "high/medium/low",
                "resolution": "how to resolve"
            }}
        ]
    }}"""
) 