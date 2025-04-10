"""
Prompt templates for research tasks.
"""
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any

DEPTH_ANALYSIS_PROMPT = PromptTemplate.from_template(
    """You are a depth analysis expert. Return ONLY a valid JSON object.
    
    QUERY: {query}
    CURRENT_RESULTS: {current_results}
    DEPTH: {depth}
    
    TASK:
    Perform a depth analysis of the research results.
    
    Return ONLY a JSON object with this exact structure:
    {{
        "analysis": {{
            "key_findings": [
                {{
                    "finding": "finding description",
                    "confidence": 0.8,
                    "sources": ["source1", "source2"]
                }}
            ],
            "gaps": ["gap1", "gap2"],
            "biases": ["bias1", "bias2"],
            "overall_quality": 0.8
        }},
        "recommendations": [
            {{
                "action": "action description",
                "priority": "high/medium/low",
                "reason": "reason description"
            }}
        ],
        "confidence_metrics": {{
            "overall_confidence": 0.8,
            "source_reliability": 0.9,
            "data_completeness": 0.7
        }}
    }}"""
)

RESEARCH_PLAN_PROMPT = PromptTemplate.from_template(
    """You are a research planning expert. Return ONLY a valid JSON object.
    
    QUERY: {query}
    CONTEXT: {context}
    
    TASK:
    Create a detailed research plan.
    
    REQUIREMENTS:
    1. Define research objectives
    2. Identify key areas to investigate
    3. Plan search strategies
    4. Set evaluation criteria
    5. Establish timeline and milestones
    
    Return ONLY a JSON object with this exact structure:
    {{
        "objectives": ["objective1", "objective2"],
        "key_areas": [
            {{
                "area": "area name",
                "importance": 0.8,
                "investigation_method": "method description"
            }}
        ],
        "search_strategies": [
            {{
                "strategy": "strategy description",
                "resources": ["resource1", "resource2"],
                "expected_outcomes": "outcome description"
            }}
        ],
        "evaluation_criteria": ["criterion1", "criterion2"],
        "timeline": {{
            "total_duration": "duration in minutes",
            "milestones": [
                {{
                    "name": "milestone name",
                    "duration": "duration in minutes",
                    "deliverables": ["deliverable1", "deliverable2"]
                }}
            ]
        }}
    }}"""
)

SEARCH_STRATEGY_PROMPT = PromptTemplate.from_template(
    """You are a search strategy expert. Return ONLY a valid JSON object.
    
    QUERY: {query}
    CONTEXT: {context}
    
    TASK:
    Develop an effective search strategy.
    
    Return ONLY a JSON object with this exact structure:
    {{
        "search_terms": [
            {{
                "term": "term text",
                "importance": 0.8,
                "synonyms": ["synonym1", "synonym2"]
            }}
        ],
        "sources": ["source1", "source2"],
        "parameters": {{
            "time_range": "time range",
            "language": "language",
            "region": "region"
        }},
        "query_variations": ["query1", "query2"],
        "constraints": {{
            "max_results": 10,
            "min_relevance": 0.7
        }}
    }}"""
)

RESULT_EVALUATION_PROMPT = PromptTemplate.from_template(
    """You are a research evaluation expert providing a concise qualitative analysis.

    QUERY: {query}
    RESULTS (Snippets):
    ```json
    {results} # Contains titles, URLs, content snippets, and scores
    ```

    TASK:
    Write a brief narrative analysis (1-2 paragraphs) evaluating the overall set of search results provided in relation to the user's QUERY. Focus on:
    1.  **General Relevance & Coverage:** Comment on how well the results, as a whole, seem to address the query. Are the main aspects covered?
    2.  **Source Landscape:** Describe the types and apparent quality/credibility of the sources found (e.g., mix of official sites, news, blogs, forums?). Avoid giving numerical scores.
    3.  **Key Themes & Gaps:** Briefly mention the main themes or points emerging from the results. Point out any obvious information gaps based *only* on the provided snippets.
    4.  **Potential Biases:** Note any potential general biases suggested by the collection of sources or their content snippets.

    Keep the language clear, objective, and focus on the overall picture presented by the results. Do NOT list results individually. Do NOT use numerical scores. Do NOT output JSON.

    NARRATIVE ANALYSIS:
    """
)

SYNTHESIS_REPORT_PROMPT = PromptTemplate.from_template(
    """You are a research synthesis expert. Return ONLY a valid JSON object.
    
    QUERY: {query}
    FINDINGS: {findings}
    
    TASK:
    Synthesize research findings into a comprehensive report.
    
    Return ONLY a JSON object with this exact structure:
    {{
        "key_findings": [
            {{
                "finding": "finding description",
                "confidence": 0.8,
                "sources": ["source1", "source2"]
            }}
        ],
        "insights": ["insight1", "insight2"],
        "research_questions": {{
            "question1": {{
                "question": "question text",
                "answer": "answer text",
                "confidence": 0.8
            }}
        }},
        "limitations": ["limitation1", "limitation2"],
        "gaps": ["gap1", "gap2"],
        "recommendations": ["recommendation1", "recommendation2"]
    }}"""
)

# Prompt for processing a single result
RESULT_PROCESSING_PROMPT = """
You are a result processing expert. Return ONLY a valid JSON object.

Result:
{result}

Context:
{context}

Return ONLY a JSON object with this exact structure:
{
    "key_points": ["point1", "point2"],
    "supporting_evidence": ["evidence1", "evidence2"],
    "contradicting_evidence": ["contradiction1", "contradiction2"],
    "confidence_score": 0.8,
    "source_quality": "quality description",
    "relevance_score": 0.9
}"""

# Prompt for extracting information from multiple results
INFORMATION_EXTRACTION_PROMPT = """
You are an information extraction expert. Return ONLY a valid JSON object.

Results:
{results}

Context:
{context}

Return ONLY a JSON object with this exact structure:
{
    "findings": [
        {
            "topic": "topic name",
            "description": "description text",
            "supporting_evidence": ["evidence1", "evidence2"],
            "contradicting_evidence": ["contradiction1", "contradiction2"],
            "confidence_score": 0.8,
            "source_quality": "quality description"
        }
    ],
    "topics": ["topic1", "topic2"],
    "relationships": [
        {
            "topic1": "topic name",
            "topic2": "topic name",
            "relationship": "relationship description"
        }
    ],
    "gaps": ["gap1", "gap2"],
    "confidence_scores": {
        "topic1": 0.8,
        "topic2": 0.7
    },
    "recommendations": ["recommendation1", "recommendation2"]
}"""

# Prompt for analyzing relationships between findings
RELATIONSHIP_ANALYSIS_PROMPT = """
You are a relationship analysis expert. Return ONLY a valid JSON object.

Findings:
{findings}

Sources:
{sources}

Return ONLY a JSON object with this exact structure:
{
    "topics": {
        "topic1": {
            "findings": ["finding1", "finding2"],
            "supporting_sources": ["source1", "source2"],
            "contradicting_sources": ["source3", "source4"],
            "consensus_level": 0.8
        }
    },
    "overall_consensus": 0.7,
    "contradictions": [
        {
            "topic": "topic name",
            "finding1": "finding description",
            "finding2": "contradicting finding",
            "severity": 0.8
        }
    ],
    "supporting_evidence": [
        {
            "topic": "topic name",
            "evidence": "evidence description",
            "strength": 0.9
        }
    ],
    "confidence_levels": {
        "topic1": 0.8,
        "topic2": 0.7
    }
}"""

# Prompt for synthesizing findings
SYNTHESIS_PROMPT = """
Synthesize the following research findings into a coherent summary:

Findings:
{findings}

Relationships:
{relationships}

Confidence Metrics:
{confidence_metrics}

Return a JSON object with the following structure:
{
    "summary": str,  # Overall summary
    "key_points": List[str],  # Key points
    "confidence_level": float,  # Overall confidence level
    "limitations": List[str],  # Limitations of the research
    "recommendations": List[str]  # Recommendations for further research
}
""" 