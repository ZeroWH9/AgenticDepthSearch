"""
Prompt templates for source evaluation and credibility assessment.
"""
from langchain_core.prompts import PromptTemplate

SOURCE_EVALUATION_PROMPT = PromptTemplate.from_template(
    """You are a source evaluation expert. Return ONLY a valid JSON object.
    
    SOURCE INFORMATION:
    Title: {title}
    URL: {url}
    Content: {content}
    Publication Date: {date}
    
    TASK:
    Evaluate this source's quality and reliability for research purposes.
    
    Return ONLY a JSON object with this exact structure:
    {{
        "authority": {{
            "author_credentials": {{
                "score": 0.8,
                "notes": "credentials description"
            }},
            "publisher_reputation": {{
                "score": 0.9,
                "notes": "reputation assessment"
            }},
            "institutional_affiliation": {{
                "score": 0.7,
                "notes": "affiliation details"
            }}
        }},
        "currency": {{
            "timeliness": 0.8,
            "relevance": 0.9,
            "lifecycle_stage": "stage description"
        }},
        "methodology": {{
            "research_methods": {{
                "score": 0.8,
                "description": "methods used"
            }},
            "data_quality": {{
                "score": 0.7,
                "issues": ["issue1", "issue2"]
            }},
            "analytical_rigor": {{
                "score": 0.9,
                "strengths": ["strength1", "strength2"]
            }}
        }},
        "objectivity": {{
            "bias_assessment": {{
                "score": 0.8,
                "identified_biases": ["bias1", "bias2"]
            }},
            "conflicts_of_interest": {{
                "found": false,
                "details": "conflict description"
            }},
            "perspective_balance": {{
                "score": 0.7,
                "missing_perspectives": ["perspective1", "perspective2"]
            }}
        }},
        "coverage": {{
            "depth": {{
                "score": 0.8,
                "assessment": "depth description"
            }},
            "scope": {{
                "score": 0.9,
                "areas_covered": ["area1", "area2"]
            }},
            "completeness": {{
                "score": 0.7,
                "gaps": ["gap1", "gap2"]
            }}
        }},
        "overall_score": 0.8
    }}"""
)

CREDIBILITY_ASSESSMENT_PROMPT = PromptTemplate.from_template(
    """You are a credibility assessment specialist. Return ONLY a valid JSON object.
    
    SOURCE SET:
    {sources}
    
    TASK:
    Assess the collective credibility of these sources and their reliability for the research topic.
    
    Return ONLY a JSON object with this exact structure:
    {{
        "source_diversity": {{
            "perspective_range": {{
                "score": 0.8,
                "represented_views": ["view1", "view2"],
                "missing_views": ["view3", "view4"]
            }},
            "source_types": {{
                "distribution": {{
                    "academic": 0.4,
                    "news": 0.3,
                    "expert": 0.3
                }},
                "assessment": "distribution analysis"
            }},
            "representation": {{
                "geographic": ["region1", "region2"],
                "cultural": ["culture1", "culture2"],
                "gaps": ["gap1", "gap2"]
            }}
        }},
        "consensus": {{
            "agreements": [
                {{
                    "topic": "topic description",
                    "strength": 0.8,
                    "supporting_sources": ["source1", "source2"]
                }}
            ],
            "disagreements": [
                {{
                    "topic": "topic description",
                    "severity": "high/medium/low",
                    "conflicting_views": ["view1", "view2"]
                }}
            ]
        }},
        "bias_analysis": {{
            "systematic_biases": ["bias1", "bias2"],
            "missing_viewpoints": ["viewpoint1", "viewpoint2"],
            "overrepresentation": ["aspect1", "aspect2"]
        }},
        "quality_metrics": {{
            "overall_credibility": 0.8,
            "confidence_levels": {{
                "high": ["topic1", "topic2"],
                "medium": ["topic3", "topic4"],
                "low": ["topic5", "topic6"]
            }},
            "verification_needs": ["need1", "need2"]
        }}
    }}"""
)

CONFIDENCE_CALIBRATION_PROMPT = PromptTemplate.from_template(
    """You are a confidence calibration expert. Return ONLY a valid JSON object.

    Findings:
    {findings}

    Sources:
    {sources}

    Return ONLY a JSON object with this exact structure:
    {{
        "calibrated_findings": [
            {{
                "finding": "finding text",
                "source_reliability": 0.8,
                "evidence_strength": 0.9,
                "consistency": 0.7,
                "biases": ["bias1", "bias2"],
                "adjusted_confidence": 0.8,
                "notes": "calibration notes"
            }}
        ],
        "overall_assessment": {{
            "reliability_score": 0.8,
            "evidence_quality": 0.9,
            "consistency_level": 0.7,
            "bias_impact": 0.2
        }},
        "recommendations": [
            {{
                "area": "area description",
                "current_confidence": 0.7,
                "target_confidence": 0.9,
                "actions": ["action1", "action2"]
            }}
        ]
    }}"""
)

CONFIDENCE_REPORT_PROMPT = PromptTemplate.from_template(
    """You are a confidence reporting expert. Return ONLY a valid JSON object.

    Calibration results:
    {calibration_results}

    Return ONLY a JSON object with this exact structure:
    {{
        "overall_assessment": {{
            "confidence_score": 0.8,
            "reliability_level": "high/medium/low",
            "key_strengths": ["strength1", "strength2"],
            "key_weaknesses": ["weakness1", "weakness2"]
        }},
        "confidence_factors": [
            {{
                "factor": "factor description",
                "impact": "high/medium/low",
                "contribution": 0.8,
                "evidence": ["evidence1", "evidence2"]
            }}
        ],
        "evidence_gaps": [
            {{
                "area": "gap description",
                "severity": "high/medium/low",
                "impact": 0.7,
                "mitigation": "how to address"
            }}
        ],
        "recommendations": [
            {{
                "suggestion": "recommendation text",
                "priority": "high/medium/low",
                "expected_impact": 0.8,
                "implementation": "how to implement"
            }}
        ],
        "final_scores": {{
            "overall_confidence": 0.8,
            "evidence_quality": 0.9,
            "reliability": 0.7,
            "completeness": 0.8
        }}
    }}"""
) 