"""
Prompt templates for processing and analysis.
"""
from langchain_core.prompts import PromptTemplate

RESULT_ANALYSIS_PROMPT = PromptTemplate.from_template(
    """You are a research analysis expert.
    
    QUERY: {query}
    RESULTS: {results}
    
    TASK:
    Analyze the search results and extract key information.
    
    REQUIREMENTS:
    1. Extract key facts and findings
    2. Identify relationships between different pieces of information
    3. Note any contradictions or gaps
    4. Assess the reliability of sources
    5. Synthesize the information into coherent insights
    
    Return your analysis as a JSON object with the following structure:
    {{
        "key_facts": List[Dict[str, str]],  # Key facts and findings
        "relationships": List[Dict[str, str]],  # Relationships between facts
        "contradictions": List[Dict[str, str]],  # Contradictions found
        "gaps": List[str],  # Information gaps
        "source_reliability": Dict[str, float],  # Source reliability scores
        "synthesis": str  # Overall synthesis
    }}
    
    Each fact should have:
    - fact: str
    - source: str
    - confidence: float
    
    Each relationship should have:
    - fact1: str
    - fact2: str
    - relationship_type: str
    
    ANALYSIS:"""
)

CONFIDENCE_REPORT_PROMPT = PromptTemplate.from_template(
    """You are a confidence assessment expert.
    
    ANALYSIS: {analysis}
    
    TASK:
    Assess the confidence level of the research findings.
    
    REQUIREMENTS:
    1. Evaluate source reliability
    2. Consider information consistency
    3. Assess evidence strength
    4. Identify potential biases
    5. Calculate overall confidence score
    
    Return your confidence assessment as a JSON object with the following structure:
    {{
        "source_reliability": Dict[str, float],  # Reliability scores by source
        "consistency_score": float,  # Information consistency score
        "evidence_strength": float,  # Strength of evidence
        "potential_biases": List[str],  # Identified biases
        "overall_confidence": float  # Overall confidence score
    }}
    
    CONFIDENCE ASSESSMENT:"""
)

SUBQUERY_GENERATION_PROMPT = PromptTemplate.from_template(
    """Generate {max_subqueries} specific sub-queries to research the following topic:

    Main query: {query}

    {context}

    Rules:
    1. Each sub-query should be specific and focused
    2. Sub-queries should cover different aspects of the topic
    3. Avoid duplicate or overlapping queries
    4. Each sub-query should be at least {min_query_length} characters long

    Return only the sub-queries, one per line.
    """
)

SOURCE_INFO_EXTRACTION_PROMPT = PromptTemplate.from_template(
    """Extract source information from these results:
    
    {results}
    
    For each source, identify:
    1. Source name/domain
    2. Source type (news, academic, blog, etc)
    3. Publication date
    4. Author information
    5. Source reputation indicators
    
    Return in format:
    SOURCE: [source_name]
    TYPE: [source_type]
    DATE: [publication_date]
    AUTHOR: [author_info]
    REPUTATION: [reputation_indicators]
    ---
    """
)

SOURCE_RELIABILITY_PROMPT = PromptTemplate.from_template(
    """Evaluate reliability of these sources:
    
    {sources}
    
    Consider:
    1. Source reputation
    2. Author credibility
    3. Publication history
    4. Fact-checking record
    
    Return reliability scores (0-1) for each source in format:
    SOURCE: [source_name]
    SCORE: [reliability_score]
    NOTES: [reliability_notes]
    ---
    """
)

CONFIDENCE_METRICS_PROMPT = PromptTemplate.from_template(
    """Calculate confidence metrics for these results:
    
    {results}
    
    Consider:
    1. Source reliability
    2. Information consistency
    3. Evidence strength
    4. Coverage depth
    
    Return in format:
    SOURCE_RELIABILITY: [score]
    INFORMATION_CONSISTENCY: [score]
    EVIDENCE_STRENGTH: [score]
    COVERAGE_DEPTH: [score]
    OVERALL_CONFIDENCE: [score]
    NOTES: [confidence_notes]
    ---
    """
)

FACT_CHECKING_ADJUSTMENT_PROMPT = PromptTemplate.from_template(
    """Adjust confidence metrics based on fact checking:
    
    Base confidence: {base_confidence}
    Fact checking: {fact_checking}
    
    Consider:
    1. Agreement level with fact checking
    2. Number of verified claims
    3. Contradictions found
    4. Evidence quality
    
    Return in format:
    SOURCE_RELIABILITY: [adjusted_score]
    CONSISTENCY: [adjusted_score]
    EVIDENCE_STRENGTH: [adjusted_score]
    COVERAGE_DEPTH: [adjusted_score]
    NOTES: [adjustment_notes]
    ---
    """
)

SOURCE_EVALUATION_PROMPT = PromptTemplate.from_template(
    """You are a source evaluation expert. Respond ONLY with a valid JSON object.

    SOURCE: {source}
    CONTENT: {content}

    TASK:
    Evaluate the reliability and quality of the source based ONLY on the provided SOURCE metadata and CONTENT. DO NOT search for external information.

    REQUIREMENTS:
    1. Assess source credibility (consider domain, author if available in SOURCE metadata)
    2. Evaluate content quality (clarity, depth based on CONTENT)
    3. Check for potential biases (based on tone, language in CONTENT and SOURCE metadata)
    4. Consider recency and relevance (if available in SOURCE metadata, relative to a general context)
    5. Calculate reliability score (float between 0.0 and 1.0)

    Return ONLY a JSON object with this exact structure:
    {{
        "source_credibility": float,  # Source credibility score (0.0-1.0)
        "content_quality": float,  # Content quality score (0.0-1.0)
        "potential_biases": List[str],  # Identified biases (list of strings)
        "recency_score": float,  # Recency score (0.0-1.0, estimate if date not present)
        "relevance_score": float,  # Relevance score (0.0-1.0, estimate based on content)
        "overall_reliability": float  # Overall reliability score (0.0-1.0)
    }}

    SOURCE EVALUATION:
    """
)

FACT_CHECKING_PROMPT = PromptTemplate.from_template(
    """You are a fact-checking expert.
    
    FACT: {fact}
    SOURCES: {sources}
    
    TASK:
    Verify the accuracy of the fact using multiple sources.
    
    REQUIREMENTS:
    1. Cross-reference with multiple sources
    2. Check for consistency
    3. Identify supporting evidence
    4. Note any contradictions
    5. Calculate verification score
    
    Return your verification as a JSON object with the following structure:
    {{
        "supporting_sources": List[str],  # Sources supporting the fact
        "contradicting_sources": List[str],  # Sources contradicting the fact
        "consistency_score": float,  # Consistency score
        "evidence_strength": float,  # Strength of evidence
        "verification_score": float  # Overall verification score
    }}
    
    FACT VERIFICATION:"""
)

SYNTHESIS_PROMPT = PromptTemplate.from_template(
    """You are a research synthesis expert.
    
    ANALYSIS: {analysis}
    CONFIDENCE: {confidence}
    
    TASK:
    Synthesize the research findings into a coherent report.
    
    REQUIREMENTS:
    1. Organize key findings
    2. Highlight important relationships
    3. Address contradictions and gaps
    4. Provide confidence levels
    5. Draw conclusions
    
    Return your synthesis as a JSON object with the following structure:
    {{
        "key_findings": List[Dict[str, str]],  # Key findings
        "relationships": List[Dict[str, str]],  # Important relationships
        "contradictions": List[Dict[str, str]],  # Contradictions addressed
        "gaps": List[str],  # Information gaps
        "confidence_levels": Dict[str, float],  # Confidence levels
        "conclusions": List[str]  # Conclusions drawn
    }}
    
    Each finding should have:
    - finding: str
    - confidence: float
    - sources: List[str]
    
    Each relationship should have:
    - finding1: str
    - finding2: str
    - relationship_type: str
    
    SYNTHESIS:"""
)

FACT_CHECK_REPORT_PROMPT = PromptTemplate.from_template(
    """Generate a fact-checking report based on:

    Verification results:
    {verification_results}

    Include:
    1. Overall verification status
    2. Key findings and evidence
    3. Areas of uncertainty
    4. Recommendations for further verification
    5. Confidence level in verification

    Return in format:
    OVERALL_STATUS: [status]
    KEY_FINDINGS: [findings]
    UNCERTAINTIES: [uncertainties]
    RECOMMENDATIONS: [recommendations]
    CONFIDENCE: [confidence_score]
    ---
    """
) 