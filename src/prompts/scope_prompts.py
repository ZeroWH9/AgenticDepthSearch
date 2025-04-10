"""
Prompt templates for research scope definition and query decomposition.
"""
from langchain_core.prompts import PromptTemplate

SCOPE_DEFINITION_PROMPT = PromptTemplate.from_template(
    """You are a research scope definition expert.
    
    USER QUERY: {user_query}
    
    INSTRUCTIONS:
    1. Define the central research question clearly
    2. Identify 3-5 specific sub-questions that compose the main question
    3. For each sub-question, specify:
       - Relevant time boundaries
       - Geographic or contextual boundaries
       - Technical terms that need definition
    4. Suggest 2-3 different angles or aspects to consider
    
    Prioritize precision and completeness over simplification.
    Preserve the complexity of the topic.
    
    Return your scope definition as a JSON object with the following structure:
    {{
        "central_question": str,  # Main research question
        "sub_questions": List[Dict[str, str]],  # Sub-questions
        "boundaries": Dict[str, str],  # Research boundaries
        "angles": List[str]  # Different aspects to consider
    }}
    
    Each sub-question should have:
    - question: str
    - time_boundaries: str
    - geographic_boundaries: str
    - technical_terms: List[str]
    
    SCOPE DEFINITION:"""
)

SUBQUERY_GENERATION_PROMPT = PromptTemplate.from_template(
    """You are a query decomposition specialist.
    
    MAIN QUERY: {main_query}
    CONTEXT: {context}
    SCOPE: {scope}
    
    TASK:
    Generate a set of focused sub-queries that will help explore this topic thoroughly.
    
    REQUIREMENTS:
    1. Each sub-query should:
       - Be specific and targeted
       - Focus on a distinct aspect
       - Be searchable on its own
    2. Sub-queries should collectively cover:
       - Core concepts and definitions
       - Key relationships and dependencies
       - Historical context and evolution
       - Current state and trends
       - Challenges and limitations
    
    Return your sub-queries as a JSON object with the following structure:
    {{
        "main_query": str,  # Original main query
        "sub_queries": List[Dict[str, str]],  # Generated sub-queries
        "coverage": Dict[str, bool]  # Coverage of different aspects
    }}
    
    Each sub-query should have:
    - query: str
    - aspect: str
    - priority: int
    - dependencies: List[str]
    
    Number of sub-queries: {num_queries}
    
    SUB-QUERIES:"""
) 