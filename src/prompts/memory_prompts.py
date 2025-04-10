"""
Prompt templates for memory management and summarization.
"""
from langchain_core.prompts import PromptTemplate

MEMORY_SUMMARY_PROMPT = PromptTemplate.from_template(
    """You are a memory summarization expert.
    
    TOPIC: {topic}
    MEMORIES: {memories}
    
    TASK:
    Generate a comprehensive summary of the provided memories.
    
    REQUIREMENTS:
    1. Key Information:
       - Extract main points
       - Identify patterns
       - Note important details
       - Highlight relationships
    
    2. Organization:
       - Group related information
       - Create logical flow
       - Maintain context
       - Preserve chronology
    
    3. Quality:
       - Ensure accuracy
       - Maintain objectivity
       - Avoid redundancy
       - Include relevant context
    
    MEMORY SUMMARY:"""
)

TOPIC_EXTRACTION_PROMPT = PromptTemplate.from_template(
    """You are a topic extraction specialist.
    
    CONTENT: {content}
    
    TASK:
    Extract and categorize topics from the content.
    
    REQUIREMENTS:
    1. Topic Identification:
       - Main topics
       - Subtopics
       - Related concepts
       - Key themes
    
    2. Categorization:
       - Primary category
       - Secondary categories
       - Topic relationships
       - Hierarchical structure
    
    3. Metadata:
       - Topic importance
       - Topic frequency
       - Topic relationships
       - Topic evolution
    
    TOPIC ANALYSIS:"""
)

MEMORY_RELEVANCE_PROMPT = PromptTemplate.from_template(
    """You are a memory relevance assessor.
    
    QUERY: {query}
    MEMORY: {memory}
    
    TASK:
    Assess the relevance of this memory to the query.
    
    REQUIREMENTS:
    1. Relevance Analysis:
       - Direct relevance
       - Indirect relevance
       - Contextual relevance
       - Temporal relevance
    
    2. Quality Assessment:
       - Information quality
       - Source reliability
       - Completeness
       - Accuracy
    
    3. Utility Evaluation:
       - Potential usefulness
       - Information gaps
       - Additional context needed
       - Limitations
    
    RELEVANCE ASSESSMENT:"""
) 