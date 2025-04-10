"""
Prompt templates for information synthesis and cross-validation.
"""
from langchain_core.prompts import PromptTemplate

INFORMATION_SYNTHESIS_PROMPT = PromptTemplate.from_template(
    """You are an information synthesis specialist.
    
    RESEARCH TOPIC: {topic}
    COLLECTED INFORMATION:
    {information}
    
    TASK:
    Synthesize the collected information into a coherent and comprehensive analysis.
    
    SYNTHESIS REQUIREMENTS:
    1. Key Findings:
       - Main discoveries
       - Critical insights
       - Emerging patterns
    
    2. Evidence Quality:
       - Strength of support
       - Confidence levels
       - Knowledge gaps
    
    3. Relationships:
       - Connections between concepts
       - Cause-effect relationships
       - Dependencies
    
    4. Context Integration:
       - Historical perspective
       - Current relevance
       - Future implications
    
    5. Nuance Preservation:
       - Complexity acknowledgment
       - Uncertainty areas
       - Alternative interpretations
    
    SYNTHESIS REPORT:"""
)

CROSS_VALIDATION_PROMPT = PromptTemplate.from_template(
    """You are a cross-validation specialist.
    
    FINDINGS TO VALIDATE:
    {findings}
    
    AVAILABLE SOURCES:
    {sources}
    
    TASK:
    Cross-validate the findings against multiple sources to ensure accuracy and reliability.
    
    VALIDATION REQUIREMENTS:
    1. Agreement Analysis:
       - Consistent findings
       - Supporting evidence
       - Confidence levels
    
    2. Contradiction Detection:
       - Conflicting information
       - Discrepancies
       - Resolution attempts
    
    3. Source Assessment:
       - Independent verification
       - Methodology comparison
       - Bias consideration
    
    4. Confidence Calibration:
       - Evidence strength
       - Verification level
       - Uncertainty areas
    
    VALIDATION REPORT:"""
)

FINAL_SUMMARY_PROMPT = PromptTemplate.from_template(
    """You are a sophisticated research analyst and critical synthesizer. Your task is to transform the provided research data into a **highly detailed, extensive, and comprehensive** final report, going far beyond simple summarization. **Length and thoroughness are paramount.**

    **First, analyze the user's Original Query: '{query}'**
    Determine the *intended output format* (e.g., detailed report, script, story, summary).

    **Then, critically synthesize the AVAILABLE DATA below to generate the final response.** Do not merely repeat the Narrative Analysis; use it as a starting point and integrate it with the Raw Search Results.

    AVAILABLE DATA:
    - Original Query: {query}
    - Narrative Analysis:
    {analysis}
    - Raw Search Results (Titles/URLs/Snippets):
    {search_results_summary}
    - Fact-Checking Overview:
    {fact_checks}
    - Confidence Metrics:
    {confidence_metrics}

    **TASK: Generate the Final Response**

    1.  **Format Adaptation:**
        *   Structure your output **according to the format requested or implied in the Original Query.**
    2.  **Default Structure (Fallback - Use IF query is standard research question AND no other format requested):**
        *   **IF** applicable, structure the response as a **long, detailed, and extensive** research report using the following subcategories (use Markdown headings):
            *   **### Key Findings & Synthesis**
                *   **Critically synthesize** the main answers/findings related to the query **at length**.
                *   **Compare and contrast** information from the Narrative Analysis and Raw Search Results snippets **in detail**.
                *   **Evaluate the significance and practical implications** of the findings, don't just list them. **Provide extensive explanation.**
                *   Provide specific examples or quotes from the results/snippets to support key points.
            *   **### Critical Assessment: Nuances, Limitations & Conflicts**
                *   **Thoroughly and extensively** discuss important nuances, confidence levels, and limitations.
                *   **Explicitly identify and analyze in detail** any conflicting information or discrepancies found.
                *   Analyze potential biases and their impact on the findings **with comprehensive reasoning**.
            *   **### Unanswered Questions & Research Gaps**
                *   Clearly outline significant information gaps or unresolved questions **in full detail**.
                *   Suggest specific, actionable areas for further investigation.
            *   **### Deeper Implications & Interpretations**
                *   Based *strictly* on the synthesized information, discuss potential *deeper* implications, underlying assumptions, or interpretations. **Elaborate significantly**, going beyond the surface level.
            *   **### Illustrative Code Examples (If Applicable to Topic)**
                *   Only include if the query *specifically* relates to software/programming. Provide concise, relevant, and correct code snippets (use Markdown) that *clearly illustrate* a key concept or finding.
            *   **### Overall Conclusion & Outlook**
                *   Provide a concluding paragraph that summarizes the *most critical insights* and offers a brief outlook based on the **extensive research conducted**.
        *   **OTHERWISE (Specific format requested/implied):** Generate the response in that format, ensuring it is still **long, comprehensive, and detailed**, incorporating insights synthesized from the AVAILABLE DATA.
    3.  **General Requirements (Apply to ALL formats):**
        *   **Length and Detail:** The final output **must be extensive and detailed.** Avoid overly brief or superficial responses.
        *   **Depth over Breadth:** Prioritize deep analysis and synthesis of the core topic.
        *   **Go Beyond Summarization:** Provide value through synthesis, comparison, critical evaluation, and identification of deeper connections.
        *   **Evidence-Based:** Ground all claims and interpretations firmly in the provided AVAILABLE DATA.
        *   Maintain clarity, logical flow, and objectivity.

    FINAL RESPONSE (Ensure this response is long, detailed, extensive, and in the determined format, demonstrating critical synthesis and depth):
    """
) 