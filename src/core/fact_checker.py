"""
Fact checking module
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from src.integrations.llm import LLMFactory
from src.integrations.search import CustomTavilySearch
from src.prompts.evaluation_prompts import (
    SOURCE_EVALUATION_PROMPT,
    CREDIBILITY_ASSESSMENT_PROMPT
)
from src.prompts.research_prompts import (
    RESULT_PROCESSING_PROMPT,
    INFORMATION_EXTRACTION_PROMPT,
    RELATIONSHIP_ANALYSIS_PROMPT,
    SYNTHESIS_PROMPT
)
from src.prompts.processing_prompts import (
    FACT_CHECK_REPORT_PROMPT,
    FACT_CHECKING_PROMPT
)
from src.utils.config import settings
from src.utils.metrics import metrics_collector

@dataclass
class FactValidation:
    """
    Class to represent the validation status of a fact
    """
    claim: str
    status: str  # verified, partially_verified, unverified, refuted
    evidence: List[str]
    corroboration: List[str]
    biases: List[str]
    confidence: float
    sources: List[str]

class FactChecker:
    """
    Fact checker class
    """
    def __init__(
        self,
        llm_factory: Optional[LLMFactory] = None,
        search_client: Optional[CustomTavilySearch] = None
    ):
        """
        Initialize fact checker
        
        Args:
            llm_factory: LLM factory to use
            search_client: Search client to use
        """
        self.llm_factory = llm_factory or LLMFactory()
        self.search_client = search_client or CustomTavilySearch(settings.tavily.api_key)
        self.metrics = metrics_collector
        
    async def check_facts(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        """
        Check facts in search results
        
        Args:
            results: List of search results
            query: Original research query
            
        Returns:
            Dictionary with fact checking results
        """
        try:
            # Process each result
            processed_results = []
            for result in results:
                processed = await self._process_result(result, query)
                processed_results.append(processed)
            
            # Extract information from all results
            extracted_info = await self._extract_information(processed_results, query)
            
            # Analyze relationships between findings
            relationships = await self._analyze_relationships(
                extracted_info["findings"],
                results
            )
            
            # Synthesize findings
            synthesis = await self._synthesize_findings(
                extracted_info["findings"],
                relationships,
                extracted_info["confidence_scores"]
            )
            
            return {
                "processed_results": processed_results,
                "extracted_info": extracted_info,
                "relationships": relationships,
                "synthesis": synthesis
            }
            
        except Exception as e:
            return {"error": str(e)}
            
    async def _process_result(
        self,
        result: Dict[str, Any],
        query: str
    ) -> Dict[str, Any]:
        """
        Process a single result
        
        Args:
            result: Search result
            query: Original query
            
        Returns:
            Processed result
        """
        prompt = RESULT_PROCESSING_PROMPT.format(
            result=result,
            context={"query": query}
        )
        
        llm = self.llm_factory.get_llm(streaming=False)
        response = await llm.ainvoke(prompt)
        return eval(response)
        
    async def _extract_information(
        self,
        processed_results: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        """
        Extract information from processed results
        
        Args:
            processed_results: List of processed results
            query: Original query
            
        Returns:
            Extracted information
        """
        prompt = INFORMATION_EXTRACTION_PROMPT.format(
            results=processed_results,
            context={"query": query}
        )
        
        llm = self.llm_factory.get_llm(streaming=False)
        response = await llm.ainvoke(prompt)
        return eval(response)
        
    async def _analyze_relationships(
        self,
        findings: List[Dict[str, Any]],
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze relationships between findings
        
        Args:
            findings: List of findings
            sources: List of sources
            
        Returns:
            Relationship analysis
        """
        prompt = RELATIONSHIP_ANALYSIS_PROMPT.format(
            findings=findings,
            sources=sources
        )
        
        llm = self.llm_factory.get_llm(streaming=False)
        response = await llm.ainvoke(prompt)
        return eval(response)
        
    async def _synthesize_findings(
        self,
        findings: List[Dict[str, Any]],
        relationships: Dict[str, Any],
        confidence_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Synthesize findings into a coherent summary
        
        Args:
            findings: List of findings
            relationships: Relationship analysis
            confidence_metrics: Confidence scores
            
        Returns:
            Synthesis of findings
        """
        prompt = SYNTHESIS_PROMPT.format(
            findings=findings,
            relationships=relationships,
            confidence_metrics=confidence_metrics
        )
        
        llm = self.llm_factory.get_llm(streaming=False)
        response = await llm.ainvoke(prompt)
        return eval(response)
        
    async def _extract_claims(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Extract claims from search results
        
        Args:
            results: List of search results
            query: Original research query
            
        Returns:
            List of extracted claims
        """
        # Use SOURCE_EVALUATION_PROMPT for each result
        claims = []
        llm = self.llm_factory.get_llm(streaming=False)
        
        for result in results:
            prompt = SOURCE_EVALUATION_PROMPT.format(
                title=result.get("title", ""),
                url=result.get("url", ""),
                content=result.get("content", ""),
                date=result.get("date", "")
            )
            
            response = await llm.ainvoke(prompt)
            
            # Parse response into claim format
            claims.append({
                "claim": response,
                "source": result.get("url", ""),
                "evidence": [result.get("content", "")],
                "contradictions": []
            })
            
        return claims
        
    async def _verify_claims(
        self,
        claims: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Verify claims using multiple sources
        
        Args:
            claims: List of claims to verify
            
        Returns:
            List of verified claims
        """
        # Use CREDIBILITY_ASSESSMENT_PROMPT for verification
        prompt = CREDIBILITY_ASSESSMENT_PROMPT.format(
            sources=claims
        )
        
        llm = self.llm_factory.get_llm(streaming=False)
        response = await llm.ainvoke(prompt)
        return eval(response)
        
    async def _generate_verification_report(
        self,
        verified_claims: List[Dict[str, Any]]
    ) -> str:
        """
        Generate verification report
        
        Args:
            verified_claims: List of verified claims
            
        Returns:
            Report text
        """
        # Use CREDIBILITY_ASSESSMENT_PROMPT for report generation
        prompt = CREDIBILITY_ASSESSMENT_PROMPT.format(
            sources=verified_claims
        )
        
        response = await self.llm_factory.get_llm().agenerate(prompt)
        return response
        
    def _calculate_confidence_scores(
        self,
        verified_claims: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate confidence scores for claims
        
        Args:
            verified_claims: List of verified claims
            
        Returns:
            Dictionary with confidence scores
        """
        scores = {}
        for claim in verified_claims:
            score = 0.0
            if claim["status"] == "verified":
                score += 0.5
            if len(claim["evidence"]) > 0:
                score += 0.2
            if len(claim["corroboration"]) > 0:
                score += 0.3
            scores[claim["claim"]] = min(score, 1.0)
        return scores
        
    async def _generate_fact_check_report(
        self,
        findings: List[Dict],
        sources: List[Dict]
    ) -> Dict:
        """
        Generate fact checking report
        
        Args:
            findings: List of research findings
            sources: List of sources used
            
        Returns:
            Dict: Fact checking report
        """
        # Use template for report
        prompt = FACT_CHECKING_PROMPT.format(
            findings=findings,
            sources=sources
        )
        
        # Generate report using LLM
        response = await self.llm_factory.get_llm().agenerate(prompt)
        
        # Parse and return report
        return self._parse_fact_check_report(response) 