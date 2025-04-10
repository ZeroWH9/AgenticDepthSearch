"""
Source evaluation module
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from src.integrations.llm import LLMFactory
from src.prompts.evaluation_prompts import SOURCE_EVALUATION_PROMPT
from src.prompts import (
    CREDIBILITY_ASSESSMENT_PROMPT
)
from src.utils.config import settings
from src.prompts.processing_prompts import (
    SOURCE_EVALUATION_PROMPT,
    SOURCE_INFO_EXTRACTION_PROMPT,
    SOURCE_RELIABILITY_PROMPT
)
from src.utils.metrics import metrics_collector
import json
import logging
import re

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SourceEvaluation:
    """
    Source evaluation results
    """
    credibility_score: float  # Score de credibilidade (0-1)
    relevance_score: float  # Score de relevância (0-1)
    recency_score: float  # Score de atualidade (0-1)
    bias_level: str  # Nível de viés (low, medium, high)
    quality_indicators: Dict[str, float]  # Indicadores específicos
    issues: List[str]  # Problemas identificados
    recommendations: List[str]  # Recomendações de uso

@dataclass
class CredibilityAssessment:
    """Credibility assessment results"""
    sources: List[Dict]
    diversity_score: float
    consensus_level: float
    bias_assessment: Dict[str, float]
    overall_credibility: float
    recommendations: List[str]

class SourceEvaluator:
    """
    Evaluates source quality and credibility
    """
    
    def __init__(
        self,
        llm_factory: Optional[LLMFactory] = None
    ):
        """
        Initialize source evaluator
        
        Args:
            llm_factory: LLM factory to use
        """
        self.llm_factory = llm_factory or LLMFactory()
        self.min_credibility = settings.source_evaluation.min_credibility
        self.max_age_days = settings.source_evaluation.max_age_days
        self.require_citations = settings.source_evaluation.require_citations
        self.metrics = metrics_collector
    
    async def evaluate_source(
        self,
        source: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a source
        
        Args:
            source: Source to evaluate (dict containing 'title', 'url', 'content', etc.)
            context: Additional context (not currently used by prompt)
            
        Returns:
            Dict[str, Any]: Evaluation results with 'evaluation' and 'credibility' keys.
        """
        # Extract content and create a string representation of source metadata
        source_content = str(source.get("content", ""))
        # Create a string representation of the source excluding the (potentially large) content
        source_metadata_str = str({k: v for k, v in source.items() if k != 'content'})

        # Generate evaluation using LLM with streaming disabled
        try:
            prompt = SOURCE_EVALUATION_PROMPT.format(
                source=source_metadata_str, # Pass metadata string
                content=source_content       # Pass content string
            )
        except KeyError as e:
            logger.error(f"Missing key in SOURCE_EVALUATION_PROMPT format: {e}. Source keys: {list(source.keys())}")
            return {"evaluation": {"error": f"Prompt formatting error: Missing key {e}"}, "credibility": 0.0}

        llm = self.llm_factory.get_llm(streaming=False)
        try:
            response = await llm.ainvoke(prompt)
        except Exception as llm_error:
            logger.error(f"LLM invocation failed during source evaluation: {llm_error}")
            return {"evaluation": {"error": f"LLM invocation failed: {llm_error}"}, "credibility": 0.0}

        # Parse the JSON response from the LLM
        try:
            # Attempt to extract JSON potentially wrapped in markdown
            json_str = self._extract_json(response)
            eval_data = json.loads(json_str)
            if not isinstance(eval_data, dict):
                 eval_data = {"error": "Parsed JSON is not a dictionary"}
        except Exception as parse_error:
            logger.error(f"Error parsing evaluation response: {parse_error}. Response: {response}")
            eval_data = {"error": f"Failed to parse evaluation JSON: {parse_error}"}

        # Return evaluation data and derive credibility score from it
        credibility = 0.0
        if isinstance(eval_data, dict) and "overall_reliability" in eval_data:
            try:
                credibility = float(eval_data["overall_reliability"])
            except (ValueError, TypeError):
                logger.warning(f"Could not convert overall_reliability '{eval_data['overall_reliability']}' to float.")

        return {
            "evaluation": eval_data,
            "credibility": credibility
        }
    
    def _extract_json(self, text: str) -> str:
        """Extracts the first JSON object found within a string, handling markdown code blocks."""
        try:
            # Attempt to find JSON within markdown code blocks
            match = re.search(r"```(?:json)?\s*({.*?})\s*```", text, re.DOTALL | re.IGNORECASE)
            if match:
                json_str = match.group(1)
                logger.debug(f"Extracted JSON from markdown block.")
                return json_str.strip()

            # If no markdown block, find the first opening brace
            start_brace = text.find('{')
            if start_brace == -1:
                logger.warning(f"No JSON object start found in response: {text[:200]}...")
                return "{}" # Return empty JSON if no start brace

            # Find the matching end brace
            level = 0
            in_string = False
            for i in range(start_brace, len(text)):
                char = text[i]
                if char == '"' and (i == 0 or text[i-1] != '\\'):
                    in_string = not in_string
                if not in_string:
                    if char == '{':
                        level += 1
                    elif char == '}':
                        level -= 1
                        if level == 0:
                            json_str = text[start_brace:i+1]
                            logger.debug(f"Extracted JSON by finding matching braces.")
                            return json_str.strip()

            logger.warning(f"Could not find matching end brace for JSON: {text[start_brace:start_brace+200]}...")
            return "{}" # Return empty JSON if matching end not found
        except Exception as e:
            logger.error(f"Error during JSON extraction: {e}. Text: {text[:200]}...")
            return "{}"
    
    async def evaluate_source_batch(
        self,
        sources: List[Dict[str, Any]]
    ) -> List[SourceEvaluation]:
        """
        Evaluate multiple sources
        
        Args:
            sources: List of source dictionaries
            
        Returns:
            List of SourceEvaluation objects
        """
        evaluations = []
        for source in sources:
            eval_result = await self.evaluate_source(source)
            evaluations.append(SourceEvaluation(
                credibility_score=eval_result["credibility"],
                relevance_score=eval_result["evaluation"]["relevance_score"],
                recency_score=eval_result["evaluation"]["recency_score"],
                bias_level=eval_result["evaluation"]["bias_level"],
                quality_indicators=eval_result["evaluation"]["quality_indicators"],
                issues=eval_result["evaluation"]["issues"],
                recommendations=eval_result["evaluation"]["recommendations"]
            ))
        return evaluations
    
    async def assess_credibility(
        self,
        sources: List[Dict]
    ) -> CredibilityAssessment:
        """
        Assess credibility of a set of sources
        
        Args:
            sources: List of source dictionaries
            
        Returns:
            CredibilityAssessment: Assessment results
        """
        # Generate assessment using LLM with streaming disabled
        llm = self.llm_factory.get_llm(streaming=False)
        assessment_prompt = CREDIBILITY_ASSESSMENT_PROMPT.format(
            sources=sources
        )
        
        assessment_response = await llm.ainvoke(assessment_prompt)
        
        try:
            # TODO: Implement proper assessment parsing
            # For now, return a basic assessment
            return CredibilityAssessment(
                sources=sources,
                diversity_score=0.0,
                consensus_level=0.0,
                bias_assessment={},
                overall_credibility=0.0,
                recommendations=[]
            )
        except Exception as e:
            print(f"[DEBUG] Error assessing credibility: {str(e)}")
            return CredibilityAssessment(
                sources=sources,
                diversity_score=0.0,
                consensus_level=0.0,
                bias_assessment={},
                overall_credibility=0.0,
                recommendations=["Assessment failed"]
            )
    
    def check_source_requirements(
        self,
        source: Dict
    ) -> bool:
        """
        Check if a source meets minimum requirements
        
        Args:
            source: Source dictionary
            
        Returns:
            bool: Whether source meets requirements
        """
        # Check credibility score
        if source.get("score", 0) < self.min_credibility:
            return False
            
        # Check age if date is available
        if date_str := source.get("date"):
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
                age_days = (datetime.now() - date).days
                if age_days > self.max_age_days:
                    return False
            except:
                pass
                
        # Check citations if required
        if self.require_citations:
            if not source.get("citations", []):
                return False
                
        return True
    
    async def evaluate_sources(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate sources from search results
        
        Args:
            results: List of search results
            
        Returns:
            Dictionary with source evaluation results
        """
        try:
            # Extract source information
            sources = await self._extract_source_info(results)
            
            # Evaluate source reliability
            reliability = await self._evaluate_reliability(sources)
            
            # Generate evaluation report
            report = await self._generate_evaluation_report(sources)
            
            return {
                "sources": sources,
                "reliability": reliability,
                "report": report
            }
            
        except Exception as e:
            return {"error": str(e)}
            
    async def _extract_source_info(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract source information from results
        
        Args:
            results: List of search results
            
        Returns:
            List of source information dictionaries
        """
        # Use template for source info extraction
        prompt = SOURCE_INFO_EXTRACTION_PROMPT.format(
            results=results
        )
        
        response = await self.llm_factory.get_llm().agenerate(prompt)
        return self._parse_source_info(response.generations[0].text)
        
    async def _evaluate_reliability(
        self,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate source reliability
        
        Args:
            sources: List of source information
            
        Returns:
            Dictionary with reliability scores
        """
        # Use template for reliability evaluation
        prompt = SOURCE_RELIABILITY_PROMPT.format(
            sources=sources
        )
        
        response = await self.llm_factory.get_llm().agenerate(prompt)
        return self._parse_reliability_scores(response.generations[0].text)
        
    def _parse_reliability_scores(self, text: str) -> Dict[str, float]:
        """
        Parse LLM text to extract reliability scores
        
        Args:
            text: LLM text
            
        Returns:
            Dict[str, float]: Dictionary with source reliability scores
        """
        scores = {}
        current_source = None
        
        for line in text.strip().split('\n'):
            line = line.strip()
            
            if not line:
                continue
                
            if line == '---':
                current_source = None
                continue
                
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == 'source':
                    current_source = value
                elif key == 'score' and current_source:
                    try:
                        scores[current_source] = float(value)
                    except:
                        scores[current_source] = 0.0
                        
        return scores
        
    async def _generate_evaluation_report(
        self,
        sources: List[Dict]
    ) -> Dict:
        """
        Generate source evaluation report
        
        Args:
            sources: List of sources to evaluate
            
        Returns:
            Dict: Source evaluation report
        """
        # Use template for report
        prompt = SOURCE_EVALUATION_PROMPT.format(
            sources=sources
        )
        
        # Generate report using LLM
        response = await self.llm_factory.get_llm().agenerate(prompt)
        
        # Parse and return report
        return self._parse_evaluation_report(response)
    
    def _extract_score(self, line: str) -> float:
        """
        Extract score from evaluation line
        
        Args:
            line: Evaluation line
            
        Returns:
            Extracted score
        """
        try:
            # Look for score in format "X/Y" or "X%"
            if "/" in line:
                parts = line.split("/")
                return float(parts[0]) / float(parts[1])
            elif "%" in line:
                return float(line.split("%")[0]) / 100
            else:
                return 0.0
        except:
            return 0.0
    
    def _parse_source_info(self, text: str) -> List[Dict]:
        """
        Parse LLM text to extract source information
        
        Args:
            text: LLM text
            
        Returns:
            List[Dict]: List of source information
        """
        sources = []
        current_source = {}
        
        for line in text.strip().split('\n'):
            line = line.strip()
            
            if not line:
                continue
                
            if line == '---':
                if current_source:
                    sources.append(current_source)
                    current_source = {}
                continue
                
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == 'source':
                    current_source['name'] = value
                elif key == 'type':
                    current_source['type'] = value
                elif key == 'date':
                    current_source['date'] = value
                elif key == 'author':
                    current_source['author'] = value
                elif key == 'reputation':
                    current_source['reputation'] = value
                    
        if current_source:
            sources.append(current_source)
            
        return sources
        
    def _parse_evaluation(self, text: str) -> Dict:
        """
        Parse LLM text to extract evaluation
        
        Args:
            text: LLM text
            
        Returns:
            Dict: Evaluation results
        """
        evaluation = {}
        
        for line in text.strip().split('\n'):
            line = line.strip()
            
            if not line:
                continue
                
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == 'credibility':
                    try:
                        evaluation['credibility'] = float(value)
                    except:
                        evaluation['credibility'] = 0.0
                elif key == 'reliability':
                    try:
                        evaluation['reliability'] = float(value)
                    except:
                        evaluation['reliability'] = 0.0
                elif key == 'bias':
                    evaluation['bias'] = value
                elif key == 'notes':
                    evaluation['notes'] = value
                    
        return evaluation
    
    def _parse_evaluation_report(self, response: str) -> Dict:
        """
        Parse LLM text to extract evaluation report
        
        Args:
            response: LLM text
            
        Returns:
            Dict: Evaluation report
        """
        # Implement parsing logic based on the format of the response
        # This is a placeholder and should be replaced with the actual implementation
        return {}
        
    async def _evaluate_source(self, source_info: Dict) -> Dict:
        """
        Evaluate a single source
        
        Args:
            source_info: Source information
            
        Returns:
            Dict: Source evaluation
        """
        # Use template for source evaluation
        prompt = SOURCE_EVALUATION_PROMPT.format(
            source_info=source_info
        )
        
        response = await self.llm_factory.get_llm().agenerate(prompt)
        return self._parse_evaluation(response.generations[0].text) 