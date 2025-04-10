"""
Results processing module
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from src.integrations.llm import LLMFactory
from src.core.memory import DeepResearchMemory
from src.utils.config import settings
from src.core.source_evaluator import SourceEvaluator
from src.core.fact_checker import FactChecker
from src.core.confidence_calibrator import ConfidenceCalibrator
from src.prompts.research_prompts import (
    RESULT_PROCESSING_PROMPT,
    INFORMATION_EXTRACTION_PROMPT,
    RELATIONSHIP_ANALYSIS_PROMPT,
    SYNTHESIS_PROMPT
)
from src.core.plan_generator import PlanGenerator
from src.prompts.processing_prompts import (
    RESULT_ANALYSIS_PROMPT,
    SUBQUERY_GENERATION_PROMPT
)

@dataclass
class ProcessedResult:
    """Processed search result"""
    content: str
    title: Optional[str] = None
    url: Optional[str] = None
    score: float = 0.0
    timestamp: datetime = datetime.now()
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Post-creation initialization"""
        if self.metadata is None:
            self.metadata = {}

class QueryProcessor:
    """Query and results processor"""
    
    def __init__(
        self,
        memory: Optional[DeepResearchMemory] = None,
        llm_factory: Optional[LLMFactory] = None
    ):
        """
        Initialize the processor
        
        Args:
            memory: Memory system
            llm_factory: LLM factory
        """
        self.memory = memory or DeepResearchMemory()
        self.llm = (llm_factory or LLMFactory()).create_llm()
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict] = None,
        depth: int = 3,
        breadth: int = 2
    ) -> List[ProcessedResult]:
        """
        Process a query and return results
        
        Args:
            query: Main query
            context: Additional context
            depth: Search depth
            breadth: Search breadth
            
        Returns:
            List[ProcessedResult]: Processed results
        """
        # Generate sub-queries
        sub_queries = await self._generate_sub_queries(query, context)
        
        # Process results
        results = []
        for sub_query in sub_queries[:breadth]:
            # Search relevant history
            history = self.memory.get_relevant_history(sub_query)
            
            # Process results
            processed = await self._process_results(
                sub_query,
                history,
                depth
            )
            results.extend(processed)
        
        # Update memory
        self.memory.add_to_context({
            "query": query,
            "sub_queries": sub_queries,
            "results": results
        })
        
        return results
    
    async def _generate_sub_queries(
        self,
        query: str,
        context: Optional[Dict] = None
    ) -> List[str]:
        """
        Generate sub-queries based on main query
        
        Args:
            query: Main query
            context: Additional context
            
        Returns:
            List[str]: List of sub-queries
        """
        # Retrieve relevant history
        history = self.memory.get_relevant_history(query)
        
        # Use template for sub-query generation
        prompt = SUBQUERY_GENERATION_PROMPT.format(
            query=query,
            context=context or {},
            max_subqueries=5,
            min_query_length=10
        )
        
        response = await self.llm.agenerate(prompt)
        return self._parse_sub_queries(response.generations[0].text)
    
    async def _process_results(
        self,
        query: str,
        history: List[Dict],
        depth: int
    ) -> List[ProcessedResult]:
        """
        Process raw results
        
        Args:
            query: Sub-query
            history: Relevant history
            depth: Search depth
            
        Returns:
            List[ProcessedResult]: Processed results
        """
        # Use template for analysis
        prompt = RESULT_ANALYSIS_PROMPT.format(
            query=query,
            history=history,
            depth=depth
        )
        
        # Process results using LLM
        response = await self.llm.agenerate(prompt)
        
        # Parse and return processed results
        return self._parse_processed_results(response)
    
    def _parse_sub_queries(self, text: str) -> List[str]:
        """
        Parse LLM text to extract sub-queries
        
        Args:
            text: LLM text
            
        Returns:
            List[str]: List of sub-queries
        """
        lines = text.strip().split('\n')
        queries = [
            line.strip().strip('- ').strip('"\'')
            for line in lines
            if line.strip()
        ]
        return queries[:5]  # Limit to 5 sub-queries
    
    def _parse_processed_results(self, response) -> List[ProcessedResult]:
        """
        Parse LLM text to extract results
        
        Args:
            response: LLM response
            
        Returns:
            List[ProcessedResult]: List of processed results
        """
        results = []
        current_result = {}
        
        for line in response.generations[0].text.strip().split('\n'):
            line = line.strip()
            
            if not line:
                continue
                
            if line == '---':
                if current_result:
                    results.append(ProcessedResult(**current_result))
                    current_result = {}
                continue
                
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == 'title':
                    current_result['title'] = value
                elif key == 'url':
                    current_result['url'] = value
                elif key == 'content':
                    current_result['content'] = value
                elif key == 'score':
                    try:
                        current_result['score'] = float(value)
                    except:
                        current_result['score'] = 0.0
                elif key == 'metadata':
                    try:
                        current_result['metadata'] = eval(value)
                    except:
                        current_result['metadata'] = {}
        
        if current_result:
            results.append(ProcessedResult(**current_result))
            
        return results

class ResearchProcessor:
    """
    Processes and analyzes research results
    """
    
    def __init__(self, llm_factory: LLMFactory):
        """
        Initialize research processor
        
        Args:
            llm_factory: LLM factory
        """
        self.llm_factory = llm_factory
        self.plan_generator = PlanGenerator(llm_factory)
        self.source_evaluator = SourceEvaluator(llm_factory)
        self.fact_checker = FactChecker(llm_factory)
        self.confidence_calibrator = ConfidenceCalibrator(llm_factory)
        
    async def process_research(
        self,
        query: str,
        results: Dict[str, Any],
        time_budget: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process research results
        
        Args:
            query: Research query
            results: Research results
            time_budget: Time budget in seconds
            
        Returns:
            Processed results
        """
        try:
            # Generate research plan
            plan = await self.plan_generator.generate_plan(query, time_budget)
            
            # Evaluate sources
            evaluated_results = await self.source_evaluator.evaluate_sources(results)
            
            # Cross-validate findings
            validated_results = await self.fact_checker.cross_validate(evaluated_results)
            
            # Calibrate confidence
            calibrated_results = await self.confidence_calibrator.calibrate(validated_results)
            
            # Update results with enhanced information
            processed_results = {
                **calibrated_results,
                "research_plan": plan,
                "confidence_metrics": {
                    "source_quality": calibrated_results.get("source_quality", 0),
                    "fact_consistency": calibrated_results.get("fact_consistency", 0),
                    "overall_confidence": calibrated_results.get("overall_confidence", 0)
                }
            }
            
            return processed_results
            
        except Exception as e:
            return {"error": str(e)}
            
    async def filter_by_constraints(
        self,
        results: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter results by constraints
        
        Args:
            results: Research results
            constraints: Research constraints
            
        Returns:
            Filtered results
        """
        try:
            filtered_results = results.copy()
            
            # Apply time constraints
            if "timeframe" in constraints:
                filtered_results["results"] = [
                    r for r in filtered_results["results"]
                    if self._is_in_timeframe(r, constraints["timeframe"])
                ]
                
            # Apply geography constraints
            if "geography" in constraints:
                filtered_results["results"] = [
                    r for r in filtered_results["results"]
                    if self._is_in_geography(r, constraints["geography"])
                ]
                
            return filtered_results
            
        except Exception as e:
            return {"error": str(e)}
            
    def _is_in_timeframe(self, result: Dict[str, Any], timeframe: Dict[str, Any]) -> bool:
        """
        Check if result is within timeframe
        
        Args:
            result: Search result
            timeframe: Timeframe constraints
            
        Returns:
            Whether result is within timeframe
        """
        if "timestamp" not in result:
            return False
            
        timestamp = result["timestamp"]
        
        if "start" in timeframe and timestamp < timeframe["start"]:
            return False
            
        if "end" in timeframe and timestamp > timeframe["end"]:
            return False
            
        return True
        
    def _is_in_geography(self, result: Dict[str, Any], geography: Dict[str, Any]) -> bool:
        """
        Check if result is within geography
        
        Args:
            result: Search result
            geography: Geography constraints
            
        Returns:
            Whether result is within geography
        """
        if "location" not in result:
            return False
            
        location = result["location"].lower()
        
        if "countries" in geography:
            return any(country.lower() in location for country in geography["countries"])
            
        if "regions" in geography:
            return any(region.lower() in location for region in geography["regions"])
            
        return True

def get_default_processor() -> QueryProcessor:
    """
    Returns a processor instance with default configuration
    
    Returns:
        QueryProcessor: Configured processor
    """
    return QueryProcessor() 