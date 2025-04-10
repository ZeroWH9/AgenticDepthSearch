"""
Main research class that coordinates the research process
"""
from typing import List, Dict, Optional, Any
import time
from src.core.agents import DeepResearchAgent
from src.core.memory import DeepResearchMemory
from src.core.plan_generator import PlanGenerator, ResearchPlan
from src.core.source_evaluator import SourceEvaluator, SourceEvaluation
from src.core.fact_checker import FactChecker, FactValidation
from src.core.confidence_calibrator import ConfidenceCalibrator, ConfidenceMetrics
from src.integrations.search import CustomTavilySearch
from src.utils.config import settings
from src.utils.cache import cache, Cache
from src.utils.metrics import SearchMetrics, metrics_collector
from src.integrations.llm import LLMFactory
from src.prompts.research_prompts import (
    DEPTH_ANALYSIS_PROMPT,
    RESEARCH_PLAN_PROMPT,
    RESULT_PROCESSING_PROMPT,
    INFORMATION_EXTRACTION_PROMPT,
    RELATIONSHIP_ANALYSIS_PROMPT,
    SYNTHESIS_PROMPT
)
from src.prompts.synthesis_prompts import FINAL_SUMMARY_PROMPT
from src.core.processor import ResearchProcessor
import asyncio
import logging
import json

logger = logging.getLogger(__name__)

class DeepResearcher:
    """Main research class"""
    
    def __init__(
        self,
        llm_factory: Optional[LLMFactory] = None,
        search_client: Optional[CustomTavilySearch] = None,
        memory: Optional[DeepResearchMemory] = None,
        processor: Optional[ResearchProcessor] = None,
        plan_generator: Optional[PlanGenerator] = None,
        source_evaluator: Optional[SourceEvaluator] = None,
        fact_checker: Optional[FactChecker] = None,
        confidence_calibrator: Optional[ConfidenceCalibrator] = None,
        agent: Optional[DeepResearchAgent] = None
    ):
        """
        Initialize the researcher
        
        Args:
            llm_factory: LLM factory instance
            search_client: Search client instance
            memory: Memory system instance
            processor: Research processor instance
            plan_generator: Research plan generator
            source_evaluator: Source evaluation system
            fact_checker: Fact checking system
            confidence_calibrator: Confidence calibration system
            agent: Research agent instance
        """
        logger.debug("Initializing DeepResearcher...")
        
        # Core components
        self.memory = memory or DeepResearchMemory()
        self.search_client = search_client or CustomTavilySearch(settings.tavily.api_key)
        self.llm_factory = llm_factory or LLMFactory()
        
        # Research enhancement components
        self.plan_generator = plan_generator or PlanGenerator(self.llm_factory)
        self.source_evaluator = source_evaluator or SourceEvaluator(self.llm_factory)
        self.fact_checker = fact_checker or FactChecker(self.llm_factory, self.search_client)
        self.confidence_calibrator = confidence_calibrator or ConfidenceCalibrator(self.llm_factory)
        
        # Main agent
        self.agent = agent or DeepResearchAgent(
            memory=self.memory,
            search_client=self.search_client,
            llm_factory=self.llm_factory
        )
        
        # Utilities
        self.metrics = metrics_collector
        self.cache = Cache()
        logger.debug("DeepResearcher initialized successfully")
    
    async def research(
        self,
        query: str,
        depth: int = 3,
        breadth: int = 2,
        time_budget: Optional[int] = None,
        constraints: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Perform research
        
        Args:
            query: Research query
            depth: Search depth
            breadth: Number of sub-queries
            time_budget: Time budget in minutes
            constraints: Research constraints
            
        Returns:
            Dict with research results
        """
        logger.debug("Starting research in DeepResearcher...")
        logger.debug(f"Query: {query}")

        # 1. Sanitize and Validate Query Length EARLY
        query = query.strip()
        if not query:
            logger.debug("Empty query provided.")
            return {
                "error": "Empty query provided",
                "query": "",
                "results": [],
                "analysis": {}, "plan": {}, "confidence_metrics": {}
            }

        # Assuming min_query_length is accessible via agent or settings
        # Let's default to 10 based on the previous error message if not easily accessible
        min_len = getattr(self.agent, 'min_query_length', 10)
        if len(query) < min_len:
            logger.debug(f"Query too short: {len(query)} < {min_len}")
            error_msg = f"Query must be at least {min_len} characters long"
            return {
                "error": error_msg,
                "query": query,
                "results": [],
                "analysis": {"gaps": [error_msg]},
                "plan": {},
                "confidence_metrics": {}
            }

        cache_key = f"{query}_{depth}_{breadth}"
        cached_results = self.cache.get(cache_key)
        if cached_results:
            logger.debug("Results found in cache")
            return cached_results
        
        start_time = self.metrics.start_timer()
        
        try:
            # 1. Generate research plan
            logger.debug("Generating research plan...")
            try:
                plan = await self.plan_generator.generate_plan(
                    query=query,
                    time_budget=time_budget,
                    constraints=constraints
                )
            except Exception as plan_error:
                logger.warning(f"Error generating research plan: {str(plan_error)}")
                plan = {"topic": query, "scope": {}, "steps": []}
            
            # 2. Execute research with plan
            agent = DeepResearchAgent(
                memory=self.memory,
                search_client=self.search_client,
                llm_factory=self.llm_factory
            )
            
            # Passar breadth para o método research do agente
            results = await agent.research(query=query, breadth=breadth)

            # analysis is now the narrative analysis string
            analysis_narrative = results.get("analysis", "Analysis not available.")

            # Check if there was an error in narrative analysis
            if analysis_narrative.startswith("Error generating narrative analysis"):
                logger.error(f"Error returned by agent analysis: {analysis_narrative}")
                has_analysis_errors = True
            else:
                has_analysis_errors = False

            if "error" not in results and "results" in results and results["results"] and not has_analysis_errors:
                # 3. Evaluate sources (already done, resulted in analysis_narrative)
                logger.debug("Narrative analysis of sources completed.")

                # 4 & 5. FactCheck and Confidence Calibration - SKIPPED
                # Would require extracting claims/findings from narrative analysis or raw results
                logger.debug("Skipping FactCheck and Confidence Calibration (requires refactoring with narrative analysis).")
                fact_check_results = {"status": "Skipped"}
                confidence_metrics = {"status": "Skipped"}
                results["fact_checks"] = fact_check_results
                results["confidence_metrics"] = confidence_metrics

                # 6. Update results with enhanced information
                results["plan"] = plan
                # 'analysis' já contém a string narrativa

                # 7. Generate Final Summary using narrative analysis
                logger.debug("Generating final summary...")
                summary = "Summary generation skipped due to errors."
                
             
                search_results_summary = "" 
                raw_results_list = results.get("results", [])
                if raw_results_list:
                     search_results_summary = "\n".join([
                         f"- {res.get('title', 'No Title')}: {res.get('url', 'No URL')}"
                         for res in raw_results_list[:10] # LIMIT 10
                     ])
                else:
                     search_results_summary = "No raw search results available."

                try:
                    summary_prompt = FINAL_SUMMARY_PROMPT.format(
                        query=query,
                        analysis=analysis_narrative, # Passa a string narrativa
                        search_results_summary=search_results_summary,
                        fact_checks=json.dumps(fact_check_results, indent=2, ensure_ascii=False), # Passa status "Skipped"
                        confidence_metrics=json.dumps(confidence_metrics, indent=2, ensure_ascii=False) # Passa status "Skipped"
                    )
                    summary_llm = self.llm_factory.get_llm(streaming=False)
                    summary = await summary_llm.ainvoke(summary_prompt)
                    logger.debug("Final summary generated.")
                except Exception as summary_error:
                    logger.error(f"Error generating final summary: {summary_error}")
                    summary = f"Error generating final summary: {summary_error}"
                results["summary"] = summary

            elif "error" in results:
                 logger.warning(f"Error returned by agent: {results['error']}")
                 results["summary"] = "Could not generate summary due to agent error."
            else:
                 logger.warning("Agent did not return valid results or analysis failed.")
                 results["summary"] = "Could not generate summary due to lack of results or analysis failure."

            # 8. Add to memory (Raw results only)
            logger.debug("Adding raw results to memory...")
            if "results" in results and results["results"]:
                self.memory.add_memory(query, results["results"])

            self.cache.set(cache_key, results)
            
            self.metrics.add_metrics(
                query=query,
                duration=self.metrics.get_duration(start_time),
                num_results=len(results.get("results", []))
            )
            
            logger.debug("Research completed")
            return results
            
        except Exception as e:
            self.metrics.add_metrics(
                query=query,
                duration=self.metrics.get_duration(start_time),
                num_results=0,
                error=str(e)
            )
            
            logger.exception(f"Error during research: {str(e)}")
            return {
                "error": f"An error occurred during research: {str(e)}",
                "results": [],
                "analysis": {},
                "plan": {},
                "confidence_metrics": {}
            }
    
    async def analyze_results(
        self,
        query: str,
        results: List[Dict],
        depth: int
    ) -> Dict[str, Any]:
        """
        Analisa os resultados da pesquisa usando o template DEPTH_ANALYSIS_PROMPT
        """
        prompt = DEPTH_ANALYSIS_PROMPT.format(
            query=query,
            current_results=results,
            depth=depth
        )
        
        # Use non-streaming LLM
        llm = self.llm_factory.get_llm(streaming=False)
        response = await llm.ainvoke(prompt)
        return self._parse_analysis_response(response)
    
    async def synthesize_findings(
        self,
        analysis: Dict[str, Any],
        validations: List[FactValidation],
        confidence_metrics: ConfidenceMetrics
    ) -> Dict[str, Any]:
        """
        Synthesize research findings
        
        Args:
            analysis: Research analysis
            validations: Fact validations
            confidence_metrics: Confidence metrics
            
        Returns:
            Dict[str, Any]: Synthesized findings
        """
        prompt = SYNTHESIS_PROMPT.format(
            analysis=str(analysis),
            validations=str([v.__dict__ for v in validations]),
            confidence_metrics=str(confidence_metrics.__dict__)
        )
        
        # Use non-streaming LLM
        llm = self.llm_factory.get_llm(streaming=False)
        response = await llm.ainvoke(prompt)
        return eval(response)
    
    def get_relevant_history(
        self,
        query: str,
        max_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get relevant history for a query
        
        Args:
            query: Query to get history for
            max_results: Maximum number of results
            
        Returns:
            List[Dict[str, Any]]: Relevant history
        """
        return self.memory.get_relevant_memories(query, max_results)
        
    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """
        Parse analysis response
        
        Args:
            response: Analysis response
            
        Returns:
            Dict[str, Any]: Parsed response
        """
        try:
            return eval(response)
        except:
            return {"analysis": response}

def get_default_researcher() -> DeepResearcher:
    """
    Returns a researcher instance with default configuration
    
    Returns:
        DeepResearcher: Configured researcher
    """
    return DeepResearcher() 