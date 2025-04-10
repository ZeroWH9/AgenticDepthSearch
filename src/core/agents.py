"""
Agent module for the research system
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from src.core.memory import DeepResearchMemory, MemoryManager
from src.integrations.search import CustomTavilySearch
from src.integrations.llm import LLMFactory
from src.core.processor import ResearchProcessor
from src.prompts.agent_prompts import (
    AGENT_SYSTEM_PROMPT,
    AGENT_TOOL_PROMPT,
    AGENT_PLANNING_PROMPT,
    AGENT_EVALUATION_PROMPT,
    AGENT_SYNTHESIS_PROMPT
)
from src.prompts.processing_prompts import SUBQUERY_GENERATION_PROMPT, FACT_CHECKING_PROMPT
from src.prompts.research_prompts import (
    DEPTH_ANALYSIS_PROMPT,
    SEARCH_STRATEGY_PROMPT,
    RESULT_EVALUATION_PROMPT,
    RESEARCH_PLAN_PROMPT,
    SYNTHESIS_REPORT_PROMPT
)
from src.utils.config import settings
from src.utils.metrics import metrics_collector
import asyncio
import json
import logging
import re # Manter import re

# Configure logging based on settings
log_level_from_settings = getattr(logging, settings.log_level, logging.INFO)
logging.basicConfig(level=log_level_from_settings, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepResearchAgent:
    """
    Main agent for the research system
    """
    
    def __init__(
        self,
        llm_factory: Optional[LLMFactory] = None,
        processor: Optional[ResearchProcessor] = None,
        memory: Optional[DeepResearchMemory] = None,
        search_client: Optional[CustomTavilySearch] = None
    ):
        """
        Initialize the agent
        
        Args:
            llm_factory: LLM factory to use
            processor: Research processor to use
            memory: Memory system to use
            search_client: Search client to use
        """
        self.llm_factory = llm_factory or LLMFactory()
        self.processor = processor or ResearchProcessor(self.llm_factory)
        self.memory = memory or DeepResearchMemory()
        self.search_client = search_client or CustomTavilySearch(settings.tavily.api_key)
        self.metrics = metrics_collector
        
        # Settings for sub-query generation
        self.max_subqueries = settings.research.max_sources // 2  # Half of max sources
        self.min_query_length = 10  # Minimum length for each sub-query
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create agent
        self.agent = self._create_agent()
    
    def _create_tools(self) -> List[Tool]:
        """
        Create tools for the agent
        
        Returns:
            List[Tool]: List of tools
        """
        return [
            Tool(
                name="search",
                func=self._search,
                description="Search for information"
            ),
            Tool(
                name="analyze",
                func=self._analyze,
                description="Analyze information"
            ),
            Tool(
                name="generate_subqueries",
                func=self._generate_subqueries,
                description="Generate sub-queries"
            ),
            Tool(
                name="get_memory",
                func=self._get_memory,
                description="Get memory"
            )
        ]
    
    def _create_agent(self) -> AgentExecutor:
        """
        Create the agent executor
        
        Returns:
            AgentExecutor: Agent executor
        """
        # Prepare tool names and descriptions
        tool_names = [tool.name for tool in self.tools]
        tool_descriptions = [tool.description for tool in self.tools]
        
        # Create the system message template
        system_template = AGENT_SYSTEM_PROMPT.template
        system_message = SystemMessagePromptTemplate.from_template(
            system_template,
            input_variables=["tools", "tool_names"]
        )
        
        # Create the human message template
        human_template = "{input}"
        human_message = HumanMessagePromptTemplate.from_template(human_template)
        
        # Create the chat prompt template
        prompt = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            human_message,
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Get LLM with streaming disabled for multiple prompts
        llm = self.llm_factory.get_llm(streaming=False)
        
        # Create agent with streaming disabled
        agent = create_structured_chat_agent(
            llm=llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create executor with streaming disabled
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=settings.agent.verbose,
            handle_parsing_errors=True,
            max_iterations=settings.agent.max_iterations
        )
    
    async def _search(self, query: str) -> Dict[str, Any]:
        """
        Search for information
        
        Args:
            query: Search query
            
        Returns:
            Dict[str, Any]: Search results
        """
        try:
            logger.debug(f"Starting search for query: {query}")
            results = await self.search_client.parallel_search([query], max_results=settings.research.max_sources)
            logger.debug(f"Raw search results: {results}")
            
            if not results or not results[0]:
                logger.debug("No results found")
                return {
                    "query": query,
                    "results": [],
                    "error": "No results found"
                }
            
            # Format results
            formatted_results = []
            for result in results[0]:
                if isinstance(result, dict):
                    formatted_result = {
                        "title": str(result.get("title", "No title")),
                        "url": str(result.get("url", "")),
                        "content": str(result.get("content", "No content")),
                        "score": float(result.get("score", 0.0))
                    }
                    formatted_results.append(formatted_result)
            
            logger.debug(f"Formatted results: {formatted_results}")
            return {
                "query": query,
                "results": formatted_results
            }
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return {
                "query": query,
                "results": [],
                "error": str(e)
            }
    
    async def _analyze(self, results: List[Dict[str, Any]], query: Optional[str] = None) -> str:
        """Analyze search results using the narrative prompt and return a string."""
        try:
   
            if not isinstance(results, list):
                logger.error(f"Results must be a list, received: {type(results)}")
                results = []

            # Sanitize results to ensure valid JSON for the prompt
            sanitized_results = []
            for result in results:
                if isinstance(result, dict):
                    sanitized_result = {
                        "title": str(result.get("title", "")),
                        "url": str(result.get("url", "")),
                        "content_snippet": str(result.get("content", ""))[:500] + "...",
                        "score": float(result.get("score", 0.0))
                    }
                    sanitized_results.append(sanitized_result)

      
            prompt = RESULT_EVALUATION_PROMPT.format(
                query=query or "General research",
                results=json.dumps(sanitized_results, ensure_ascii=False, indent=2)
            )

            logger.debug("Sending prompt for narrative analysis")
            response = await self.llm_factory.get_llm(streaming=False).ainvoke(prompt)
            logger.debug("Narrative analysis received.")

          
            analysis_narrative = response.strip()
           
            analysis_narrative = re.sub(r"^```(?:json|\s*)?\n?", "", analysis_narrative)
            analysis_narrative = re.sub(r"\n?```$", "", analysis_narrative)

            return analysis_narrative.strip()

        except Exception as e:
            logger.error(f"Error during narrative analysis: {e}")
            return f"Error generating narrative analysis: {str(e)}"
    
    async def _generate_subqueries(self, query: str, memories: List[Dict[str, Any]]) -> List[str]:
        """Generate sub-queries using the JSON prompt."""
        try:
          
            query = query.strip()
            if not query:
                logger.warning("Empty query received")
                return []
            
    
            sanitized_memories = []
            for memory in memories:
                if isinstance(memory, dict):
                    sanitized_memory = {
                        "query": str(memory.get("query", "")),
                        "timestamp": str(memory.get("timestamp", "")),
                        "content": str(memory.get("content", ""))
                    }
                    sanitized_memories.append(sanitized_memory)

            prompt = SEARCH_STRATEGY_PROMPT.format(
                query=query,
                context=json.dumps(sanitized_memories, ensure_ascii=False)
            )
            

            logger.debug("Sending prompt for sub-query generation")
            response_text = await self.llm_factory.get_llm(streaming=False).ainvoke(prompt)
            

  
            cleaned_response = response_text.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                 cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()


            try:
                strategy = json.loads(cleaned_response)
                logger.debug(f"Parsed strategy: {strategy}")
                

                subqueries = []
                if "query_variations" in strategy:
                    subqueries.extend(strategy["query_variations"])
                if "search_terms" in strategy:
                    for term in strategy["search_terms"]:
                        if isinstance(term, dict) and "term" in term:
                            subqueries.append(term["term"])
                

                if not subqueries:
                    subqueries = [query]
                    

                valid_subqueries = []
                for subquery in subqueries:
                    if isinstance(subquery, str) and len(subquery.strip()) >= self.min_query_length:
                        valid_subqueries.append(subquery.strip())
                        
                if not valid_subqueries:
                    valid_subqueries = [query]
                    
                return valid_subqueries
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing cleaned JSON response: {e}")
                logger.debug(f"Cleaned response that caused error: {cleaned_response}")
                logger.debug(f"Original response: {response_text}")
                
     
                try:
                    start = cleaned_response.find("{")
                    end = cleaned_response.rfind("}") + 1
                    if start >= 0 and end > start:
                        json_str = cleaned_response[start:end]
                        strategy = json.loads(json_str)
                        subqueries = strategy.get("query_variations", [query])
                        return [q for q in subqueries if isinstance(q, str) and len(q.strip()) >= self.min_query_length]
                    else:
                         logger.warning("Could not find JSON object within the cleaned response.")
                         return [query] # Fallback para query original
                except Exception as e2:
                    logger.error(f"Error trying to recover JSON after cleaning: {e2}")
                    return [query]
                    
        except Exception as e:
            logger.error(f"Error during sub-query generation: {e}")
            return [query]  # Return original query in case of error
    
    def _parse_subqueries(self, text: str) -> List[str]:
        """
        Parse LLM text to extract sub-queries
        
        Args:
            text: LLM text
            
        Returns:
            List[str]: List of sub-queries
        """
        print(f"[DEBUG] Parsing text for sub-queries: {text}")
        subqueries = []
        
        for line in text.strip().split('\n'):
            line = line.strip()
            
            if not line:
                continue
                
            # Skip common formatting markers
            if any(marker in line.lower() for marker in ['---', 'sub-queries:', 'subqueries:', 'queries:']):
                continue
                
            # Skip lines that are too short
            if len(line) < self.min_query_length:
                continue
                
            subqueries.append(line)
            
        print(f"[DEBUG] Parsed sub-queries: {subqueries}")
        return subqueries
    
    def _get_memory(self, query: str) -> Dict[str, Any]:
        """
        Get memory
        
        Args:
            query: Query to search for
            
        Returns:
            Dict[str, Any]: Memory data
        """
        return self.memory.get_memory(query)
    
    async def research(self, query: str, breadth: int = 2, time_budget: int = 60) -> Dict[str, Any]:
        """Conduct research using the JSON prompts, limiting sub-queries by breadth."""
        try:
            # Sanitize a query
            query = query.strip()
            if not query:
                return {
                    "query": "",
                    "results": [],
                    "analysis": {
                        "relevance_scores": {},
                        "source_credibility": {},
                        "quality_assessment": {},
                        "gaps": ["Empty query provided"],
                        "biases": [],
                        "overall_quality": 0.0
                    }
                }
            
            logger.debug("Validating query...")
            # Validate query
            if len(query) < self.min_query_length:
                return {
                    "query": query,
                    "results": [],
                    "analysis": {
                        "relevance_scores": {},
                        "source_credibility": {},
                        "quality_assessment": {},
                        "gaps": [f"Query must be at least {self.min_query_length} characters long"],
                        "biases": [],
                        "overall_quality": 0.0
                    }
                }
            
            logger.debug("Getting relevant memories...")
            memories = self.memory.get_relevant_memories(query)
            
            logger.debug("Generating sub-queries...")
            # Gera *todas* as sub-queries que o LLM sugere
            all_suggested_subqueries = await self._generate_subqueries(query, memories)
            if not all_suggested_subqueries:
                logger.warning("No sub-queries generated, using original query")
                all_suggested_subqueries = [query]
            
            # ** Limitar o número de sub-queries com base no breadth **
            subqueries = all_suggested_subqueries[:breadth] 
            logger.debug(f"Generated sub-queries (limited by breadth={breadth}): {subqueries}")
            
            logger.debug("Starting search for each sub-query...")
            all_results = []
            search_tasks = [self._search(subq) for subq in subqueries]
            search_results_list = await asyncio.gather(*search_tasks)
            for result_dict in search_results_list:
                 if isinstance(result_dict, dict) and "results" in result_dict:
                    all_results.extend(result_dict["results"])
            logger.debug(f"Aggregated search results: {len(all_results)} results.")
            
            if not all_results:
                return {
                    "query": query, "results": [], "analysis": {"gaps": ["No results found"]}
                }
            
            logger.debug("Analyzing results (narrative)...")
            analysis = await self._analyze(all_results, query)
            logger.debug("Narrative analysis completed.")
            
            # Check for errors during analysis before storing
            if isinstance(analysis, dict) and 'error' in analysis:
                logger.warning(f"Skipping memory storage due to analysis error: {analysis['error']}")
                # Retorna o erro mas mantém a estrutura esperada
                return {"query": query, "results": all_results, "analysis": analysis}
            
            logger.debug("Storing in memory...")
            self.memory.add_memory(query, all_results)
            
            # Format results for display
            formatted_results = {
                "query": query,
                "results": all_results,
                "analysis": analysis, # Narrative analysis string
                "sub_queries": subqueries # Add the list of sub-queries used
            }
            logger.debug("Results formatted by the agent.")
            return formatted_results
            
        except Exception as e:
            logger.exception(f"Unexpected error in DeepResearchAgent.research: {e}") 
            return {
                "query": query,
                "results": [],
                "analysis": f"Unexpected error during agent research: {str(e)}", 
                "sub_queries": [],
                "summary": "Summary generation failed due to an unexpected error." 
            }
    
    async def research_with_constraints(
        self,
        query: str,
        time_budget: int = 60,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform research with constraints
        
        Args:
            query: Query to research
            time_budget: Time budget in seconds
            constraints: Research constraints
            
        Returns:
            Research results
        """
        try:
            # Perform base research
            results = await self.research(query, time_budget)
            
            if "error" in results:
                return results
                
            # Apply constraints if provided
            if constraints:
                filtered_results = await self._filter_results_by_constraints(
                    results["results"],
                    constraints
                )
                results["results"] = filtered_results
                
            return results
            
        except Exception as e:
            print(f"Error during constrained research: {str(e)}")
            return {
                "error": f"Error during constrained research: {str(e)}"
            }
    
    async def _filter_results_by_constraints(
        self,
        results: List[Dict[str, Any]],
        constraints: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Filter results by constraints
        
        Args:
            results: Results to filter
            constraints: Filter constraints
            
        Returns:
            Filtered results
        """
        filtered = []
        
        for result in results:
            # Check time constraints
            if "timeframe" in constraints:
                if not self._is_in_timeframe(result, constraints["timeframe"]):
                    continue
                    
            # Check geography constraints
            if "geography" in constraints:
                if not self._is_in_geography(result, constraints["geography"]):
                    continue
                    
            # Check other constraints
            if "other" in constraints:
                if not self._matches_other_constraints(result, constraints["other"]):
                    continue
                    
            filtered.append(result)
            
        return filtered
        
    def _is_in_timeframe(self, result: Dict[str, Any], timeframe: Dict[str, Any]) -> bool:
        """
        Check if result is within timeframe
        
        Args:
            result: Result to check
            timeframe: Timeframe constraints
            
        Returns:
            Whether result is within timeframe
        """
        if "timestamp" not in result:
            return False
            
        result_time = datetime.fromisoformat(result["timestamp"])
        
        if "start" in timeframe:
            start_time = datetime.fromisoformat(timeframe["start"])
            if result_time < start_time:
                return False
                
        if "end" in timeframe:
            end_time = datetime.fromisoformat(timeframe["end"])
            if result_time > end_time:
                return False
                
        return True
        
    def _is_in_geography(self, result: Dict[str, Any], geography: Dict[str, Any]) -> bool:
        """
        Check if result is within geography
        
        Args:
            result: Result to check
            geography: Geography constraints
            
        Returns:
            Whether result is within geography
        """
        if "location" not in result:
            return False
            
        result_location = result["location"].lower()
        
        if "countries" in geography:
            countries = [c.lower() for c in geography["countries"]]
            if result_location not in countries:
                return False
                
        if "regions" in geography:
            regions = [r.lower() for r in geography["regions"]]
            if result_location not in regions:
                return False
                
        return True
        
    def _matches_other_constraints(self, result: Dict[str, Any], constraints: Dict[str, Any]) -> bool:
        """
        Check if result matches other constraints
        
        Args:
            result: Result to check
            constraints: Other constraints
            
        Returns:
            Whether result matches constraints
        """
        for key, value in constraints.items():
            if key not in result:
                return False
                
            if isinstance(value, list):
                if result[key] not in value:
                    return False
            else:
                if result[key] != value:
                    return False
                    
        return True

def get_default_agent() -> DeepResearchAgent:
    """
    Returns an agent instance with default configuration
    
    Returns:
        DeepResearchAgent: Configured agent
    """
    return DeepResearchAgent(
        memory=MemoryManager(),
        search_client=CustomTavilySearch(settings.tavily.api_key),
        llm_factory=LLMFactory(),
        processor=ResearchProcessor()
    ) 