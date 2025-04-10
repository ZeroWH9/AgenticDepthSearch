"""
Research plan generation module.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from src.prompts.research_prompts import (
    RESEARCH_PLAN_PROMPT,
    DEPTH_ANALYSIS_PROMPT
)
from src.prompts.scope_prompts import (
    SCOPE_DEFINITION_PROMPT,
    SUBQUERY_GENERATION_PROMPT
)
from src.prompts.processing_prompts import (
    SOURCE_INFO_EXTRACTION_PROMPT,
    SOURCE_RELIABILITY_PROMPT
)
from src.utils.config import settings
from src.integrations.llm import LLMFactory
import json
import logging
import re

logger = logging.getLogger(__name__)

@dataclass
class ResearchPlan:
    """Research plan data structure"""
    topic: str
    scope: Dict[str, str]
    steps: List[Dict[str, str]]
    time_allocation: Dict[str, int]
    success_criteria: Dict[str, str]
    quality_requirements: Dict[str, float]

class PlanGenerator:
    """Research plan generation system"""
    
    def __init__(self, llm_factory: LLMFactory):
        """
        Initialize plan generator
        
        Args:
            llm_factory: LLM factory
        """
        self.llm_factory = llm_factory
        self.llm = llm_factory.get_llm(streaming=False)
    
    async def define_scope(self, query: str) -> Dict[str, str]:
        """
        Define research scope for a query
        
        Args:
            query: Research query
            
        Returns:
            Dict: Research scope definition (with potential 'error' key)
        """
        # Generate scope using LLM
        try:
            scope_response = await self.llm.ainvoke(
                SCOPE_DEFINITION_PROMPT.format(user_query=query)
            )
            
            # Parse and validate scope
            return self._parse_scope(scope_response, query)
        except Exception as e:
            logger.error(f"Error generating scope: {str(e)}")
            # Return fallback scope with error
            return {
                "topic": query,
                "scope": "Research scope could not be generated",
                "central_question": query
            }
    
    async def generate_plan(
        self,
        query: str,
        time_budget: int = 30,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a research plan
        
        Args:
            query: Research query
            time_budget: Time budget in minutes
            constraints: Research constraints
            
        Returns:
            Dict: Research plan
        """
        # Define scope first
        scope = await self.define_scope(query)
        
        # Generate initial plan
        initial_plan = await self._generate_initial_plan(
            query,
            time_budget,
            constraints or {}
        )
        
        # Optimize plan
        optimized_plan = await self._optimize_plan(initial_plan)
        
        # Generate execution strategy
        strategy = await self._generate_execution_strategy(optimized_plan)
        
        # Estimate times
        time_estimates = self._estimate_times(optimized_plan)
        
        return {
            "plan": optimized_plan,
            "strategy": strategy,
            "time_estimates": time_estimates
        }
    
    async def _generate_initial_plan(
        self,
        query: str,
        time_budget: int,
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate initial research plan
        
        Args:
            query: Research query
            time_budget: Time budget in minutes
            constraints: Research constraints
            
        Returns:
            Dictionary with initial plan (with potential 'error' key)
        """
        try:
            # Use template for plan generation
            prompt = RESEARCH_PLAN_PROMPT.format(
                query=query,
                context=json.dumps({"query": query, "constraints": constraints}, ensure_ascii=False)
            )
            
            # Generate plan using LLM
            response = await self.llm.ainvoke(prompt)
            
            # Parse and return plan
            plan = self._parse_plan(response)
            

            if "topic" not in plan:
                plan["topic"] = query
            
            return plan
        except Exception as e:
            logger.error(f"Error generating initial plan: {str(e)}")
            # Return basic fallback plan with error
            return {
                "topic": query,
                "objectives": ["Research the given topic"],
                "key_areas": [{"area": "General information", "importance": 1.0}],
                "search_strategies": [{"strategy": "Web search", "resources": ["Search engines"]}]
            }
    
    async def _optimize_plan(
        self,
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize research plan
        
        Args:
            plan: Initial research plan
            
        Returns:
            Dictionary with optimized plan
        """
        # Use DEPTH_ANALYSIS_PROMPT for optimization
        prompt = DEPTH_ANALYSIS_PROMPT.format(
            query=plan.get("topic", ""),
            current_results=plan,
            depth="optimization"
        )
        
        response = await self.llm.ainvoke(prompt)
        return self._parse_optimization(response)
    
    async def _generate_execution_strategy(
        self,
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate execution strategy for plan
        
        Args:
            plan: Research plan
            
        Returns:
            Dictionary with execution strategy
        """
        try:
           
            query = plan.get("topic", "")
            if not query:
                return {"error": "No topic found in plan"}
            
            context = {
                "objectives": plan.get("objectives", []),
                "key_areas": plan.get("key_areas", []),
                "constraints": plan.get("constraints", {})
            }
            
            # Use RESEARCH_PLAN_PROMPT for strategy
            prompt = RESEARCH_PLAN_PROMPT.format(
                query=query,
                context=json.dumps(context, ensure_ascii=False)
            )
            
            # Generate strategy using LLM
            response = await self.llm.ainvoke(prompt)
            strategy = self._parse_strategy(response)
            
            if "topic" not in strategy:
                strategy["topic"] = query
            
            return strategy
        except Exception as e:
            logger.error(f"Error generating execution strategy: {str(e)}")
            # Return fallback strategy with error
            return {
                "topic": plan.get("topic", ""),
                "search_strategies": [{"strategy": "Web search", "resources": ["Search engines"]}],
                "timeline": {"total_duration": "30 minutes"}
            }
    
    def _estimate_times(
        self,
        plan: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Estimate time for each component
        
        Args:
            plan: Research plan
            
        Returns:
            Dictionary with time estimates
        """
        # Simple time estimation based on plan complexity
        total_time = plan.get("time_budget", 30)
        steps = plan.get("steps", [])
        
        if not steps:
            return {}
            
        # Distribute time based on step complexity
        time_per_step = total_time / len(steps)
        return {
            step.get("name", f"step_{i}"): int(time_per_step * (1 + i * 0.1))
            for i, step in enumerate(steps)
        }
    
    def _clean_and_extract_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Helper method to clean markdown and extract JSON object."""
        cleaned_response = response.strip()
        # Remove potential markdown code block fences
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        # Attempt to parse directly
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            # If direct parse fails, try finding the first '{' and last '}'
            try:
                start = cleaned_response.find("{")
                end = cleaned_response.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = cleaned_response[start:end]
                    return json.loads(json_str)
                else:
                    return None # JSON object not found
            except json.JSONDecodeError:
                return None # Still couldn't parse

    def _parse_scope(self, response: str, query: str = "") -> Dict[str, Any]:
        """Parse scope definition response, returning dict with error on failure."""
        data = self._clean_and_extract_json(response)
        if data and isinstance(data, dict) and "central_question" in data:
            if "topic" not in data:
                 data["topic"] = data["central_question"]
            return data
        else:
            logger.error(f"Error parsing scope. Could not extract valid JSON. Received response: {response[:500]}...")
            return {
                "topic": query,
                "scope": "Could not parse scope from LLM response",
                "central_question": query,
                "error": "Invalid or missing JSON in LLM response"
            }
    
    def _parse_plan(self, response: str) -> Dict[str, Any]:
        """Parse plan response, returning dict with error on failure."""
        data = self._clean_and_extract_json(response)
        if data and isinstance(data, dict) and "objectives" in data:
            return data
        else:
            logger.error(f"Error parsing plan. Could not extract valid JSON. Received response: {response[:500]}...")
            return {
                "topic": "Unknown (parse error)",
                "objectives": ["Could not parse plan from LLM response"],
                "error": "Invalid or missing JSON in LLM response"
            }
    
    def _parse_optimization(self, response: str) -> Dict[str, Any]:
        """Parse optimization response, returning dict with error on failure."""
        data = self._clean_and_extract_json(response)
        if data and isinstance(data, dict) and "analysis" in data:
            return data
        else:
            logger.error(f"Error parsing optimization. Could not extract valid JSON. Received response: {response[:500]}...")
            return {"error": "Could not parse optimization response: Invalid or missing JSON"}
    
    def _parse_strategy(self, response: str) -> Dict[str, Any]:
        """Parse strategy response, returning dict with error on failure."""
        data = self._clean_and_extract_json(response)
        if data and isinstance(data, dict) and "search_strategies" in data:
            return data
        else:
            logger.error(f"Error parsing strategy. Could not extract valid JSON. Received response: {response[:500]}...")
            return {"error": "Could not parse strategy response: Invalid or missing JSON"} 