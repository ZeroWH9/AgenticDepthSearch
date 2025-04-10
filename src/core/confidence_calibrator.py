"""
Confidence calibration module
"""
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from src.integrations.llm import LLMFactory
from src.prompts.evaluation_prompts import CONFIDENCE_CALIBRATION_PROMPT
from src.prompts.processing_prompts import (
    CONFIDENCE_REPORT_PROMPT,
    CONFIDENCE_METRICS_PROMPT,
    FACT_CHECKING_ADJUSTMENT_PROMPT
)

@dataclass
class ConfidenceMetrics:
    """Data structure for confidence metrics"""
    overall_confidence: float
    source_reliability: float
    consistency_score: float
    evidence_strength: float
    recommendations: List[str]  # Recommendations to increase confidence (Tradução mantida como no original)

class ConfidenceCalibrator:
    """Confidence calibration system"""
    
    def __init__(self, llm_factory: LLMFactory):
        """
        Initialize confidence calibrator
        
        Args:
            llm_factory: LLM factory
        """
        self.llm_factory = llm_factory
        self.llm = llm_factory.get_llm(streaming=False)
    
    async def calibrate(
        self,
        results: List[Dict[str, Any]],
        fact_checking: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calibrate confidence levels
        
        Args:
            results: List of search results
            fact_checking: Fact checking results
            
        Returns:
            Dictionary with calibrated confidence metrics
        """
        try:
            # Calculate base confidence
            base_confidence = await self._calculate_base_confidence(results)
            
            # Adjust for fact checking
            adjusted_confidence = await self._adjust_for_fact_checking(
                base_confidence,
                fact_checking
            )
            
            # Generate confidence report
            report = await self._generate_confidence_report(
                base_confidence,
                adjusted_confidence
            )
            
            return {
                "base_confidence": base_confidence,
                "adjusted_confidence": adjusted_confidence,
                "report": report
            }
            
        except Exception as e:
            return {"error": str(e)}
            
    async def _calculate_base_confidence(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate base confidence metrics
        
        Args:
            results: List of search results
            
        Returns:
            Dictionary with base confidence metrics
        """
        # Use template for confidence metrics
        prompt = CONFIDENCE_METRICS_PROMPT.format(
            results=results
        )
        
        response = await self.llm.ainvoke(prompt)
        return self._parse_confidence_metrics(response)
        
    def _parse_confidence_metrics(self, text: str) -> Dict[str, float]:
        """
        Parse LLM text to extract confidence metrics
        
        Args:
            text: LLM text
            
        Returns:
            Dict[str, float]: Confidence metrics
        """
        metrics = {}
        
        for line in text.strip().split('\n'):
            line = line.strip()
            
            if not line:
                continue
                
            if line == '---':
                break
                
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key in ['source_reliability', 'information_consistency', 
                          'evidence_strength', 'coverage_depth', 'overall_confidence']:
                    try:
                        metrics[key] = float(value)
                    except:
                        metrics[key] = 0.0
                elif key == 'notes':
                    metrics['notes'] = value
                    
        return metrics
        
    async def _adjust_for_fact_checking(
        self,
        base_confidence: Dict[str, float],
        fact_checking: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Adjust confidence based on fact checking
        
        Args:
            base_confidence: Base confidence metrics
            fact_checking: Fact checking results
            
        Returns:
            Dictionary with adjusted confidence metrics
        """
        # Use template for fact checking adjustment
        prompt = FACT_CHECKING_ADJUSTMENT_PROMPT.format(
            base_confidence=base_confidence,
            fact_checking=fact_checking
        )
        
        response = await self.llm.ainvoke(prompt)
        return self._parse_fact_checking_adjustment(response)
        
    def _parse_fact_checking_adjustment(self, text: str) -> Dict[str, float]:
        """
        Parse LLM text to extract fact checking adjustments
        
        Args:
            text: LLM text
            
        Returns:
            Dict[str, float]: Adjusted confidence metrics
        """
        metrics = {}
        
        for line in text.strip().split('\n'):
            line = line.strip()
            
            if not line:
                continue
                
            if line == '---':
                break
                
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                if key in ['source_reliability', 'consistency', 
                          'evidence_strength', 'coverage_depth']:
                    try:
                        metrics[key] = float(value)
                    except:
                        metrics[key] = 0.0
                elif key == 'notes':
                    metrics['notes'] = value
                    
        return metrics
        
    async def _generate_confidence_report(
        self,
        findings: List[Dict],
        sources: List[Dict]
    ) -> Dict:
        """
        Generate confidence calibration report
        
        Args:
            findings: List of research findings
            sources: List of sources used
            
        Returns:
            Dict: Confidence calibration report
        """
        # Use template for report
        prompt = CONFIDENCE_CALIBRATION_PROMPT.format(
            findings=findings,
            sources=sources
        )
        
        # Generate report using LLM
        response = await self.llm.ainvoke(prompt)
        
        # Parse and return report
        return self._parse_confidence_report(response)
    
    async def calibrate_confidence(
        self,
        findings: List[Dict[str, Any]],
        sources: List[Dict[str, Any]],
        validations: List[Dict[str, Any]]
    ) -> ConfidenceMetrics:
        """
        Calibrate confidence level for research findings
        
        Args:
            findings: List of research findings
            sources: List of sources used
            validations: List of fact validations
            
        Returns:
            ConfidenceMetrics: Calculated confidence metrics
        """
        prompt = CONFIDENCE_CALIBRATION_PROMPT.format(
            findings=str(findings),
            sources=str(sources),
            validations=str(validations)
        )
        
        response = await self.llm.ainvoke(prompt)
        
        # Parse response and create ConfidenceMetrics object
        try:
            data = eval(response)
            return ConfidenceMetrics(
                overall_confidence=data.get("overall_confidence", 0.0),
                source_quality_score=data.get("source_quality_score", 0.0),
                cross_validation_score=data.get("cross_validation_score", 0.0),
                consistency_score=data.get("consistency_score", 0.0),
                uncertainty_areas=data.get("uncertainty_areas", []),
                confidence_factors=data.get("confidence_factors", {}),
                recommendations=data.get("recommendations", [])
            )
        except:
            # Fallback to default values
            return ConfidenceMetrics(
                overall_confidence=0.5,
                source_quality_score=0.0,
                cross_validation_score=0.0,
                consistency_score=0.0,
                uncertainty_areas=["Could not parse confidence metrics"],
                confidence_factors={},
                recommendations=["Consider validating information manually"]
            )
    
    async def adjust_confidence(
        self,
        metrics: ConfidenceMetrics,
        new_evidence: Dict[str, Any]
    ) -> ConfidenceMetrics:
        """
        Adjust confidence based on new evidence
        
        Args:
            metrics: Current confidence metrics
            new_evidence: New evidence to consider
            
        Returns:
            Updated ConfidenceMetrics object
        """
        prompt = CONFIDENCE_CALIBRATION_PROMPT.format(
            current_metrics=str(metrics.__dict__),
            new_evidence=str(new_evidence)
        )
        
        response = await self.llm.ainvoke(prompt)
        updated_metrics = eval(response)
        
        return ConfidenceMetrics(
            overall_confidence=updated_metrics["overall_confidence"],
            source_quality_score=updated_metrics["source_quality_score"],
            cross_validation_score=updated_metrics["cross_validation_score"],
            consistency_score=updated_metrics["consistency_score"],
            uncertainty_areas=updated_metrics["uncertainty_areas"],
            confidence_factors=updated_metrics["confidence_factors"],
            recommendations=updated_metrics["recommendations"]
        )

    def calculate_source_quality(
        self,
        sources: List[Dict]
    ) -> float:
        """
        Calculate overall source quality score
        
        Args:
            sources: List of source dictionaries
            
        Returns:
            float: Source quality score (0-1)
        """
        if not sources:
            return 0.0
            
        total_score = 0.0
        weights = {
            "credibility": 0.4,
            "relevance": 0.3,
            "recency": 0.2,
            "depth": 0.1
        }
        
        for source in sources:
            score = 0.0
            score += source.get("credibility", 0.0) * weights["credibility"]
            score += source.get("relevance", 0.0) * weights["relevance"]
            score += source.get("recency", 0.0) * weights["recency"]
            score += source.get("depth", 0.0) * weights["depth"]
            total_score += score
            
        return min(1.0, total_score / len(sources))
    
    def calculate_evidence_strength(
        self,
        findings: List[Dict],
        validations: List[Dict]
    ) -> float:
        """
        Calculate evidence strength score
        
        Args:
            findings: Research findings
            validations: Validation results
            
        Returns:
            float: Evidence strength score (0-1)
        """
        if not findings or not validations:
            return 0.0
            
        total_strength = 0.0
        for finding in findings:
            # Find corresponding validation
            validation = next(
                (v for v in validations if v.get("claim") == finding.get("claim")),
                None
            )
            
            if validation:
                # Calculate strength based on validation metrics
                strength = 0.0
                strength += validation.get("agreement_level", 0.0) * 0.4
                strength += validation.get("confidence", 0.0) * 0.4
                strength += (1 - len(validation.get("contradictions", [])) * 0.1) * 0.2
                total_strength += strength
                
        return min(1.0, total_strength / len(findings))
    
    def identify_uncertainty_areas(
        self,
        findings: List[Dict],
        validations: List[Dict]
    ) -> List[str]:
        """
        Identify areas of uncertainty in the research
        
        Args:
            findings: Research findings
            validations: Validation results
            
        Returns:
            List[str]: List of uncertainty areas
        """
        uncertainties = []
        
        for finding in findings:
            validation = next(
                (v for v in validations if v.get("claim") == finding.get("claim")),
                None
            )
            
            if validation:
                if validation.get("confidence", 0.0) < self.min_confidence:
                    uncertainties.append(
                        f"Low confidence in: {finding.get('claim', 'Unknown claim')}"
                    )
                if validation.get("contradictions", []):
                    uncertainties.append(
                        f"Contradictions found in: {finding.get('claim', 'Unknown claim')}"
                    )
                    
        return uncertainties 