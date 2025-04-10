"""
Metrics collection and analysis module
"""

import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class SearchMetrics:
    query: str
    duration: float
    num_results: int
    error: Optional[str] = None

@dataclass
class SystemMetrics:
    total_searches: int = 0
    avg_duration: float = 0
    avg_results: float = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class MetricsCollector:
    def __init__(self):
        self._search_metrics: List[SearchMetrics] = []
        self._system_metrics: SystemMetrics = SystemMetrics()
        self._start_time: Optional[float] = None

    def start_timer(self) -> float:
        self._start_time = time.time()
        return self._start_time

    def get_duration(self, start_time: float) -> float:
        return time.time() - start_time

    def add_metrics(
        self,
        query: str,
        duration: float,
        num_results: int,
        error: Optional[str] = None
    ):
        metrics = SearchMetrics(
            query=query,
            duration=duration,
            num_results=num_results,
            error=error
        )
        self._search_metrics.append(metrics)
        self._update_system_metrics()

    def get_search_metrics(self) -> List[SearchMetrics]:
        return self._search_metrics

    def get_system_metrics(self) -> SystemMetrics:
        return self._system_metrics

    def _update_system_metrics(self):
        total_searches = len(self._search_metrics)
        total_duration = sum(m.duration for m in self._search_metrics)
        total_results = sum(m.num_results for m in self._search_metrics)
        errors = [m.error for m in self._search_metrics if m.error]

        self._system_metrics = SystemMetrics(
            total_searches=total_searches,
            avg_duration=total_duration / total_searches if total_searches > 0 else 0,
            avg_results=total_results / total_searches if total_searches > 0 else 0,
            errors=errors
        )

metrics_collector = MetricsCollector() 