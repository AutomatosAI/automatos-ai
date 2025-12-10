"""
Visualization Suggester
=======================

Automatically suggests the best visualization type based on data characteristics.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Supported chart types."""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    AREA = "area"
    TABLE = "table"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    DONUT = "donut"
    TREEMAP = "treemap"


class VisualizationSuggester:
    """
    Analyzes data and suggests the most appropriate visualization.
    """
    
    def __init__(self):
        pass
    
    def suggest(
        self,
        data: List[Dict[str, Any]],
        query: str = ""
    ) -> Tuple[ChartType, Dict[str, Any]]:
        """
        Suggest the best visualization for the data.
        
        Args:
            data: Query result data
            query: Original query for context
            
        Returns:
            Tuple of (chart_type, config)
        """
        if not data:
            return ChartType.TABLE, {"reason": "No data"}
        
        # Analyze data characteristics
        analysis = self._analyze_data(data)
        
        # Get query hints
        query_hints = self._parse_query_hints(query)
        
        # Apply selection rules
        chart_type = self._select_chart_type(analysis, query_hints)
        
        # Build configuration
        config = self._build_chart_config(chart_type, analysis, data)
        
        return chart_type, config
    
    def _analyze_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze data characteristics.
        """
        if not data:
            return {}
        
        columns = list(data[0].keys())
        row_count = len(data)
        
        analysis = {
            'row_count': row_count,
            'column_count': len(columns),
            'columns': {},
            'has_time': False,
            'has_category': False,
            'has_numeric': False,
            'numeric_columns': [],
            'category_columns': [],
            'time_columns': [],
        }
        
        for col in columns:
            col_analysis = self._analyze_column(col, data)
            analysis['columns'][col] = col_analysis
            
            if col_analysis['type'] == 'temporal':
                analysis['has_time'] = True
                analysis['time_columns'].append(col)
            elif col_analysis['type'] == 'numeric':
                analysis['has_numeric'] = True
                analysis['numeric_columns'].append(col)
            elif col_analysis['type'] == 'categorical':
                analysis['has_category'] = True
                analysis['category_columns'].append(col)
        
        return analysis
    
    def _analyze_column(
        self,
        column_name: str,
        data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze a single column's characteristics.
        """
        values = [row.get(column_name) for row in data]
        non_null_values = [v for v in values if v is not None]
        
        col_info = {
            'name': column_name,
            'null_count': len(values) - len(non_null_values),
            'unique_count': len(set(str(v) for v in non_null_values)),
            'type': 'unknown'
        }
        
        # Determine type
        if not non_null_values:
            col_info['type'] = 'null'
            return col_info
        
        # Check for temporal
        time_keywords = ['date', 'time', 'created', 'updated', 'timestamp', 'day', 'month', 'year']
        if any(kw in column_name.lower() for kw in time_keywords):
            col_info['type'] = 'temporal'
            return col_info
        
        # Check for numeric
        numeric_count = sum(1 for v in non_null_values if isinstance(v, (int, float)))
        if numeric_count > len(non_null_values) * 0.8:
            col_info['type'] = 'numeric'
            numeric_values = [v for v in non_null_values if isinstance(v, (int, float))]
            col_info['min'] = min(numeric_values)
            col_info['max'] = max(numeric_values)
            col_info['sum'] = sum(numeric_values)
            col_info['avg'] = col_info['sum'] / len(numeric_values)
            return col_info
        
        # Check for categorical
        if col_info['unique_count'] <= min(20, len(data) * 0.5):
            col_info['type'] = 'categorical'
            # Get value counts
            value_counts = {}
            for v in non_null_values:
                v_str = str(v)
                value_counts[v_str] = value_counts.get(v_str, 0) + 1
            col_info['value_counts'] = value_counts
            return col_info
        
        col_info['type'] = 'text'
        return col_info
    
    def _parse_query_hints(self, query: str) -> Dict[str, bool]:
        """
        Extract visualization hints from the query.
        """
        query_lower = query.lower()
        
        return {
            'wants_trend': any(w in query_lower for w in ['trend', 'over time', 'daily', 'weekly', 'monthly']),
            'wants_comparison': any(w in query_lower for w in ['compare', 'versus', 'vs', 'breakdown']),
            'wants_distribution': any(w in query_lower for w in ['distribution', 'spread', 'histogram']),
            'wants_proportion': any(w in query_lower for w in ['percentage', 'proportion', 'share', 'pie']),
            'wants_correlation': any(w in query_lower for w in ['correlation', 'relationship', 'scatter']),
            'wants_chart': any(w in query_lower for w in ['chart', 'graph', 'plot', 'visualize']),
        }
    
    def _select_chart_type(
        self,
        analysis: Dict[str, Any],
        hints: Dict[str, bool]
    ) -> ChartType:
        """
        Select the best chart type based on analysis and hints.
        """
        row_count = analysis.get('row_count', 0)
        has_time = analysis.get('has_time', False)
        has_numeric = analysis.get('has_numeric', False)
        has_category = analysis.get('has_category', False)
        category_cols = analysis.get('category_columns', [])
        numeric_cols = analysis.get('numeric_columns', [])
        
        # User explicitly wants a trend/time series
        if hints.get('wants_trend') and has_time and has_numeric:
            return ChartType.LINE
        
        # User wants proportion/percentage
        if hints.get('wants_proportion') and has_category:
            # Check if categories are small enough for pie
            for col in category_cols:
                col_info = analysis['columns'].get(col, {})
                if col_info.get('unique_count', 0) <= 8:
                    return ChartType.PIE
            return ChartType.BAR
        
        # User wants correlation/scatter
        if hints.get('wants_correlation') and len(numeric_cols) >= 2:
            return ChartType.SCATTER
        
        # User wants distribution
        if hints.get('wants_distribution') and has_numeric:
            return ChartType.HISTOGRAM
        
        # Auto-detect based on data structure
        
        # Time series data → Line chart
        if has_time and has_numeric and row_count > 1:
            return ChartType.LINE
        
        # Categorical + numeric → Bar chart
        if has_category and has_numeric:
            # Small categories → Pie, else Bar
            for col in category_cols:
                col_info = analysis['columns'].get(col, {})
                if col_info.get('unique_count', 0) <= 6:
                    return ChartType.PIE
            return ChartType.BAR
        
        # Just numeric columns → Bar (comparing values)
        if has_numeric and not has_category and not has_time:
            if row_count <= 20:
                return ChartType.BAR
        
        # Large dataset → Table
        if row_count > 50:
            return ChartType.TABLE
        
        # Default to table for raw data
        return ChartType.TABLE
    
    def _build_chart_config(
        self,
        chart_type: ChartType,
        analysis: Dict[str, Any],
        data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build chart configuration.
        """
        config = {
            'type': chart_type.value,
            'reason': '',
            'x_axis': None,
            'y_axis': None,
            'series': None,
            'title': '',
        }
        
        time_cols = analysis.get('time_columns', [])
        numeric_cols = analysis.get('numeric_columns', [])
        category_cols = analysis.get('category_columns', [])
        
        if chart_type == ChartType.LINE:
            config['x_axis'] = time_cols[0] if time_cols else category_cols[0] if category_cols else None
            config['y_axis'] = numeric_cols[0] if numeric_cols else None
            config['reason'] = "Time series data best shown as line chart"
            config['title'] = f"{config['y_axis']} over {config['x_axis']}"
        
        elif chart_type == ChartType.BAR:
            config['x_axis'] = category_cols[0] if category_cols else time_cols[0] if time_cols else None
            config['y_axis'] = numeric_cols[0] if numeric_cols else None
            config['reason'] = "Categorical comparison best shown as bar chart"
            config['title'] = f"{config['y_axis']} by {config['x_axis']}"
        
        elif chart_type == ChartType.PIE:
            config['series'] = category_cols[0] if category_cols else None
            config['y_axis'] = numeric_cols[0] if numeric_cols else None
            config['reason'] = "Proportion/share data best shown as pie chart"
            config['title'] = f"Distribution of {config['series']}"
        
        elif chart_type == ChartType.SCATTER:
            config['x_axis'] = numeric_cols[0] if len(numeric_cols) > 0 else None
            config['y_axis'] = numeric_cols[1] if len(numeric_cols) > 1 else None
            config['reason'] = "Correlation between numeric values best shown as scatter plot"
            config['title'] = f"{config['y_axis']} vs {config['x_axis']}"
        
        elif chart_type == ChartType.TABLE:
            config['reason'] = "Raw data or large dataset best shown as table"
            config['title'] = "Data Table"
        
        return config
    
    def get_chart_prompt(
        self,
        chart_type: ChartType,
        config: Dict[str, Any],
        data: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a prompt for PandasAI to create the chart.
        """
        prompts = {
            ChartType.LINE: f"Create a line chart showing {config.get('y_axis')} over {config.get('x_axis')} with clear labels and a legend",
            ChartType.BAR: f"Create a bar chart comparing {config.get('y_axis')} across different {config.get('x_axis')} categories",
            ChartType.PIE: f"Create a pie chart showing the distribution of {config.get('series')} with percentage labels",
            ChartType.SCATTER: f"Create a scatter plot showing the relationship between {config.get('x_axis')} and {config.get('y_axis')}",
            ChartType.HISTOGRAM: f"Create a histogram showing the distribution of values",
            ChartType.AREA: f"Create an area chart showing {config.get('y_axis')} over {config.get('x_axis')}",
        }
        
        return prompts.get(chart_type, "Create an appropriate visualization for this data")

