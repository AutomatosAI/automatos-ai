"""
Statistical Analysis Mathematical Foundation
Real implementations for trend analysis and performance prediction
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrendAnalysis:
    """Result of trend analysis"""
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0.0 to 1.0
    slope: float
    confidence: float
    predicted_next: Optional[float] = None
    volatility: float = 0.0


class StatisticalAnalysis:
    """Statistical algorithms for performance analysis"""
    
    @staticmethod
    def calculate_mean(values: List[float]) -> float:
        """Calculate arithmetic mean"""
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    @staticmethod
    def calculate_std(values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        mean = StatisticalAnalysis.calculate_mean(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    @staticmethod
    def calculate_trend(values: List[float], time_points: Optional[List[float]] = None) -> TrendAnalysis:
        """
        Analyze trend in time series data using simple linear regression.
        
        Args:
            values: List of values (e.g., success rates over time)
            time_points: Optional time points (defaults to 0, 1, 2, ...)
            
        Returns:
            TrendAnalysis with direction, strength, and prediction
        """
        if len(values) < 2:
            return TrendAnalysis(
                trend_direction="stable",
                trend_strength=0.0,
                slope=0.0,
                confidence=0.0
            )
        
        # Use simple indices if no time points given
        if time_points is None:
            time_points = list(range(len(values)))
        
        n = len(values)
        x = np.array(time_points)
        y = np.array(values)
        
        # Linear regression: y = slope * x + intercept
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator
        
        intercept = y_mean - slope * x_mean
        
        # Calculate R² for confidence
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))  # Clamp to [0, 1]
        
        # Determine trend direction
        if abs(slope) < 0.01:  # Threshold for "stable"
            direction = "stable"
            strength = 0.0
        elif slope > 0:
            direction = "increasing"
            strength = min(1.0, abs(slope))
        else:
            direction = "decreasing"
            strength = min(1.0, abs(slope))
        
        # Predict next value
        next_time = x[-1] + 1
        predicted_next = slope * next_time + intercept
        
        # Calculate volatility (coefficient of variation)
        std = np.std(values)
        mean = np.mean(values)
        volatility = (std / mean) if mean > 0 else 0.0
        
        return TrendAnalysis(
            trend_direction=direction,
            trend_strength=strength,
            slope=slope,
            confidence=r_squared,
            predicted_next=float(predicted_next),
            volatility=float(volatility)
        )
    
    @staticmethod
    def calculate_moving_average(values: List[float], window_size: int = 3) -> List[float]:
        """
        Calculate moving average for smoothing trends.
        
        Args:
            values: List of values
            window_size: Size of the moving window
            
        Returns:
            List of moving averages (same length as input, padded with None)
        """
        if len(values) < window_size:
            return values
        
        result = []
        for i in range(len(values)):
            if i < window_size - 1:
                result.append(values[i])  # Not enough data yet
            else:
                window = values[i - window_size + 1:i + 1]
                result.append(sum(window) / window_size)
        
        return result
    
    @staticmethod
    def detect_anomalies(values: List[float], threshold_std: float = 2.0) -> List[int]:
        """
        Detect anomalies using standard deviation method.
        
        Args:
            values: List of values
            threshold_std: Number of standard deviations for anomaly threshold
            
        Returns:
            List of indices where anomalies were detected
        """
        if len(values) < 3:
            return []
        
        mean = StatisticalAnalysis.calculate_mean(values)
        std = StatisticalAnalysis.calculate_std(values)
        
        if std == 0:
            return []
        
        anomalies = []
        for i, value in enumerate(values):
            z_score = abs((value - mean) / std)
            if z_score > threshold_std:
                anomalies.append(i)
        
        return anomalies
    
    @staticmethod
    def calculate_percentile(values: List[float], percentile: float) -> float:
        """
        Calculate percentile of values.
        
        Args:
            values: List of values
            percentile: Percentile to calculate (0-100)
            
        Returns:
            Value at the given percentile
        """
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            fraction = index - int(index)
            return lower + (upper - lower) * fraction
    
    @staticmethod
    def calculate_correlation(x_values: List[float], y_values: List[float]) -> float:
        """
        Calculate Pearson correlation coefficient.
        
        Args:
            x_values: First variable
            y_values: Second variable
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        x = np.array(x_values)
        y = np.array(y_values)
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))
        
        if denominator == 0:
            return 0.0
        
        correlation = numerator / denominator
        return float(np.clip(correlation, -1.0, 1.0))

