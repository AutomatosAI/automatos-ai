"""
Probability Theory Mathematical Foundation
Real implementations for confidence intervals and uncertainty quantification
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class ConfidenceInterval:
    """Confidence interval result"""
    mean: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    standard_error: float
    sample_size: int


class ProbabilityTheory:
    """Probability algorithms for uncertainty quantification"""
    
    @staticmethod
    def calculate_confidence_interval(
        values: List[float],
        confidence_level: float = 0.95
    ) -> ConfidenceInterval:
        """
        Calculate confidence interval for a sample.
        
        Uses t-distribution for small samples (n < 30) or normal distribution for large samples.
        
        Args:
            values: Sample values
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            ConfidenceInterval with bounds
        """
        if not values or len(values) < 2:
            return ConfidenceInterval(
                mean=values[0] if values else 0.0,
                lower_bound=values[0] if values else 0.0,
                upper_bound=values[0] if values else 0.0,
                confidence_level=confidence_level,
                standard_error=0.0,
                sample_size=len(values)
            )
        
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # Sample standard deviation
        standard_error = std / np.sqrt(n)
        
        # Determine critical value
        if n < 30:
            # Use t-distribution for small samples
            from scipy import stats
            degrees_of_freedom = n - 1
            t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
            margin_of_error = t_critical * standard_error
        else:
            # Use normal distribution for large samples
            # Z-score for common confidence levels
            z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
            z_critical = z_scores.get(confidence_level, 1.96)
            margin_of_error = z_critical * standard_error
        
        return ConfidenceInterval(
            mean=float(mean),
            lower_bound=float(mean - margin_of_error),
            upper_bound=float(mean + margin_of_error),
            confidence_level=confidence_level,
            standard_error=float(standard_error),
            sample_size=n
        )
    
    @staticmethod
    def calculate_probability(
        successes: int,
        total: int
    ) -> Tuple[float, ConfidenceInterval]:
        """
        Calculate probability with confidence interval (proportion).
        
        Args:
            successes: Number of successes
            total: Total number of trials
            
        Returns:
            (probability, confidence_interval)
        """
        if total == 0:
            return 0.0, ConfidenceInterval(
                mean=0.0, lower_bound=0.0, upper_bound=0.0,
                confidence_level=0.95, standard_error=0.0, sample_size=0
            )
        
        p = successes / total
        
        # Wilson score interval (better for proportions than normal approximation)
        z = 1.96  # 95% confidence
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        margin = z * np.sqrt((p * (1 - p) / total + z**2 / (4 * total**2))) / denominator
        
        ci = ConfidenceInterval(
            mean=float(p),
            lower_bound=float(max(0.0, center - margin)),
            upper_bound=float(min(1.0, center + margin)),
            confidence_level=0.95,
            standard_error=float(np.sqrt(p * (1 - p) / total)) if total > 0 else 0.0,
            sample_size=total
        )
        
        return float(p), ci
    
    @staticmethod
    def bayesian_update(
        prior_probability: float,
        likelihood_given_true: float,
        likelihood_given_false: float
    ) -> float:
        """
        Bayesian probability update.
        
        P(H|E) = P(E|H) * P(H) / P(E)
        
        Args:
            prior_probability: Prior P(H)
            likelihood_given_true: P(E|H) - probability of evidence given hypothesis is true
            likelihood_given_false: P(E|¬H) - probability of evidence given hypothesis is false
            
        Returns:
            Updated posterior probability
        """
        # Calculate total probability of evidence
        p_evidence = (
            likelihood_given_true * prior_probability +
            likelihood_given_false * (1 - prior_probability)
        )
        
        if p_evidence == 0:
            return prior_probability
        
        # Bayes theorem
        posterior = (likelihood_given_true * prior_probability) / p_evidence
        return float(np.clip(posterior, 0.0, 1.0))
    
    @staticmethod
    def calculate_expected_value(
        outcomes: List[float],
        probabilities: List[float]
    ) -> float:
        """
        Calculate expected value E[X] = Σ(x_i * p_i)
        
        Args:
            outcomes: Possible outcomes
            probabilities: Probability of each outcome
            
        Returns:
            Expected value
        """
        if len(outcomes) != len(probabilities):
            return 0.0
        
        expected = sum(outcome * prob for outcome, prob in zip(outcomes, probabilities))
        return float(expected)
    
    @staticmethod
    def calculate_variance(
        outcomes: List[float],
        probabilities: List[float]
    ) -> float:
        """
        Calculate variance Var[X] = E[X²] - E[X]²
        
        Args:
            outcomes: Possible outcomes
            probabilities: Probability of each outcome
            
        Returns:
            Variance
        """
        if len(outcomes) != len(probabilities):
            return 0.0
        
        expected_value = ProbabilityTheory.calculate_expected_value(outcomes, probabilities)
        expected_squared = sum(outcome**2 * prob for outcome, prob in zip(outcomes, probabilities))
        variance = expected_squared - expected_value**2
        
        return float(max(0.0, variance))
    
    @staticmethod
    def monte_carlo_simulation(
        simulation_func: callable,
        num_simulations: int = 1000
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation.
        
        Args:
            simulation_func: Function that returns a single simulation result
            num_simulations: Number of simulations to run
            
        Returns:
            Dictionary with simulation statistics
        """
        results = [simulation_func() for _ in range(num_simulations)]
        
        return {
            'mean': float(np.mean(results)),
            'median': float(np.median(results)),
            'std': float(np.std(results)),
            'min': float(np.min(results)),
            'max': float(np.max(results)),
            'confidence_interval': ProbabilityTheory.calculate_confidence_interval(results),
            'num_simulations': num_simulations
        }
    
    @staticmethod
    def exponential_decay(
        initial_value: float,
        decay_rate: float,
        time: float
    ) -> float:
        """
        Calculate exponential decay: V(t) = V₀ * e^(-λt)
        
        Useful for memory decay, importance degradation, etc.
        
        Args:
            initial_value: Initial value V₀
            decay_rate: Decay rate λ (higher = faster decay)
            time: Time elapsed
            
        Returns:
            Decayed value
        """
        return float(initial_value * np.exp(-decay_rate * time))
    
    @staticmethod
    def sigmoid(x: float, steepness: float = 1.0, midpoint: float = 0.0) -> float:
        """
        Sigmoid function: S(x) = 1 / (1 + e^(-k(x-x₀)))
        
        Useful for smooth transitions, activation functions, etc.
        
        Args:
            x: Input value
            steepness: Steepness parameter k (higher = steeper)
            midpoint: Midpoint x₀
            
        Returns:
            Value between 0 and 1
        """
        return float(1.0 / (1.0 + np.exp(-steepness * (x - midpoint))))
