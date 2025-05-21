from dataclasses import dataclass
from typing import List
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

@dataclass
class TensionPoint:
    """A single point in the tension curve"""
    time: float  # relative time position (0 to 1)
    value: float  # tension value (0 to 1)
    weight: float = 1.0  # importance of this tension point

@dataclass
class TensionCurve:
    """Represents a tension curve over time"""
    points: List[TensionPoint]
    duration: float = 1.0  # normalized duration
    
    def interpolate(self, num_points: int = 100) -> np.ndarray:
        """Convert discrete tension points to a continuous curve"""
        if not self.points:
            return np.zeros(num_points)
        
        sorted_points = sorted(self.points, key=lambda x: x.time)
        times = np.array([p.time for p in sorted_points])
        values = np.array([p.value for p in sorted_points])
        weights = np.array([p.weight for p in sorted_points])
        
        f = CubicSpline(times, values * weights, bc_type='natural')
 
        x = np.linspace(0, 1, num_points)
        return x, f(x)
    
    def plot(self):
        """Plot the tension curve"""
        x, y = self.interpolate()
        plt.plot(x, y, label='Tension Curve')
        
        # Plot the original points
        times = [p.time for p in self.points]
        values = [p.value * p.weight for p in self.points]
        plt.scatter(times, values, color='red', label='Tension Points')
        
        plt.xlabel('Time')
        plt.ylabel('Value * Weight')
        plt.title('Tension Curve')
        plt.legend()
        plt.show()