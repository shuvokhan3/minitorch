"""
Dataset generators for classification experiments
"""

import random
from typing import List, Tuple


class DataView:
    """Wrapper to provide .X, .y, .N interface for datasets"""
    def __init__(self, points: List[Tuple[float, float]], labels: List[int]):
        self.X = points
        self.y = labels
        self.N = len(points)

    def simple(N: len) -> Tuple[List[Tuple[float, float]], List[int]]:
        """
        simple linear dataset - points on left vs right ,
        classicication rule: x_coord >= 0.5 -> class 1 , else class 0 

        Arge: 
            N: number of points to generate

        Returns:
            Tuple of (points, labels)

        """

        points = []
        labels = []
        for _ in range(N):
            x = random.random()
            y = random.random()
            points.append((x, y))
            labels.append(1 if x >= 0.5 else 0)
        return points, labels   
    
    