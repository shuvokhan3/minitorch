"""
Dataset generators for classification experiments
"""

import random
from typing import List, Tuple

#This class acts as a lightweight dataset container
class DataView:
    """Wrapper to provide .X, .y, .N interface for datasets"""
    def __init__(self, points: List[Tuple[float, float]], labels: List[int]):
        """ self.X store point or input vector feature like :
        [
           (0.1, 0.7),
           (0.9, 0.2),
        ]
        """
        self.X = points

        """
        self.y store label like : [ 0 , 1 ] 
        """
        self.y = labels # store labels

        #input featuer length
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
    
    def diag(N: int) -> Tuple[List[Tuple[float, float]], List[int]]:
        """
        Diagonal dataset - points above vs below the diagonal .

        Classification rule : x + y >= 1.0 -> class 1 , else class 0

        Args:
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
            labels.append(1 if x + y >= 1.0 else 0)
        return points, labels
    
    def split(N: int) -> Tuple[List[Tuple[float, float]], List[int]]:
        """
        Split dataset - points in center vs edges
        classification rule: 0.2 <= x <= 0.8 -> class 0 , else class 1
        Args:
            N: Number of points to generate 
        Returns:
            Tuple of (points, labels)

        """
        points = []
        labels = []
        for _ in range(N):
            x = random.random()
            y = random.random()
            points.append((x, y))
            labels.append(0 if 0.2 <= x <= 0.8 else 1)
        return points, labels

    def xor(N : int) -> Tuple[List[Tuple[float, float]], List[int]]:
        """
        xor dataset - requires non-linear separation.

        classification rule:
        (x < 0.5 and y < 0.5 ) or (x >= 0.5 and y >= 0.5 ) -> class 0 else -> class 1

        Args :
          N : Number of points to generate
        returns :
           Tuple of (points, labels)

        """
        points = []
        labels = []

        for _ in range(N):
            x = random.random()
            y = random.random()
            points.append((x, y))
            same_side = (x < 0.5) == (y < 0.5)
            labels.append(0 if same_side else 1)
        return points, labels






























