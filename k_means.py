"""Given a n vectors of d size, implement a k-means clustering algorithm. Values within the dimensions can range from -1 to 1"""
import math
from typing import Dict, List
import numpy as np
from enum import Enum
from collections import defaultdict

class DistanceType(Enum):
    EUCLIDEAN = 0
    MANHATTAN = 1

def distance(a: np.array, b: np.array, distance_type: DistanceType = DistanceType.EUCLIDEAN):
    if distance_type.value == 0:
        d = sum([abs(x[0] - x[1]) for x in zip(a,b)])/ len(a)
    else:
        d = sum([(x[0] - x[1]) ** 2 for x in zip(a,b)])/ len(a)
    return d

def get_shortest_k(vec: np.array, k_locations: Dict[int, List[np.array]], distance_type: DistanceType = DistanceType.EUCLIDEAN):
    closest_k = min(
        [
            (
                x[0], 
                distance(vec, x[1])
            ) 
            for x in k_locations.items()
        ], key=lambda x: x[1]
    )
    return closest_k[0]

def k_means(m: np.array, k: int, max_iter: int):
    """[k means algo to]
    
    Parameters
    ----------
    m : np.array
        [matrix containing n vectors of size d]
    k : int
        [number of clusters]
    max_iter : int
        [parameter for early stopping]
    """
    d = m.shape[1]
    starting_points = {
        i :  np.array([np.random.uniform(-1.0, 1.0) for _ in range(d)])
        for i in range(k)
    }

    for iteration in range(max_iter):
        assigned_cluster = []
        vecs_closest_to_k = defaultdict(list)
        for vec in m:
            closest_k = get_shortest_k(vec, starting_points)
            assigned_cluster.append(closest_k)
            vecs_closest_to_k[closest_k].append(vec)
        starting_points = {
            i: starting_points[i] if i not in vecs_closest_to_k else np.average(np.array(vecs_closest_to_k[i]), axis=1)
            for i in range(k)
        }

    return m, assigned_cluster

if __name__ == "__main__":
    test_data = np.array(
        [
            np.array([np.random.uniform(-1.0, 1.0) for _ in range (3)]),
            np.array([np.random.uniform(-1.0, 1.0) for _ in range (3)]),
            np.array([np.random.uniform(-1.0, 1.0) for _ in range (3)]),
        ]
    )
    x, y = k_means(test_data, 3, 1)
    print(y)
    