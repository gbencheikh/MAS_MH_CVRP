import threading
from typing import List, Dict, Tuple
import time

from src.solution import VRPSolution


class SolutionPool:
    """
    Pool partagé de solutions pour la communication entre agents.
    Thread-safe pour permettre la communication asynchrone.
    """
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.solutions: List[Tuple[VRPSolution, float, str]] = []
        self.lock = threading.Lock()  # ← CRITIQUE : Initialiser le lock
        self.history: List[Dict] = []
    
    def add_solution(self, solution: VRPSolution, distance: float, agent_name: str):
        """Ajoute une solution au pool (thread-safe)."""
        with self.lock:
            # Ajouter la solution
            self.solutions.append((solution._copy_solution(), distance, agent_name))
            
            # Trier par distance (meilleure en premier)
            self.solutions.sort(key=lambda x: x[1])
            
            # Garder seulement les meilleures
            self.solutions = self.solutions[:self.max_size]
            
            # Historique
            self.history.append({
                'timestamp': time.time(),
                'agent': agent_name,
                'distance': distance,
                'pool_size': len(self.solutions)
            })
    
    def get_best(self) -> Tuple[VRPSolution, float]:
        """Récupère la meilleure solution du pool."""
        with self.lock:
            if not self.solutions:
                return None, float('inf')
            return self.solutions[0][0]._copy_solution(), self.solutions[0][1]
    
    def get_top_k(self, k: int = 3) -> List[Tuple[VRPSolution, float, str]]:
        """Récupère les k meilleures solutions."""
        with self.lock:
            return [(sol._copy_solution(), dist, agent) for sol, dist, agent in self.solutions[:k]]
    
    def get_all(self) -> List[Tuple[VRPSolution, float, str]]:
        """Récupère toutes les solutions du pool."""
        with self.lock:
            return [(sol._copy_solution(), dist, agent) for sol, dist, agent in self.solutions]
    
    def get_stats(self) -> Dict:
        """Récupère les statistiques du pool."""
        with self.lock:
            if not self.solutions:
                return {'count': 0, 'best': float('inf'), 'worst': float('inf'), 'avg': 0, 'agents': {}}
            
            distances = [dist for _, dist, _ in self.solutions]
            agents_count = {}
            for _, _, agent in self.solutions:
                agents_count[agent] = agents_count.get(agent, 0) + 1
            
            return {
                'count': len(self.solutions),
                'best': min(distances),
                'worst': max(distances),
                'avg': sum(distances) / len(distances),
                'agents': agents_count
            }
    
    def clear(self):
        """Vide le pool."""
        with self.lock:
            self.solutions = []
            self.history = []