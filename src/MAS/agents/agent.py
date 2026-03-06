import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import time 
from typing import List, Dict, Tuple
from queue import Queue
import json

from src.instance import VRPInstance
from src.solution import VRPSolution
from src.MAS.solution_pool import SolutionPool

class Agent:
    """
    Classe de base pour un agent résolvant le CVRP.
    """
    
    def __init__(self, name: str, instance: VRPInstance, solver, solver_params: Dict):
        self.name = name
        self.instance = instance
        self.solver = solver
        self.solver_params = solver_params
        self.best_solution = None
        self.best_distance = float('inf')
        self.computation_time = 0
        self.is_running = False
    
    def run(self, solution_pool: SolutionPool = None, verbose: bool = False) -> VRPSolution:
        """
        Exécute l'algorithme de l'agent.
        
        Args:
            solution_pool: Pool de solutions partagé (optionnel)
            verbose: Afficher les informations
        
        Returns:
            Meilleure solution trouvée
        """
        raise NotImplementedError("Les sous-classes doivent implémenter run()")
    
    def get_best_solution(self) -> Tuple[VRPSolution, float]:
        """Retourne la meilleure solution trouvée."""
        return self.best_solution, self.best_distance