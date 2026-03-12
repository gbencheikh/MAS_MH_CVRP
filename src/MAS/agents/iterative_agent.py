import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import time
from src.MAS.solution_pool import SolutionPool
from src.instance import VRPInstance
from src.solution import VRPSolution
from src.MAS.agents.agent import Agent
from src.notifier import *

class IterativeAgent(Agent):
    """
    Agent basé sur une métaheuristique itérative (Hill Climbing, Tabu Search, Simulated Annealing).
    """
    
    def run(self, solution_pool: SolutionPool = None, verbose: bool = False) -> VRPSolution:
        self.is_running = True
        start_time = time.time()
        
        if verbose:
            section(f"Agent {self.name} (Itératif) - DÉMARRAGE")
        
        # Créer l'instance du solver
        solver_instance = self.solver(self.instance, **self.solver_params)
        
        # Exécuter l'algorithme
        solution = solver_instance.run(verbose=verbose)
        
        self.best_solution = solution
        self.best_distance = solution.total_distance
        self.computation_time = time.time() - start_time
        
        # Ajouter au pool si disponible
        if solution_pool:
            solution_pool.add_solution(solution, self.best_distance, self.name)
            if verbose:
                stats = solution_pool.get_stats()
                print(f"\n{self.name} a ajouté sa solution au pool")
                print(f"Pool stats: {stats['count']} solutions, meilleure: {stats['best']:.2f}")
        
        if verbose:
            print(f"\n{self.name} TERMINÉ - Distance: {self.best_distance:.2f}, Temps: {self.computation_time:.2f}s")
        
        self.is_running = False
        return solution
