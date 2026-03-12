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

class PopulationAgent(Agent):
    """
    Agent basé sur une métaheuristique à population (GA, ACO, PSO, GB).
    Peut être initialisé avec les solutions du pool.
    """
    
    def run(self, solution_pool: SolutionPool = None, verbose: bool = False) -> VRPSolution:
        self.is_running = True
        
        if verbose:
            section(f"Agent {self.name} (Population) - DÉMARRAGE")
        
        # Récupérer les meilleures solutions du pool
        initial_solutions = []
        if solution_pool:
            top_solutions = solution_pool.get_top_k(k=5)
            initial_solutions = [sol for sol, _, _ in top_solutions]
            
            if verbose and initial_solutions:
                print(f"{self.name} a récupéré {len(initial_solutions)} solutions du pool")
                distances = [f'{dist:.2f}' for _, dist, _ in top_solutions]
                print(f"Distances: {distances}")
        
        # Créer l'instance du solver
        solver_instance = self.solver(self.instance, **self.solver_params)
        
        # Injecter les solutions si supporté
        if hasattr(solver_instance, 'set_initial_solutions') and initial_solutions:
            solver_instance.set_initial_solutions(initial_solutions)
            if verbose:
                print(f"{self.name} initialisé avec les solutions du pool")
        
        # Préparer les paramètres pour run()
        run_kwargs = self.run_params.copy()
        run_kwargs['verbose'] = verbose
        
        # Exécuter l'algorithme
        solution = solver_instance.run(**run_kwargs)
        
        self.best_solution = solution
        self.best_distance = solution.total_distance
        self.computation_time = solution.computation_time
        
        # Ajouter au pool
        if solution_pool:
            solution_pool.add_solution(solution, self.best_distance, self.name)
            if verbose:
                stats = solution_pool.get_stats()
                print(f"\n{self.name} a ajouté sa solution au pool")
                print(f"Pool: {stats['count']} solutions, meilleure: {stats['best']:.2f}")
        
        if verbose:
            print(f"{self.name} TERMINÉ - Distance: {self.best_distance:.2f}, Temps: {self.computation_time:.2f}s")
        
        self.is_running = False
        return solution
