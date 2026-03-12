import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import time
import threading

from typing import List, Dict, Tuple, Set

from src.MAS.solution_pool import SolutionPool
from src.instance import VRPInstance
from src.solution import VRPSolution
from src.MAS.agents.agent import Agent
from src.notifier import *

class AsyncCollaborativeAgent(threading.Thread):
    """
    Agent qui s'exécute en parallèle et collabore via le pool.
    """
    
    def __init__(self, name: str, instance: VRPInstance, solver_class, 
                 solver_params: Dict, run_params: Dict, 
                 is_population_based: bool, max_cycles: int = 10):
        super().__init__()
        self.name = name
        self.instance = instance
        self.solver_class = solver_class
        self.solver_params = solver_params
        self.run_params = run_params
        self.is_population_based = is_population_based
        self.max_cycles = max_cycles
        
        self.pool = None  # Sera assigné avant start()
        self.best_solution = None
        self.best_distance = float('inf')
        self.cycles_completed = 0
        self.solutions_deposited = 0
        self.running = True
    
    def run(self):
        """Thread principal de l'agent."""
        print(f"[{self.name}] Démarrage...")
        
        for cycle in range(self.max_cycles):
            if not self.running or self.pool.is_stagnant():
                print(f"[{self.name}] Arrêt anticipé (cycle {cycle}, stagnation ou arrêt global)")
                break
            
            self.cycles_completed = cycle + 1
            
            if self.is_population_based:
                self._run_population_cycle(cycle)
            else:
                self._run_iterative_cycle(cycle)
            
            # Pause courte pour laisser les autres agents travailler
            time.sleep(0.1)
        
        print(f"[{self.name}] Terminé - {self.cycles_completed} cycles, {self.solutions_deposited} solutions déposées, best: {self.best_distance:.2f}")
    
    def _run_iterative_cycle(self, cycle: int):
        """
        Cycle pour métaheuristiques itératives (HC, TS, SA).
        1. Récupérer une solution du pool
        2. Relancer depuis cette solution
        3. Déposer si meilleure
        """
        # 1. Récupérer une solution non utilisée
        initial_solution, sol_id = self.pool.get_unused_solution(self.name)
        
        if initial_solution is None:
            print(f"[{self.name}] Cycle {cycle}: Aucune solution disponible")
            return
        
        # 2. Créer le solver et lancer avec cette solution initiale
        solver = self.solver_class(self.instance, **self.solver_params)
        
        # Vérifier si le solver a une méthode improve()
        if hasattr(solver, 'improve'):
            new_solution = solver.improve(
                initial_solution=initial_solution,
                verbose=False,
                **self.run_params
            )
        else:
            # Sinon, on doit modifier run() pour accepter initial_solution
            # Pour l'instant, on génère une nouvelle solution
            new_solution = solver.run(verbose=False, **self.run_params)
        
        # 3. Déposer si meilleure que la précédente
        if new_solution.total_distance < self.best_distance:
            added = self.pool.add_solution(new_solution, self.name)
            if added:
                self.solutions_deposited += 1
                self.best_distance = new_solution.total_distance
                self.best_solution = new_solution
                print(f"[{self.name}] Cycle {cycle}: Nouvelle meilleure {self.best_distance:.2f} (depuis {sol_id})")
    
    def _run_population_cycle(self, cycle: int):
        """
        Cycle pour métaheuristiques à population (GA, ACO, PSO, GB).
        1. Récupérer K solutions du pool
        2. Injecter dans la population
        3. Continuer l'exécution
        4. Déposer si meilleure
        """
        # 1. Récupérer K solutions non utilisées
        k = 5
        new_solutions = self.pool.get_k_unused_solutions(self.name, k=k)
        
        if not new_solutions:
            print(f"[{self.name}] Cycle {cycle}: Aucune solution disponible")
            return
        
        # 2 & 3. Créer le solver et injecter les solutions
        if cycle == 0:
            # Premier cycle : créer le solver
            self.solver_instance = self.solver_class(self.instance, **self.solver_params)
            
            # Initialiser avec les solutions du pool
            if hasattr(self.solver_instance, 'set_initial_solutions'):
                self.solver_instance.set_initial_solutions(new_solutions)
            
            # Lancer
            new_solution = self.solver_instance.run(verbose=False, **self.run_params)
        else:
            # Cycles suivants : injecter et continuer
            if hasattr(self.solver_instance, 'inject_and_continue'):
                new_solution = self.solver_instance.inject_and_continue(
                    new_solutions=new_solutions,
                    verbose=False,
                    **self.run_params
                )
            else:
                # Fallback : relancer avec nouvelles solutions
                if hasattr(self.solver_instance, 'set_initial_solutions'):
                    self.solver_instance.set_initial_solutions(new_solutions)
                new_solution = self.solver_instance.run(verbose=False, **self.run_params)
        
        # 4. Déposer si meilleure
        if new_solution.total_distance < self.best_distance:
            added = self.pool.add_solution(new_solution, self.name)
            if added:
                self.solutions_deposited += 1
                self.best_distance = new_solution.total_distance
                self.best_solution = new_solution
                print(f"[{self.name}] Cycle {cycle}: Nouvelle meilleure {self.best_distance:.2f}")
    
    def stop(self):
        """Arrêt de l'agent."""
        self.running = False