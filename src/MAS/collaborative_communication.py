import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import time
import threading
from src.MAS.solution_pool import SolutionPool
from src.instance import VRPInstance
from src.solution import VRPSolution
from src.MAS.agents.agent import Agent

from src.notifier import * 

from typing import List, Dict, Tuple
from collections import defaultdict
import json

from src.instance import VRPInstance
from src.solution import VRPSolution
from src.MAS.collaborative_solution_pool import CollaborativeSolutionPool
from src.MAS.agents.async_collaborative_agent import AsyncCollaborativeAgent
class AsyncCollaborativeMultiAgentSystem:
    """
    Système multi-agent asynchrone avec pool de solutions partagé.
    """
    
    def __init__(self, instance: VRPInstance, pool_size: int = 20, 
                 stagnation_limit: int = 5, initial_pool_size: int = 10):
        self.instance = instance
        self.pool = CollaborativeSolutionPool(max_size=pool_size, stagnation_limit=stagnation_limit)
        self.agents: List[AsyncCollaborativeAgent] = []
        self.initial_pool_size = initial_pool_size
    
    def add_agent(self, name: str, solver_class, solver_params: Dict = None, 
                  run_params: Dict = None, is_population_based: bool = False, 
                  max_cycles: int = 10):
        """
        Ajoute un agent au système.
        
        Args:
            name: Nom de l'agent
            solver_class: Classe du solver
            solver_params: Paramètres du constructeur
            run_params: Paramètres de run()
            is_population_based: True si GA/ACO/PSO/GB, False si HC/TS/SA
            max_cycles: Nombre maximum de cycles
        """
        solver_params = solver_params or {}
        run_params = run_params or {}
        
        agent = AsyncCollaborativeAgent(
            name=name,
            instance=self.instance,
            solver_class=solver_class,
            solver_params=solver_params,
            run_params=run_params,
            is_population_based=is_population_based,
            max_cycles=max_cycles
        )
        
        self.agents.append(agent)
    
    def run(self, verbose: bool = True) -> Dict:
        """
        Lance le système multi-agent asynchrone.
        
        Returns:
            Dictionnaire avec les résultats
        """
        start_time = time.time()
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"SYSTÈME MULTI-AGENT ASYNCHRONE COLLABORATIF")
            print(f"Instance: {self.instance.num_customers} clients")
            print(f"Agents: {len(self.agents)}")
            print(f"{'='*80}\n")
        
        # 1. Initialiser le pool avec des solutions aléatoires
        self.pool.initialize_with_random(self.instance, self.initial_pool_size)
        
        # 2. Assigner le pool à chaque agent
        for agent in self.agents:
            agent.pool = self.pool
        
        # 3. Démarrer tous les agents en parallèle
        for agent in self.agents:
            agent.start()
        
        # 4. Attendre que tous terminent
        for agent in self.agents:
            agent.join()
        
        end_time = time.time()
        
        # 5. Collecter les résultats
        best_solution, best_distance = self.pool.get_best()
        pool_stats = self.pool.get_stats()
        
        results = {
            'best_solution': best_solution,
            'best_distance': best_distance,
            'total_time': end_time - start_time,
            'pool_stats': pool_stats,
            'agents_results': {}
        }
        
        for agent in self.agents:
            results['agents_results'][agent.name] = {
                'cycles': agent.cycles_completed,
                'deposited': agent.solutions_deposited,
                'best_distance': agent.best_distance,
                'best_solution': agent.best_solution
            }
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"RÉSULTATS FINAUX")
            print(f"{'='*80}")
            print(f"Temps total: {results['total_time']:.2f}s")
            print(f"Meilleure solution: {best_distance:.2f}")
            print(f"Cycles globaux: {pool_stats['global_cycle']}")
            print(f"\nPool final:")
            print(f"  Count: {pool_stats['count']}")
            print(f"  Best: {pool_stats['best']:.2f}")
            print(f"  Avg: {pool_stats['avg']:.2f}")
            print(f"\nPerformance par agent:")
            for agent_name, data in results['agents_results'].items():
                print(f"  {agent_name:10s}: {data['cycles']} cycles, {data['deposited']} déposées, best: {data['best_distance']:.2f}")
            print(f"{'='*80}\n")
        
        return results