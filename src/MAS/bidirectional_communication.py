
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
from src.MAS.agents.iterative_agent import IterativeAgent
from src.MAS.agents.population_agent import PopulationAgent 

from src.notifier import * 

from typing import List, Dict, Tuple
from collections import defaultdict
import json

from src.instance import VRPInstance
from src.solution import VRPSolution

class BidirectionalMultiAgentSystem:
    """
    Système multi-agent pour résoudre le CVRP de manière collaborative.
    """
    
    def __init__(self, instance: VRPInstance, pool_size: int = 20):
        self.instance = instance
        self.solution_pool = SolutionPool(max_size=pool_size)
        self.iterative_agents: List[IterativeAgent] = []
        self.population_agents: List[PopulationAgent] = []
        self.results: Dict = {}
    
    def add_iterative_agent(self, name: str, solver, solver_params: Dict = None, run_params: Dict = None):
        """Ajoute un agent itératif (HC, Tabu, SA)."""
        agent = IterativeAgent(name, self.instance, solver, solver_params, run_params)
        self.iterative_agents.append(agent)
    
    def add_population_agent(self, name: str, solver, solver_params: Dict = None, run_params: Dict = None):
        """Ajoute un agent à population (GA, ACO, PSO, GB)."""
        agent = PopulationAgent(name, self.instance, solver, solver_params, run_params)
        self.population_agents.append(agent)
    
    def run_phase_parallel(self, agents: List[Agent], phase_name: str, verbose: bool = True) -> Dict:
        """Exécute une phase avec plusieurs agents en parallèle."""
        if verbose:
            print(f"\n{'#'*60}")
            print(f"# {phase_name}")
            print(f"# {len(agents)} agents en parallèle")
            print(f"{'#'*60}")
        
        start_time = time.time()
        threads = []
        
        # Créer un thread pour chaque agent
        for agent in agents:
            thread = threading.Thread(
                target=agent.run,
                kwargs={'solution_pool': self.solution_pool, 'verbose': False}
            )
            threads.append(thread)
            thread.start()
        
        # Attendre que tous les threads se terminent
        for thread in threads:
            thread.join()
        
        phase_time = time.time() - start_time
        
        # Récupérer les résultats
        results = {}
        for agent in agents:
            results[agent.name] = {
                'distance': agent.best_distance,
                'time': agent.computation_time,
                'solution': agent.best_solution,
                'feasible': agent.best_solution.is_feasible()[0] if agent.best_solution else False
            }
        
        if verbose:
            print(f"\n{phase_name} TERMINÉE - Temps total: {phase_time:.2f}s")
            print(f"Résultats:")
            for agent_name, result in sorted(results.items(), key=lambda x: x[1]['distance']):
                feasible_str = "✓" if result['feasible'] else "✗"
                print(f"  {agent_name:25s}: {result['distance']:8.2f} ({result['time']:5.2f}s) {feasible_str}")
        
        return results
    
    def run_sequential(self, verbose: bool = True) -> Dict:
        """
        Exécute le système multi-agent en mode séquentiel.
        
        Phase 1: Agents itératifs → Pool
        Phase 2: Agents population (utilisent Pool) → Pool
        
        Returns:
            Dictionnaire des résultats
        """
        overall_start = time.time()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"SYSTÈME MULTI-AGENT - DÉMARRAGE")
            print(f"Instance: {self.instance.num_customers} clients, {self.instance.num_vehicule} véhicules")
            print(f"Capacité: {self.instance.vehicle_capacity}")
            print(f"{'='*60}")
        
        # PHASE 1: Agents itératifs
        phase1_results = {}
        if self.iterative_agents:
            phase1_results = self.run_phase_parallel(
                self.iterative_agents,
                "PHASE 1: Agents Itératifs (HC, Tabu, SA)",
                verbose
            )
            
            if verbose:
                pool_stats = self.solution_pool.get_stats()
                print(f"\nPool après Phase 1: {pool_stats}")

        
        # PHASE 2: Agents à population
        phase2_results = {}
        if self.population_agents:
            phase2_results = self.run_phase_parallel(
                self.population_agents,
                "PHASE 2: Agents à Population (GA, ACO, PSO, GB)",
                verbose
            )
        
        overall_time = time.time() - overall_start
        
        # Meilleure solution globale
        best_solution, best_distance = self.solution_pool.get_best()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"SYSTÈME MULTI-AGENT - RÉSUMÉ FINAL")
            print(f"{'='*60}")
            print(f"Temps total: {overall_time:.2f}s")
            print(f"Meilleure solution: {best_distance:.2f}")
            
            pool_stats = self.solution_pool.get_stats()
            print(f"\nPool final:")
            print(f"  Nombre de solutions: {pool_stats['count']}")
            print(f"  Meilleure distance: {pool_stats['best']:.2f}")
            print(f"  Distance moyenne: {pool_stats['avg']:.2f}")
            print(f"  Contributions: {pool_stats['agents']}")
            
            if best_solution:
                feasible, msg = best_solution.is_feasible()
                print(f"  Faisabilité: {feasible} - {msg}")
            print(f"{'='*60}")
        
        self.results = {
            'phase1': phase1_results,
            'phase2': phase2_results,
            'best_solution': best_solution,
            'best_distance': best_distance,
            'total_time': overall_time,
            'pool_stats': self.solution_pool.get_stats()
        }
        
        return self.results
    
    def get_summary_dict(self) -> Dict:
        """
        Retourne un dictionnaire résumé pour Excel/DataFrame.
        Compatible avec votre format de tests.
        """
        if not self.results:
            return {}
        
        summary = {
            'MAS_Distance': self.results['best_distance'],
            'MAS_CPU': self.results['total_time'],
        }
        
        # Ajouter les résultats individuels
        all_results = {**self.results.get('phase1', {}), **self.results.get('phase2', {})}
        for agent_name, data in all_results.items():
            prefix = agent_name.replace('_', '')
            summary[f'{prefix}_Distance'] = data['distance']
            summary[f'{prefix}_CPU'] = data['time']
        
        # Statistiques du pool
        pool_stats = self.results.get('pool_stats', {})
        summary['Pool_Best'] = pool_stats.get('best', 0)
        summary['Pool_Avg'] = pool_stats.get('avg', 0)
        summary['Pool_Count'] = pool_stats.get('count', 0)
        
        # Statistiques de visite si disponibles
        if self.results['best_solution'] and hasattr(self.results['best_solution'], 'visit_counter'):
            vc = self.results['best_solution'].visit_counter
            if vc:
                summary['MAS_total_visites'] = sum(vc.values())
                summary['MAS_num_unique'] = len(vc)
                summary['MAS_max_visits'] = max(vc.values())
        
        return summary