import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


import os
sys.path.append(str(Path(__file__).parent.parent))


from src.instance import VRPInstance
from src.solution import VRPSolution

from src.metaheuristics.hill_climbing import Hill_Climbing
from src.metaheuristics.tabu_search import Tabu_Search
from src.metaheuristics.simulated_annealing import Simulated_Annealing
from src.metaheuristics.genetic_algorithm import Genetic_Algorithm
from src.metaheuristics.ant_colony_am import Ant_Colony_Optimization, Ant
from src.metaheuristics.golden_ball import Golden_Ball
from src.metaheuristics.particle_swarm import Particle_Swarm_Optimization
from src.notifier import *
from src.MAS.collaborative_communication import AsyncCollaborativeMultiAgentSystem

# Exemple d'utilisation
if __name__ == "__main__":    
    # Charger une instance
    instance = VRPInstance.load_from_file("data/instances/A-n32-k5.vrp")
    
    # Créer le système
    mas = AsyncCollaborativeMultiAgentSystem(
        instance=instance,
        pool_size=20,
        stagnation_limit=5,
        initial_pool_size=10
    )
    
    # Ajouter les agents itératifs (cycles courts)
    mas.add_agent("HC", Hill_Climbing, 
                  run_params={'max_iterations': 100, 'log_history': False},
                  is_population_based=False, max_cycles=10)
    
    mas.add_agent("TS", Tabu_Search,
                  solver_params={'tabu_tenure': 20},
                  run_params={'max_iterations': 50, 'aspiration': True, 'log_history': False},
                  is_population_based=False, max_cycles=10)
    
    mas.add_agent("SA", Simulated_Annealing,
                  solver_params={'initial_temperature': 100, 'cooling_rate': 0.995},
                  run_params={'max_iterations': 500, 'seed': 42, 'log_history': False},
                  is_population_based=False, max_cycles=10)
    
    # Ajouter les agents à population
    mas.add_agent("GA", Genetic_Algorithm,
                  solver_params={'population_size': 30, 'elite_size': 5},
                  run_params={'max_generations': 20, 'seed': 42, 'log_history': False},
                  is_population_based=True, max_cycles=10)
    
    mas.add_agent("ACO", Ant_Colony_Optimization,
                  solver_params={'num_ants': 50, 'alpha': 1.0, 'beta': 5.0},
                  run_params={'max_iterations': 20, 'local_search': True, 'seed': 42, 'log_history': False},
                  is_population_based=True, max_cycles=10)
    
    mas.add_agent("PSO", Particle_Swarm_Optimization,
                  solver_params={'num_particles': 20},
                  run_params={'max_iterations': 20, 'seed': 42, 'log_history': False},
                  is_population_based=True, max_cycles=10)
    
    # Lancer le système
    results = mas.run(verbose=True)
    
    # Sauvegarder la meilleure solution
    if results['best_solution']:
        results['best_solution'].save_to_json("best_solution_async_mas.json")
        print("Meilleure solution sauvegardée.")