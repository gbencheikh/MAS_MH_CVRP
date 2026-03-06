import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import gc

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

def run_tests(): 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    directory_path = os.path.join(current_dir, "..", "data", "instances")
    results_dir = os.path.join(current_dir, "..", "results")
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"resultats_{date_str}.xlsx"

    os.makedirs(results_dir, exist_ok=True)

    Summary_outputfile = os.path.join(results_dir, filename)
    results = []
    # Liste des métaheuristiques  --> "HC", "TS","SA","GB","GA","PSO","ACO"
    algorithms = ["HC", "TS","SA","GA","PSO","ACO"]
    
    for file_name in os.listdir(directory_path):
        if not file_name.endswith(".vrp"):
            continue

        filepath = os.path.join(directory_path, file_name)
        instance = VRPInstance.load_from_file(filepath)

        row = {"Instance": file_name}
        if instance.optimal_value != None:
            row["Optimal"] = instance.optimal_value
        else: 
            row["Optimal"] = "None"

        for algo in algorithms: 
            gc.collect() # Nettoyage de la mémoire 

            match(algo):
                case "HC": 
                    instance = VRPInstance.load_from_file(filepath)
                    MH = Hill_Climbing(instance)
                    solution = MH.run(max_iterations= 1000, log_history= True, verbose = False)
                case "TS":
                    instance = VRPInstance.load_from_file(filepath)
                    MH = Tabu_Search(instance)
                    solution = MH.run(max_iterations=500, aspiration=True, log_history=True, verbose=False)
                case "SA":
                    instance = VRPInstance.load_from_file(filepath)
                    MH = Simulated_Annealing(instance, initial_temperature=100.0, cooling_rate=0.995, min_temperature=0.0001)
                    solution = MH.run(max_iterations=5000, seed=42, log_history=True, verbose=False)
                case "GB":
                    instance = VRPInstance.load_from_file(filepath)
                    MH = Golden_Ball(instance, population_size=instance.num_customers//3, num_rounds=5, cooperation_rate=0.7, mutation_rate=0.3)
                    solution = MH.run(max_iterations=100, seed=42, log_history=True, verbose=False)
                case "GA": 
                    instance = VRPInstance.load_from_file(filepath)
                    MH = Genetic_Algorithm(instance, population_size=instance.num_customers//2, elite_size=5, mutation_rate=0.4, crossover_rate=0.9)
                    solution = MH.run(max_generations=100, seed=42, log_history=True, verbose=False)
                case "PSO": 
                    instance = VRPInstance.load_from_file(filepath)
                    MH = Particle_Swarm_Optimization(instance, num_particles=30, w=0.5, c1=1.5, c2=1.5)
                    solution = MH.run(max_iterations=100, seed=42, log_history=True, verbose=False)
                case "ACO":
                    instance = VRPInstance.load_from_file(filepath)
                    MH = Ant_Colony_Optimization(instance, num_ants=100, alpha=1.0, beta=5.0, evaporation_rate=0.1, q=100.0)
                    solution = MH.run(max_iterations=100, local_search=True, seed=42, log_history=True, verbose=False)
            
            feasible, msg = solution.is_feasible()
            print(f"{algo} Solution faisable ? {feasible} - {msg}")
            print(f"Agent {solution.agent_name} terminé en {solution.computation_time:.2f} secondes")
            print(solution)

            # Statistiques globales dans le titre
            total_visits = sum(solution.visit_counter.values())
            num_unique = len(solution.visit_counter)
            max_visits = max(solution.visit_counter.values())

            row[f"{algo}_Distance"] = solution.compute_total_distance()
            row[f"{algo}_CPU"] = solution.computation_time
            if instance.optimal_value != None: 
                row[f"{algo}_gap"] = solution.total_distance - instance.optimal_value
            else: 
                row[f"{algo}_gap"] = "None"
            row[f"{algo}_total_visites"] = total_visits
            row[f"{algo}_num_unique"] = num_unique
            row[f"{algo}_max_visits"] = max_visits
            row[f"{algo}_mean_visits"] = total_visits / num_unique

        results.append(row)

    df_summary = pd.DataFrame(results)
    df_summary.to_excel(Summary_outputfile, index=False)

    print(f"Résultats sauvegardés dans {Summary_outputfile}")

    df_summary = pd.DataFrame(results)
    df_summary.to_excel(Summary_outputfile, index=False)
    
if __name__ == "__main__":
    run_tests()