import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import copy
from collections import defaultdict

from src.instance import VRPInstance
from src.solution import VRPSolution
from src.utils import * 

class Hill_Climbing: 
    def __init__(self, instance: VRPInstance): 
        self.instance = instance

    def run(self, max_iterations: int = 1000, log_history: bool = True, verbose: bool = True) -> VRPSolution:
        """
        Algorithme Hill Climbing pour résoudre le CVRP.
        
        Args:
            max_iterations: Nombre maximum d'itérations
            verbose: Afficher les informations de progression
        
        Returns:
            La meilleure solution trouvée
        """
        if log_history:
            history = [] # historique des distances
            visit_counter = defaultdict(int)

        start_time = time.time()
        
        # Générer la solution initiale
        current_solution = VRPSolution(self.instance).generate_solution()
        current_distance = current_solution.total_distance
        if log_history:
            sig = current_solution.solution_signature()
            visit_counter[sig] += 1

            history.append({
                "iteration": 0,
                "signature": sig,
                "distance": current_distance
            })

        if verbose:
            print(f"Solution initiale: distance = {current_distance:.2f}")
            print(f"Nombre de routes: {len(current_solution.routes)}")
        
        iteration = 0
        no_improvement_count = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Générer le voisinage
            neighbors = generate_neighborhood(current_solution)
            
            if not neighbors:
                if verbose:
                    print("Aucun voisin généré, arrêt.")
                break
            
            # Trouver le meilleur voisin
            best_neighbor = min(neighbors, key=lambda sol: sol.total_distance)
            best_neighbor_distance = best_neighbor.total_distance
            
            # Si le meilleur voisin est meilleur que la solution courante
            if best_neighbor_distance < current_distance:
                current_solution = best_neighbor
                current_distance = best_neighbor_distance
                no_improvement_count = 0
                
                if log_history:
                    sig = current_solution.solution_signature()
                    visit_counter[sig] += 1

                    history.append({
                        "iteration": iteration,
                        "signature": sig,
                        "distance": current_distance
                    })
                
                if verbose and iteration % 10 == 0:
                    print(f"Itération {iteration}: nouvelle meilleure distance = {current_distance:.2f}")
            else:
                # Aucune amélioration trouvée (optimum local atteint)
                no_improvement_count += 1
                if no_improvement_count >= 1:  # Arrêt dès le premier optimum local
                    if verbose:
                        print(f"Optimum local atteint à l'itération {iteration}")
                    break
        
        end_time = time.time()
        current_solution.computation_time = end_time - start_time
        current_solution.agent_name = "Hill Climbing"
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Hill Climbing terminé en {current_solution.computation_time:.2f} secondes")
            print(f"Nombre d'itérations: {iteration}")
            print(f"Distance finale: {current_distance:.2f}")
            print(f"{'='*60}")
            
        if log_history:
            current_solution.history = history
            current_solution.visit_counter = visit_counter
        
        return current_solution

    def improve(self, initial_solution: VRPSolution, max_iterations: int = 100, 
            log_history: bool = False, verbose: bool = False) -> VRPSolution:
        """
        Améliore une solution existante avec Hill Climbing.
        
        Args:
            initial_solution: Solution de départ (du pool)
            max_iterations: Nombre d'itérations
            log_history: Sauvegarder l'historique
            verbose: Afficher les informations
        
        Returns:
            Solution améliorée
        """
        if log_history:
            history = []
            visit_counter = defaultdict(int)
        
        start_time = time.time()
        
        # Partir de la solution donnée (au lieu de generate_solution())
        current_solution = initial_solution._copy_solution()
        current_distance = current_solution.total_distance
        
        if log_history:
            sig = current_solution.solution_signature()
            visit_counter[sig] += 1
            history.append({
                "iteration": 0,
                "signature": sig,
                "distance": current_distance
            })
        
        if verbose:
            print(f"Solution initiale (du pool): distance = {current_distance:.2f}")
        
        iteration = 0
        no_improvement_count = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Générer le voisinage
            neighbors = generate_neighborhood(current_solution)
            
            if not neighbors:
                break
            
            # Trouver le meilleur voisin
            best_neighbor = min(neighbors, key=lambda sol: sol.total_distance)
            best_neighbor_distance = best_neighbor.total_distance
            
            # Si amélioration
            if best_neighbor_distance < current_distance:
                current_solution = best_neighbor
                current_distance = best_neighbor_distance
                no_improvement_count = 0
                
                if log_history:
                    sig = current_solution.solution_signature()
                    visit_counter[sig] += 1
                    history.append({
                        "iteration": iteration,
                        "signature": sig,
                        "distance": current_distance
                    })
            else:
                no_improvement_count += 1
                if no_improvement_count >= 1:
                    break
        
        end_time = time.time()
        current_solution.computation_time = end_time - start_time
        current_solution.agent_name = "Hill Climbing (improve)"
        
        if log_history:
            current_solution.history = history
            current_solution.visit_counter = visit_counter
        
        return current_solution