import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
from collections import deque
from collections import defaultdict

from src.instance import VRPInstance
from src.solution import VRPSolution
from src.utils import * 

class Tabu_Search: 
    def __init__(self, instance: VRPInstance, tabu_tenure: int = 10): 
        self.instance = instance
        self.tabu_list: deque = deque(maxlen=tabu_tenure)


    def _solution_hash(self, solution: VRPSolution) -> Tuple:
        """
        Hash basé sur les ensembles de clients par route (ordre indépendant).
        """
        route_sets = []
        for route in solution.routes:
            customers = frozenset(node for node in route if node != 0)
            route_sets.append(customers)
        
        # Trier pour avoir un ordre canonique
        return tuple(sorted(route_sets, key=lambda x: min(x) if x else 0))
    
    def _is_tabu(self, solution: VRPSolution) -> bool:
        """Vérifie si une solution est dans la liste taboue."""
        solution_hash = self._solution_hash(solution)
        return solution_hash in self.tabu_list
    
    def _add_to_tabu(self, solution: VRPSolution):
        """Ajoute une solution à la liste taboue."""
        solution_hash = self._solution_hash(solution)
        self.tabu_list.append(solution_hash)

    def run(self, max_iterations: int = 1000, aspiration: bool = True, log_history: bool = True, verbose: bool = True) -> VRPSolution:
        """
        Algorithme Tabu Search pour résoudre le CVRP.
        
        Args:
            max_iterations: Nombre maximum d'itérations
            aspiration: Activer le critère d'aspiration (accepter une solution taboue si elle améliore le meilleur)
            verbose: Afficher les informations de progression
            log_history: Sauvegarder l'historique et le compteur de visites
        
        Returns:
            La meilleure solution trouvée
        """
        start_time = time.time()
        
        # Générer la solution initiale
        current_solution = VRPSolution(self.instance).generate_solution()
        current_distance = current_solution.total_distance
        
        # Meilleure solution globale
        best_solution = current_solution._copy_solution()
        best_distance = current_distance
        if log_history:
            history = []
            visit_counter = defaultdict(int)

            def solution_signature(solution: VRPSolution):
                """Signature hashable pour le suivi des visites"""
                route_tuples = [tuple(route) for route in solution.routes]
                return tuple(sorted(route_tuples))

            # Sauvegarde solution initiale
            sig = solution_signature(current_solution)
            visit_counter[sig] += 1
            history.append({
                "iteration": 0,
                "signature": sig,
                "distance": current_distance
            })

        if verbose:
            print(f"Solution initiale: distance = {current_distance:.2f}")
            print(f"Nombre de routes: {len(current_solution.routes)}")
            print(f"Critère d'aspiration: {'Activé' if aspiration else 'Désactivé'}")
        
        iteration = 0
        iterations_since_improvement = 0
        max_no_improvement = max(50, max_iterations // 10)  # Arrêt après un certain nombre d'itérations sans amélioration
        
        while iteration < max_iterations:
            iteration += 1
            
            # Générer le voisinage (même fonction que Hill Climbing)
            neighbors = generate_neighborhood(current_solution)
            
            if not neighbors:
                if verbose:
                    print("Aucun voisin généré, arrêt.")
                break
            
            # Trouver le meilleur voisin non-tabou (ou qui satisfait l'aspiration)
            best_neighbor = None
            best_neighbor_distance = float('inf')
            
            for neighbor in neighbors:
                is_tabu = self._is_tabu(neighbor)
                
                # Accepter si non-tabou OU si critère d'aspiration satisfait
                if not is_tabu or (aspiration and neighbor.total_distance < best_distance):
                    if neighbor.total_distance < best_neighbor_distance:
                        best_neighbor = neighbor
                        best_neighbor_distance = neighbor.total_distance
            
            # Si aucun voisin acceptable trouvé
            if best_neighbor is None:
                if verbose:
                    print(f"Aucun voisin acceptable à l'itération {iteration}, arrêt.")
                break
            
            # Mettre à jour la solution courante
            current_solution = best_neighbor
            current_distance = best_neighbor_distance
            
            # Ajouter la solution courante à la liste taboue
            self._add_to_tabu(current_solution)
            
            if log_history:
                sig = solution_signature(current_solution)
                visit_counter[sig] += 1
                history.append({
                    "iteration": iteration,
                    "signature": sig,
                    "distance": current_distance
                })

            # Mettre à jour la meilleure solution globale
            if current_distance < best_distance:
                best_solution = current_solution._copy_solution()
                best_distance = current_distance
                iterations_since_improvement = 0
                
                if verbose:
                    print(f"Itération {iteration}: NOUVELLE MEILLEURE distance = {best_distance:.2f}")
            else:
                iterations_since_improvement += 1
            
            # Affichage périodique
            if verbose and iteration % 50 == 0:
                print(f"Itération {iteration}: distance courante = {current_distance:.2f}, "
                      f"meilleure = {best_distance:.2f}, taille tabu = {len(self.tabu_list)}")
            
            # Critère d'arrêt: trop d'itérations sans amélioration
            if iterations_since_improvement >= max_no_improvement:
                if verbose:
                    print(f"Arrêt après {iterations_since_improvement} itérations sans amélioration")
                break
        
        end_time = time.time()
        best_solution.computation_time = end_time - start_time
        best_solution.agent_name = "Tabu Search"
        
        if log_history:
            best_solution.history = history
            best_solution.visit_counter = visit_counter

        if verbose:
            print(f"\n{'='*60}")
            print(f"Tabu Search terminé en {best_solution.computation_time:.2f} secondes")
            print(f"Nombre d'itérations: {iteration}")
            print(f"Distance finale: {best_distance:.2f}")
            if current_distance > 0:
                improvement = ((current_distance - best_distance) / current_distance * 100)
                print(f"Amélioration depuis l'initial: {improvement:.2f}%")
            print(f"{'='*60}")
        
        return best_solution