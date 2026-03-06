import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import random
from typing import List, Tuple, Dict

from src.instance import VRPInstance
from src.solution import VRPSolution
from src.utils import * 
from collections import defaultdict

class Ant:
    """
    Représente une fourmi qui construit une solution pour le CVRP.
    """
    
    def __init__(self, instance: VRPInstance, pheromone_matrix: Dict, alpha: float, beta: float):
        self.instance = instance
        self.pheromone_matrix = pheromone_matrix
        self.alpha = alpha
        self.beta = beta
        self.solution = VRPSolution(instance)
    
    def _calculate_visibility(self, i: int, j: int) -> float:
        """Calcule la visibilité (heuristique) entre deux nœuds."""
        distance = self.instance.distance_matrix[i][j]
        if distance > 0:
            return 1.0 / distance
        return 0.0
    
    def _calculate_probabilities(self, current_node: int, unvisited: set, current_load: int) -> List[Tuple[int, float]]:
        """
        Calcule les probabilités de transition vers les clients non visités.
        Amélioration: Prend en compte la demande pour favoriser les clients compatibles.
        """
        feasible_customers = []
        pheromone_values = []
        heuristic_values = []
        
        # Filtrer les clients faisables (respectant la capacité)
        for customer in unvisited:
            demand = self.instance.demands[customer]
            if current_load + demand <= self.instance.vehicle_capacity:
                pheromone = self.pheromone_matrix.get((current_node, customer), 0.1)
                distance = self.instance.distance_matrix[current_node][customer]
                
                # Heuristique améliorée : distance + ratio de demande
                if distance > 0:
                    # Favoriser les clients proches ET avec une demande appropriée
                    heuristic = (1.0 / distance) * (1.0 + demand / self.instance.vehicle_capacity)
                else:
                    heuristic = 1.0
                
                feasible_customers.append(customer)
                pheromone_values.append(pheromone)
                heuristic_values.append(heuristic)
        
        if not feasible_customers:
            return []
        
        # Calculer le total pour normalisation avec protection contre valeurs nulles
        total = sum(
            max(pheromone, 0.001) ** self.alpha * max(heuristic, 0.001) ** self.beta
            for pheromone, heuristic in zip(pheromone_values, heuristic_values)
        )
        
        # Calculer les probabilités
        probabilities = []
        for customer, pheromone, heuristic in zip(feasible_customers, pheromone_values, heuristic_values):
            if total > 0:
                prob = (max(pheromone, 0.001) ** self.alpha * max(heuristic, 0.001) ** self.beta) / total
            else:
                prob = 1.0 / len(feasible_customers)
            probabilities.append((customer, prob))
        
        return probabilities
    
    def _select_next_customer(self, probabilities: List[Tuple[int, float]]) -> int:
        """Sélectionne le prochain client selon les probabilités."""
        if not probabilities:
            return None
        
        customers, probs = zip(*probabilities)
        selected = random.choices(customers, weights=probs, k=1)[0]
        return selected
    
    def construct_solution(self):
        """
        Construit une solution complète en suivant les phéromones et l'heuristique.
        """
        self.solution = VRPSolution(self.instance)
        unvisited = set(range(1, self.instance.num_nodes))
        
        while unvisited:
            route = [0]  # Commencer au dépôt
            current_node = 0
            current_load = 0
            
            while unvisited:
                # Calculer les probabilités
                probabilities = self._calculate_probabilities(current_node, unvisited, current_load)
                
                if not probabilities:
                    # Aucun client ne peut être ajouté, terminer cette route
                    break
                
                # Sélectionner le prochain client
                next_customer = self._select_next_customer(probabilities)
                
                if next_customer is None:
                    break
                
                # Ajouter le client à la route
                route.append(next_customer)
                current_load += self.instance.demands[next_customer]
                current_node = next_customer
                unvisited.remove(next_customer)
            
            # Retourner au dépôt
            route.append(0)
            self.solution.add_route(route)
        
        self.solution.compute_total_distance()
    
    def get_solution(self) -> VRPSolution:
        """Retourne la solution construite par la fourmi."""
        return self.solution


class Ant_Colony_Optimization:
    """
    Algorithme de Colonie de Fourmis (ACO) pour résoudre le CVRP.
    Version collaborative avec solution pool optionnel.
    """
    
    def __init__(self, instance: VRPInstance,
                 num_ants: int = 20,
                 alpha: float = 1.0,
                 beta: float = 3.0,
                 evaporation_rate: float = 0.2,
                 q: float = 100.0):
        """
        Args:
            instance: Instance du problème CVRP
            num_ants: Nombre de fourmis par itération
            alpha: Importance des phéromones
            beta: Importance de la visibilité/heuristique
            evaporation_rate: Taux d'évaporation des phéromones
            q: Constante pour le dépôt de phéromones
        """
        self.instance = instance
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        
        # Initialiser la matrice de phéromones
        self.pheromone_matrix = self._initialize_pheromones()
    
    def _initialize_pheromones(self) -> Dict[Tuple[int, int], float]:
        """
        Initialise la matrice de phéromones avec des valeurs basées sur une solution gloutonne.
        Amélioration: Utiliser une heuristique pour l'initialisation.
        """
        pheromones = {}
        n = self.instance.num_nodes
        
        # Créer une solution initiale gloutonne pour estimer les bonnes valeurs
        initial_solution = VRPSolution(self.instance).generate_solution()
        
        if initial_solution.total_distance > 0:
            initial_pheromone = self.q / initial_solution.total_distance
        else:
            initial_pheromone = 1.0
        
        # Initialiser toutes les arêtes avec une valeur de base
        for i in range(n):
            for j in range(n):
                if i != j:
                    pheromones[(i, j)] = initial_pheromone * 0.1  # 10% de la valeur optimale estimée
        
        # Renforcer les arêtes de la solution initiale
        for route in initial_solution.routes:
            for i in range(len(route) - 1):
                edge = (route[i], route[i + 1])
                if edge in pheromones:
                    pheromones[edge] = initial_pheromone
        
        return pheromones
    
    def _evaporate_pheromones(self):
        """
        Évapore les phéromones sur tous les arcs.
        """
        for edge in self.pheromone_matrix:
            self.pheromone_matrix[edge] *= (1 - self.evaporation_rate)
    
    def _deposit_pheromones(self, ants: List[Ant]):
        """
        Dépose des phéromones sur les arcs utilisés par les fourmis.
        """
        for ant in ants:
            solution = ant.get_solution()
            
            solution.compute_total_distance() 

            if solution.total_distance > 0:
                delta_pheromone = self.q / solution.total_distance
            else:
                continue
            
            # Déposer sur chaque arc de chaque route
            for route in solution.routes:
                for i in range(len(route) - 1):
                    node_from = route[i]
                    node_to = route[i + 1]
                    
                    # Dépôt bidirectionnel
                    if (node_from, node_to) in self.pheromone_matrix:
                        self.pheromone_matrix[(node_from, node_to)] += delta_pheromone
                    if (node_to, node_from) in self.pheromone_matrix:
                        self.pheromone_matrix[(node_to, node_from)] += delta_pheromone
    
    def _deposit_pheromones_elite(self, solution: VRPSolution, factor: float = 3.0):
        """
        Dépose des phéromones supplémentaires pour une solution élite.
        """
        solution.compute_total_distance()
        if solution.total_distance > 0:
            delta_pheromone = (self.q / solution.total_distance) * factor
        else:
            return
        
        for route in solution.routes:
            for i in range(len(route) - 1):
                node_from = route[i]
                node_to = route[i + 1]
                
                if (node_from, node_to) in self.pheromone_matrix:
                    self.pheromone_matrix[(node_from, node_to)] += delta_pheromone
                if (node_to, node_from) in self.pheromone_matrix:
                    self.pheromone_matrix[(node_to, node_from)] += delta_pheromone
    
    def _apply_local_search(self, solution: VRPSolution) -> VRPSolution:
        """
        Applique une recherche locale AGRESSIVE sur une solution.
        Amélioration: Plus d'itérations et meilleure exploration.
        """
        current = solution._copy_solution()
        max_iterations = 20  # Augmenté de 5 à 20
        
        for iteration in range(max_iterations):
            improved = False
            neighbors = generate_neighborhood(current)
            
            if not neighbors:
                break
            
            # Prendre le MEILLEUR voisin (Hill Climbing dans la recherche locale)
            best_neighbor = min(neighbors, key=lambda sol: sol.total_distance)


            if best_neighbor.total_distance < current.total_distance:
                current = best_neighbor
                improved = True
            
            if not improved:
                break
        
        return current
    
    def run(self, max_iterations: int = 100, 
            local_search: bool = True,
            seed: int = None, 
            log_history: bool = True,
            verbose: bool = True) -> VRPSolution:
        """
        Algorithme de Colonie de Fourmis standard pour résoudre le CVRP.
        
        Args:
            max_iterations: Nombre maximum d'itérations
            local_search: Appliquer une recherche locale aux meilleures solutions
            seed: Graine aléatoire pour la reproductibilité
            verbose: Afficher les informations de progression
        
        Returns:
            La meilleure solution trouvée
        """
        if seed is not None:
            random.seed(seed)
        
        start_time = time.time()
        
        # Meilleure solution globale
        best_solution = None
        best_distance = float('inf')
        
        if log_history:
            history = [] # historique des distances
            visit_counter = defaultdict(int)

        if verbose:
            print(f"Nombre de fourmis: {self.num_ants}")
            print(f"Alpha (phéromones): {self.alpha}")
            print(f"Beta (visibilité): {self.beta}")
            print(f"Taux d'évaporation: {self.evaporation_rate}")
            print(f"Q (constante dépôt): {self.q}")
            print(f"Recherche locale: {'Activée' if local_search else 'Désactivée'}")
        
        iterations_since_improvement = 0
        max_no_improvement = max(20, max_iterations // 5)
        
        for iteration in range(max_iterations):
            # Créer les fourmis et construire leurs solutions
            ants = []
            for _ in range(self.num_ants):
                ant = Ant(self.instance, self.pheromone_matrix, self.alpha, self.beta)
                ant.construct_solution()
                ants.append(ant)
            
            # Trouver la meilleure solution de cette itération
            iteration_best = None
            iteration_best_distance = float('inf')
            
            for ant in ants:
                solution = ant.get_solution()
                
                # DEBUG: Afficher les premières solutions
                if iteration == 0 and verbose:
                    print(f"  Debug - Solution fourmi: distance={solution.total_distance:.2f}, "
                        f"routes={len(solution.routes)}, "
                        f"faisable={solution.is_feasible()[0]}")
                
                if solution.total_distance < iteration_best_distance:
                    iteration_best = solution
                    iteration_best_distance = solution.total_distance
            
            if iteration == 0 and verbose:
                print(f"  Debug - Meilleure itération: distance={iteration_best.total_distance:.2f}")
                print(f"  Debug - Routes: {iteration_best.routes}")

            # Recherche locale (si activée)
            if local_search and iteration_best:
                before_ls = iteration_best.total_distance
                iteration_best = self._apply_local_search(iteration_best)
                after_ls = iteration_best.total_distance
                
                if iteration == 0 and verbose:
                    print(f"  Debug - Après recherche locale: avant={before_ls:.2f}, après={after_ls:.2f}")
            
            if log_history:
                sig = iteration_best.solution_signature()
                visit_counter[sig] += 1

                history.append({
                    "iteration": iteration,
                    "signature": sig,
                    "distance": iteration_best.total_distance
                })

            # Mettre à jour best_solution
            if iteration_best and iteration_best.total_distance < best_distance:
                if verbose : 
                    print(f"  Debug - MISE À JOUR: nouvelle meilleure={iteration_best.total_distance:.2f}")
                    
                best_solution = iteration_best._copy_solution()
                best_solution.compute_total_distance() 

                best_distance = best_solution.total_distance
                        
                if verbose:
                    print(f"Itération {iteration + 1}: NOUVELLE MEILLEURE distance = {best_distance:.2f}")
            else:
                iterations_since_improvement += 1
            
            # Évaporation des phéromones
            self._evaporate_pheromones()
            
            # Dépôt de phéromones (toutes les fourmis)
            self._deposit_pheromones(ants)
            
            # Dépôt élitiste pour la meilleure solution globale
            if best_solution:
                self._deposit_pheromones_elite(best_solution, factor=3.0)
            
            # Affichage périodique
            if verbose and (iteration + 1) % 10 == 0:
                avg_distance = sum(ant.get_solution().total_distance for ant in ants) / len(ants)
                print(f"Itération {iteration + 1}: meilleure = {best_distance:.2f}, "
                      f"moyenne = {avg_distance:.2f}")
            
            # Critère d'arrêt
            if iterations_since_improvement >= max_no_improvement:
                if verbose:
                    print(f"Arrêt après {iterations_since_improvement} itérations sans amélioration")
                break
        
        end_time = time.time()
        
        if best_solution:
            best_solution.computation_time = end_time - start_time
            best_solution.agent_name = "Ant Colony Optimization"
        
        if log_history:
            best_solution.history = history
            best_solution.visit_counter = visit_counter

        if verbose:
            print(f"\n{'='*60}")
            print(f"ACO terminé en {best_solution.computation_time:.2f} secondes")
            print(f"Nombre d'itérations: {iteration + 1}")
            print(f"Distance finale: {best_distance:.2f}")
            print(f"{'='*60}")
        
        if verbose:
            print(f"\n  Debug FINAL - best_solution existe? {best_solution is not None}")
            if best_solution:
                print(f"  Debug FINAL - best_distance: {best_distance:.2f}")
                print(f"  Debug FINAL - best_solution.total_distance: {best_solution.total_distance:.2f}")
                print(f"  Debug FINAL - Nombre de routes: {len(best_solution.routes)}")
    
        return best_solution