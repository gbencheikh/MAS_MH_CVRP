import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import random
from typing import List, Tuple

from src.instance import VRPInstance
from src.solution import VRPSolution
from src.utils import * 
from collections import defaultdict

class Genetic_Algorithm:
    """
    Algorithme Génétique pour résoudre le CVRP.
    
    Utilise une population de solutions qui évoluent par sélection,
    croisement et mutation.
    """
    
    def __init__(self, instance: VRPInstance,
                 population_size: int = 50,
                 elite_size: int = 10,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.8):
        """
        Args:
            instance: Instance du problème CVRP
            population_size: Taille de la population
            elite_size: Nombre d'individus élites préservés
            mutation_rate: Probabilité de mutation (entre 0 et 1)
            crossover_rate: Probabilité de croisement (entre 0 et 1)
        """
        self.instance = instance
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def set_initial_solutions(self, solutions: List[VRPSolution]):
        """
        Initialiser la population .
        """
        self.initial_pool_solutions = solutions

    def _initialize_population(self) -> List[VRPSolution]:
        """
        Crée une population initiale de solutions.
        
        Returns:
            Liste de solutions initiales
        """
        population = []
        # 1. D'abord ajouter les solutions du pool (si disponibles)
        for sol in self.initial_pool_solutions:
            population.append(sol._copy_solution())
        
        # 2. Compléter avec des solutions aléatoires
        while len(population) < self.population_size:
            solution = VRPSolution(self.instance).generate_solution()
            solution.compute_total_distance()
            population.append(solution)
        return population
    
    def _evaluate_fitness(self, population: List[VRPSolution]) -> List[float]:
        """
        Calcule le fitness de chaque individu (inverse de la distance).
        
        Args:
            population: Liste de solutions
        
        Returns:
            Liste des fitness (plus élevé = meilleur)
        """
        fitness_scores = []
        for solution in population:
            # Fitness = 1 / distance (meilleure distance = meilleur fitness)
            if solution.total_distance > 0:
                fitness = 1.0 / solution.total_distance
            else:
                fitness = float('inf')
            fitness_scores.append(fitness)
        return fitness_scores
    
    def _tournament_selection(self, population: List[VRPSolution], 
                             fitness_scores: List[float], 
                             tournament_size: int = 3) -> VRPSolution:
        """
        Sélection par tournoi : choisit le meilleur parmi un échantillon aléatoire.
        
        Args:
            population: Population actuelle
            fitness_scores: Scores de fitness
            tournament_size: Taille du tournoi
        
        Returns:
            Solution sélectionnée
        """
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx]
    
    def _crossover(self, parent1: VRPSolution, parent2: VRPSolution) -> VRPSolution:
        """
        Croisement amélioré : Order Crossover (OX) adapté au VRP.
        """
        child = VRPSolution(self.instance)
        
        # Stratégie : choisir aléatoirement entre plusieurs méthodes
        method = random.choice(['route_based', 'customer_based', 'hybrid'])
        
        if method == 'route_based':
            # Méthode 1: Sélection aléatoire de routes complètes
            num_routes_p1 = random.randint(1, max(1, len(parent1.routes) - 1))
            selected_routes_p1 = random.sample(parent1.routes, num_routes_p1)
            
            visited = set()
            for route in selected_routes_p1:
                customers = [node for node in route if node != 0]
                if all(c not in visited for c in customers):
                    if child._route_feasible(route):
                        child.routes.append(route.copy())
                        visited.update(customers)
            
            # Compléter avec les routes de parent2
            for route in parent2.routes:
                customers = [node for node in route if node != 0]
                new_customers = [c for c in customers if c not in visited]
                
                if new_customers:
                    new_route = [0] + new_customers + [0]
                    if child._route_feasible(new_route):
                        child.routes.append(new_route)
                        visited.update(new_customers)
        
        elif method == 'customer_based':
            # Méthode 2: Ordre des clients de parent1, structure de parent2
            all_customers_p1 = []
            for route in parent1.routes:
                all_customers_p1.extend([node for node in route if node != 0])
            
            # Reconstruire en suivant l'ordre de parent1 mais la structure de parent2
            customer_index = 0
            for route in parent2.routes:
                num_customers = len([n for n in route if n != 0])
                new_route = [0]
                load = 0
                
                while customer_index < len(all_customers_p1) and len(new_route) - 1 < num_customers:
                    customer = all_customers_p1[customer_index]
                    demand = self.instance.demands[customer]
                    
                    if load + demand <= self.instance.vehicle_capacity:
                        new_route.append(customer)
                        load += demand
                        customer_index += 1
                    else:
                        break
                
                if len(new_route) > 1:
                    new_route.append(0)
                    child.routes.append(new_route)
        
        else:  # hybrid
            # Méthode 3: Mélange aléatoire avec probabilité 50/50
            all_routes = parent1.routes + parent2.routes
            random.shuffle(all_routes)
            
            visited = set()
            for route in all_routes:
                customers = [node for node in route if node != 0]
                new_customers = [c for c in customers if c not in visited]
                
                if new_customers and random.random() < 0.5:  # 50% de garder cette route
                    new_route = [0] + new_customers + [0]
                    if child._route_feasible(new_route):
                        child.routes.append(new_route)
                        visited.update(new_customers)
        
        # Ajouter les clients manquants
        missing = set(range(1, self.instance.num_nodes)) - set()
        for route in child.routes:
            for node in route:
                if node != 0:
                    missing.discard(node)
        
        for customer in missing:
            # Essayer d'insérer dans une route existante
            inserted = False
            for route in child.routes:
                route_load = sum(self.instance.demands[node] for node in route[1:-1])
                if route_load + self.instance.demands[customer] <= self.instance.vehicle_capacity:
                    # Insérer à une position aléatoire (pas toujours à la fin!)
                    insert_pos = random.randint(1, len(route) - 1)
                    route.insert(insert_pos, customer)
                    inserted = True
                    break
            
            if not inserted:
                child.routes.append([0, customer, 0])
        
        # Appliquer une petite perturbation aléatoire
        if random.random() < 0.3 and child.routes:
            neighbors = generate_neighborhood(child)
            if neighbors:
                child = random.choice(neighbors[:min(5, len(neighbors))])
        
        child.compute_total_distance()
        return child
    
    def _mutate(self, solution: VRPSolution) -> VRPSolution:
        """
        Applique une mutation à une solution.
        Choisit aléatoirement un voisin.
        
        Args:
            solution: Solution à muter
        
        Returns:
            Solution mutée
        """
        neighbors = generate_neighborhood(solution)
        
        if neighbors:
            # Choisir un voisin aléatoire
            mutated = random.choice(neighbors)
            return mutated
        else:
            return solution._copy_solution()
    
    def run(self, max_generations: int = 100, seed: int = None, log_history: bool = True, verbose: bool = True) -> VRPSolution:
        """
        Algorithme Génétique pour résoudre le CVRP.
        
        Args:
            max_generations: Nombre maximum de générations
            seed: Graine aléatoire pour la reproductibilité
            verbose: Afficher les informations de progression
        
        Returns:
            La meilleure solution trouvée
        """
        if seed is not None:
            random.seed(seed)
        
        start_time = time.time()
        
        # Initialiser la population
        population = self._initialize_population()
        
        # Trouver la meilleure solution initiale
        best_solution = min(population, key=lambda sol: sol.total_distance)._copy_solution()
        best_solution.compute_total_distance()
        best_distance = best_solution.total_distance
        
        if log_history:
            history = [] # historique des distances
            visit_counter = defaultdict(int)

            sig = best_solution.solution_signature()
            visit_counter[sig] += 1

            history.append({
                "iteration": 0,
                "signature": sig,
                "distance": best_distance
            })

        if verbose:
            print(f"Population initiale: {self.population_size} individus")
            print(f"Meilleure distance initiale: {best_distance:.2f}")
            print(f"Taille élite: {self.elite_size}")
            print(f"Taux de mutation: {self.mutation_rate}")
            print(f"Taux de croisement: {self.crossover_rate}")
        
        generations_since_improvement = 0
        max_no_improvement = max(20, max_generations // 5)
        
        for generation in range(max_generations):
            fitness_scores = self._evaluate_fitness(population)
            
            # Trier par fitness
            sorted_indices = sorted(range(len(population)), 
                                key=lambda i: fitness_scores[i], 
                                reverse=True)
            sorted_population = [population[i] for i in sorted_indices]
            
            # AMÉLIORATION 1: Élite plus petite pour plus de diversité
            elite_size = max(2, self.elite_size // 2)  # Réduire l'élite
            new_population = sorted_population[:elite_size]
            
            # AMÉLIORATION 2: Vérifier la diversité
            if generation % 10 == 0:
                # Calculer la diversité (écart-type des distances)
                distances = [sol.total_distance for sol in population]
                avg_dist = sum(distances) / len(distances)
                diversity = sum((d - avg_dist) ** 2 for d in distances) ** 0.5
                
                # Si diversité trop faible, réinjecter des solutions aléatoires
                if diversity < avg_dist * 0.05:  # Moins de 5% de variation
                    if verbose:
                        print(f"Génération {generation}: Diversité faible ({diversity:.2f}), réinjection!")
                    # Remplacer 30% de la population par de nouvelles solutions
                    num_new = len(population) // 3
                    for _ in range(num_new):
                        new_sol = VRPSolution(self.instance).generate_solution()
                        new_population.append(new_sol)
            
            # Générer le reste
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1._copy_solution()
                
                # AMÉLIORATION 3: Mutation plus agressive
                if random.random() < self.mutation_rate:
                    # Appliquer plusieurs mutations
                    for _ in range(random.randint(1, 3)):
                        child = self._mutate(child)
                
                child.compute_total_distance()
                new_population.append(child)
            
            population = new_population
            
            # Mettre à jour la meilleure solution
            current_best = min(population, key=lambda sol: sol.total_distance)
            if current_best.total_distance < best_distance:
                best_solution = current_best._copy_solution()
                best_solution.compute_total_distance()
                best_distance = best_solution.total_distance
                generations_since_improvement = 0
                
                if log_history:
                    sig = best_solution.solution_signature()
                    visit_counter[sig] += 1

                    history.append({
                        "iteration": generation,
                        "signature": sig,
                        "distance": best_distance
                    })

                if verbose:
                    print(f"Génération {generation + 1}: NOUVELLE MEILLEURE distance = {best_distance:.2f}")
            else:
                generations_since_improvement += 1
            
            # Affichage périodique
            if verbose and (generation + 1) % 10 == 0:
                avg_distance = sum(sol.total_distance for sol in population) / len(population)
                print(f"Génération {generation + 1}: meilleure = {best_distance:.2f}, "
                      f"moyenne = {avg_distance:.2f}")

            # Critère d'arrêt
            if generations_since_improvement >= max_no_improvement:
                if verbose:
                    print(f"Arrêt après {generations_since_improvement} générations sans amélioration")
                break
        
        end_time = time.time()
        best_solution.computation_time = end_time - start_time
        best_solution.agent_name = "Genetic Algorithm"
        
        if log_history:
            best_solution.history = history
            best_solution.visit_counter = visit_counter

        if verbose:
            print(f"\n{'='*60}")
            print(f"Algorithme Génétique terminé en {best_solution.computation_time:.2f} secondes")
            print(f"Nombre de générations: {generation + 1}")
            print(f"Distance finale: {best_distance:.2f}")
            print(f"{'='*60}")
        
        return best_solution
    
    def inject_and_continue(self, new_solutions: List[VRPSolution], 
                       max_generations: int = 20, seed: int = None,
                       log_history: bool = False, verbose: bool = False) -> VRPSolution:
        """
        Injecte de nouvelles solutions dans la population et continue l'exécution.
        
        Args:
            new_solutions: Nouvelles solutions du pool
            max_generations: Nombre de générations à exécuter
            
        Returns:
            Meilleure solution trouvée
        """
        if seed is not None:
            random.seed(seed)
        
        # Si c'est le premier appel, initialiser la population
        if not hasattr(self, 'population') or not self.population:
            self.population = self._initialize_population()
            self.generation_count = 0
        
        # Remplacer les K pires par les nouvelles solutions
        if new_solutions:
            # Trier la population par fitness (pire en dernier)
            self.population.sort(key=lambda sol: sol.total_distance)
            
            # Retirer les K pires
            num_to_replace = min(len(new_solutions), len(self.population))
            self.population = self.population[:-num_to_replace]
            
            # Ajouter les nouvelles
            for sol in new_solutions:
                self.population.append(sol._copy_solution())
        
        # Continuer l'évolution
        best_solution = min(self.population, key=lambda sol: sol.total_distance)._copy_solution()
        best_distance = best_solution.total_distance
        
        for gen in range(max_generations):
            self.generation_count += 1
            
            # Évaluer le fitness
            fitness_scores = self._evaluate_fitness(self.population)
            
            # Trier par fitness
            sorted_indices = sorted(range(len(self.population)), 
                                key=lambda i: fitness_scores[i], 
                                reverse=True)
            sorted_population = [self.population[i] for i in sorted_indices]
            
            # Élite
            new_population = sorted_population[:self.elite_size]
            
            # Générer le reste
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(self.population, fitness_scores)
                parent2 = self._tournament_selection(self.population, fitness_scores)
                
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1._copy_solution()
                
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                child.compute_total_distance()
                new_population.append(child)
            
            self.population = new_population
            
            # Mettre à jour la meilleure
            current_best = min(self.population, key=lambda sol: sol.total_distance)
            if current_best.total_distance < best_distance:
                best_solution = current_best._copy_solution()
                best_distance = best_solution.total_distance
        
        best_solution.agent_name = "Genetic Algorithm (continue)"
        return best_solution