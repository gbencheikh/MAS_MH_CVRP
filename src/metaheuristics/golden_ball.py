import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import random
from typing import List, Tuple, Set
from collections import defaultdict
from src.instance import VRPInstance
from src.solution import VRPSolution
from src.utils import * 

class Golden_Ball:
    """
    Algorithme Golden Ball pour résoudre le CVRP.
    
    Inspiré du jeu télévisé "Golden Balls", cet algorithme utilise une approche
    compétitive où des solutions "jouent" entre elles. Les meilleures solutions
    survivent et se combinent, tandis que les moins bonnes sont éliminées.
    """
    
    def __init__(self, instance: VRPInstance,
                 population_size: int = 30,
                 num_rounds: int = 5,
                 cooperation_rate: float = 0.7,
                 mutation_rate: float = 0.3):
        """
        Args:
            instance: Instance du problème CVRP
            population_size: Taille de la population initiale
            num_rounds: Nombre de rounds de compétition par génération
            cooperation_rate: Probabilité de coopération vs compétition
            mutation_rate: Probabilité de mutation après combinaison
        """
        self.instance = instance
        self.population_size = population_size
        self.num_rounds = num_rounds
        self.cooperation_rate = cooperation_rate
        self.mutation_rate = mutation_rate
    
    def set_initial_solutions(self, solutions: List[VRPSolution]):
        """
        Initialiser la population .
        """
        self.initial_pool_solutions = solutions

    def _initialize_population(self) -> List[VRPSolution]:
        """
        Crée une population initiale diverse de solutions.
        
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
    
    def _cooperate(self, sol1: VRPSolution, sol2: VRPSolution) -> Tuple[VRPSolution, VRPSolution]:
        """
        Mode coopération: les deux solutions échangent des informations
        et créent deux nouvelles solutions améliorées.
        
        Args:
            sol1: Première solution
            sol2: Deuxième solution
        
        Returns:
            Tuple de deux nouvelles solutions
        """
        # Créer deux enfants par croisement
        child1 = self._crossover(sol1, sol2)
        child2 = self._crossover(sol2, sol1)
        
        # Améliorer avec recherche locale
        neighbors1 = generate_neighborhood(child1)
        if neighbors1:
            child1 = min(neighbors1, key=lambda s: s.total_distance)
        
        neighbors2 = generate_neighborhood(child2)
        if neighbors2:
            child2 = min(neighbors2, key=lambda s: s.total_distance)
        
        return child1, child2
    
    def _compete(self, sol1: VRPSolution, sol2: VRPSolution) -> Tuple[VRPSolution, VRPSolution]:
        """
        Mode compétition: seule la meilleure solution survit,
        l'autre est remplacée par une mutation de la gagnante.
        
        Args:
            sol1: Première solution
            sol2: Deuxième solution
        
        Returns:
            Tuple de deux solutions (gagnante + mutation)
        """
        # Déterminer le gagnant
        if sol1.total_distance < sol2.total_distance:
            winner = sol1
        else:
            winner = sol2
        
        # Le gagnant survit
        survivor = winner._copy_solution()
        
        # Créer une mutation du gagnant
        neighbors = generate_neighborhood(winner)
        if neighbors:
            mutant = random.choice(neighbors)
        else:
            mutant = winner._copy_solution()
        
        return survivor, mutant
    
    def _crossover(self, parent1: VRPSolution, parent2: VRPSolution) -> VRPSolution:
        """
        Croisement de deux solutions (similaire à l'algorithme génétique).
        
        Args:
            parent1: Première solution parente
            parent2: Deuxième solution parente
        
        Returns:
            Solution enfant
        """
        child = VRPSolution(self.instance)
        
        # Combiner des routes des deux parents
        all_routes = parent1.routes + parent2.routes
        random.shuffle(all_routes)
        
        visited = set()
        
        for route in all_routes:
            customers = [node for node in route if node != 0]
            new_customers = [c for c in customers if c not in visited]
            
            if new_customers:
                new_route = [0] + new_customers + [0]
                
                if child._route_feasible(new_route):
                    child.routes.append(new_route)
                    visited.update(new_customers)
                else:
                    # Diviser si trop chargé
                    temp_route = [0]
                    temp_load = 0
                    
                    for customer in new_customers:
                        demand = self.instance.demands[customer]
                        if temp_load + demand <= self.instance.vehicle_capacity:
                            temp_route.append(customer)
                            temp_load += demand
                            visited.add(customer)
                        else:
                            if len(temp_route) > 1:
                                temp_route.append(0)
                                child.routes.append(temp_route)
                            temp_route = [0, customer]
                            temp_load = demand
                            visited.add(customer)
                    
                    if len(temp_route) > 1:
                        temp_route.append(0)
                        child.routes.append(temp_route)
        
        # Ajouter les clients manquants
        missing = set(range(1, self.instance.num_nodes)) - visited
        for customer in missing:
            inserted = False
            for route in child.routes:
                route_load = sum(self.instance.demands[node] for node in route[1:-1])
                if route_load + self.instance.demands[customer] <= self.instance.vehicle_capacity:
                    route.insert(-1, customer)
                    inserted = True
                    break
            
            if not inserted:
                child.routes.append([0, customer, 0])
        
        child.compute_total_distance()
        return child
    
    def _play_round(self, population: List[VRPSolution]) -> List[VRPSolution]:
        """
        Joue un round de Golden Ball: apparie les solutions et les fait jouer.
        
        Args:
            population: Population actuelle
        
        Returns:
            Nouvelle population après le round
        """
        # Mélanger la population pour créer des paires aléatoires
        shuffled = population.copy()
        random.shuffle(shuffled)
        
        new_population = []
        
        # Créer des paires et les faire jouer
        for i in range(0, len(shuffled) - 1, 2):
            sol1 = shuffled[i]
            sol2 = shuffled[i + 1]
            
            # Décider: coopération ou compétition
            if random.random() < self.cooperation_rate:
                # Coopération: les deux gagnent
                child1, child2 = self._cooperate(sol1, sol2)
            else:
                # Compétition: un seul gagne vraiment
                child1, child2 = self._compete(sol1, sol2)
            
            # Mutation possible
            if random.random() < self.mutation_rate:
                neighbors = generate_neighborhood(child1)
                if neighbors:
                    child1 = random.choice(neighbors)
            
            if random.random() < self.mutation_rate:
                neighbors = generate_neighborhood(child2)
                if neighbors:
                    child2 = random.choice(neighbors)
            
            new_population.extend([child1, child2])
        
        # Si nombre impair, ajouter le dernier
        if len(shuffled) % 2 == 1:
            new_population.append(shuffled[-1])
        
        return new_population

    def _select_survivors(self, population: List[VRPSolution]) -> List[VRPSolution]:
        """
        Sélectionne les meilleurs individus pour la prochaine génération.
        
        Args:
            population: Population actuelle
        
        Returns:
            Population réduite aux meilleurs
        """
        # Trier par fitness (meilleur en premier)
        sorted_pop = sorted(population, key=lambda sol: sol.total_distance)
        
        # Garder les meilleurs
        survivors = sorted_pop[:self.population_size]
        
        return survivors
    
    def run(self, max_iterations: int = 100, seed: int = None, log_history: bool = True, verbose: bool = True) -> VRPSolution:
        """
        Algorithme Golden Ball pour résoudre le CVRP.
        
        Args:
            max_iterations: Nombre maximum d'itérations
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

        # Meilleure solution globale
        current_solution = min(population, key=lambda sol: sol.total_distance)
    
        best_solution = current_solution._copy_solution()
        best_solution.compute_total_distance()
        best_distance = best_solution.total_distance
        
        if log_history:
            history = []
            visit_counter = defaultdict(int)

            sig = current_solution.solution_signature()
            visit_counter[sig] += 1
            history.append({
                "iteration": 0, 
                "signature": sig, 
                "distance": best_distance
            })

        if verbose:
            print(f"Population initiale: {self.population_size} individus")
            print(f"Meilleure distance initiale: {best_distance:.2f}")
            print(f"Nombre de rounds par itération: {self.num_rounds}")
            print(f"Taux de coopération: {self.cooperation_rate}")
            print(f"Taux de mutation: {self.mutation_rate}")
        
        iterations_since_improvement = 0
        max_no_improvement = max(50, max_iterations // 5)
        
        for iteration in range(max_iterations):
            iteration += 1

            # Jouer plusieurs rounds 
            for _ in range(self.num_rounds): 
                population = self._play_round(population) 
                
                # Sélectionner les survivants 
                population = self._select_survivors(population) 
                

                # Mettre à jour la meilleure solution 
                current_best = min(population, key=lambda sol: sol.total_distance) 
                current_best.compute_total_distance()
                
                if current_best.total_distance < best_distance: 
                    best_solution = current_best._copy_solution() 
                    best_solution.compute_total_distance()
                    best_distance = best_solution.total_distance 
                    iterations_since_improvement = 0

                    if verbose:
                        print(f"Itération {iteration}: NOUVELLE MEILLEURE = {best_distance:.2f}")
            else:
                iterations_since_improvement += 1
            
            # Historique
            if log_history:
                sig = current_best.solution_signature()
                visit_counter[sig] += 1
                history.append({
                    "iteration": iteration, 
                    "signature": sig, 
                    "distance": current_best.total_distance
                })

            # Affichage périodique
            if verbose and (iteration + 1) % 10 == 0:
                avg_distance = sum(sol.total_distance for sol in population) / len(population)
                print(f"Itération {iteration + 1}: meilleure = {best_distance:.2f}, "
                      f"moyenne = {avg_distance:.2f}")
            
            # Critère d'arrêt
            if iterations_since_improvement >= max_no_improvement:
                if verbose:
                    print(f"Arrêt après {iterations_since_improvement} itérations sans amélioration")
                break
        
        end_time = time.time()
        best_solution.computation_time = end_time - start_time
        best_solution.agent_name = "Golden Ball"
        
        if log_history:
            best_solution.history = history
            best_solution.visit_counter = visit_counter

        if verbose:
            print(f"\n{'='*60}")
            print(f"Golden Ball terminé en {best_solution.computation_time:.2f} secondes")
            print(f"Nombre d'itérations: {iteration + 1}")
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