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
        self.instance = instance
        self.population_size = population_size
        self.num_rounds = num_rounds
        self.cooperation_rate = cooperation_rate
        self.mutation_rate = mutation_rate
        self.initial_pool_solutions = []
        # État persistant pour inject_and_continue
        self.population = []
        self.generation_count = 0

    def set_initial_solutions(self, solutions: List[VRPSolution]):
        """Initialiser la population."""
        self.initial_pool_solutions = solutions

    def _initialize_population(self) -> List[VRPSolution]:
        population = []
        for sol in self.initial_pool_solutions:
            population.append(sol._copy_solution())
        while len(population) < self.population_size:
            solution = VRPSolution(self.instance).generate_solution()
            solution.compute_total_distance()
            population.append(solution)
        return population

    def _quick_improve(self, solution: VRPSolution) -> VRPSolution:
        """
        Amélioration locale rapide : applique un seul opérateur aléatoire
        sur une paire de positions choisie aléatoirement.
        Évite de générer tout le voisinage.
        """
        if not solution.routes:
            return solution

        op = random.randint(0, 2)

        if op == 0:
            # Swap intra-route sur une route aléatoire non triviale
            candidates = [r for r in solution.routes if len(r) > 3]
            if candidates:
                route = random.choice(candidates)
                route_idx = solution.routes.index(route)
                positions = list(range(1, len(route) - 1))
                if len(positions) >= 2:
                    i, j = random.sample(positions, 2)
                    neighbor = solution._copy_solution()
                    neighbor.routes[route_idx][i], neighbor.routes[route_idx][j] =                         neighbor.routes[route_idx][j], neighbor.routes[route_idx][i]
                    if neighbor._is_route_feasible(route_idx):
                        neighbor.compute_total_distance()
                        return neighbor if neighbor.total_distance < solution.total_distance else solution

        elif op == 1:
            # 2-opt sur une route aléatoire non triviale
            candidates = [r for r in solution.routes if len(r) > 3]
            if candidates:
                route = random.choice(candidates)
                route_idx = solution.routes.index(route)
                positions = list(range(1, len(route) - 1))
                if len(positions) >= 2:
                    i, j = sorted(random.sample(positions, 2))
                    neighbor = solution._copy_solution()
                    neighbor.routes[route_idx][i:j+1] = reversed(neighbor.routes[route_idx][i:j+1])
                    if neighbor._is_route_feasible(route_idx):
                        neighbor.compute_total_distance()
                        return neighbor if neighbor.total_distance < solution.total_distance else solution

        else:
            # Relocate : déplacer un client aléatoire vers une autre route
            if len(solution.routes) >= 2:
                from_idx = random.randrange(len(solution.routes))
                from_route = solution.routes[from_idx]
                if len(from_route) > 3:
                    to_idx = random.choice([i for i in range(len(solution.routes)) if i != from_idx])
                    pos_from = random.randint(1, len(from_route) - 2)
                    pos_to = random.randint(1, len(solution.routes[to_idx]))
                    neighbor = solution._copy_solution()
                    client = neighbor.routes[from_idx].pop(pos_from)
                    neighbor.routes[to_idx].insert(pos_to, client)
                    if (neighbor._is_route_feasible(from_idx) and
                            neighbor._is_route_feasible(to_idx)):
                        neighbor.compute_total_distance()
                        return neighbor if neighbor.total_distance < solution.total_distance else solution

        return solution

    def _cooperate(self, sol1: VRPSolution, sol2: VRPSolution) -> Tuple[VRPSolution, VRPSolution]:
        child1 = self._crossover(sol1, sol2)
        child2 = self._crossover(sol2, sol1)
        child1 = self._quick_improve(child1)
        child2 = self._quick_improve(child2)
        return child1, child2

    def _compete(self, sol1: VRPSolution, sol2: VRPSolution) -> Tuple[VRPSolution, VRPSolution]:
        winner = sol1 if sol1.total_distance < sol2.total_distance else sol2
        survivor = winner._copy_solution()
        # Mutation rapide : un seul opérateur aléatoire au lieu du voisinage complet
        mutant = self._quick_improve(winner._copy_solution())
        return survivor, mutant

    def _crossover(self, parent1: VRPSolution, parent2: VRPSolution) -> VRPSolution:
        child = VRPSolution(self.instance)
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
        shuffled = population.copy()
        random.shuffle(shuffled)
        new_population = []
        for i in range(0, len(shuffled) - 1, 2):
            sol1 = shuffled[i]
            sol2 = shuffled[i + 1]
            if random.random() < self.cooperation_rate:
                child1, child2 = self._cooperate(sol1, sol2)
            else:
                child1, child2 = self._compete(sol1, sol2)
            if random.random() < self.mutation_rate:
                child1 = self._quick_improve(child1)
            if random.random() < self.mutation_rate:
                child2 = self._quick_improve(child2)
            new_population.extend([child1, child2])
        if len(shuffled) % 2 == 1:
            new_population.append(shuffled[-1])
        return new_population

    def _select_survivors(self, population: List[VRPSolution]) -> List[VRPSolution]:
        sorted_pop = sorted(population, key=lambda sol: sol.total_distance)
        return sorted_pop[:self.population_size]

    def run(self, max_iterations: int = 100, seed: int = None, log_history: bool = True, verbose: bool = True) -> VRPSolution:
        if seed is not None:
            random.seed(seed)
        
        start_time = time.time()
        population = self._initialize_population()

        current_solution = min(population, key=lambda sol: sol.total_distance)
        best_solution = current_solution._copy_solution()
        best_solution.compute_total_distance()
        best_distance = best_solution.total_distance
        
        if log_history:
            history = []
            visit_counter = defaultdict(int)
            sig = current_solution.solution_signature()
            visit_counter[sig] += 1
            history.append({"iteration": 0, "signature": sig, "distance": best_distance})

        if verbose:
            print(f"Population initiale: {self.population_size} individus")
            print(f"Meilleure distance initiale: {best_distance:.2f}")

        iterations_since_improvement = 0
        max_no_improvement = max(50, max_iterations // 5)
        
        for iteration in range(max_iterations):
            iteration += 1
            for _ in range(self.num_rounds):
                population = self._play_round(population)
                population = self._select_survivors(population)
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

            if log_history:
                sig = current_best.solution_signature()
                visit_counter[sig] += 1
                history.append({"iteration": iteration, "signature": sig, "distance": current_best.total_distance})

            if verbose and (iteration + 1) % 10 == 0:
                avg_distance = sum(sol.total_distance for sol in population) / len(population)
                print(f"Itération {iteration + 1}: meilleure = {best_distance:.2f}, moyenne = {avg_distance:.2f}")

            if iterations_since_improvement >= max_no_improvement:
                if verbose:
                    print(f"Arrêt après {iterations_since_improvement} itérations sans amélioration")
                break

        # Sauvegarder l'état pour inject_and_continue
        self.population = population
        self.generation_count = iteration

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
                            max_iterations: int = 20, seed: int = None,
                            log_history: bool = False, verbose: bool = False) -> VRPSolution:
        """
        Injecte de nouvelles solutions dans la population et continue l'exécution
        avec la logique propre à Golden Ball (_play_round / _select_survivors).

        Args:
            new_solutions: Nouvelles solutions du pool à injecter
            max_iterations: Nombre d'itérations Golden Ball à exécuter
            seed: Graine aléatoire
            log_history: Activer l'historique
            verbose: Affichage

        Returns:
            Meilleure solution trouvée
        """
        if seed is not None:
            random.seed(seed)

        # Initialiser la population si premier appel
        if not self.population:
            self.population = self._initialize_population()
            self.generation_count = 0

        # Injecter les nouvelles solutions en remplaçant les K pires
        if new_solutions:
            self.population.sort(key=lambda sol: sol.total_distance)
            num_to_replace = min(len(new_solutions), len(self.population))
            self.population = self.population[:-num_to_replace]
            for sol in new_solutions:
                self.population.append(sol._copy_solution())

        best_solution = min(self.population, key=lambda sol: sol.total_distance)._copy_solution()
        best_distance = best_solution.total_distance

        for iteration in range(max_iterations):
            self.generation_count += 1

            # Jouer les rounds Golden Ball
            for _ in range(self.num_rounds):
                self.population = self._play_round(self.population)
                self.population = self._select_survivors(self.population)

                current_best = min(self.population, key=lambda sol: sol.total_distance)
                current_best.compute_total_distance()

                if current_best.total_distance < best_distance:
                    best_solution = current_best._copy_solution()
                    best_distance = best_solution.total_distance

                    if verbose:
                        print(f"[GB inject] Itération {self.generation_count}: NOUVELLE MEILLEURE = {best_distance:.2f}")

        best_solution.agent_name = "Golden Ball (continue)"
        return best_solution