import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
import math
import random
from collections import defaultdict

from src.instance import VRPInstance
from src.solution import VRPSolution
from src.utils import *


class Simulated_Annealing:
    def __init__(self, instance: VRPInstance,
                 initial_temperature: float = 1000.0,
                 cooling_rate: float = 0.95,
                 min_temperature: float = 0.01):
        """
        Args:
            instance: Instance du problème CVRP
            initial_temperature: Température initiale
            cooling_rate: Taux de refroidissement (entre 0 et 1)
            min_temperature: Température minimale avant arrêt
        """
        self.instance = instance
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature

    def _acceptance_probability(self, current_distance: float,
                                neighbor_distance: float,
                                temperature: float) -> float:
        if neighbor_distance < current_distance:
            return 1.0
        delta = neighbor_distance - current_distance
        return math.exp(-delta / temperature)

    def _random_neighbor(self, solution: VRPSolution) -> VRPSolution:
        """
        Génère UN seul voisin en appliquant un opérateur aléatoire
        sur des positions aléatoires. O(1) au lieu de O(n²).

        Opérateurs :
            0 - swap intra-route
            1 - 2-opt intra-route
            2 - relocate (inter-route)
            3 - swap inter-route
        """
        routes = solution.routes
        if not routes:
            return solution._copy_solution()

        op = random.randint(0, 3)

        if op == 0:
            # Swap intra-route
            candidates = [i for i, r in enumerate(routes) if len(r) > 3]
            if candidates:
                ri = random.choice(candidates)
                route = routes[ri]
                i, j = random.sample(range(1, len(route) - 1), 2)
                neighbor = solution._copy_solution()
                neighbor.routes[ri][i], neighbor.routes[ri][j] = \
                    neighbor.routes[ri][j], neighbor.routes[ri][i]
                if neighbor._is_route_feasible(ri):
                    neighbor.compute_total_distance()
                    return neighbor

        elif op == 1:
            # 2-opt intra-route
            candidates = [i for i, r in enumerate(routes) if len(r) > 3]
            if candidates:
                ri = random.choice(candidates)
                route = routes[ri]
                i, j = sorted(random.sample(range(1, len(route) - 1), 2))
                neighbor = solution._copy_solution()
                neighbor.routes[ri][i:j + 1] = reversed(neighbor.routes[ri][i:j + 1])
                if neighbor._is_route_feasible(ri):
                    neighbor.compute_total_distance()
                    return neighbor

        elif op == 2:
            # Relocate inter-route
            if len(routes) >= 2:
                from_i = random.randrange(len(routes))
                if len(routes[from_i]) > 3:
                    to_i = random.choice([i for i in range(len(routes)) if i != from_i])
                    pos_from = random.randint(1, len(routes[from_i]) - 2)
                    pos_to = random.randint(1, len(routes[to_i]))
                    neighbor = solution._copy_solution()
                    client = neighbor.routes[from_i].pop(pos_from)
                    neighbor.routes[to_i].insert(pos_to, client)
                    if (neighbor._is_route_feasible(from_i) and
                            neighbor._is_route_feasible(to_i)):
                        neighbor.compute_total_distance()
                        return neighbor

        else:
            # Swap inter-route
            if len(routes) >= 2:
                i, j = random.sample(range(len(routes)), 2)
                if len(routes[i]) > 2 and len(routes[j]) > 2:
                    pi = random.randint(1, len(routes[i]) - 2)
                    pj = random.randint(1, len(routes[j]) - 2)
                    neighbor = solution._copy_solution()
                    neighbor.routes[i][pi], neighbor.routes[j][pj] = \
                        neighbor.routes[j][pj], neighbor.routes[i][pi]
                    if (neighbor._is_route_feasible(i) and
                            neighbor._is_route_feasible(j)):
                        neighbor.compute_total_distance()
                        return neighbor

        # Fallback : retourner une copie si l'opérateur a échoué
        return solution._copy_solution()

    def run(self, max_iterations: int = 10000, seed: int = None,
            log_history: bool = True, verbose: bool = True) -> VRPSolution:
        """
        Algorithme de Recuit Simulé pour résoudre le CVRP.
        """
        if seed is not None:
            random.seed(seed)

        start_time = time.time()

        current_solution = VRPSolution(self.instance).generate_solution()
        current_distance = current_solution.total_distance

        best_solution = current_solution._copy_solution()
        best_distance = current_distance

        temperature = self.initial_temperature

        if log_history:
            history = []
            visit_counter = defaultdict(int)
            sig = current_solution.solution_signature()
            visit_counter[sig] += 1
            history.append({"iteration": 0, "signature": sig, "distance": current_distance})

        if verbose:
            print(f"Solution initiale: distance = {current_distance:.2f}")
            print(f"Nombre de routes: {len(current_solution.routes)}")
            print(f"Température initiale: {self.initial_temperature}")
            print(f"Taux de refroidissement: {self.cooling_rate}")
            print(f"Température minimale: {self.min_temperature}")

        iteration = 0
        accepted_moves = 0
        rejected_moves = 0

        while iteration < max_iterations and temperature > self.min_temperature:
            iteration += 1

            neighbor = self._random_neighbor(current_solution)
            neighbor_distance = neighbor.total_distance

            accept_prob = self._acceptance_probability(
                current_distance, neighbor_distance, temperature)

            if random.random() < accept_prob:
                current_solution = neighbor
                current_distance = neighbor_distance
                accepted_moves += 1

                if current_distance < best_distance:
                    best_solution = current_solution._copy_solution()
                    best_distance = current_distance

                    if verbose:
                        print(f"Itération {iteration}: NOUVELLE MEILLEURE distance = {best_distance:.2f} "
                              f"(T = {temperature:.2f})")
            else:
                rejected_moves += 1

            if log_history:
                sig = current_solution.solution_signature()
                visit_counter[sig] += 1
                history.append({"iteration": iteration, "signature": sig, "distance": current_distance})

            temperature *= self.cooling_rate

            if verbose and iteration % 500 == 0:
                accept_rate = (accepted_moves / iteration) * 100
                print(f"Itération {iteration}: distance courante = {current_distance:.2f}, "
                      f"meilleure = {best_distance:.2f}, T = {temperature:.2f}, "
                      f"taux d'acceptation = {accept_rate:.1f}%")

        end_time = time.time()
        best_solution.computation_time = end_time - start_time
        best_solution.agent_name = "Simulated Annealing"

        if log_history:
            best_solution.history = history
            best_solution.visit_counter = visit_counter

        if verbose:
            total_moves = accepted_moves + rejected_moves
            accept_rate = (accepted_moves / total_moves * 100) if total_moves > 0 else 0
            print(f"\n{'='*60}")
            print(f"Recuit Simulé terminé en {best_solution.computation_time:.2f} secondes")
            print(f"Nombre d'itérations: {iteration}")
            print(f"Température finale: {temperature:.4f}")
            print(f"Distance finale: {best_distance:.2f}")
            print(f"Mouvements acceptés: {accepted_moves} ({accept_rate:.1f}%)")
            print(f"Mouvements rejetés: {rejected_moves}")
            print(f"{'='*60}")

        return best_solution

    def improve(self, initial_solution: VRPSolution, max_iterations: int = 500,
                seed: int = None, log_history: bool = False,
                verbose: bool = False) -> VRPSolution:
        """
        Améliore une solution existante avec Simulated Annealing.
        """
        if seed is not None:
            random.seed(seed)

        start_time = time.time()

        current_solution = initial_solution._copy_solution()
        current_distance = current_solution.total_distance

        best_solution = current_solution._copy_solution()
        best_distance = current_distance

        temperature = self.initial_temperature

        if log_history:
            history = []
            visit_counter = defaultdict(int)
            sig = current_solution.solution_signature()
            visit_counter[sig] += 1
            history.append({"iteration": 0, "signature": sig, "distance": current_distance})

        iteration = 0

        while iteration < max_iterations and temperature > self.min_temperature:
            iteration += 1

            neighbor = self._random_neighbor(current_solution)
            neighbor_distance = neighbor.total_distance

            accept_prob = self._acceptance_probability(
                current_distance, neighbor_distance, temperature)

            if random.random() < accept_prob:
                current_solution = neighbor
                current_distance = neighbor_distance

                if log_history:
                    sig = current_solution.solution_signature()
                    visit_counter[sig] += 1
                    history.append({"iteration": iteration, "signature": sig, "distance": current_distance})

                if current_distance < best_distance:
                    best_solution = current_solution._copy_solution()
                    best_distance = current_distance

            temperature *= self.cooling_rate

        end_time = time.time()
        best_solution.computation_time = end_time - start_time
        best_solution.agent_name = "Simulated Annealing (improve)"

        if log_history:
            best_solution.history = history
            best_solution.visit_counter = visit_counter

        return best_solution