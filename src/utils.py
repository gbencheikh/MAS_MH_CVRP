import random
import sys
from pathlib import Path

from matplotlib import pyplot as plt
sys.path.append(str(Path(__file__).parent.parent))

from typing import List, Tuple
from src.instance import VRPInstance
from src.solution import VRPSolution

def generate_neighborhood(solution: VRPSolution) -> List[VRPSolution]:
    """
    Génère le voisinage d'une solution en appliquant différents opérateurs:
    1. Swap intra-route: échanger deux clients dans la même route
    2. Swap inter-route: échanger deux clients de routes différentes
    3. Relocate: déplacer un client d'une route à une autre
    4. 2-opt intra-route: inverser un segment dans une route
    """
    neighbors = []
    
    # 1. Swap intra-route
    for route_idx, route in enumerate(solution.routes):
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                neighbor = solution._copy_solution()
                neighbor.routes[route_idx][i], neighbor.routes[route_idx][j] = \
                    neighbor.routes[route_idx][j], neighbor.routes[route_idx][i]
                
                if neighbor._is_route_feasible(route_idx):
                    neighbor.compute_total_distance()
                    neighbors.append(neighbor)
    
    # 2. Swap inter-route
    for i in range(len(solution.routes)):
        for j in range(i + 1, len(solution.routes)):
            route_i = solution.routes[i]
            route_j = solution.routes[j]
            
            for pos_i in range(1, len(route_i) - 1):
                for pos_j in range(1, len(route_j) - 1):
                    neighbor = solution._copy_solution()
                    neighbor.routes[i][pos_i], neighbor.routes[j][pos_j] = \
                        neighbor.routes[j][pos_j], neighbor.routes[i][pos_i]
                    
                    if (neighbor._is_route_feasible(i) and 
                        neighbor._is_route_feasible(j)):
                        neighbor.compute_total_distance()
                        neighbors.append(neighbor)
    
    # 3. Relocate
    for from_route_idx in range(len(solution.routes)):
        for to_route_idx in range(len(solution.routes)):
            if from_route_idx == to_route_idx:
                continue
            
            from_route = solution.routes[from_route_idx]
            to_route = solution.routes[to_route_idx]
            
            for pos_from in range(1, len(from_route) - 1):
                for pos_to in range(1, len(to_route)):
                    neighbor = solution._copy_solution()
                    
                    # Retirer le client de from_route
                    client = neighbor.routes[from_route_idx][pos_from]
                    neighbor.routes[from_route_idx].pop(pos_from)
                    
                    # Insérer le client dans to_route
                    neighbor.routes[to_route_idx].insert(pos_to, client)
                    
                    # Vérifier si les deux routes sont toujours valides
                    if (neighbor._is_route_feasible(from_route_idx) and
                        neighbor._is_route_feasible(to_route_idx)):
                        neighbor.compute_total_distance()
                        neighbors.append(neighbor)
    
    # 4. 2-opt intra-route
    for route_idx, route in enumerate(solution.routes):
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                neighbor = solution._copy_solution()
                # Inverser le segment [i:j+1]
                neighbor.routes[route_idx][i:j+1] = reversed(neighbor.routes[route_idx][i:j+1])
                
                if neighbor._is_route_feasible(route_idx):
                    neighbor.compute_total_distance()
                    neighbors.append(neighbor)
    
    return neighbors

def plot_visit_distribution(history, visit_counter, algo_name="Algorithme"):
    unique_signatures = []
    seen = set()

    for h in history:
        sig = h["signature"]
        if sig not in seen:
            unique_signatures.append(sig)
            seen.add(sig)

    visit_counts = [visit_counter[sig] for sig in unique_signatures]

    total_visits = sum(visit_counter.values())
    unique_solutions = len(visit_counter)
    max_visits = max(visit_counter.values())
    mean_visits = total_visits / unique_solutions

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(visit_counts)), visit_counts)

    plt.xlabel("Sommet (solution distincte, ordre de découverte)")
    plt.ylabel("Nombre de visites")

    plt.title(
        f"{algo_name} – Visites des sommets\n"
        f"Visites totales = {total_visits} | "
        f"Sommets distincts = {unique_solutions} | "
        f"Visites moyennes = {mean_visits:.2f} | "
        f"Max visites = {max_visits}"
    )

    plt.tight_layout()
    plt.show()