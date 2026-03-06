"""
VRP Solution - Représente une solution simple pour le VRP.
"""
from __future__ import annotations

from typing import List, Tuple, Dict
import json
import copy
import os
from src.instance import VRPInstance
import matplotlib.pyplot as plt

class VRPSolution:
    """
    Représente un ensemble de tournées. Chaque tournée commence et finit au dépôt (0).
    Example: routes = [[0, 2, 1, 0], [0, 3, 4, 0]]
    """
    
    def __init__(self, instance: VRPInstance):
        self.instance = instance
        self.routes: List[List[int]] = []
        self.total_distance: float = 0.0
        self.computation_time: float = 0.0
        self.agent_name: str = "heuristic"
        self.history = None 
        self.visit_counter = None

    def add_route(self, route: List[int]):
        """Ajoute une route complète (doit commencer et finir par 0)."""
        if not route or route[0] != 0 or route[-1] != 0:
            raise ValueError("Une route doit commencer et terminer au dépôt (index 0).")
        self.routes.append(route)

    def compute_total_distance(self) -> float:
        """Calcule la distance totale de toutes les routes."""
        dist = 0.0
        for route in self.routes:
            for i in range(len(route) - 1):
                dist += self.instance.distance_matrix[route[i]][route[i + 1]]
        self.total_distance = dist
        return self.total_distance

    def __str__(self) -> str:
        if not self.routes:
            return "VRP Solution: (vide)"
        self.compute_total_distance()
        lines = [f"VRP Solution - distance totale: {self.total_distance:.2f}"]
        for idx, route in enumerate(self.routes, 1):
            # Calculer la charge de la route (exclure le dépôt)
            route_load = sum(self.instance.demands[node] for node in route[1:-1])
            capacity = self.instance.vehicle_capacity

            lines.append(f" Route {idx}: {' -> '.join(map(str, route))} | Charge: {route_load}/{capacity}")
        return "\n".join(lines)

    def display(self) -> None:
        """
        Affiche la solution VRP de manière lisible (texte + faisabilité).
        """
        feasible, msg = self.is_feasible()
        print(self)
        print(f"Statut : {msg} (faisable = {feasible})")

    def solution_signature(self):
        """
        Retourne une signature hashable de la solution
        """
        # Chaque route convertie en tuple
        routes = []
        for route in self.routes:
            # Assure-toi que la route est un iterable de clients (int)
            routes.append(tuple(route))
        
        # Tri des routes pour éviter que l'ordre des routes change la signature
        routes_sorted = tuple(sorted(routes))
        return routes_sorted

    def to_dict(self) -> Dict:
        """
        Convertit la solution en dictionnaire sérialisable (pour JSON).
        """
        feasible, msg = self.is_feasible()
        self.compute_total_distance()
        return {
            "agent": self.agent_name,
            "total_distance": self.total_distance,
            "computation_time": self.computation_time,
            "capacity": self.instance.vehicle_capacity,
            "num_customers": self.instance.num_customers,
            "is_feasible": feasible,
            "status_msg": msg,
            "routes": self.routes,
        }

    def save_to_json(self, filepath: str) -> None:
        """
        Sauvegarde la solution VRP dans un fichier JSON.
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def generate_solution(self):
        """
        Génère une solution initiale pour le CVRP
        Méthode: Nearest Neighbor avec respect de la capacité.
        
        Args:
            instance: Une instance du VRP
        
        Returns:
            solution: liste de routes, chaque route est une liste d'IDs de clients
        """
        solution = VRPSolution(self.instance)
        
        unvisited = set(range(1, self.instance.num_nodes))  # Extraire tous les clients (exclure le dépôt)
        
        while unvisited:
            route = [0]  # Commence au dépôt
            current_load = 0
            current_node = 0
            
            while unvisited:
                # Trouver le client non visité le plus proche qui respecte la capacité
                best_node = None
                best_distance = float('inf')
                
                for node in unvisited:
                    demand = self.instance.demands[node]
                    if current_load + demand <= self.instance.vehicle_capacity:
                        dist = self.instance.distance_matrix[current_node][node]
                        if dist < best_distance:
                            best_distance = dist
                            best_node = node
                
                # Si aucun client ne peut être ajouté, terminer cette route
                if best_node is None:
                    break
                
                # Ajouter le client à la route
                route.append(best_node)
                current_load += self.instance.demands[best_node]
                current_node = best_node
                unvisited.remove(best_node)
            
            # Retourner au dépôt
            route.append(0)
            solution.add_route(route)

        solution.compute_total_distance()
        return solution

    def is_feasible(self) -> Tuple[bool, str]:
        """
        Vérifie la faisabilité basique :
        - Chaque route commence et se termine au dépôt (0)
        - La capacité n'est jamais dépassée sur une route
        - Aucun client n'est visité plusieurs fois
        - Tous les clients présents dans les routes sont valides (1..N)
        """
        if not self.routes:
            return False, "Aucune route"

        visited = set()
        for idx, route in enumerate(self.routes, 1):
            if not route or route[0] != 0 or route[-1] != 0:
                return False, f"Route {idx} ne commence/pas au dépôt"

            load = 0
            for node in route[1:-1]:  # exclure le dépôt au début/fin
                if node <= 0 or node >= self.instance.num_nodes:
                    return False, f"Route {idx} contient un client invalide: {node}"
                if node in visited:
                    return False, f"Client {node} visité plusieurs fois"
                demand = self.instance.demands[node]
                load += demand
                if load > self.instance.vehicle_capacity:
                    return False, f"Capacité dépassée sur route {idx}"
                visited.add(node)

        # vérifier que tous les clients sont servis
        expected_customers = set(range(1, self.instance.num_nodes))
        if visited != expected_customers:
            missing = expected_customers - visited
            extra = visited - expected_customers
            if missing:
                return False, f"Clients non servis: {sorted(missing)}"
            if extra:
                return False, f"Clients inconnus: {sorted(extra)}"

        return True, "Solution VRP faisable"

    def _is_route_feasible(self, route_idx: int) -> bool:
        """Vérifie si une route respecte la contrainte de capacité."""

        route = self.routes[route_idx]
        if not route or route[0] != 0 or route[-1] != 0:
            return False
        
        total_load = sum(self.instance.demands[node] for node in route[1:-1])
        return total_load <= self.instance.vehicle_capacity
    
    def _route_feasible(self, route: List[int]) -> bool:
        """Vérifie si une route respecte la contrainte de capacité."""

        if not route or route[0] != 0 or route[-1] != 0:
            return False
        
        total_load = sum(self.instance.demands[node] for node in route[1:-1])
        return total_load <= self.instance.vehicle_capacity

    def _copy_solution(self) -> VRPSolution:
        """Crée une copie profonde d'une solution."""
        new_sol = VRPSolution(self.instance)
        new_sol.routes = [route.copy() for route in self.routes]
        new_sol.total_distance = self.total_distance
        return new_sol

    def plot_history(self): 
        iters = [h["iteration"] for h in self.history]
        distances = [h["distance"] for h in self.history]

        plt.plot(iters, distances)
        plt.xlabel("Iteration")
        plt.ylabel("Distance totale")
        plt.title(f"Convergence {self.agent_name}")
        plt.show()

    def plot_history_with_visits(self, save_path: str = None):
        """
        Affiche la convergence de l'algorithme et le nombre de visites par solution.
        """
        if not hasattr(self, "history") or not hasattr(self, "visit_counter"):
            print("Historique ou compteur de visites non disponible")
            return

        # Préparer les données
        iters = [h["iteration"] for h in self.history]
        distances = [h["distance"] for h in self.history]

        # Obtenir les solutions uniques et leur nombre de visites
        unique_signatures = []
        seen = set()
        visit_counts = []

        for h in self.history:
            sig = h["signature"]
            if sig not in seen:
                unique_signatures.append(sig)
                visit_counts.append(self.visit_counter[sig])
                seen.add(sig)

        # Créer le graphique double-axe
        fig, ax1 = plt.subplots(figsize=(12, 5))

        # Axe gauche : distance
        ax1.plot(iters, distances, color='blue', label='Distance')
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Distance totale", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Axe droit : visites des solutions uniques
        ax2 = ax1.twinx()
        
        bar_x = [h["iteration"] for h in self.history]
        bar_y = [self.visit_counter[h["signature"]] for h in self.history]

        ax2.bar(bar_x, bar_y, color='orange', alpha=0.3, width=0.8)

        ax2.set_ylabel("Nombre de visites", color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        # Statistiques globales dans le titre
        total_visits = sum(self.visit_counter.values())
        num_unique = len(self.visit_counter)
        max_visits = max(self.visit_counter.values())
        mean_visits = total_visits / num_unique

        plt.title(
            f"{self.agent_name} – Convergence et visites des solutions\n"
            f"Visites totales = {total_visits} | "
            f"Sommets distincts = {num_unique} | "
            f"Visites moyennes = {mean_visits:.2f} | "
            f"Max visites = {max_visits}"
        )

        fig.tight_layout()
        # Sauvegarde ou affichage
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300)
            print(f"Figure sauvegardée dans : {save_path}")
        else:
            plt.show()
