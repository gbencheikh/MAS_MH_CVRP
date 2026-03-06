"""
VRP Instance - Représentation d'un problème de Vehicle Routing simplifié.

Hypothèses simples :
- Un dépôt indexé 0
- Un seul type de véhicule (capacité uniforme)
- Distances euclidiennes entre points (peuvent être pré-calculées)
"""
from __future__ import annotations

import math
from typing import List, Tuple
import matplotlib.pyplot as plt
import re 

class VRPInstance:
    """
    Représente une instance simplifiée de VRP.

    Attributes:
        coords: Liste des coordonnées (x, y) pour dépôt + clients (index 0 = dépôt)
        demands: Liste des demandes (index 0 = 0 pour le dépôt)
        vehicle_capacity: Capacité du véhicule
    """

    def __init__(self, coords: List[Tuple[float, float]], demands: List[int], num_vehicule: int, vehicle_capacity: int, optimal_value: int):
        assert len(coords) == len(demands), "coords et demands doivent être de même longueur"
        self.coords = coords
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.optimal_value = optimal_value
        self.num_nodes = len(coords)
        self.num_customers = self.num_nodes - 1
        self.num_vehicule = num_vehicule
        self.distance_matrix = self._compute_distance_matrix()

    def _compute_distance_matrix(self) -> List[List[float]]:
        """Calcule la matrice de distances euclidiennes."""
        dist = [[0.0] * self.num_nodes for _ in range(self.num_nodes)]
        for i in range(self.num_nodes):
            xi, yi = self.coords[i]
            for j in range(i + 1, self.num_nodes):
                xj, yj = self.coords[j]
                d = math.hypot(xi - xj, yi - yj)
                dist[i][j] = d
                dist[j][i] = d
        return dist

    @staticmethod
    def create_simple_instance() -> "VRPInstance":
        """
        Petite instance jouet : 1 dépôt + 4 clients.
        Capacité = 5.
        """
        coords = [
            (0, 0),   # dépôt
            (1, 3),
            (4, 3),
            (2, 1),
            (5, 0),
        ]
        demands = [0, 2, 2, 1, 3]
        num_vehicules = 2
        vehicle_capacity = 5
        return VRPInstance(coords, demands, num_vehicules, vehicle_capacity)

    @staticmethod
    def create_random_instance(
        num_customers: int,
        num_vehicules: int, 
        vehicle_capacity: int,
        demand_min: int = 1,
        demand_max: int = 10,
        coord_min: int = 0,
        coord_max: int = 100,
        seed: int | None = None,
    ) -> "VRPInstance":
        """
        Génère une instance aléatoire simple.

        Args:
            num_customers: nombre de clients (hors dépôt)
            vehicle_capacity: capacité du véhicule
            demand_min / demand_max: bornes des demandes clients
            coord_min / coord_max: bornes des coordonnées (carré)
            seed: graine aléatoire optionnelle
        """
        import random

        if seed is not None:
            random.seed(seed)

        # Dépôt à l'origine
        coords = [(0.0, 0.0)]
        demands = [0]

        for _ in range(num_customers):
            x = random.uniform(coord_min, coord_max)
            y = random.uniform(coord_min, coord_max)
            d = random.randint(demand_min, demand_max)
            coords.append((x, y))
            demands.append(d)

        return VRPInstance(coords, demands, num_vehicules, vehicle_capacity)

    @staticmethod
    def load_from_file(filepath: str) -> "VRPInstance":
        """
        Charge une instance au format CVRPLIB simplifié, par ex. :

            NAME : A-n5-k2
            CAPACITY : 85
            NODE_COORD_SECTION
            1   107   87
            ...
            DEMAND_SECTION
            1   0
            ...
            DEPOT_SECTION
                1
                -1
            EOF
        """
        name = None
        capacity = None
        nb_vehicules= None
        coords: list[tuple[float, float]] = []
        demands: list[int] = []
        
        # Extract number of vehicles from filename
        match = re.search(r'k(\d+)', filepath)
        nb_vehicules = int(match.group(1)) if match else None

        section = None  # None | "NODE_COORD" | "DEMAND" | "DEPOT"

        with open(filepath, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                upper = line.upper()
                if upper.startswith("NAME"):
                    # Optionnel : name = ...
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        name = parts[1].strip()
                    continue
                
                if upper.startswith("COMMENT"):
                    # Recherche de "Optimal value: XXX"
                    match = re.search(r'OPTIMAL VALUE\s*:\s*(\d+)', upper)
                    if match:
                        optimal_value = int(match.group(1))
                    else:
                        optimal_value = None

                if upper.startswith("CAPACITY"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        capacity = int(parts[1])
                    continue

                if upper.startswith("NODE_COORD_SECTION"):
                    section = "NODE_COORD"
                    continue
                if upper.startswith("DEMAND_SECTION"):
                    section = "DEMAND"
                    continue
                if upper.startswith("DEPOT_SECTION"):
                    section = "DEPOT"
                    continue
                if upper.startswith("EOF"):
                    break

                # Lecture selon la section courante
                if section == "NODE_COORD":
                    parts = line.split()
                    if len(parts) >= 3:
                        # idx = int(parts[0])  # ignoré, on suppose 1..N
                        x = float(parts[1])
                        y = float(parts[2])
                        coords.append((x, y))
                    continue

                if section == "DEMAND":
                    parts = line.split()
                    if len(parts) >= 2:
                        # idx = int(parts[0])
                        d = int(parts[1])
                        demands.append(d)
                    continue

                if section == "DEPOT":
                    # On lit les indices de dépôts jusqu'à -1, mais
                    # pour l'instant on suppose un unique dépôt 1 déjà dans les données.
                    if line == "-1":
                        section = None
                    continue

        if capacity is None:
            raise ValueError(f"CAPACITY manquante dans le fichier {filepath}")
        if not coords or not demands:
            raise ValueError(f"Données NODE_COORD_SECTION ou DEMAND_SECTION manquantes dans {filepath}")
        if len(coords) != len(demands):
            raise ValueError("Nombre de coordonnées différent du nombre de demandes")

        return VRPInstance(coords, demands, nb_vehicules, capacity, optimal_value)
    
    def __str__(self) -> str:
        lines = ["-" * 50]
        lines.append(f"VRP Instance: {self.num_customers} clients, {self.num_vehicule} vehicules, Capacité : {self.vehicle_capacity} ")
        lines.append("-" * 50)

        lines.append("Coordonnées (0 = dépôt) :")
        for idx, (x, y) in enumerate(self.coords):
            lines.append(f"  {idx}: ({x}, {y})  demande={self.demands[idx]}")
        lines.append("-" * 50)
        return "\n".join(lines)
    
    def _visualiser_instance(self):
        """
        Crée une visualisation graphique de l'instance
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Préparation des données
        coords_dict = {id_ville: (x, y) for id_ville, x, y in self.NODE_COORD_SECTION}
        demandes_dict = {id_ville: demande for id_ville, demande in self.demandes}
        
        # Graphique 1: Carte des villes avec taille proportionnelle aux demandes
        ax1.set_title("Carte des clients et du dépôt", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Coordonnée X")
        ax1.set_ylabel("Coordonnée Y")
        ax1.grid(True, alpha=0.3)
        
        # Afficher le dépôt
        depot_coord = coords_dict.get(0)
        if depot_coord:
            ax1.scatter(depot_coord[0], depot_coord[1], 
                    c='red', s=300, marker='s', 
                    edgecolors='black', linewidths=2,
                    label='Dépôt', zorder=5)
            ax1.annotate('DEPOT', (depot_coord[0], depot_coord[1]), 
                        fontsize=10, fontweight='bold',
                        ha='center', va='bottom', color='red')
        
        # Afficher les clients
        client_coords = [(x, y) for id_ville, x, y in self.NODE_COORD_SECTION if id_ville != 0]
        client_demandes = [demandes_dict.get(id_ville, 0) for id_ville, x, y in self.NODE_COORD_SECTION if id_ville != 0]
        
        if client_coords:
            x_coords, y_coords = zip(*client_coords)
            
            # Normaliser les tailles pour la visualisation
            if max(client_demandes) > 0:
                sizes = [100 + (d / max(client_demandes)) * 300 for d in client_demandes]
            else:
                sizes = [100] * len(client_demandes)
            
            scatter = ax1.scatter(x_coords, y_coords, 
                                c=client_demandes, s=sizes,
                                cmap='viridis', alpha=0.6,
                                edgecolors='black', linewidths=1,
                                label='Clients')
            
            # Ajouter une barre de couleur
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('Demande', rotation=270, labelpad=20)
        
        ax1.legend(loc='upper right')
        ax1.set_aspect('equal', adjustable='box')
        
        # Graphique 2: Distribution des demandes
        ax2.set_title("Distribution des demandes", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Demande")
        ax2.set_ylabel("Nombre de clients")
        ax2.grid(True, alpha=0.3, axis='y')
        
        if client_demandes:
            ax2.hist(client_demandes, bins=20, color='steelblue', 
                    edgecolor='black', alpha=0.7)
            
            # Ajouter une ligne pour la capacité du véhicule
            ax2.axvline(self.capacite_vehicule, color='red', 
                    linestyle='--', linewidth=2, 
                    label=f'Capacité véhicule ({self.capacite_vehicule})')
            ax2.legend()
        
        plt.tight_layout()
        plt.show()
    

