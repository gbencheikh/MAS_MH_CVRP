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

class Particle:
    """
    Représente une particule dans l'essaim.
    Chaque particule a une position (solution) et se souvient de sa meilleure position.
    """
    
    def __init__(self, instance: VRPInstance):
        self.instance = instance
        self.position = VRPSolution(instance).generate_solution()
        self.position.compute_total_distance()
        
        # Meilleure position personnelle
        self.best_position = self.position._copy_solution()
        self.best_position.compute_total_distance()
        self.best_distance = self.position.total_distance
    
    def update_best(self):
        """Met à jour la meilleure position personnelle si nécessaire."""
        if self.position.total_distance < self.best_distance:
            self.best_position = self.position._copy_solution()
            self.best_position.compute_total_distance()
            self.best_distance = self.position.total_distance


class Particle_Swarm_Optimization:
    """
    Algorithme d'Optimisation par Essaim Particulaire (PSO) pour résoudre le CVRP.
    
    Les particules se déplacent dans l'espace de recherche en suivant:
    - Leur propre meilleure position (composante cognitive)
    - La meilleure position globale de l'essaim (composante sociale)
    """
    
    def __init__(self, instance: VRPInstance,
                 num_particles: int = 30,
                 w: float = 0.5,
                 c1: float = 1.5,
                 c2: float = 1.5):
        """
        Args:
            instance: Instance du problème CVRP
            num_particles: Nombre de particules dans l'essaim
            w: Coefficient d'inertie (exploration vs exploitation)
            c1: Coefficient cognitif (attraction vers meilleure position personnelle)
            c2: Coefficient social (attraction vers meilleure position globale)
        """
        self.instance = instance
        self.num_particles = num_particles
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Initialiser l'essaim
        self.swarm: List[Particle] = []
        self.global_best_position = None
        self.global_best_distance = float('inf')
    
    def _initialize_swarm(self):
        """Initialise l'essaim de particules."""
        self.swarm = []
        for _ in range(self.num_particles):
            particle = Particle(self.instance)
            self.swarm.append(particle)
            
            # Mettre à jour le meilleur global
            if particle.best_distance < self.global_best_distance:
                self.global_best_position = particle.best_position._copy_solution()
                self.global_best_position.compute_total_distance()
                self.global_best_distance = particle.best_distance
    
    def _move_towards(self, current: VRPSolution, target: VRPSolution, influence: float) -> VRPSolution:
        """
        Déplace une solution vers une autre avec plus de variabilité.
        """
        if random.random() > influence:
            return current._copy_solution()
        
        new_solution = VRPSolution(self.instance)
        
        # AMÉLIORATION: Plus de randomisation
        # Décider combien de routes viennent de chaque parent
        num_routes_target = max(1, int(len(target.routes) * influence))
        num_routes_current = max(1, int(len(current.routes) * (1 - influence)))
        
        # Échantillonner DIFFÉRENTES routes à chaque fois
        target_routes = [r.copy() for r in random.sample(target.routes, min(num_routes_target, len(target.routes)))]
        current_routes = [r.copy() for r in random.sample(current.routes, min(num_routes_current, len(current.routes)))]
        
        # IMPORTANT: Mélanger vraiment
        all_routes = target_routes + current_routes
        random.shuffle(all_routes)
        
        # Appliquer aussi une petite mutation aléatoire
        for i in range(len(all_routes)):
            if random.random() < 0.2:  # 20% de chance de modifier
                route = all_routes[i]
                if len(route) > 3:  # Au moins 2 clients
                    # Échanger deux clients dans la route
                    idx1, idx2 = random.sample(range(1, len(route) - 1), 2)
                    route[idx1], route[idx2] = route[idx2], route[idx1]
        
        # Reconstruire la solution
        visited = set()
        for route in all_routes:
            customers = [node for node in route if node != 0]
            new_customers = [c for c in customers if c not in visited]
            
            if new_customers:
                new_route = [0] + new_customers + [0]
                if new_solution._route_feasible(new_route):
                    new_solution.routes.append(new_route)
                    visited.update(new_customers)
                else:
                    # Diviser
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
                                new_solution.routes.append(temp_route)
                            temp_route = [0, customer]
                            temp_load = demand
                            visited.add(customer)
                    
                    if len(temp_route) > 1:
                        temp_route.append(0)
                        new_solution.routes.append(temp_route)
        
        # Ajouter les manquants
        missing = set(range(1, self.instance.num_nodes)) - visited
        for customer in missing:
            inserted = False
            for route in new_solution.routes:
                route_load = sum(self.instance.demands[node] for node in route[1:-1])
                if route_load + self.instance.demands[customer] <= self.instance.vehicle_capacity:
                    route.insert(-1, customer)
                    inserted = True
                    break
            if not inserted:
                new_solution.routes.append([0, customer, 0])
        
        new_solution.compute_total_distance()
        return new_solution
    
    def _update_particle_position(self, particle: Particle):
        """
        Met à jour la position d'une particule selon l'équation PSO.
        
        La nouvelle position est influencée par:
        - Composante d'inertie (w): tendance à continuer dans la même direction
        - Composante cognitive (c1): attraction vers sa meilleure position
        - Composante sociale (c2): attraction vers la meilleure position globale
        
        Args:
            particle: Particule à mettre à jour
        """
        # Générer des facteurs aléatoires
        r1 = random.random()
        r2 = random.random()
        
        # Calculer les influences
        inertia_influence = self.w
        cognitive_influence = self.c1 * r1
        social_influence = self.c2 * r2
        
        # Normaliser les influences
        total_influence = inertia_influence + cognitive_influence + social_influence
        if total_influence > 0:
            inertia_influence /= total_influence
            cognitive_influence /= total_influence
            social_influence /= total_influence
        
        # Nouvelle position = combinaison des trois composantes
        new_position = particle.position._copy_solution()
        
        # Appliquer la composante cognitive (vers meilleure position personnelle)
        if cognitive_influence > 0 and random.random() < cognitive_influence:
            new_position = self._move_towards(new_position, particle.best_position, cognitive_influence)
        
        # Appliquer la composante sociale (vers meilleure position globale)
        if social_influence > 0 and self.global_best_position and random.random() < social_influence:
            new_position = self._move_towards(new_position, self.global_best_position, social_influence)
        
        new_position.compute_total_distance()

        # Appliquer une recherche locale occasionnelle (exploration)
        if random.random() < 0.3:
            neighbors = generate_neighborhood(new_position)
            if neighbors:
                # Choisir un voisin aléatoire ou le meilleur
                if random.random() < 0.5:
                    new_position = random.choice(neighbors)._copy_solution()
                    new_position.compute_total_distance()
                else:
                    best_neighbor = min(neighbors, key=lambda s: s.total_distance)
                    new_position = best_neighbor._copy_solution()
                    new_position.compute_total_distance()
        
        particle.position = new_position
        particle.update_best()
    
    def run(self, max_iterations: int = 100, seed: int = None, log_history: bool = True, verbose: bool = True) -> VRPSolution:
        """
        Algorithme PSO pour résoudre le CVRP.
        
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
        
        # Initialiser l'essaim
        self._initialize_swarm()
        
        if log_history:
            history = [] # historique des distances
            visit_counter = defaultdict(int)

            sig = self.global_best_position.solution_signature()
            visit_counter[sig] += 1

            history.append({
                "iteration": 0,
                "signature": sig,
                "distance": self.global_best_distance
            })

        if verbose:
            print(f"Nombre de particules: {self.num_particles}")
            print(f"Meilleure distance initiale: {self.global_best_distance:.2f}")
            print(f"Coefficient d'inertie (w): {self.w}")
            print(f"Coefficient cognitif (c1): {self.c1}")
            print(f"Coefficient social (c2): {self.c2}")
        
        iterations_since_improvement = 0
        max_no_improvement = max(20, max_iterations // 5)
        
        for iteration in range(max_iterations):
            improvement_this_iteration = False 

            # Mettre à jour chaque particule
            for particle in self.swarm:
                self._update_particle_position(particle)
                particle.best_position.compute_total_distance()

                # Mettre à jour le meilleur global
                if particle.best_distance < self.global_best_distance:
                    improvement_this_iteration = True 

                    self.global_best_position = particle.best_position._copy_solution()
                    self.global_best_position.compute_total_distance()
                    self.global_best_distance = particle.best_distance
                    iterations_since_improvement = 0
                    
                    if log_history:
                        sig = self.global_best_position.solution_signature()
                        visit_counter[sig] += 1

                        history.append({
                            "iteration": iteration,
                            "signature": sig,
                            "distance": self.global_best_distance
                        })

                    if verbose:
                        print(f"Itération {iteration + 1}: NOUVELLE MEILLEURE distance = {self.global_best_distance:.2f}")
            
            if not improvement_this_iteration:
                iterations_since_improvement += 1
            
            # Affichage périodique
            if verbose and (iteration + 1) % 10 == 0:
                avg_distance = sum(p.position.total_distance for p in self.swarm) / len(self.swarm)
                avg_best = sum(p.best_distance for p in self.swarm) / len(self.swarm)
                print(f"Itération {iteration + 1}: meilleure globale = {self.global_best_distance:.2f}, "
                      f"moyenne actuelle = {avg_distance:.2f}, moyenne best = {avg_best:.2f}")
            
            # Critère d'arrêt
            if iterations_since_improvement >= max_no_improvement:
                if verbose:
                    print(f"Arrêt après {iterations_since_improvement} itérations sans amélioration")
                break
        
        end_time = time.time()
        
        if self.global_best_position:
            self.global_best_position.computation_time = end_time - start_time
            self.global_best_position.agent_name = "Particle Swarm Optimization"
        
        if log_history:
                self.global_best_position.history = history
                self.global_best_position.visit_counter = visit_counter

        if verbose:
            print(f"\n{'='*60}")
            print(f"PSO terminé en {self.global_best_position.computation_time:.2f} secondes")
            print(f"Nombre d'itérations: {iteration + 1}")
            print(f"Distance finale: {self.global_best_distance:.2f}")
            print(f"{'='*60}")
        
        self.global_best_position.compute_total_distance()
        return self.global_best_position
