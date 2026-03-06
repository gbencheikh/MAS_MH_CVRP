import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import time
from collections import deque
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
        """
        Calcule la probabilité d'accepter une solution moins bonne.
        
        Args:
            current_distance: Distance de la solution courante
            neighbor_distance: Distance de la solution voisine
            temperature: Température courante
        
        Returns:
            Probabilité d'acceptation (entre 0 et 1)
        """
        if neighbor_distance < current_distance:
            return 1.0  # Toujours accepter une amélioration
        
        # Formule du recuit simulé: exp(-ΔE / T)
        delta = neighbor_distance - current_distance
        return math.exp(-delta / temperature)
    
    def run(self, max_iterations: int = 10000, seed: int = None, log_history: bool = True, verbose: bool = True) -> VRPSolution:
        """
        Algorithme de Recuit Simulé pour résoudre le CVRP.
        
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
        
        # Générer la solution initiale
        current_solution = VRPSolution(self.instance).generate_solution()
        current_distance = current_solution.total_distance
        
        # Meilleure solution globale
        best_solution = current_solution._copy_solution()
        best_distance = current_distance
        
        # Température initiale
        temperature = self.initial_temperature
        
        # Historique et compteur
        if log_history:
            history = []
            visit_counter = defaultdict(int)

            sig = current_solution.solution_signature()
            visit_counter[sig] += 1
            history.append({
                "iteration": 0, 
                "signature": sig, 
                "distance": current_distance
            })
            
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
            
            # Générer le voisinage
            neighbors = generate_neighborhood(current_solution)
            
            if not neighbors:
                if verbose:
                    print("Aucun voisin généré, arrêt.")
                break
            
            # Choisir un voisin aléatoire
            neighbor = random.choice(neighbors)
            neighbor_distance = neighbor.total_distance
            
            # Calculer la probabilité d'acceptation
            accept_prob = self._acceptance_probability(current_distance, 
                                                       neighbor_distance, 
                                                       temperature)
            
            # Décider d'accepter ou non le voisin
            if random.random() < accept_prob:
                # Accepter le voisin
                current_solution = neighbor
                current_distance = neighbor_distance
                accepted_moves += 1
                
                # Mettre à jour la meilleure solution si nécessaire
                if current_distance < best_distance:
                    best_solution = current_solution._copy_solution()
                    best_distance = current_distance
                    
                    if verbose:
                        print(f"Itération {iteration}: NOUVELLE MEILLEURE distance = {best_distance:.2f} "
                              f"(T = {temperature:.2f})")
            else:
                rejected_moves += 1
            
            # Historique
            if log_history:
                sig = current_solution.solution_signature()
                visit_counter[sig] += 1
                history.append({
                    "iteration": iteration, 
                    "signature": sig, 
                    "distance": current_distance
                })

            # Refroidissement
            temperature *= self.cooling_rate
            
            # Affichage périodique
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
            
            if current_solution.total_distance > 0:
                initial_distance = VRPSolution(self.instance).generate_solution().total_distance
                improvement = ((initial_distance - best_distance) / initial_distance * 100)
                print(f"Amélioration depuis l'initial: {improvement:.2f}%")
            print(f"{'='*60}")
        
        return best_solution