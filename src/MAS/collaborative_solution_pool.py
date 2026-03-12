import threading
from typing import List, Dict, Tuple
import time

from src.solution import VRPSolution
from src.instance import VRPInstance
from typing import List, Dict, Tuple, Set
from collections import defaultdict

class CollaborativeSolutionPool:
    """
    Pool de solutions partagé avec tracking d'utilisation par agent.
    Thread-safe pour accès concurrent.
    """
    
    def __init__(self, max_size: int = 20, stagnation_limit: int = 5):
        self.max_size = max_size
        self.stagnation_limit = stagnation_limit
        self.solutions: List[Tuple[VRPSolution, float, str, str]] = []  # (solution, distance, agent_source, solution_id)
        self.usage_tracker: Dict[str, Set[str]] = defaultdict(set)  # {agent_name: {solution_ids_used}}
        self.best_distance_history: List[float] = []
        self.lock = threading.Lock()
        self.global_cycle = 0
        self.solution_counter = 0
    
    def initialize_with_random(self, instance: VRPInstance, num_solutions: int = 10):
        """
        Initialise le pool avec des solutions aléatoires.
        
        Args:
            instance: Instance du problème
            num_solutions: Nombre de solutions aléatoires à générer
        """
        with self.lock:
            print(f"Initialisation du pool avec {num_solutions} solutions aléatoires...")
            for i in range(num_solutions):
                solution = VRPSolution(instance).generate_solution()
                solution.compute_total_distance()
                solution_id = f"INIT_{i}"
                self.solutions.append((solution, solution.total_distance, "RANDOM", solution_id))
            
            # Trier par distance
            self.solutions.sort(key=lambda x: x[1])
            self.solutions = self.solutions[:self.max_size]
            
            # Initialiser l'historique
            if self.solutions:
                self.best_distance_history.append(self.solutions[0][1])
            
            print(f"Pool initialisé avec {len(self.solutions)} solutions")
            print(f"Meilleure distance initiale: {self.solutions[0][1]:.2f}")
    
    def get_unused_solution(self, agent_name: str) -> Tuple[VRPSolution, str]:
        """
        Récupère UNE solution pas encore utilisée par cet agent.
        
        Returns:
            (solution_copy, solution_id) ou (None, None) si toutes utilisées
        """
        with self.lock:
            used_ids = self.usage_tracker[agent_name]
            
            # Chercher une solution non utilisée
            for solution, dist, source, sol_id in self.solutions:
                if sol_id not in used_ids:
                    # Marquer comme utilisée
                    self.usage_tracker[agent_name].add(sol_id)
                    return solution._copy_solution(), sol_id
            
            # Si toutes utilisées, prendre la meilleure
            if self.solutions:
                solution, dist, source, sol_id = self.solutions[0]
                return solution._copy_solution(), sol_id
            
            return None, None
    
    def get_k_unused_solutions(self, agent_name: str, k: int = 5) -> List[VRPSolution]:
        """
        Récupère K solutions pas encore utilisées par cet agent.
        
        Returns:
            Liste de solutions (peut être < k si pas assez disponibles)
        """
        with self.lock:
            used_ids = self.usage_tracker[agent_name]
            unused_solutions = []
            
            # Chercher des solutions non utilisées
            for solution, dist, source, sol_id in self.solutions:
                if sol_id not in used_ids and len(unused_solutions) < k:
                    self.usage_tracker[agent_name].add(sol_id)
                    unused_solutions.append(solution._copy_solution())
            
            # Si pas assez, compléter avec les meilleures
            if len(unused_solutions) < k and self.solutions:
                remaining = k - len(unused_solutions)
                for solution, dist, source, sol_id in self.solutions[:remaining]:
                    unused_solutions.append(solution._copy_solution())
            
            return unused_solutions
    
    def add_solution(self, solution: VRPSolution, agent_name: str) -> bool:
        """
        Ajoute une solution au pool si elle est meilleure que la pire.
        
        Returns:
            True si ajoutée, False sinon
        """
        with self.lock:
            self.solution_counter += 1
            solution_id = f"{agent_name}_{self.solution_counter}"
            distance = solution.total_distance
            
            # Vérifier si meilleure que la pire du pool
            if len(self.solutions) < self.max_size:
                self.solutions.append((solution._copy_solution(), distance, agent_name, solution_id))
                self.solutions.sort(key=lambda x: x[1])
                
                # Mettre à jour l'historique
                self.best_distance_history.append(self.solutions[0][1])
                self.global_cycle += 1
                return True
            
            elif distance < self.solutions[-1][1]:
                self.solutions.append((solution._copy_solution(), distance, agent_name, solution_id))
                self.solutions.sort(key=lambda x: x[1])
                self.solutions = self.solutions[:self.max_size]
                
                # Mettre à jour l'historique
                self.best_distance_history.append(self.solutions[0][1])
                self.global_cycle += 1
                return True
            
            return False
    
    def is_stagnant(self) -> bool:
        """
        Vérifie si le pool stagne (aucune amélioration depuis N cycles).
        """
        with self.lock:
            if len(self.best_distance_history) < self.stagnation_limit:
                return False
            
            recent_best = self.best_distance_history[-self.stagnation_limit:]
            return len(set(recent_best)) == 1  # Tous identiques = stagnation
    
    def get_best(self) -> Tuple[VRPSolution, float]:
        """Retourne la meilleure solution du pool."""
        with self.lock:
            if not self.solutions:
                return None, float('inf')
            return self.solutions[0][0]._copy_solution(), self.solutions[0][1]
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du pool."""
        with self.lock:
            if not self.solutions:
                return {'count': 0, 'best': float('inf'), 'avg': 0, 'worst': float('inf')}
            
            distances = [dist for _, dist, _, _ in self.solutions]
            return {
                'count': len(self.solutions),
                'best': min(distances),
                'avg': sum(distances) / len(distances),
                'worst': max(distances),
                'global_cycle': self.global_cycle
            }
