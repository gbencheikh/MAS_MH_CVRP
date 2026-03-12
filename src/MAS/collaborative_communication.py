import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import time
from typing import List, Dict
from collections import defaultdict

from src.instance import VRPInstance
from src.solution import VRPSolution
from src.MAS.collaborative_solution_pool import CollaborativeSolutionPool
from src.MAS.agents.async_collaborative_agent import AsyncCollaborativeAgent


class AsyncCollaborativeMultiAgentSystem:
    """
    Système multi-agent asynchrone avec pool de solutions partagé.
    """

    def __init__(self, instance: VRPInstance,
                 pool_size: int = 20,
                 stagnation_limit: int = 5,
                 initial_pool_size: int = 10):

        self.instance = instance
        self.pool = CollaborativeSolutionPool(
            max_size=pool_size,
            stagnation_limit=stagnation_limit
        )

        self.agents: List[AsyncCollaborativeAgent] = []
        self.agent_scores = defaultdict(float)

        self.initial_pool_size = initial_pool_size

    # ---------------------------------------------------------

    def add_agent(self,
                  name: str,
                  solver_class,
                  solver_params: Dict = None,
                  run_params: Dict = None,
                  is_population_based: bool = False,
                  max_cycles: int = 10,
                  role: str = "balanced"):
        """
        Ajoute un agent au système
        """

        solver_params = solver_params or {}
        run_params = run_params or {}

        agent = AsyncCollaborativeAgent(
            name=name,
            instance=self.instance,
            solver_class=solver_class,
            solver_params=solver_params,
            run_params=run_params,
            is_population_based=is_population_based,
            max_cycles=max_cycles,
            role=role
        )

        self.agents.append(agent)

    # ---------------------------------------------------------

    def pool_diversity(self) -> float:
        """
        Mesure simple de diversité du pool
        """

        solutions = self.pool.get_all()

        if len(solutions) < 2:
            return 0

        signatures = []

        for s in solutions:
            try:
                signatures.append(s.solution_signature())
            except:
                continue

        distances = []

        for i in range(len(signatures)):
            for j in range(i + 1, len(signatures)):
                d = sum(a != b for a, b in zip(signatures[i], signatures[j]))
                distances.append(d)

        if not distances:
            return 0

        return sum(distances) / len(distances)

    # ---------------------------------------------------------

    def adapt_agents(self):
        """
        Ajuste les cycles des agents selon la diversité
        """

        div = self.pool_diversity()

        for agent in self.agents:

            if agent.role == "explore" and div < 3:
                agent.max_cycles += 2

            elif agent.role == "exploit" and div > 5:
                agent.max_cycles += 1

    # ---------------------------------------------------------

    def run(self, verbose: bool = True) -> Dict:

        start_time = time.time()

        if verbose:
            print("\n" + "=" * 80)
            print("SYSTÈME MULTI-AGENT ASYNCHRONE COLLABORATIF")
            print(f"Instance: {self.instance.num_customers} clients")
            print(f"Agents: {len(self.agents)}")
            print("=" * 80 + "\n")

        # -------------------------------------------------
        # 1. Initialisation du pool
        # -------------------------------------------------

        self.pool.initialize_with_random(
            self.instance,
            self.initial_pool_size
        )

        # -------------------------------------------------
        # 2. Adapter les agents selon la diversité
        # -------------------------------------------------

        self.adapt_agents()

        # -------------------------------------------------
        # 3. Assigner le pool aux agents
        # -------------------------------------------------

        for agent in self.agents:
            agent.pool = self.pool

        # -------------------------------------------------
        # 4. Lancer les agents en parallèle
        # -------------------------------------------------

        for agent in self.agents:
            agent.start()

        # -------------------------------------------------
        # 5. Attendre la fin
        # -------------------------------------------------

        for agent in self.agents:
            agent.join()

        end_time = time.time()

        # -------------------------------------------------
        # 6. Résultats
        # -------------------------------------------------

        best_solution, best_distance = self.pool.get_best()
        pool_stats = self.pool.get_stats()

        diversity = self.pool_diversity()

        results = {
            "best_solution": best_solution,
            "best_distance": best_distance,
            "total_time": end_time - start_time,
            "pool_stats": pool_stats,
            "agents_results": {}
        }

        # -------------------------------------------------
        # 7. Statistiques agents
        # -------------------------------------------------

        for agent in self.agents:

            results["agents_results"][agent.name] = {
                "cycles": agent.cycles_completed,
                "deposited": agent.solutions_deposited,
                "best_distance": agent.best_distance,
                "best_solution": agent.best_solution
            }

            if agent.best_solution:

                improvement = max(
                    0,
                    best_distance - agent.best_solution.total_distance
                )

                self.agent_scores[agent.name] += improvement

        results["diversity"] = diversity
        results["agent_scores"] = dict(self.agent_scores)

        # -------------------------------------------------
        # 8. Affichage
        # -------------------------------------------------

        if verbose:

            print("\n" + "=" * 80)
            print("RÉSULTATS FINAUX")
            print("=" * 80)

            print(f"Temps total: {results['total_time']:.2f}s")
            print(f"Meilleure solution: {best_distance:.2f}")
            print(f"Diversité finale: {diversity:.2f}")

            print("\nPool final:")
            print(f"  Count: {pool_stats['count']}")
            print(f"  Best: {pool_stats['best']:.2f}")
            print(f"  Avg: {pool_stats['avg']:.2f}")

            print("\nPerformance par agent:")

            for agent_name, data in results["agents_results"].items():

                print(
                    f"  {agent_name:10s} | "
                    f"cycles={data['cycles']:3d} | "
                    f"deposited={data['deposited']:3d} | "
                    f"best={data['best_distance']:.2f}"
                )

            print("=" * 80 + "\n")

        return results