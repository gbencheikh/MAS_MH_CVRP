import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


from src.instance import VRPInstance
from src.solution import VRPSolution
from src.metaheuristics.hill_climbing import Hill_Climbing

class Hill_Climbing_Agent: 
    def __init__(self, instance): 
        self.instance = instance
        self.solution = None
        self.agentName = "Hill_Climbing"
        self.MH = Hill_Climbing(instance)

    def run(self): 
        self.solution = self.MH.run(max_iterations= 1000, log_history= True, verbose = False)

    def improve(self, solution): 
        self.solution = self.MH.improve(solution, max_iterations= 1000, log_history= True, verbose = False)
