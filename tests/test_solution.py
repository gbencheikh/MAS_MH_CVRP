import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.instance import VRPInstance
from src.solution import VRPSolution

if __name__ == "__main__":
    instance = VRPInstance.load_from_file("data/sample/E-n22-k4.vrp")
    
    print(instance)

    # Construire une solution simple à partir d'une ou deux routes
    solution = VRPSolution(instance)

    # Test d'une solution réalisable
    solution.add_route([0, 1, 4, 0])
    solution.add_route([0, 2, 3, 0])
    solution.add_route([0, 5, 0])
    
    feasible, msg = solution.is_feasible()
    print(f"Solution faisable ? {feasible} - {msg}")
    print(solution)


    # Test d'une solution générée aléatoirement 
    solution2 = VRPSolution(instance).generate_solution()

    
    feasible, msg = solution2.is_feasible()
    print(f"Solution faisable ? {feasible} - {msg}")
    print(solution2)
