import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.instance import VRPInstance

if __name__ == "__main__":
    instance = VRPInstance.load_from_file("data/sample/E-n22-k4.vrp")
    
    print(instance)