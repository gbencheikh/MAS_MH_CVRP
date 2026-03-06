# Multi-Metaheuristics-for-CVRP

A Python framework for solving combinatorial problem in this case the **Capacitated Vehicle Routing Problem (CVRP)** 🚚 by combining multiple **metaheuristics** within a **multi-agent system**, using two interaction strategies: **sequential** and **collaborative**.  
This project embraces the **No Free Lunch Theorem** by avoiding any bias toward a single metaheuristic, allowing diverse methods to explore the solution space collectively or iteratively.

---

## 🎯 Objectives

- Provide a flexible and extensible platform to solve CVRP instances using multiple metaheuristics.
- Integrate different collaboration strategies between metaheuristics through a **multi-agent system**.
- Enable comparative and collaborative evaluation of optimization algorithms.
- Demonstrate the **No Free Lunch** principle in combinatorial optimization.

---

## Metaheuristics Implemented

Each metaheuristic is encapsulated within an autonomous agent:

- Genetic Algorithm (GA)
- Ant Colony Optimization (ACO)
- Particle Swarm Optimization (PSO)
- Golden Ball Algorithm
- Hill Climbing
- Tabu Search
- Simulated Annealing

---

## Multi-Agent Architectures

### 1. **Sequential Architecture**
Agents are divided into two categories:
- **Single-solution agents** (e.g., Hill Climbing, Simulated Annealing)
- **Population-based agents** (e.g., GA, ACO, PSO)

Each single-solution agent improves an initial solution and passes it to the next agent. Once all individual agents have contributed, the best solution is handed to population-based agents for further optimization.

### 2. **Collaborative Architecture**
All agents run **simultaneously**, sharing a common **environment** where they:
- Access and evaluate shared solutions.
- Select promising solutions to improve using their own strategy.
- Post the improved solutions back to the environment.

Each agent acts autonomously but contributes to a **shared evolving population**.

---

## 📁 Project Structure

```bash
CVRP-MultiMetaheuristics/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/                      # Problem instances
│   ├── instances/
│   └── sample/ 
│
├── docs/                      # Documentation
│   ├── architecture.md
│   ├── metaheuristics.md
│   └── usage.md
│
├── src/
│   ├── core/                  # Core problem components
│   ├── metaheuristics/        # Metaheuristic algorithms
│   ├── agents/                # Agent definitions
│   ├── manager/               # Execution managers
│   ├── utils/                 # Logging, metrics, etc.
│   └── main.py                # Entry point
│
├── tests/                     # Unit tests
│
└── examples/                  # Example usage scripts

# Getting Started
```
git clone https://github.com/gbencheikh/MAS_MH_CVRP
cd MAS_MH_CVRP
pip install -r requirements.txt
``` 

# The output includes:

- Final best solution
- Total cost (distance)
- Saving results in .csv file 
- Solution plot (with matplotlib)

## 📚 Documentation

Detailed documentation is available in the [`/docs`](./docs) folder, including:

- [`architecture.md`](./docs/architecture.md): Details about sequential vs collaborative systems
- [`metaheuristics.md`](./docs/metaheuristics.md): All implemented algorithms
- [`usage.md`](./docs/usage.md): How to configure and run experiments

---

## No Free Lunch Principle

This project is designed to **avoid relying on a single metaheuristic**. Instead, it enables:

- Synergy between strategies  
- Metaheuristic interoperability  
- Performance variation handling depending on instance type  

Each agent contributes a different search behavior, and their **collaboration** leads to **more robust and adaptable solutions**.

---

## Roadmap

- [ ] Add logging system  
- [ ] Collaborative environment with thread-safe access  
- [ ] Support for dynamic CVRP *(future work)*  
- [ ] Visual dashboard *(Streamlit)*  
- [ ] Automated benchmarking over CVRPLIB  

## Authors 
Manal 
Ph.D. student

Ghita 
Lecturer & Researcher 
[LinkedIn](https://www.linkedin.com/in/ghita-bencheikh/)
[ORCID](https://orcid.org/my-orcid?orcid=0000-0003-3256-1796)

Ghizlane
Professor 