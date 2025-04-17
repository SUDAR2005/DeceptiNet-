# backend.py
import time
import random
import csv
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Assuming pyMetaheuristic is installed
from pyMetaheuristic.algorithm import whale_optimization_algorithm
from pyMetaheuristic.algorithm import harris_hawks_optimization
import numpy as np

# --- Enums and Pydantic Models ---
class AlgorithmType(str, Enum):
    GA = "GA"
    WOA = "WOA"
    HHO = "HHO"

class NetworkNode(BaseModel):
    """Represents a single node from the frontend data."""
    name: str
    ip: str
    connections: Optional[List[str]] = Field(default_factory=list)
    # Add asset value, potentially derived or from CSV on frontend side
    asset_value: float = Field(default=1.0, ge=0)
    # Add simulated OS type for vulnerability calculation (optional)
    os_type: Optional[str] = Field(default="Ubuntu 20.04")
     # Add simulated open ports (optional)
    open_ports: Optional[List[int]] = Field(default_factory=lambda: [22, 80])


class PredictionRequest(BaseModel):
    """The structure of the data sent from the frontend."""
    network_data: List[NetworkNode]
    algorithm_type: AlgorithmType = AlgorithmType.GA # Default to GA
    max_honeypots: int = Field(default=3, gt=0)
    # Optional GA/WOA/HHO parameters (can add more)
    ga_generations: Optional[int] = Field(default=30, gt=0)
    ga_pop_size: Optional[int] = Field(default=50, gt=0)
    woa_iterations: Optional[int] = Field(default=50, gt=0) # Reduced default for faster API response
    hho_iterations: Optional[int] = Field(default=50, gt=0) # Reduced default

class PredictionResponse(BaseModel):
    """The structure of the prediction result sent back."""
    predicted_honeypots: List[str]
    message: str
    algorithm_used: str
    fitness_score: Optional[float] = None # Add fitness score

# --- FastAPI App Setup ---

app = FastAPI(
    title="Adaptive Network Honeypot Predictor",
    description="API to predict optimal honeypot placements using different algorithms.",
    version="0.2.0",
)

# --- CORS Middleware ---
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:5500", # Example for VS Code Live Server
    "null", # For file:// origin
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions (Adapted from Adaptive_deception.py) ---

def simulate_scan_host(node_info: NetworkNode) -> Dict[str, Any]:
    """
    Simulates scanning a host based on provided NetworkNode info.
    This replaces the Mininet-based scan_host.
    """
    detected_os = node_info.os_type if node_info.os_type else "Ubuntu 20.04"
    open_ports = node_info.open_ports if node_info.open_ports else [22, 80]

    # Simulate OS age based on name (very basic)
    os_age = 3.0 if "XP" in detected_os or "18.04" in detected_os else \
             2.0 if "7" in detected_os or "CentOS" in detected_os else 1.0

    # Simulate CVE count based on OS age
    cve_count = 0
    if "XP" in detected_os:
        cve_count = random.randint(3, 5)
    elif "7" in detected_os or "18.04" in detected_os:
        cve_count = random.randint(2, 4)
    elif "CentOS" in detected_os:
        cve_count = random.randint(1, 3)
    else:
        cve_count = random.randint(0, 2)

    # Simulate service score based on common vulnerable ports
    service_score = 1.0
    if 21 in open_ports: service_score += 0.5 # FTP
    if 23 in open_ports: service_score += 0.5 # Telnet
    if 135 in open_ports or 139 in open_ports or 445 in open_ports: service_score += 0.5 # SMB/NetBIOS
    if 3389 in open_ports: service_score += 0.5 # RDP

    return {
        'cve_count': cve_count,
        'open_ports': len(open_ports),
        'os_age': os_age,
        'service_score': service_score,
        'detected_os': detected_os, # Keep for context if needed
        'open_port_list': open_ports # Keep for context
    }

def calculate_cvss_score(vuln_data: Dict[str, Any]) -> float:
    """
    Calculate simulated CVSS score based on simulated vulnerability data.
    (Same logic as in Adaptive_deception.py)
    """
    # Base metrics
    if vuln_data['os_age'] >= 3.0: av_score = 0.85 # Network
    elif vuln_data['os_age'] >= 2.0: av_score = 0.62 # Adjacent
    else: av_score = 0.55 # Local

    if vuln_data['cve_count'] >= 3: ac_score = 0.77 # Low complexity
    else: ac_score = 0.44 # High complexity

    if vuln_data['service_score'] >= 2.5: pr_score = 0.85 # None required
    elif vuln_data['service_score'] >= 1.5: pr_score = 0.62 # Low required
    else: pr_score = 0.27 # High required

    ui_score = 0.85  # Assume None for simplicity

    scope_changed = vuln_data['open_ports'] >= 5

    # Impact metrics
    if vuln_data['cve_count'] >= 4 or vuln_data['os_age'] >= 3.0: c_score = 0.56 # High
    elif vuln_data['cve_count'] >= 2 or vuln_data['os_age'] >= 2.0: c_score = 0.22 # Low
    else: c_score = 0.0   # None

    if vuln_data['cve_count'] >= 3 or vuln_data['os_age'] >= 2.5: i_score = 0.56 # High
    elif vuln_data['cve_count'] >= 1 or vuln_data['os_age'] >= 1.5: i_score = 0.22 # Low
    else: i_score = 0.0   # None

    if vuln_data['service_score'] >= 2.0 or vuln_data['open_ports'] >= 7: a_score = 0.56 # High
    elif vuln_data['service_score'] >= 1.5 or vuln_data['open_ports'] >= 3: a_score = 0.22 # Low
    else: a_score = 0.0   # None

    # Calculations (simplified slightly from original spec for brevity)
    exploitability = 8.22 * av_score * ac_score * pr_score * ui_score
    impact_base = 1 - ((1 - c_score) * (1 - i_score) * (1 - a_score))

    if scope_changed:
        impact = 7.52 * (impact_base - 0.029) - 3.25 * pow(max(0, impact_base - 0.02), 15)
    else:
        impact = 6.42 * impact_base

    if impact <= 0:
        return 0.0

    if scope_changed:
        score = min(10.0, 1.08 * (exploitability + impact))
    else:
        score = min(10.0, exploitability + impact)

    # Round up to one decimal place
    return round(score * 10) / 10

def simulate_network_scan(network_data: List[NetworkNode]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Simulates scanning the entire network based on input data."""
    vulnerability_scores = {}
    asset_values = {}

    print("Simulating vulnerability scan...")
    for node_info in network_data:
        node_id = node_info.ip # Use IP as the unique identifier internally
        if not node_id:
            print(f"Warning: Skipping node with missing IP: {node_info.name}")
            continue

        # Simulate scan
        scan_result = simulate_scan_host(node_info)
        cvss_score = calculate_cvss_score(scan_result)
        vulnerability_scores[node_id] = cvss_score
        asset_values[node_id] = node_info.asset_value

        print(f"  - Node {node_info.name} ({node_id}): Score = {cvss_score:.1f}, Asset = {node_info.asset_value}")

    return vulnerability_scores, asset_values


# --- Honeypot Optimizer Class (Refactored) ---

class HoneypotOptimizer:
    def __init__(self, network_data: List[NetworkNode],
                 vulnerability_scores: Dict[str, float],
                 asset_values: Dict[str, float],
                 max_honeypots: int):

        self.max_honeypots = max_honeypots
        self.vulnerability_scores = vulnerability_scores
        self.asset_values = asset_values

        # Build topology and map IP to index
        self.node_ips = [node.ip for node in network_data if node.ip] # List of IPs in order
        self.ip_to_index = {ip: i for i, ip in enumerate(self.node_ips)}
        self.num_nodes = len(self.node_ips)

        # Store connections using indices for faster lookup
        self.node_connections_indices = [[] for _ in range(self.num_nodes)]
        for i, node in enumerate(network_data):
             if node.ip and node.connections:
                 current_ip = node.ip
                 current_idx = self.ip_to_index.get(current_ip)
                 if current_idx is None: continue

                 for conn_ip in node.connections:
                     conn_idx = self.ip_to_index.get(conn_ip)
                     if conn_idx is not None:
                        # Check for duplicates before adding
                        if conn_idx not in self.node_connections_indices[current_idx]:
                            self.node_connections_indices[current_idx].append(conn_idx)
                        if current_idx not in self.node_connections_indices[conn_idx]:
                            self.node_connections_indices[conn_idx].append(current_idx)


        print(f"Optimizer initialized for {self.num_nodes} nodes.")
        # print(f"Node IPs: {self.node_ips}")
        # print(f"Vuln Scores: {self.vulnerability_scores}") # Use node_id (IP)
        # print(f"Asset Values: {self.asset_values}") # Use node_id (IP)
        # print(f"Connections (Indices): {self.node_connections_indices}")


    def get_covered_nodes_indices(self, individual: List[int]) -> Set[int]:
        """Gets the set of node *indices* covered by the honeypot placement."""
        covered_indices = set()
        for node_idx, has_honeypot in enumerate(individual):
            if has_honeypot == 1:
                covered_indices.add(node_idx)
                # Add connected nodes (indices)
                if node_idx < len(self.node_connections_indices):
                    for connected_idx in self.node_connections_indices[node_idx]:
                        covered_indices.add(connected_idx)
        return covered_indices

    def evaluate_fitness(self, individual: List[int]) -> float:
        """Calculates fitness based on coverage and constraints."""
        honeypot_count = sum(individual)

        if honeypot_count > self.max_honeypots or honeypot_count == 0:
             # Penalize heavily for exceeding limits or placing none
            return -10000.0 * (1 + abs(honeypot_count - self.max_honeypots / 2))

        covered_indices = self.get_covered_nodes_indices(individual)

        vuln_coverage = 0.0
        asset_coverage = 0.0
        for node_idx in covered_indices:
            if 0 <= node_idx < self.num_nodes:
                node_ip = self.node_ips[node_idx]
                vuln_coverage += self.vulnerability_scores.get(node_ip, 0.0)
                asset_coverage += self.asset_values.get(node_ip, 0.0)

        # Weights (can be tuned)
        alpha = 0.4  # Vulnerability coverage weight
        beta = 0.6   # Asset coverage weight
        gamma = 0.1  # Resource usage penalty (per honeypot)

        # Normalize coverage scores roughly (assuming max possible score ~ num_nodes * 10 for vuln, num_nodes * max_asset for asset)
        max_possible_vuln = sum(self.vulnerability_scores.values()) or 1.0
        max_possible_asset = sum(self.asset_values.values()) or 1.0

        # Avoid division by zero and handle empty network case
        norm_vuln_coverage = (vuln_coverage / max_possible_vuln) if max_possible_vuln > 0 else 0
        norm_asset_coverage = (asset_coverage / max_possible_asset) if max_possible_asset > 0 else 0


        # Fitness: Maximize weighted normalized coverage, penalize resource usage
        fitness = (alpha * norm_vuln_coverage + beta * norm_asset_coverage) * 100 - (gamma * honeypot_count)
        # print(f"Individual: {individual}, VulnCov: {vuln_coverage:.2f}, AssetCov: {asset_coverage:.2f}, Fitness: {fitness:.4f}")

        return fitness


    # --- GA Specific Methods ---
    def tournament_selection(self, population: List[List[int]], fitness_scores: List[float], tournament_size=3) -> List[int]:
        """Selects an individual using tournament selection."""
        best_idx = -1
        best_fitness = -float('inf')
        for _ in range(tournament_size):
            idx = random.randrange(len(population))
            if fitness_scores[idx] > best_fitness:
                best_idx = idx
                best_fitness = fitness_scores[idx]
        return population[best_idx][:] # Return a copy

    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        """Performs two-point crossover."""
        length = len(parent1)
        if length < 2:
            return parent1[:], parent2[:]
        p1, p2 = sorted(random.sample(range(length), 2))
        child1 = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
        child2 = parent2[:p1] + parent1[p1:p2] + parent2[p2:]
        return child1, child2

    def mutate(self, individual: List[int], mutation_rate=0.1):
        """Flips bits with a given mutation rate."""
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] = 1 - individual[i]


    # --- Optimization Algorithm Runners ---

    def genetic_algorithm(self, pop_size=50, num_generations=30) -> Tuple[List[str], float]:
        """Runs the Genetic Algorithm."""
        if self.num_nodes == 0: return [], -float('inf')
        print(f"Running GA: Pop={pop_size}, Gens={num_generations}, MaxHP={self.max_honeypots}")
        population = [[random.randint(0, 1) for _ in range(self.num_nodes)] for _ in range(pop_size)]
        best_fitness = -float('inf')
        best_solution = [0] * self.num_nodes

        start_time = time.time()
        for generation in range(num_generations):
            fitness_scores = [self.evaluate_fitness(ind) for ind in population]

            current_best_gen_idx = np.argmax(fitness_scores)
            current_best_gen_fitness = fitness_scores[current_best_gen_idx]

            if current_best_gen_fitness > best_fitness:
                best_fitness = current_best_gen_fitness
                best_solution = population[current_best_gen_idx][:]
                # print(f"  Gen {generation}: New best fitness: {best_fitness:.4f}")


            new_population = []
            # Elitism: Keep the best individual from the current generation
            elite_idx = np.argmax(fitness_scores)
            new_population.append(population[elite_idx][:])

            # Fill the rest of the population
            while len(new_population) < pop_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)
                new_population.append(child1)
                if len(new_population) < pop_size:
                    new_population.append(child2)

            population = new_population
            if generation % 10 == 0: # Print progress periodically
                 print(f"  GA Generation {generation}, Best Fitness so far: {best_fitness:.4f}")


        end_time = time.time()
        print(f"GA finished in {end_time - start_time:.2f}s. Final best fitness: {best_fitness:.4f}")

        honeypot_ips = [self.node_ips[i] for i, val in enumerate(best_solution) if val == 1]
        return honeypot_ips, best_fitness

    def whale_optimization(self, iterations=50) -> Tuple[List[str], float]:
        """Runs the Whale Optimization Algorithm."""
        if self.num_nodes == 0: return [], -float('inf')
        print(f"Running WOA: Iterations={iterations}, MaxHP={self.max_honeypots}")

        # Objective function for minimization
        def objective_function(solution: np.ndarray) -> float:
            # Convert continuous solution to binary
            individual = [1 if x > 0.5 else 0 for x in solution]
            # Return negative fitness because WOA minimizes
            return -self.evaluate_fitness(individual)

        dim = self.num_nodes
        min_values = [0.0] * dim
        max_values = [1.0] * dim

        start_time = time.time()
        # Note: pyMetaheuristic returns variables + fitness as a list
        result = whale_optimization_algorithm(
            target_function=objective_function,
            hunting_party=max(10, self.num_nodes), # Adjust party size based on network
            min_values=min_values,
            max_values=max_values,
            iterations=iterations,
            verbose=False # Keep API logs cleaner
        )
        end_time = time.time()

        best_binary_solution = [1 if x > 0.5 else 0 for x in result[:-1]]
        # Fitness is the last element, negate it back to maximization value
        best_fitness = -result[-1]

        print(f"WOA finished in {end_time - start_time:.2f}s. Final best fitness: {best_fitness:.4f}")
        honeypot_ips = [self.node_ips[i] for i, val in enumerate(best_binary_solution) if val == 1]
        return honeypot_ips, best_fitness


    def harris_hawks_optimization(self, iterations=50) -> Tuple[List[str], float]:
        """Runs the Harris Hawks Optimization Algorithm."""
        if self.num_nodes == 0: return [], -float('inf')
        print(f"Running HHO: Iterations={iterations}, MaxHP={self.max_honeypots}")

        # Objective function for minimization
        def objective_function(solution: np.ndarray) -> float:
            individual = [1 if x > 0.5 else 0 for x in solution]
            return -self.evaluate_fitness(individual)

        dim = self.num_nodes
        min_values = [0.0] * dim
        max_values = [1.0] * dim

        start_time = time.time()
        result = harris_hawks_optimization(
            target_function=objective_function,
            hawks=max(10, self.num_nodes), # Adjust hawks size
            min_values=min_values,
            max_values=max_values,
            iterations=iterations,
            verbose=False
        )
        end_time = time.time()

        best_binary_solution = [1 if x > 0.5 else 0 for x in result[:-1]]
        best_fitness = -result[-1]

        print(f"HHO finished in {end_time - start_time:.2f}s. Final best fitness: {best_fitness:.4f}")
        honeypot_ips = [self.node_ips[i] for i, val in enumerate(best_binary_solution) if val == 1]
        return honeypot_ips, best_fitness


# --- API Endpoints ---

@app.get("/")
async def read_root():
    """Root endpoint providing basic API info."""
    return {"message": "Welcome to the Adaptive Network Honeypot Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_honeypots(request: PredictionRequest):
    """
    Accepts network data and algorithm choice, returns honeypot prediction.
    """
    print(f"Received prediction request: Algorithm={request.algorithm_type}, MaxHP={request.max_honeypots}")

    if not request.network_data:
        raise HTTPException(status_code=400, detail="No network data provided.")

    # 1. Simulate Vulnerability Scan
    vulnerability_scores, asset_values = simulate_network_scan(request.network_data)

    if not vulnerability_scores:
         raise HTTPException(status_code=400, detail="Could not simulate scan or network is empty.")


    # 2. Initialize Optimizer
    try:
        optimizer = HoneypotOptimizer(
            network_data=request.network_data,
            vulnerability_scores=vulnerability_scores,
            asset_values=asset_values,
            max_honeypots=request.max_honeypots
        )
    except Exception as e:
         print(f"Error initializing optimizer: {e}")
         raise HTTPException(status_code=500, detail=f"Optimizer initialization failed: {e}")


    # 3. Run Selected Algorithm
    predicted_nodes = []
    best_fitness = -float('inf')
    message = "Prediction failed or no suitable honeypots found."

    try:
        if request.algorithm_type == AlgorithmType.GA:
            predicted_nodes, best_fitness = optimizer.genetic_algorithm(
                pop_size=request.ga_pop_size or 50,
                num_generations=request.ga_generations or 30
            )
            message = f"GA prediction complete. Found {predicted_nodes} honeypots."
        elif request.algorithm_type == AlgorithmType.WOA:
            predicted_nodes, best_fitness = optimizer.whale_optimization(
                iterations=request.woa_iterations or 50
            )
            message = f"WOA prediction complete. Found {predicted_nodes} honeypots."
        elif request.algorithm_type == AlgorithmType.HHO:
             predicted_nodes, best_fitness = optimizer.harris_hawks_optimization(
                iterations=request.hho_iterations or 50
            )
             message = f"HHO prediction complete. Found {predicted_nodes} honeypots."

    except Exception as e:
        print(f"Error during optimization ({request.algorithm_type}): {e}")
        # Don't raise HTTPException here, return a response indicating failure
        message = f"Error during {request.algorithm_type} optimization: {e}"
        predicted_nodes = [] # Ensure empty list on error
        best_fitness = None


    # Map internal IPs back to frontend node names/IDs if needed (optional)
    # For now, return the IPs as the identifiers used internally
    final_prediction_ids = predicted_nodes # IPs in this case

    print(f"Prediction result: {final_prediction_ids}, Fitness: {best_fitness}")

    return PredictionResponse(
        predicted_honeypots=final_prediction_ids,
        message=message,
        algorithm_used=request.algorithm_type.value,
        fitness_score=best_fitness if best_fitness > -10000 else None # Don't show penalty score
    )

# --- Run the app ---
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on http://localhost:8000")
    uvicorn.run(app, host="localhost", port=8000, reload=False) # Disable reload for stability