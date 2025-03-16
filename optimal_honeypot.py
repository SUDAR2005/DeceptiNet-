import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
import pandas as pd
import itertools

class NetworkEnvironment:
    def __init__(self, node_data_csv=None, edge_data_csv=None):
        self.G = nx.Graph()
        if node_data_csv:
            try:
                node_df = pd.read_csv(node_data_csv, index_col=0)
                for node_index, row in node_df.iterrows():
                    self.G.add_node(node_index, asset_value=row['asset_value'], vulnerability=row['vulnerability'])
            except FileNotFoundError:
                print(f"Error: Node data CSV file not found at {node_data_csv}.")
                return
            except KeyError as e:
                print(f"Error: Column '{e}' not found in node data CSV file. Using random values instead.")
                return
        if edge_data_csv:
            try:
                edge_df = pd.read_csv(edge_data_csv)
                for _, row in edge_df.iterrows():
                    source = row['source']
                    target = row['target']
                    if source in self.G.nodes() and target in self.G.nodes():
                        self.G.add_edge(source, target, vulnerability=random.uniform(0.1, 1.0))
                    else:
                        print(f"Warning: Edge ({source}, {target}) refers to non-existent nodes.")
            except FileNotFoundError:
                print(f"Error: Edge data CSV file not found at {edge_data_csv}.")
                return
            
        self.attack_paths = []
        if self.G.number_of_nodes() > 0:
            for _ in range(len(edge_df) * 2):
                source = random.choice(list(self.G.nodes()))
                target = random.choice(list(self.G.nodes()))
                if source != target and nx.has_path(self.G, source, target):
                    path = nx.shortest_path(self.G, source, target)
                    if len(path) > 2:
                        self.attack_paths.append(path)
        self.attack_paths = []
        for source, target in itertools.product(self.G.nodes(), self.G.nodes()):
            if source != target and nx.has_path(self.G, source, target):
                try:
                    paths = list(nx.all_shortest_paths(self.G, source, target))
                    for path in paths:
                        if len(path) > 2:
                            self.attack_paths.append(path)
                except nx.NetworkXNoPath:
                    pass

    
    def evaluate_placement(self,honeypot_placement):
        if sum(honeypot_placement) == 0:
            return 0.0,
        honeypot_nodes = [i for i,val in enumerate(honeypot_placement) if val == 1]
        
        path_coverage = 0
        for path in self.attack_paths:
            if any(node in honeypot_nodes for node in path):
                for i,node in enumerate(path):
                    if node in honeypot_nodes:
                        position_factor = (len(path) - i)/len(path)
                        path_coverage += position_factor
                        break
        
        asset_protection = 0
        for node in honeypot_nodes:
            if node in self.G.nodes():
                asset_protection += (self.G.nodes[node]['asset_value'] * 
                                    self.G.nodes[node]['vulnerability'])
        
        deception_score = 0
        for node in honeypot_nodes:
            if node in self.G.nodes():
                deception_score += self.G.degree(node)/self.G.number_of_nodes()
        
        cost_penalty = len(honeypot_nodes) * 2
        
        score = (path_coverage * 3 + asset_protection * 2 + deception_score * 1.5 - cost_penalty)
        
        return score,     
    def visualize_network(self,honeypot_placement=None):
        plt.figure(figsize=(12,8))
        
        pos = nx.spring_layout(self.G,seed=42)
        
        nx.draw_networkx_edges(self.G,pos,alpha=0.3)
        
        node_colors = [self.G.nodes[n]['asset_value'] for n in self.G.nodes()]
        nx.draw_networkx_nodes(self.G,pos,node_color=node_colors,
                              cmap=plt.cm.YlOrRd,node_size=300,alpha=0.8)
        
        if honeypot_placement:
            honeypot_nodes = [i for i,val in enumerate(honeypot_placement) if val == 1]
            nx.draw_networkx_nodes(self.G,pos,nodelist=honeypot_nodes,
                                  node_color='blue',node_size=400,alpha=0.8)
        
        nx.draw_networkx_labels(self.G,pos)
        
        for i,path in enumerate(self.attack_paths):
            path_edges = [(path[i],path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(self.G,pos,edgelist=path_edges,
                                  width=2,alpha=0.5,edge_color=f'C{i%9}')
        
        plt.title("Network with Honeypot Placement")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def genetic_algorithm_optimization(network,num_honeypots=5,population_size=100,generations=50):
    creator.create("FitnessMax",base.Fitness,weights=(1.0,))
    creator.create("Individual",list,fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    toolbox.register("attr_bool",random.randint,0,1)
    toolbox.register("individual",tools.initRepeat,creator.Individual,
                     toolbox.attr_bool,n=network.G.number_of_nodes())
    toolbox.register("population",tools.initRepeat,list,toolbox.individual)
    
    toolbox.register("evaluate",network.evaluate_placement)
    
    toolbox.register("mate",tools.cxTwoPoint)
    toolbox.register("mutate",tools.mutFlipBit,indpb=0.05)
    toolbox.register("select",tools.selTournament,tournsize=3)
    
    def check_constraint(individual):
        if sum(individual) > num_honeypots:
            while sum(individual) > num_honeypots:
                honeypot_indices = [i for i,gene in enumerate(individual) if gene == 1]
                remove_idx = random.choice(honeypot_indices)
                individual[remove_idx] = 0
        return individual
    
    toolbox.register("constraint",check_constraint)
    
    pop = toolbox.population(n=population_size)
    
    for ind in pop:
        toolbox.constraint(ind)
    
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg",np.mean)
    stats.register("std",np.std)
    stats.register("min",np.min)
    stats.register("max",np.max)
    
    pop,log = algorithms.eaSimple(pop,toolbox,cxpb=0.5,mutpb=0.2,
                                 ngen=generations,stats=stats,halloffame=hof,
                                 verbose=True)
    
    best_individual = toolbox.constraint(hof[0])
    
    return best_individual,log

def simulated_annealing_optimization(network,num_honeypots=5,max_iterations=1000,temp_start=100,temp_end=1):
    
    n_nodes = network.G.number_of_nodes()
    
    current_solution = [0] * n_nodes
    honeypot_indices = random.sample(range(n_nodes),num_honeypots)
    for idx in honeypot_indices:
        current_solution[idx] = 1
    
    current_score = network.evaluate_placement(current_solution)[0]
    
    best_solution = current_solution.copy()
    best_score = current_score
    
    temp = temp_start
    cooling_rate = (temp_start - temp_end)/max_iterations
    
    scores = [current_score]
    
    for iteration in range(max_iterations):
        neighbor = current_solution.copy()
        
        honeypot_indices = [i for i,val in enumerate(neighbor) if val == 1]
        non_honeypot_indices = [i for i,val in enumerate(neighbor) if val == 0]
        
        if not honeypot_indices or not non_honeypot_indices:
            continue
        
        remove_idx = random.choice(honeypot_indices)
        add_idx = random.choice(non_honeypot_indices)
        
        neighbor[remove_idx] = 0
        neighbor[add_idx] = 1
        
        neighbor_score = network.evaluate_placement(neighbor)[0]
        
        delta = neighbor_score - current_score
        if delta > 0 or random.random() < np.exp(delta/temp):
            current_solution = neighbor
            current_score = neighbor_score
            
            if current_score > best_score:
                best_solution = current_solution.copy()
                best_score = current_score
        
        temp -= cooling_rate
        scores.append(current_score)
        
        if iteration%100 == 0:
            print(f"Iteration {iteration},Score: {current_score},Best: {best_score},Temp: {temp:.2f}")
    
    return best_solution,scores

def particle_swarm_optimization(network,num_honeypots=5,num_particles=30,max_iterations=100):
    n_nodes = network.G.number_of_nodes()
    
    #/ PSO parameters
    w = 0.7  #/ Inertia weight
    c1 = 1.5  #/ Cognitive parameter
    c2 = 1.5  #/ Social parameter
    
    particles = []
    velocities = []
    p_best_positions = []
    p_best_scores = []
    for _ in range(num_particles):
        particle = [0] * n_nodes
        honeypot_indices = random.sample(range(n_nodes),num_honeypots)
        for idx in honeypot_indices:
            particle[idx] = 1
        particles.append(particle)
        p_best_positions.append(particle.copy())
        p_best_scores.append(network.evaluate_placement(particle)[0])
        velocity = [random.uniform(-0.1,0.1) for _ in range(n_nodes)]
        velocities.append(velocity)
    g_best_idx = np.argmax(p_best_scores)
    g_best_position = p_best_positions[g_best_idx].copy()
    g_best_score = p_best_scores[g_best_idx]
    best_scores = [g_best_score]
    for iteration in range(max_iterations):
        for i in range(num_particles):
            for j in range(n_nodes):
                r1,r2 = random.random(),random.random()
                velocities[i][j] = (w * velocities[i][j] + 
                                  c1 * r1 * (p_best_positions[i][j] - particles[i][j]) + 
                                  c2 * r2 * (g_best_position[j] - particles[i][j]))
            
            particle_probs = [1/(1 + np.exp(-v)) for v in velocities[i]]
            
            new_particle = [1 if random.random() < prob else 0 for prob in particle_probs]
            
            current_honeypots = sum(new_particle)
            
            if current_honeypots > num_honeypots:
                honeypot_indices = [j for j,val in enumerate(new_particle) if val == 1]
                to_remove = random.sample(honeypot_indices,current_honeypots - num_honeypots)
                for idx in to_remove:
                    new_particle[idx] = 0
            elif current_honeypots < num_honeypots:
                non_honeypot_indices = [j for j,val in enumerate(new_particle) if val == 0]
                to_add = random.sample(non_honeypot_indices,num_honeypots - current_honeypots)
        for idx in to_add:
            new_particle[idx] = 1
            particles[i] = new_particle
            score = network.evaluate_placement(particles[i])[0]
            if score > p_best_scores[i]:
                p_best_positions[i] = particles[i].copy()
                p_best_scores[i] = score
                if score > g_best_score:
                    g_best_position = particles[i].copy()
                    g_best_score = score
        best_scores.append(g_best_score)
        if iteration%10 == 0:
            print(f"Iteration {iteration},Best Score: {g_best_score}")
    
    return g_best_position,best_scores

def ant_colony_optimization(network,num_honeypots=5,num_ants=20,max_iterations=100,alpha=1.0,beta=2.0,evaporation=0.5):
    n_nodes = network.G.number_of_nodes()
    pheromone = np.ones(n_nodes)
    heuristic = np.zeros(n_nodes)
    for node in range(n_nodes):
        if node in network.G.nodes():
            node_degree = network.G.degree(node)/max(network.G.degree(),key=lambda x: x[1])[1]
            asset_value = network.G.nodes[node]['asset_value']/10 
            vulnerability = network.G.nodes[node]['vulnerability']
            heuristic[node] = (node_degree + asset_value + vulnerability)/3
    
    best_solution = None
    best_score = float('-inf')
    best_scores = []
    for iteration in range(max_iterations):
        ant_solutions = []
        ant_scores = []
        for _ in range(num_ants):
            solution = [0] * n_nodes
            remaining_nodes = list(range(n_nodes))
            for _ in range(num_honeypots):
                if not remaining_nodes:
                    break
                probabilities = []
                for node in remaining_nodes:
                    prob = (pheromone[node] ** alpha) * (heuristic[node] ** beta)
                    probabilities.append(prob)
                total = sum(probabilities)
                if total == 0: 
                    probabilities = [1/len(remaining_nodes)] * len(remaining_nodes)
                else:
                    probabilities = [p/total for p in probabilities]
                cumulative_prob = 0
                r = random.random()
                selected_idx = 0
                for i,prob in enumerate(probabilities):
                    cumulative_prob += prob
                    if r <= cumulative_prob:
                        selected_idx = i
                        break
                selected_node = remaining_nodes[selected_idx]
                solution[selected_node] = 1
                remaining_nodes.remove(selected_node)
            score = network.evaluate_placement(solution)[0]
            ant_solutions.append(solution)
            ant_scores.append(score)
            if score > best_score:
                best_solution = solution.copy()
                best_score = score
        best_scores.append(best_score)
        pheromone *= (1 - evaporation)
        for i,(solution,score) in enumerate(zip(ant_solutions,ant_scores)):
            if score > 0:
                for j in range(n_nodes):
                    if solution[j] == 1:
                        pheromone[j] += score/10
        if iteration%10 == 0:
            print(f"Iteration {iteration},Best Score: {best_score}")
    
    return best_solution,best_scores

def main(node_data_csv=None, edge_data_csv=None):
    print("Creating network environment...")
    network = NetworkEnvironment(node_data_csv=node_data_csv, edge_data_csv=edge_data_csv)
    print("Visualizing initial network...")
    network.visualize_network()
    algorithm_choice = input("Enter the Algorithm type required: ")
    honeypots = int(input("Enter the number of honeypots: "))      
    print(f"Running {algorithm_choice} algorithm for honeypot placement optimization...")
    if algorithm_choice == "genetic":
        best_solution,_ = genetic_algorithm_optimization(network,num_honeypots=honeypots)
    elif algorithm_choice == "simulated_annealing":
        best_solution,_ = simulated_annealing_optimization(network,num_honeypots=honeypots)
    elif algorithm_choice == "particle_swarm":
        best_solution,_ = particle_swarm_optimization(network,num_honeypots=honeypots)
    elif algorithm_choice == "ant_colony":
        best_solution,_ = ant_colony_optimization(network,num_honeypots=honeypots)
    
    if best_solution is None:
        print("Error: No optimal solution was found.")
        return
    print(f"Optimal honeypot locations: {[i for i,val in enumerate(best_solution) if val == 1]}")
    print("Honeypot placement optimization complete.")
    print(f"Optimal honeypot locations: {[i for i,val in enumerate(best_solution) if val == 1]}")
    print("Visualizing optimized network with honeypots...")
    network.visualize_network(best_solution)
    score = network.evaluate_placement(best_solution)[0]
    print(f"Solution score: {score}")


def colorful_welcome():
    """Prints a colorful welcome message to the terminal."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "reset": "\033[0m",
        "bold": "\033[1m",
        "underline": "\033[4m",
    }

    welcome_message = (
        f"{colors['bold']}{colors['green']}Welcome to {colors['yellow']}GhostNode{colors['magenta']}.AI{colors['reset']}\n"
        f"{colors['blue']}--------------------------------------{colors['reset']}\n"
        f"{colors['cyan']}Honeypot Placement Optimization System{colors['reset']}\n"
        f"{colors['underline']}{colors['red']}Analyzing your network...{colors['reset']}\n"
    )
    print(welcome_message)
if __name__ == "__main__":
    colorful_welcome()
    main('data.csv', 'node.csv')