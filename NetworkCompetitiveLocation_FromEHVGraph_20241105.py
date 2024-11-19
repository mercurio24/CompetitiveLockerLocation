import networkx as nx
import osmnx as ox
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from itertools import combinations, chain
import random
import numpy as np
import plotly.graph_objects as go
from time import time as current_time
import gurobipy as grb
from joblib import Parallel, delayed
import joblib
from contextlib import contextmanager
from tqdm import tqdm

random.seed(28)

@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def find_all_maxima_in_dict(dictionary):
    """
    This function finds all the maxima in a list.
    :param lst: The list of values
    :return: A list of indices of the maxima
    """
    if not dictionary:
        return []

    max_value = max(dictionary.values())
    max_indices = [i for i, value in dictionary.items() if value == max_value]
    return max_indices

def payoff_per_location_decision(location_decision, current_location_actions, player, population_per_node, utilities):
    """
    This function calculates the payoff of a player given a location decision
    :param graph: The graph of the city
    :param location_decision: The location decision of the player
    :return: The payoff of the player
    """
    current_actions = current_location_actions.copy()
    current_actions[player] = location_decision
    current_lockers = list(chain(*current_actions.values()))
    districts = {district for _, district in utilities.keys()}
    # Calculate the payoff of the player
    payoff = 0.0
    for locker in location_decision:
        for district in districts:
            # print("Payoff update")
            payoff += population_per_node[district] * utilities[district, locker] / (1 + sum(utilities[district, locker] for locker in current_lockers))
    return payoff

def best_location_action(location_actions, current_location_actions, player, population_per_node, utilities):
    payoffs = {location_decision: payoff_per_location_decision(location_decision, current_location_actions, player, population_per_node, utilities) for location_decision in location_actions}
    # print(f"Player {player} payoffs: {payoffs}")
    best_location = max(payoffs, key=payoffs.get)
    return best_location, payoffs[best_location]

def NE_two_players_by_enumeration_check(location_actions, population_per_node, utilities):
    """
    This function checks if a Nash Equilibrium exists by enumerating all possible location decisions
    :param location_actions: The possible location actions of the players
    :param all_pairs_distances: The shortest path lengths between all pairs of nodes
    :param population_per_node: The population of each node
    :param alpha: The alpha parameter of the exponential function
    :param beta: The beta parameter of the exponential function
    :return: The Nash Equilibria if it exists
    """

    assert len(location_actions.keys()) == 2, "Only two players are allowed for this function"
    symmetric_flag =  location_actions[0] == location_actions[1]
    print(f"Symmetric game: {symmetric_flag}")
    NEs_detected = []
    bimatrix_payoffs = {}
    payoff_computation_time = current_time()
    print("Payoffs computation")
    for locations_player_1 in location_actions[0]:
        for locations_player_2 in location_actions[1]:
            if not symmetric_flag or (symmetric_flag and (locations_player_1, locations_player_2) not in bimatrix_payoffs.keys()):
                # payoffs = {0: payoff_per_location_decision(locations_player_1, {0: locations_player_1, 1: locations_player_2}, 0, all_pairs_distances, population_per_node, alpha, beta),
                #            1: payoff_per_location_decision(locations_player_2, {0: locations_player_1, 1: locations_player_2}, 1, all_pairs_distances, population_per_node, alpha, beta)}
                bimatrix_payoffs[(locations_player_1, locations_player_2)] = (payoff_per_location_decision(locations_player_1, {0: locations_player_1, 1: locations_player_2}, 0, population_per_node, utilities), payoff_per_location_decision(locations_player_2, {0: locations_player_1, 1: locations_player_2}, 1, population_per_node, utilities))
            else:
                bimatrix_payoffs[(locations_player_2, locations_player_1)] = bimatrix_payoffs[(locations_player_1, locations_player_2)]
    print(f"Payoffs computation finished in {round((current_time() - payoff_computation_time)/60.0, 2)} minutes, checking for Nash Equilibrium")
    best_second_player_action_per_first_player_action = {}
    for locations_player_1 in location_actions[0]:
        best_second_player_action_per_first_player_action = find_all_maxima_in_dict({locations_player_2 : bimatrix_payoffs[(locations_player_1, locations_player_2)][0] for locations_player_2 in location_actions[1]})
        for second_player_action in best_second_player_action_per_first_player_action:
            if all(bimatrix_payoffs[(locations_player_1, second_player_action)][1] >= bimatrix_payoffs[(locations_player_1, locations_player_2)][1] for locations_player_2 in location_actions[1]):
                NEs_detected.append((locations_player_1, second_player_action))
    return NEs_detected

def find_equilibria_by_enumeration_for_two_players(location_actions, all_pairs_distances, population_per_node, alpha, beta):
    ### Check if a Nash Equilibrium exists by enumeration
    number_of_players = len(location_actions.keys())
    if number_of_players == 2:
        print("Checking for Nash Equilibrium for two players")
        NEs_two_players = NE_two_players_by_enumeration_check(location_actions, all_pairs_distances, population_per_node, alpha, beta)
        if NEs_two_players:
            print(f"Nash Equilibrium detected: {NEs_two_players}")
        else:
            print("No Nash Equilibrium detected")
        print()
        return NEs_two_players
    else:
        print("Number of players is not two, cannot check for Nash Equilibrium")
        return None

def solve_game_by_RSOC(other_player_locations, population_per_node, utilities, locker_cost, budget):
    """
    This function finds the optimal strategy through a Rotated Second Order Cone (RSOC) 
    :param location_actions: The possible location actions of the players
    :param all_pairs_distances: The shortest path lengths between all pairs of nodes
    :param population_per_node: The population of each node
    :param alpha: The alpha parameter of the exponential function
    :param beta: The beta parameter of the exponential function
    :return: The Nash Equilibria if it exists
    """

    districts = population_per_node.keys()
    locker_nodes = locker_cost.keys()

    model = grb.Model()
    model.setParam('OutputFlag', 0)  # Set silent mode

    # Variables
    x = model.addVars(locker_nodes, vtype=grb.GRB.BINARY, name="x")
    t = model.addVars(districts, lb=-np.inf, ub=0, name="t")
    z = model.addVars(districts, lb=0, name="z")

    # Constraints
    model.addConstr(grb.quicksum(cost_ll * x[ll] for ll, cost_ll in locker_cost.items()) <= budget, "budget")

    for dd in districts:
        model.addConstr(z[dd] == 1 + 
                        sum(utilities[dd, ll] for ll in other_player_locations) + 
                        grb.quicksum(utilities[dd, ll] * x[ll] for ll in locker_nodes), f"z_{dd}")

    # root = {dd : np.sqrt(0.5 + 0.5 * sum(utilities[dd, ll] for ll in other_player_locations)) 
    #         for dd in districts}

    for dd in districts:
        # model.addConstr(([-t[dd], z[dd], root[dd]] in grb.GRB.RSOC), f"cone_{dd}")
        # model.addConstr(t[dd] * z[dd] >= -1/2 * root[dd]**2, f"rsoc_{dd}")
        model.addConstr( - z[dd] * t[dd] >= 1 + sum(utilities[dd, ll] for ll in other_player_locations), f"rsoc_{dd}")

    # Objective
    model.setObjective(grb.quicksum(population_per_node[dd] * (1 + t[dd]) for dd in districts), grb.GRB.MAXIMIZE)

    # Optimize
    model.optimize()

    # Check the optimization status
    status = model.Status
    if status == grb.GRB.Status.OPTIMAL:
        pass
    elif status == grb.GRB.Status.INFEASIBLE:
        print("Model is infeasible")
    elif status == grb.GRB.Status.UNBOUNDED:
        print("Model is unbounded")
    else:
        print(f"Optimization ended with status {status}")

    # Return the values of x
    return [ll for ll in locker_nodes if x[ll].X == 1.0], model.objVal

def game_simulation_with_initial_actions_given(playing_style, solution_method, number_of_lockers_per_player, population_per_node, utilities, initial_location_actions, max_iterations):
    
    new_location_player_dict = {}
    new_payoff_player_dict = {}
    current_location_actions = initial_location_actions.copy()
    current_payoffs = {player: 0 for player in range(len(number_of_lockers_per_player))}
    history_location_actions = []
    player_for_sequential = 0
    iteration = 0
    ### Iterate over the players and find convergence of the game or cycles
    while iteration < max_iterations:
        if playing_style == 'simultaneous':
            for player in range(number_of_players):
                if solution_method == 'enumeration':
                    new_location_player, new_payoff_player = best_location_action(location_actions[player], current_location_actions, player, population_per_node, utilities)
                    if new_payoff_player > current_payoffs[player]:
                        new_location_player_dict[player] = new_location_player
                    else:
                        new_location_player_dict[player] = current_location_actions[player]
                elif solution_method == 'RSOC':
                    new_location_player_dict[player], new_payoff_player_dict[player] = solve_game_by_RSOC(current_location_actions[1-player], population_per_node, utilities, locker_cost, number_of_lockers_per_player[player])
                else:
                    print("Solution method not recognized")
                    break
            current_location_actions = new_location_player_dict.copy()
            print(f"Number of lockers set: {[len(current_location_actions[player]) for player in range(number_of_players)]}")
            assert all(len(current_location_actions[player]) - number_of_lockers_per_player[player] == 0 for player in range(number_of_players)), "Number of lockers do not match"
            for player in range(number_of_players):
                current_payoffs[player] = payoff_per_location_decision(current_location_actions[player], current_location_actions, player, population_per_node, utilities)
            new_location_player_dict, new_payoff_player_dict = {}, {}
        elif playing_style == 'sequential':
            if solution_method == 'enumeration':
                new_location_player, new_payoff_player = best_location_action(location_actions[player_for_sequential], current_location_actions, player_for_sequential, all_pairs_distances, population_per_node, alpha, beta)
                if new_payoff_player > current_payoffs[player_for_sequential]:
                    current_location_actions[player_for_sequential] = new_location_player
                    current_payoffs[player_for_sequential] = new_payoff_player
            elif solution_method == 'RSOC':
                current_location_actions[player_for_sequential], current_payoffs[player_for_sequential] = solve_game_by_RSOC(current_location_actions[1-player_for_sequential], population_per_node, utilities, locker_cost, number_of_lockers_per_player[player_for_sequential])
            else:
                print("Solution method not recognized")
                break
            player_for_sequential = 1 - player_for_sequential
        else:
            print("Playing style not recognized")
            break
        iteration += 1
        if iteration == max_iterations:
            print("Maximum iterations reached")
            convergence_or_cycle = "MAX_ITERATIONS"
        if current_location_actions in history_location_actions:
            if current_location_actions == history_location_actions[-1]:
                convergence_or_cycle = "CONVERGED"
                print(f"Iteration {iteration}: CONVERGED")
            else:
                convergence_or_cycle = "CYCLE"
                print(f"Iteration {iteration}: CYCLE DETECTED")
            break
        history_location_actions.append(current_location_actions)
        print(f"Iteration {iteration}: {current_location_actions}")
        # if number_of_players == 2:
        #     status_in_NE = True if current_location_actions.values() in NEs_two_players else False
        #     print(f"current iteration in NEs for two players: {status_in_NE}")

    # located_lockers = list(chain(*current_location_actions.values()))

    print(f"Playing style: {playing_style}")
    print(f"Number of iterations: {iteration}")
    print(f"Current location actions: {current_location_actions}")
    print(f"Current payoffs: {current_payoffs}")

    return current_location_actions, current_payoffs, history_location_actions, convergence_or_cycle

def find_equilibria_by_RSOC_for_all_initial_combinations(location_actions, population_per_node, utilities, max_iterations, find_one_or_return_all):
    
    ### Initialize the location decisions of the players
    assert find_one_or_return_all in ['one', 'all'], "Find one or return all not recognized"
    current_equilibria = []
    number_of_initial_combinations = len(location_actions[0]) * len(location_actions[1])
    iterations_counter = 0

    if find_one_or_return_all == 'all':
        with tqdm_joblib(tqdm(desc="Progress", total=number_of_initial_combinations)) as progress_bar:
            results_of_game_simulations = list(Parallel(n_jobs=1, verbose=0)(delayed(game_simulation_with_initial_actions_given)('sequential', 'RSOC', number_of_lockers_per_player, population_per_node, utilities, {0: initial_action_player_0, 1: initial_action_player_1}, max_iterations) for initial_action_player_0 in location_actions[0] for initial_action_player_1 in location_actions[1]))
            return [result[:2] for result in results_of_game_simulations if result[3] == "CONVERGED"]
    elif find_one_or_return_all == 'one':    
        for initial_action_player_0 in location_actions[0]:
            for initial_action_player_1 in location_actions[1]:
                print(f"\nCombination {iterations_counter}/{number_of_initial_combinations}")
                initial_location_actions = {0: initial_action_player_0, 1: initial_action_player_1}
                current_actions, current_payoffs, _, convergence_or_cycle = game_simulation_with_initial_actions_given('sequential', 'RSOC', number_of_lockers_per_player, population_per_node, utilities, initial_location_actions, max_iterations)
                if current_actions in current_equilibria:
                    continue
                if convergence_or_cycle == "CONVERGED":
                    print("Convergence detected: There is a pure Nash Equilibrium")
                return [(current_actions, current_payoffs)]
                    # elif find_one_or_return_all == 'all':
                    #     current_equilibria.append((current_actions, current_payoffs))
                    # else:
                    #     raise ValueError("Find one or return all not recognized")
                    # iterations_counter += 1

    if len(current_equilibria) == 0:
        print("No Nash Equilibrium detected")
    return current_equilibria

def find_social_optimum_by_RSOC(population_per_node, utilities, locker_cost, number_of_lockers_per_player):
    ### Initialize the location decisions of the players
    return solve_game_by_RSOC([], population_per_node, utilities, locker_cost, sum(number_of_lockers_per_player))    


def plot_simulation_state(G, current_actions):
    
    colors = ['blue', 'orange', 'olive', 'magenta', 'cyan', 'yellow',
                'purple', 'brown', 'pink', 'green', 'lime', 'navy',
                'teal', 'maroon', 'aqua', 'fuchsia']
    
    number_of_players = len(current_actions.keys())
    located_lockers = list(chain(*current_actions.values()))

    colors_per_node_with_players = {node: [colors[player] for player in range(number_of_players) if node in current_actions[player]] for node in G.nodes() if node in located_lockers}
    node_colors = ['red' if data.get('locker_possible') == 'locker' else 'black' for _, data in G.nodes(data=True)]
    node_sizes = [50 if data.get('locker_possible') == 'locker' else 10 for _, data in G.nodes(data=True)]
            

    print("Color and size map created")

    ### Plot the graph
    fig, ax = ox.plot_graph(G, 
                            node_color=node_colors, 
                            node_size=node_sizes, 
                            edge_color='black', 
                            bgcolor='white', 
                            show=False, 
                            close=False)
    plt.axis("equal")
    for node, players in colors_per_node_with_players.items():
        node_position = (G.nodes(data=True)[node]['x'], G.nodes(data=True)[node]['y'])  
        for idx_player in range(len(players)-1,-1,-1):
            circle_radius = 2e-4+1.5e-4*idx_player
            circle_color = players[idx_player]
            circle = plt.Circle(node_position, circle_radius, color=circle_color, zorder=number_of_players-idx_player)
            ax.add_patch(circle)
    plt.show()


if __name__ == """__main__""":

    # Get the directory name of the current file
    current_file_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file_path)
    graph_path = os.path.dirname(current_folder) + "/RealGraphCreation/eindhoven_with_districts_Binnenstad_Witte Dame_Bergen.graphml"
    playing_style = 'sequential' # 'simultaneous' or 'sequential'
    solution_method = 'RSOC' # 'enumeration' or 'RSOC'

    ### Load the graph
    G = ox.load_graphml(graph_path)
    if nx.is_directed(G):
        G = G.to_undirected()

    # Find all connected components
    connected_components = list(nx.connected_components(G))
    # Find the largest connected component
    largest_component = max(connected_components, key=len)
    # Create a subgraph containing only the largest connected component
    G = G.subgraph(largest_component).copy()
    all_nodes = list(G.nodes)

    # Calculate shortest path lengths among all pairs of nodes
    print("Calculating all pairs shortest path lengths")
    all_pairs_distances = dict(nx.all_pairs_dijkstra_path_length(G, weight='length'))
    print("Finished calculating all pairs shortest path lengths")

    ### Get the nodes with possible lockers
    nodes_with_locker_locations = [node for node, data in G.nodes(data=True) if data.get('locker_possible') == 'locker']
    population_per_node = {node: round(float(data.get('node_population'))) for node, data in G.nodes(data=True)}

    ### Define the parameters of the players: Players are 0, 1, ..., n_players-1
    number_of_lockers_per_player = [1, 1]#{player: 2 for player in range(number_of_players)}
    number_of_players = len(number_of_lockers_per_player)
    alpha = {district : 1 for district in G.nodes}
    beta = 1e-2
    utilities = {(district, locker): np.exp(alpha[district] - beta * all_pairs_distances[locker][district]) for district in all_pairs_distances.keys() for locker in nodes_with_locker_locations}


    if solution_method == 'RSOC':
        locker_cost = {node: 1 for node in nodes_with_locker_locations}

    ### Enumerate the actions
    location_actions = {player : list(combinations(nodes_with_locker_locations, number_of_lockers_per_player[player])) for player in range(number_of_players)}

    ### Check if a Nash Equilibrium exists by enumeration
    # NEs_detected = find_equilibria_by_enumeration_for_two_players(location_actions, all_pairs_distances, population_per_node, alpha, beta)

    ### Initialize the location decisions of the players
    max_iterations = 100
    find_one_or_return_all = 'all'
    experiment_start_time = current_time()
    equilibria_actions_and_payoffs = find_equilibria_by_RSOC_for_all_initial_combinations(location_actions, population_per_node, utilities, max_iterations, find_one_or_return_all)
    print(f"Computation time: {round((current_time() - experiment_start_time)/60.0, 2)} minutes")
    SO_action, SO_payoff = find_social_optimum_by_RSOC(population_per_node, utilities, locker_cost, number_of_lockers_per_player)
    if find_one_or_return_all == 'one':
        price_of_anarchy = SO_payoff / sum(payoff_per_location_decision(equilibria_actions_and_payoffs[0][0][player], equilibria_actions_and_payoffs[0][0], player, population_per_node, utilities) for player in [0,1])
        print(f"Price of Anarchy: {price_of_anarchy}")
    elif find_one_or_return_all == 'all':  
        smallest_payoff_equilibrium, largest_payoff_equilibrium = min(equilibria_actions_and_payoffs, key=lambda x: sum(x[1].values()))[1], max(equilibria_actions_and_payoffs, key=lambda x: sum(x[1].values()))[1]
        price_of_anarchy = SO_payoff / sum(smallest_payoff_equilibrium.values())
        price_of_stability = SO_payoff / sum(largest_payoff_equilibrium.values())
        print(f"Price of Anarchy: {price_of_anarchy}")
        print(f"Price of Stability: {price_of_stability}")
    else:
        print("Find one or return all not recognized")


    plot_simulation_state(G, equilibria_actions_and_payoffs[0][0])