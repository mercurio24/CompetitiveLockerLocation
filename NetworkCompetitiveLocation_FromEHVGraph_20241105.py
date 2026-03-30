import networkx as nx
import osmnx as ox
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from itertools import combinations, chain
import random
import numpy as np
import plotly.graph_objects as go
from time import time as start_timer
import gurobipy as grb

random.seed(28)

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

def payoff_per_location_decision(location_decision, current_location_actions, player, all_pair_distances, population_per_node, alpha, beta):
    """
    This function calculates the payoff of a player given a location decision
    :param graph: The graph of the city
    :param location_decision: The location decision of the player
    :return: The payoff of the player
    """
    current_actions = current_location_actions.copy()
    current_actions[player] = location_decision
    current_lockers = list(chain(*current_actions.values()))
    # Calculate the payoff of the player
    payoff = 0.0
    for locker in location_decision:
        for node in all_pair_distances.keys():
            # print("Payoff update")
            payoff += population_per_node[node] * np.exp(alpha[locker] - beta * all_pair_distances[locker][node]) / (1 + sum( np.exp(alpha[locker] - beta * all_pair_distances[locker][node]) for locker in current_lockers))
    return payoff

def best_location_action(location_actions, current_location_actions, player, all_pairs_distances, population_per_node, alpha, beta):
    payoffs = {location_decision: payoff_per_location_decision(location_decision, current_location_actions, player, all_pairs_distances, population_per_node, alpha, beta) for location_decision in location_actions}
    # print(f"Player {player} payoffs: {payoffs}")
    best_location = max(payoffs, key=payoffs.get)
    return best_location, payoffs[best_location]

def NE_two_players_by_enumeration_check(location_actions, all_pairs_distances, population_per_node, alpha, beta):
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
    payoff_computation_time = start_timer()
    print("Payoffs computation")
    for locations_player_1 in location_actions[0]:
        for locations_player_2 in location_actions[1]:
            if not symmetric_flag or (symmetric_flag and (locations_player_1, locations_player_2) not in bimatrix_payoffs.keys()):
                # payoffs = {0: payoff_per_location_decision(locations_player_1, {0: locations_player_1, 1: locations_player_2}, 0, all_pairs_distances, population_per_node, alpha, beta),
                #            1: payoff_per_location_decision(locations_player_2, {0: locations_player_1, 1: locations_player_2}, 1, all_pairs_distances, population_per_node, alpha, beta)}
                bimatrix_payoffs[(locations_player_1, locations_player_2)] = (payoff_per_location_decision(locations_player_1, {0: locations_player_1, 1: locations_player_2}, 0, all_pairs_distances, population_per_node, alpha, beta), payoff_per_location_decision(locations_player_2, {0: locations_player_1, 1: locations_player_2}, 1, all_pairs_distances, population_per_node, alpha, beta))
            else:
                bimatrix_payoffs[(locations_player_2, locations_player_1)] = bimatrix_payoffs[(locations_player_1, locations_player_2)]
    print(f"Payoffs computation finished in {round((start_timer() - payoff_computation_time)/60.0, 2)} minutes, checking for Nash Equilibrium")
    best_second_player_action_per_first_player_action = {}
    for locations_player_1 in location_actions[0]:
        best_second_player_action_per_first_player_action = find_all_maxima_in_dict({locations_player_2 : bimatrix_payoffs[(locations_player_1, locations_player_2)][0] for locations_player_2 in location_actions[1]})
        for second_player_action in best_second_player_action_per_first_player_action:
            if all(bimatrix_payoffs[(locations_player_1, second_player_action)][1] >= bimatrix_payoffs[(locations_player_1, locations_player_2)][1] for locations_player_2 in location_actions[1]):
                NEs_detected.append((locations_player_1, second_player_action))
    return NEs_detected

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
    # if status == grb.GRB.Status.OPTIMAL:
    #     print("Optimal solution found")
    if status == grb.GRB.Status.INFEASIBLE:
        print("Model is infeasible")
    elif status == grb.GRB.Status.UNBOUNDED:
        print("Model is unbounded")
    elif status == grb.GRB.Status.OPTIMAL:
        pass
    else:
        print(f"Optimization ended with status {status}")

    # Return the values of x
    return [ll for ll in locker_nodes if x[ll].X == 1.0], model.objVal

def game_simulation(playing_style, solution_method, number_of_lockers_per_player, all_pairs_distances, alpha, beta, initial_location_actions, max_iterations):
    
    computation_time = start_timer()
    new_location_player_dict = {}
    new_payoff_player_dict = {}
    current_location_actions = initial_location_actions.copy()
    current_payoffs = {player: 0 for player in range(len(number_of_lockers_per_player))}
    history_location_actions = []
    iteration = 0
    ### Iterate over the players and find convergence of the game or cycles
    while iteration < max_iterations:
        if playing_style == 'simultaneous':
            for player in range(number_of_players):
                if solution_method == 'enumeration':
                    new_location_player, new_payoff_player = best_location_action(location_actions[player], current_location_actions, player, all_pairs_distances, population_per_node, alpha, beta)
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
                current_payoffs[player] = payoff_per_location_decision(current_location_actions[player], current_location_actions, player, all_pairs_distances, population_per_node, alpha, beta)
            new_location_player_dict, new_payoff_playe_dict = {}, {}
        elif playing_style == 'sequential':
            for player in range(number_of_players):
                if solution_method == 'enumeration':
                    new_location_player, new_payoff_player = best_location_action(location_actions[player], current_location_actions, player, all_pairs_distances, population_per_node, alpha, beta)
                    if new_payoff_player > current_payoffs[player]:
                        current_location_actions[player] = new_location_player
                        current_payoffs[player] = new_payoff_player
                elif solution_method == 'RSOC':
                    current_location_actions[player], current_payoffs[player] = solve_game_by_RSOC(current_location_actions[1-player], population_per_node, utilities, locker_cost, number_of_lockers_per_player[player])
                else:
                    print("Solution method not recognized")
                    break
        else:
            print("Playing style not recognized")
            break
        iteration += 1
        if iteration == max_iterations:
            print("Maximum iterations reached")
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
        print()

    # located_lockers = list(chain(*current_location_actions.values()))

    print(f"Playing style: {playing_style}")
    print(f"Number of iterations: {iteration}")
    print(f"Current location actions: {current_location_actions}")
    print(f"Current payoffs: {current_payoffs}")
    print(f"Computation time: {round((start_timer() - computation_time)/60.0, 2)} minutes")

    return current_location_actions, history_location_actions, convergence_or_cycle



if __name__ == """__main__""":

    # Get the directory name of the current file
    current_file_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file_path)
    graph_path = os.path.dirname(current_folder) + "/RealGraphCreation/eindhoven_with_districts_Binnenstad_Witte Dame_Bergen.graphml"
    playing_style = 'simultaneous' # 'simultaneous' or 'sequential'
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
    number_of_lockers_per_player = [3, 3]#{player: 2 for player in range(number_of_players)}
    number_of_players = len(number_of_lockers_per_player)
    alpha = {district : 1 for district in G.nodes}
    beta = 1e-2

    if solution_method == 'RSOC':
        locker_cost = {node: 1 for node in nodes_with_locker_locations}
        utilities = {(district, locker): np.exp(alpha[district] - beta * all_pairs_distances[locker][district]) for district in all_pairs_distances.keys() for locker in nodes_with_locker_locations}

    ### Enumerate the actions
    location_actions = {player : list(combinations(nodes_with_locker_locations, number_of_lockers_per_player[player])) for player in range(number_of_players)}

    ### Check if a Nash Equilibrium exists by enumeration
    # if number_of_players == 2:
    #     print("Checking for Nash Equilibrium for two players")
    #     NEs_two_players = NE_two_players_by_enumeration_check(location_actions, all_pairs_distances, population_per_node, alpha, beta)
    #     if NEs_two_players:
    #         print(f"Nash Equilibrium detected: {NEs_two_players}")
    #     else:
    #         print("No Nash Equilibrium detected")
    #     print()

    ### Initialize the location decisions of the players
    max_iterations = 100

    # current_location_actions = {player: random.choice(location_actions[player]) for player in range(number_of_players)}
    # current_actions, action_history, convergence_or_cycle = game_simulation(playing_style, solution_method, number_of_lockers_per_player, all_pairs_distances, alpha, beta, current_location_actions, max_iterations)

    break_again = False
    for initial_action_player_0 in location_actions[0]:
        for initial_action_player_1 in location_actions[1]:
            initial_location_actions = {0: initial_action_player_0, 1: initial_action_player_1}
            current_actions, _, convergence_or_cycle = game_simulation(playing_style, solution_method, number_of_lockers_per_player, all_pairs_distances, alpha, beta, initial_location_actions, max_iterations)
            if convergence_or_cycle == "CONVERGED":
                print("Convergence detected: There is a pure Nash Equilibrium")
                break_again = True
                break
        if break_again:
            break

    located_lockers = list(chain(*current_actions.values()))

    colors = ['blue', 'orange', 'olive', 'magenta', 'cyan', 'yellow',
                'purple', 'brown', 'pink', 'green', 'lime', 'navy',
                'teal', 'maroon', 'aqua', 'fuchsia']

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