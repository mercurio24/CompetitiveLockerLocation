import networkx as nx
import osmnx as ox
import sys
import os
import re
from itertools import combinations, chain, product
import random
import numpy as np
from time import time as current_time
import gurobipy as grb
from gurobipy import GRB
from joblib import Parallel, delayed
import joblib
from contextlib import contextmanager
from tqdm import tqdm
from datetime import datetime
from math import isclose
from collections import Counter

RANDOM_SEED = 27
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
N_JOBS = -3

MNL_function = lambda model, p1_lockers, p2_lockers, dd, loc: (loc in p1_lockers)*(model._utilities[dd, loc] /
                                                        (1+sum(model._utilities[dd, ll] for ll in p1_lockers)+sum(model._utilities[dd, ll] for ll in p2_lockers)))
Deriv_MNL_function = lambda model, p1_lockers, p2_lockers, dd, loc1, loc_idx: (loc1 in p1_lockers) * (-model._utilities[dd, loc1] * model._utilities[dd, loc_idx] / (1+sum(model._utilities[dd, ll] for ll in p2_lockers)+sum(model._utilities[dd, ll] for ll in p1_lockers))**2) # * 

def find_float_after_word(text, word):
    # Create a regex pattern to find the word followed by a float
    pattern = rf'{word}\s*([-+]?\d*\.\d+|\d+)'
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return None

def distance_computation_for_locker(graph, locker):
    distances = nx.shortest_path_length(graph, target=locker, weight='length')
    return {((district, locker), distance) for district, distance in distances.items()}
    # try:
    #     return ((district, locker), distance)
    # except nx.NetworkXNoPath:
    #     print("Unconnected path found")
    #     return ((district, locker), float('inf'))    

def right_hand_cuts_callback(model, where):

    if where == GRB.Callback.MIPSOL:#
        x_sol = model.cbGetSolution(model._x)
        y_sol = model.cbGetSolution(model._y)
    # if where == GRB.Callback.MIPNODE:
    #     x_sol = model.cbGetNodeRel(model._x)
    #     y_sol = model.cbGetNodeRel(model._y)

        x_lockers_sol = tuple([ll for ll in model._locker_nodes if x_sol[ll] > 0.5])
        y_lockers_sol = tuple([ll for ll in model._locker_nodes if y_sol[ll] > 0.5])

        x_payoff_sol = payoff_per_location_decision(x_lockers_sol, [y_lockers_sol], model._population_per_node, model._utilities)
        y_payoff_sol = payoff_per_location_decision(y_lockers_sol, [x_lockers_sol], model._population_per_node, model._utilities)

        # location_actions = {player : list(combinations(model._locker_nodes, model._budget[player])) for player in (0,1)}
        # best_action_player_2 = max(location_actions[1], key=lambda x: payoff_per_location_decision(x, [x_lockers_sol], model._population_per_node, model._utilities))

        x_lockers_BR, x_payoff_BR = BestResponse_RSOC_optimization(y_lockers_sol, model._population_per_node, model._utilities, model._locker_cost, model._budgets[0])
        y_lockers_BR, y_payoff_BR = BestResponse_RSOC_optimization(x_lockers_sol, model._population_per_node, model._utilities, model._locker_cost, model._budgets[1])

        # Needing an IF statement
        # if set(y_lockers)!=set(y_lockers_BR):### Move the if to profit comparison instad of the open location
        if x_payoff_sol <= x_payoff_BR - 1e-4:   
            model.cbLazy(grb.quicksum(model._population_per_node[dd] * model._p[1, dd, ll] for dd in model._districts for ll in model._locker_nodes) >= 
                        grb.quicksum(model._population_per_node[dd] * (MNL_function(model, x_lockers_BR, y_lockers_sol, dd, ll) + grb.quicksum(Deriv_MNL_function(model, x_lockers_BR, y_lockers_sol, dd, ll, loc_idx)*(model._y[loc_idx] - 1*(loc_idx in y_lockers_sol)) for loc_idx in model._locker_nodes)) 
                                    for dd in model._districts for ll in model._locker_nodes) )
            # model.cbCut(grb.quicksum(model._population_per_node[dd] * model._p[1, dd, ll] for dd in model._districts for ll in model._locker_nodes) >= 
            #             grb.quicksum(model._population_per_node[dd] * (MNL_function(model, x_lockers_BR, y_lockers_sol, dd, ll) + grb.quicksum(Deriv_MNL_function(model, x_lockers_BR, y_lockers_sol, dd, ll, loc_idx)*(model._y[loc_idx] - 1*(loc_idx in y_lockers_sol)) for loc_idx in model._locker_nodes)) 
            #                         for dd in model._districts for ll in model._locker_nodes) )
        # if set(x_lockers)!=set(x_lockers_BR):### Move the if to profit comparison instad of the open location
        if y_payoff_sol <= y_payoff_BR - 1e-4:     
            model.cbLazy(grb.quicksum(model._population_per_node[dd] * model._p[2, dd, ll] for dd in model._districts for ll in model._locker_nodes) >= 
                        grb.quicksum(model._population_per_node[dd] * (MNL_function(model, y_lockers_BR, x_lockers_sol, dd, ll) + grb.quicksum(Deriv_MNL_function(model, y_lockers_BR, x_lockers_sol, dd, ll, loc_idx)*(model._x[loc_idx] - 1*(loc_idx in x_lockers_sol)) for loc_idx in model._locker_nodes)) 
                                    for dd in model._districts for ll in model._locker_nodes) )
            # model.cbCut(grb.quicksum(model._population_per_node[dd] * model._p[2, dd, ll] for dd in model._districts for ll in model._locker_nodes) >= 
            #             grb.quicksum(model._population_per_node[dd] * (MNL_function(model, y_lockers_BR, x_lockers_sol, dd, ll) + grb.quicksum(Deriv_MNL_function(model, y_lockers_BR, x_lockers_sol, dd, ll, loc_idx)*(model._x[loc_idx] - 1*(loc_idx in x_lockers_sol)) for loc_idx in model._locker_nodes)) 
            #                         for dd in model._districts for ll in model._locker_nodes) )

def right_hand_cuts_callback_Dragotto(model, where):

    if where == GRB.Callback.MIPSOL:#
        x_sol = model.cbGetSolution(model._x)
        y_sol = model.cbGetSolution(model._y)
    # if where == GRB.Callback.MIPNODE:
    #     x_sol = model.cbGetNodeRel(model._x)
    #     y_sol = model.cbGetNodeRel(model._y)

        x_lockers_sol = tuple([ll for ll in model._locker_nodes if x_sol[ll] > 0.5])
        y_lockers_sol = tuple([ll for ll in model._locker_nodes if y_sol[ll] > 0.5])

        x_payoff_sol = payoff_per_location_decision(x_lockers_sol, [y_lockers_sol], model._population_per_node, model._utilities)
        y_payoff_sol = payoff_per_location_decision(y_lockers_sol, [x_lockers_sol], model._population_per_node, model._utilities)

        # location_actions = {player : list(combinations(model._locker_nodes, model._budget[player])) for player in (0,1)}
        # best_action_player_2 = max(location_actions[1], key=lambda x: payoff_per_location_decision(x, [x_lockers_sol], model._population_per_node, model._utilities))

        x_lockers_BR, x_payoff_BR = BestResponse_RSOC_optimization(y_lockers_sol, model._population_per_node, model._utilities, model._locker_cost, model._budgets[0])
        y_lockers_BR, y_payoff_BR = BestResponse_RSOC_optimization(x_lockers_sol, model._population_per_node, model._utilities, model._locker_cost, model._budgets[1])

        # Needing an IF statement
        # if set(y_lockers)!=set(y_lockers_BR):### Move the if to profit comparison instad of the open location
        if x_payoff_sol <= x_payoff_BR - 1e-4:   
            model.cbLazy(grb.quicksum(model._population_per_node[dd] * model._utilities[dd,ll] * model._w[1, dd, ll] for dd in model._districts for ll in model._locker_nodes) >= 
                        grb.quicksum(model._population_per_node[dd] * (MNL_function(model, x_lockers_BR, y_lockers_sol, dd, ll) + grb.quicksum(Deriv_MNL_function(model, x_lockers_BR, y_lockers_sol, dd, ll, loc_idx)*(model._y[loc_idx] - 1*(loc_idx in y_lockers_sol)) for loc_idx in model._locker_nodes)) 
                                    for dd in model._districts for ll in model._locker_nodes) )
            # model.cbCut(grb.quicksum(model._population_per_node[dd] * model._p[1, dd, ll] for dd in model._districts for ll in model._locker_nodes) >= 
            #             grb.quicksum(model._population_per_node[dd] * (MNL_function(model, x_lockers_BR, y_lockers_sol, dd, ll) + grb.quicksum(Deriv_MNL_function(model, x_lockers_BR, y_lockers_sol, dd, ll, loc_idx)*(model._y[loc_idx] - 1*(loc_idx in y_lockers_sol)) for loc_idx in model._locker_nodes)) 
            #                         for dd in model._districts for ll in model._locker_nodes) )
        # if set(x_lockers)!=set(x_lockers_BR):### Move the if to profit comparison instad of the open location
        if y_payoff_sol <= y_payoff_BR - 1e-4:     
            model.cbLazy(grb.quicksum(model._population_per_node[dd] * model._utilities[dd,ll] * model._w[2, dd, ll] for dd in model._districts for ll in model._locker_nodes) >= 
                        grb.quicksum(model._population_per_node[dd] * (MNL_function(model, y_lockers_BR, x_lockers_sol, dd, ll) + grb.quicksum(Deriv_MNL_function(model, y_lockers_BR, x_lockers_sol, dd, ll, loc_idx)*(model._x[loc_idx] - 1*(loc_idx in x_lockers_sol)) for loc_idx in model._locker_nodes)) 
                                    for dd in model._districts for ll in model._locker_nodes) )
            # model.cbCut(grb.quicksum(model._population_per_node[dd] * model._p[2, dd, ll] for dd in model._districts for ll in model._locker_nodes) >= 
            #             grb.quicksum(model._population_per_node[dd] * (MNL_function(model, y_lockers_BR, x_lockers_sol, dd, ll) + grb.quicksum(Deriv_MNL_function(model, y_lockers_BR, x_lockers_sol, dd, ll, loc_idx)*(model._x[loc_idx] - 1*(loc_idx in x_lockers_sol)) for loc_idx in model._locker_nodes)) 
            #                         for dd in model._districts for ll in model._locker_nodes) )

def nogood_callback(model, where):
    # if where == GRB.Callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
    #     x_val = model.cbGetNodeRel(model._x)
    #     y_val = model.cbGetNodeRel(model._y)
    if where == GRB.Callback.MIPSOL:
        x_sol = model.cbGetSolution(model._x)
        y_sol = model.cbGetSolution(model._y)

        x_lockers = tuple([ll for ll in model._locker_nodes if x_sol[ll] > 0.5])
        y_lockers = tuple([ll for ll in model._locker_nodes if y_sol[ll] > 0.5])

        # print(f"X values: {x_lockers} - Y values: {y_lockers}")
    
        _, x_iter_payoff = BestResponse_RSOC_optimization(y_lockers, model._population_per_node, model._utilities, model._locker_cost, model._budgets[0])
        _, y_iter_payoff = BestResponse_RSOC_optimization(x_lockers, model._population_per_node, model._utilities, model._locker_cost, model._budgets[1])

        x_social_payoff = payoff_per_location_decision(x_lockers, [y_lockers], model._population_per_node, model._utilities)
        y_social_payoff = payoff_per_location_decision(y_lockers, [x_lockers], model._population_per_node, model._utilities)

        if x_iter_payoff > x_social_payoff + 1e-5 or y_iter_payoff > y_social_payoff + 1e-5:
            # if not isclose(social_payoff, x_iter_payoff + y_iter_payoff, rel_tol=1e-2):
            # print(f"x best payoff: {round(x_iter_payoff,1)}, x social payoff: {round(x_social_payoff,1)}, y best payoff: {round(y_iter_payoff,1)}, y social payoff: {round(y_social_payoff,1)}")
            # print(f"Social payoff: {round(x_social_payoff + y_social_payoff,1)}, Sum of payoffs: {round(x_iter_payoff + y_iter_payoff,1)}")
            model.cbLazy(grb.quicksum(1 - model._x[ll] for ll in x_lockers) + grb.quicksum(model._x[ll] for ll in model._locker_nodes if ll not in x_lockers) + grb.quicksum(1 - model._y[ll] for ll in y_lockers) + grb.quicksum(model._y[ll] for ll in model._locker_nodes if ll not in y_lockers) >= 1)
            
def find_first_float_after_substring(s, substring):
    import re
    pattern = re.escape(substring) + r'\s*([-+]?\d*\.\d+|\d+)'
    match = re.search(pattern, s)
    if match:
        return float(match.group(1))
    return None

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

def check_couples_first_coincide_and_second_too(tuples_list):
    for i in range(len(tuples_list)):
        for j in range(i + 1, len(tuples_list)):
            if tuples_list[i][0] == tuples_list[j][0] and tuples_list[i][1] != tuples_list[j][1]:
                return False
    return True

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

def payoff_per_location_decision(location_decision, other_location_actions, population_per_node, utilities):
    """
    This function calculates the payoff of a player given a location decision
    :param graph: The graph of the city
    :param location_decision: The location decision of the player
    :return: The payoff of the player
    """

    # all_lockers = [location_decision] + list(chain(other_location_actions))
    all_lockers = [ll for ll in location_decision] + list(chain(*other_location_actions))
    payoff = sum(population_per_node[district] * sum(utilities[district, locker_1] for locker_1 in location_decision) / (1 + sum(utilities[district, locker_2] for locker_2 in all_lockers)) for district in population_per_node.keys())
    return payoff#round(payoff, 3)

def payoff_per_location_decisions_for_two_players(locations_player_1, locations_player_2, population_per_node, utilities):
    """
    This function calculates the payoff of both players given location decisions
    :param graph: The graph of the city
    :param location_decision: The location decision of the player
    :return: The payoff of the player
    """
    return (locations_player_1, locations_player_2), (payoff_per_location_decision(locations_player_1, [locations_player_2], population_per_node, utilities), payoff_per_location_decision(locations_player_2, [locations_player_1], population_per_node, utilities))

def best_location_action(location_actions, other_location_actions, population_per_node, utilities):
    payoffs = {location_decision: payoff_per_location_decision(location_decision, other_location_actions, population_per_node, utilities) for location_decision in location_actions}
    # print(f"Player {player} payoffs: {payoffs}")
    best_location_decision = max(payoffs, key=payoffs.get)
    return best_location_decision, payoffs[best_location_decision]

def NE_two_players_by_enumeration_check(location_actions, population_per_node, utilities, find_one_or_return_all):
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
    assert find_one_or_return_all in ['one', 'all'], "Find one or return all not recognized"
    symmetric_flag =  location_actions[0] == location_actions[1]
    print(f"Symmetric game: {symmetric_flag}")
    NEs_detected = []
    bimatrix_payoffs = {}
    payoff_computation_time = current_time()
    print("Payoffs computation")
    # for locations_player_1 in location_actions[0]:
    #     for locations_player_2 in location_actions[1]:
    #         if not symmetric_flag or (symmetric_flag and (locations_player_1, locations_player_2) not in bimatrix_payoffs.keys()):
    #             bimatrix_payoffs[(locations_player_1, locations_player_2)] = (payoff_per_location_decision(locations_player_1, {0: locations_player_1, 1: locations_player_2}, 0, population_per_node, utilities), payoff_per_location_decision(locations_player_2, {0: locations_player_1, 1: locations_player_2}, 1, population_per_node, utilities))
    #         else:
    #             bimatrix_payoffs[(locations_player_2, locations_player_1)] = bimatrix_payoffs[(locations_player_1, locations_player_2)]
    with tqdm_joblib(tqdm(desc="Progress", total=len(location_actions[0])*len(location_actions[1]))):
        bimatrix_payoffs = dict(Parallel(n_jobs=N_JOBS, verbose=0)(delayed(payoff_per_location_decisions_for_two_players)(locations_player_1, locations_player_2, population_per_node, utilities) for locations_player_1 in location_actions[0] for locations_player_2 in location_actions[1]))
    print(f"Payoffs computation finished in {round((current_time() - payoff_computation_time)/60.0, 2)} minutes, checking for Nash Equilibrium")
    # for locations_player_1 in location_actions[0]:
    #     for locations_player_2 in location_actions[1]: 
    #         if all(bimatrix_payoffs[(locations_player_1, locations_player_2)][0] >= bimatrix_payoffs[(first_player_action, locations_player_2)][0] for first_player_action in location_actions[0]) and all(bimatrix_payoffs[(locations_player_1, locations_player_2)][1] >= bimatrix_payoffs[(locations_player_1, second_player_action)][1] for second_player_action in location_actions[1]):
    #             if find_one_or_return_all == 'one':
    #                 return [((locations_player_1, locations_player_2),bimatrix_payoffs[(locations_player_1, locations_player_2)])]
    #             else:
    #                 NEs_detected.append(((locations_player_1, locations_player_2),bimatrix_payoffs[(locations_player_1, locations_player_2)]))
    for locations_player_1 in location_actions[0]:
        best_action_player_2 = max(location_actions[1], key=lambda x: bimatrix_payoffs[(locations_player_1, x)][1])
        if all(bimatrix_payoffs[(locations_player_1, best_action_player_2)][0] >= bimatrix_payoffs[(first_player_action, best_action_player_2)][0] for first_player_action in location_actions[0]):
            if find_one_or_return_all == 'one':
                return [((locations_player_1, best_action_player_2),bimatrix_payoffs[(locations_player_1, best_action_player_2)])]
            else:
                NEs_detected.append(((locations_player_1, best_action_player_2),bimatrix_payoffs[(locations_player_1, best_action_player_2)]))
    return NEs_detected

def find_equilibria_by_enumeration_for_two_players(location_actions, population_per_node, utilities, find_one_or_return_all):
    ### Check if a Nash Equilibrium exists by enumeration
    if len(location_actions.keys()) == 2:
        print("Checking for Nash Equilibrium for two players")
        NEs_two_players = NE_two_players_by_enumeration_check(location_actions, population_per_node, utilities, find_one_or_return_all)
        if NEs_two_players:
            print(f"Nash Equilibrium detected: {NEs_two_players}")
        else:
            print("No Nash Equilibrium detected")
        return NEs_two_players
    else:
        print("Number of players is not two, cannot check for Nash Equilibrium")
        return None

def BestResponse_RSOC_optimization(other_player_locations, population_per_node, utilities, locker_cost, budget):
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
        model.addConstr( - z[dd] * t[dd] >= 1 + sum(utilities[dd, ll] for ll in other_player_locations), f"rsoc_{dd}")     

    # Objective
    model.setObjective(grb.quicksum(population_per_node[dd] * (1 + t[dd]) for dd in districts), grb.GRB.MAXIMIZE)
    model.setParam('MIPGap', 0)

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

    chosen_lockers = [ll for ll in locker_nodes if x[ll].X > 0.5]
    # payoff = payoff_per_location_decision(chosen_lockers, other_player_locations, population_per_node, utilities)
    
    return tuple(chosen_lockers), model.objVal

def game_simulation_with_initial_actions_given(playing_style, solution_method, number_of_lockers_per_player, population_per_node, utilities, initial_location_actions, max_iterations, printing=False):
    
    new_location_player_dict, new_payoff_player_dict = [None, None], [None,None]
    current_location_actions = list(initial_location_actions).copy()
    current_payoffs = [payoff_per_location_decision(current_location_actions[0], current_location_actions[1:], population_per_node, utilities), payoff_per_location_decision(current_location_actions[1], current_location_actions[:1], population_per_node, utilities)]
    history_location_actions = []
    player_for_sequential = 0
    iteration = 0

    ### Check if the initial actions are a Nash Equilibrium already
    initial_payoffs = [payoff_per_location_decision(initial_location_actions[0], initial_location_actions[1:], population_per_node, utilities), payoff_per_location_decision(initial_location_actions[1], initial_location_actions[:1], population_per_node, utilities)]
    if all(initial_payoffs[0] >= payoff_per_location_decision(first_player_action, [initial_location_actions[1]], population_per_node, utilities) for first_player_action in location_actions[0]) and all(initial_payoffs[1] >= payoff_per_location_decision(second_player_action, [initial_location_actions[0]], population_per_node, utilities) for second_player_action in location_actions[1]):
        return tuple(initial_location_actions), tuple(initial_payoffs), "CONVERGED"
    ### Iterate over the players and find convergence of the game or cycles
    while iteration < max_iterations:
        if playing_style == 'simultaneous':
            for player in range(number_of_players):
                # if solution_method == 'enumeration':
                #     new_location_player, new_payoff_player = best_location_action(location_actions[player], current_location_actions, population_per_node, utilities)
                #     if new_payoff_player > current_payoffs[player]:
                #         new_location_player_dict[player] = new_location_player
                #     else:
                        # new_location_player_dict[player] = current_location_actions[player]
                if solution_method == 'RSOC':
                    new_location_player_dict[player], new_payoff_player_dict[player] = BestResponse_RSOC_optimization(current_location_actions[1-player], population_per_node, utilities, locker_cost, number_of_lockers_per_player[player])
                else:
                    print("Solution method not recognized")
                    break
            current_location_actions = new_location_player_dict.copy()
            if printing:
                print(f"Number of lockers set: {[len(current_location_actions[player]) for player in range(number_of_players)]}")
            assert all(len(current_location_actions[player]) - number_of_lockers_per_player[player] == 0 for player in range(number_of_players)), "Number of lockers do not match"
            for player in range(number_of_players):
                current_payoffs[player] = payoff_per_location_decision(current_location_actions[player], current_location_actions[:player] + current_location_actions[player+1:], population_per_node, utilities)
            new_location_player_dict, new_payoff_player_dict = [None, None], [0,0]
        if playing_style == 'sequential':
            # if solution_method == 'enumeration':
            #     new_location_player, new_payoff_player = best_location_action(location_actions[player_for_sequential], current_location_actions[:player] + current_location_actions[player+1:], population_per_node, utilities)
            #     current_location_actions[player_for_sequential] = new_location_player
            #     current_payoffs[player_for_sequential] = new_payoff_player
            if solution_method == 'RSOC':
                current_location_actions[player_for_sequential], current_payoffs[player_for_sequential] = BestResponse_RSOC_optimization(current_location_actions[1-player_for_sequential], population_per_node, utilities, locker_cost, number_of_lockers_per_player[player_for_sequential])
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
        if tuple(current_location_actions) in history_location_actions:
            if tuple(current_location_actions) == history_location_actions[-1]:
                convergence_or_cycle = "CONVERGED"
                if printing:
                    print(f"Iteration {iteration}: CONVERGED")
            else:
                print(f"Iteration {iteration}: CYCLE DETECTED")
                convergence_or_cycle = "CYCLE"
                if printing:
                    print(f"Iteration {iteration}: CYCLE DETECTED")
            break
        history_location_actions.append(tuple(current_location_actions))
        if printing:
            print(f"Iteration {iteration}: {current_location_actions}")
        # if number_of_players == 2:
        #     status_in_NE = True if current_location_actions.values() in NEs_two_players else False
        #     print(f"current iteration in NEs for two players: {status_in_NE}")

    # located_lockers = list(chain(*current_location_actions.values()))

    if printing:
        print(f"Playing style: {playing_style}")
        print(f"Number of iterations: {iteration}")
        print(f"Current location actions: {current_location_actions}")
        print(f"Current payoffs: {current_payoffs}")

    return tuple(current_location_actions), tuple(current_payoffs), convergence_or_cycle

def find_equilibria_by_RSOC_for_all_initial_combinations(location_actions, population_per_node, utilities, max_iterations, find_one_or_return_all):
    
    ### Initialize the location decisions of the players
    assert find_one_or_return_all in ['one', 'all'], "Find one or return all not recognized"
    current_equilibria = []
    number_of_initial_combinations = len(location_actions[0]) * len(location_actions[1])
    iterations_counter = 0

    if find_one_or_return_all == 'all':
        results_of_game_simulations = []
        n_jobs = N_JOBS
        if n_jobs == 1:
            results_of_game_simulations = Parallel(n_jobs=1, verbose=0)(delayed(game_simulation_with_initial_actions_given)('sequential', 'RSOC', number_of_lockers_per_player, population_per_node, utilities, [initial_action_player_0, initial_action_player_1], max_iterations) for initial_action_player_0 in location_actions[0] for initial_action_player_1 in location_actions[1])
        if n_jobs != 1:
            ## REGULAR PARALLELIZATION
            with tqdm_joblib(tqdm(desc="Progress", total=number_of_initial_combinations)) as progress_bar:
                results_of_game_simulations = Parallel(n_jobs=n_jobs, verbose=0)(delayed(game_simulation_with_initial_actions_given)('sequential', 'RSOC', number_of_lockers_per_player, population_per_node, utilities, [initial_action_player_0, initial_action_player_1], max_iterations) for initial_action_player_0 in location_actions[0] for initial_action_player_1 in location_actions[1])
            ### DEBUGGING CHECK
            # results_of_game_simulations = []
            # for initial_action_player_0 in location_actions[0]:
            #     for initial_action_player_1 in location_actions[1]:
            #         simulation = game_simulation_with_initial_actions_given('sequential', 'RSOC', number_of_lockers_per_player, population_per_node, utilities, [initial_action_player_0, initial_action_player_1], max_iterations)
            #         results_of_game_simulations.append(simulation)
            #         print(f"From {(initial_action_player_0, initial_action_player_1)}\n{simulation}\n")
        assert check_couples_first_coincide_and_second_too(results_of_game_simulations), "First coincide and second not"
        return {tuple(result[:2]) for result in results_of_game_simulations if result[2] == "CONVERGED"}
    if find_one_or_return_all == 'one':    
        for initial_action_player_0 in location_actions[0]:
            for initial_action_player_1 in location_actions[1]:
                print(f"\nCombination {iterations_counter+1}/{number_of_initial_combinations}")
                initial_location_actions = [initial_action_player_0, initial_action_player_1]
                current_actions, current_payoffs, convergence_or_cycle = game_simulation_with_initial_actions_given('sequential', 'RSOC', number_of_lockers_per_player, population_per_node, utilities, initial_location_actions, max_iterations, True)
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

def find_social_optimum_by_RSOC(population_per_node, utilities, locker_cost, budgets):
    
    districts = population_per_node.keys()
    locker_nodes = locker_cost.keys()

    model = grb.Model()
    model.setParam('OutputFlag', 0)  # Set silent mode
    model.setParam('MIPGap', 0)

    # Variables
    x = model.addVars(locker_nodes, vtype=grb.GRB.BINARY, name="x")
    y = model.addVars(locker_nodes, vtype=grb.GRB.BINARY, name="y")
    t = model.addVars(districts, lb=-np.inf, ub=0, name="t")
    z = model.addVars(districts, lb=0, name="z")

    # Constraints
    model.addConstr(grb.quicksum(cost_ll * x[ll] for ll, cost_ll in locker_cost.items()) <= budgets[0], "budget1")
    model.addConstr(grb.quicksum(cost_ll * y[ll] for ll, cost_ll in locker_cost.items()) <= budgets[1], "budget2")

    for dd in districts:
        model.addConstr(z[dd] == 1 + 
                        grb.quicksum(utilities[dd, ll] * x[ll] for ll in locker_nodes) + 
                        grb.quicksum(utilities[dd, ll] * y[ll] for ll in locker_nodes), f"z_{dd}")
        model.addConstr( - z[dd] * t[dd] >= 1, f"rsoc_{dd}") 

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

    
    x_lockers = [ll for ll in locker_nodes if x[ll].X>0.5]
    y_lockers = [ll for ll in locker_nodes if y[ll].X>0.5]

    payoff1 = payoff_per_location_decision(x_lockers, [y_lockers], population_per_node, utilities)
    payoff2 = payoff_per_location_decision(y_lockers, [x_lockers], population_per_node, utilities)

    # all_lockers = x_lockers + y_lockers
    # social_payoff = 0.0
    # for locker in all_lockers:
    #     for district in districts:
    #         social_payoff += population_per_node[district] * utilities[district, locker] / (1 + sum(utilities[district, locker] for locker in all_lockers))
    # social_payoff = round(model.objVal, 3)

    return tuple([x_lockers, y_lockers]), (payoff1, payoff2)  

def Equilibrium_PoA_PoS_linearized_model(population_per_node, utilities, locker_cost, budgets, PoA_or_PoS="PoS", check_actual_equilibrium=True):
    
    assert PoA_or_PoS in ["PoA", "PoS", "both"], "Price of Anarchy or Price of Stability not recognized"
    districts = population_per_node.keys()
    locker_nodes = list(locker_cost.keys())
    # BigM = 1

    # derivative = lambda x, dd, loc: population_per_node[dd] * ((1+sum(utilities[dd, ll]*model._y[ll] for ll in locker_nodes)+sum(utilities[dd, ll]*model._x[ll] for ll in locker_nodes if ll!=loc))/
    #                                                     (1+sum(utilities[dd, ll]*model._y[ll] for ll in locker_nodes)+sum(utilities[dd, ll]*model._x[ll] for ll in locker_nodes))**2)

    print(f"Some utilities: {random.sample(list(utilities.items()), 10)}")
    print("Finding an initial equilibrium with best response behavior")
    x_eq, y_eq = [], []
    old_x_eq, old_y_eq = None, None
    iteration, max_iterations = 1, 1000
    while iteration <= 1000:
        old_x_eq = x_eq
        x_eq, _ = BestResponse_RSOC_optimization(y_eq, population_per_node, utilities, locker_cost, budgets[0])
        if x_eq == old_x_eq and y_eq == old_y_eq:
            print(f"Equilibrium found in {iteration} iterations: {x_eq} - {y_eq}\n")
            break
        old_y_eq = y_eq
        y_eq, _ = BestResponse_RSOC_optimization(x_eq, population_per_node, utilities, locker_cost, budgets[1])
        if x_eq == old_x_eq and y_eq == old_y_eq:
            print(f"Equilibrium found in {iteration} iterations: {x_eq} - {y_eq}\n")
            break
        iteration += 1
    if iteration == max_iterations:
        print("Maximum iterations reached\n")        

    with grb.Env() as env, grb.Model(env=env) as model:

        # model.setParam('LazyConstraints', 1)
        model.setParam('OutputFlag', 1)  # Set silent mode
        model.setParam('MIPGap', 0)
        # model.setParam('PreSOS1BigM', -1)

        model._locker_nodes = locker_nodes
        model._locker_cost = locker_cost
        model._districts = districts
        model._population_per_node = population_per_node
        model._utilities = utilities
        model._budgets = budgets
        # model._upperbound = {}
        # model._upperbound.update({(pp,dd,ll) : model._utilities[dd, ll]/(1+model._utilities[dd, ll]) for dd in model._districts for ll in model._locker_nodes})
        model._upperbound = {(dd,ll) : model._utilities[dd, ll]/(1+model._utilities[dd, ll]+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes])[:sum(budgets)-1])) for dd in model._districts for ll in model._locker_nodes}
        # for pp in (1,2):
        #     if budgets[pp-1] <= 1:
        #         model._upperbound.update({(pp,dd,ll) : model._utilities[dd, ll]/(1+model._utilities[dd, ll]) for dd in model._districts for ll in model._locker_nodes})
        #     else:
        #         model._upperbound.update({(pp,dd,ll) : model._utilities[dd, ll]/(1+model._utilities[dd, ll]+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes if tt!=ll])[:sum(budgets)-1])) for dd in model._districts for ll in model._locker_nodes})
        # model._upperbound = {(dd,ll) : 1 for dd in model._districts for ll in model._locker_nodes}
        # model._BigN = {(dd,ll) : 1/(1+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes])[:sum(budgets)])) for dd in model._districts for ll in model._locker_nodes}
        model._BigN = {(dd,ll) : 1 for dd in model._districts for ll in model._locker_nodes}

        # Variables
        model._x = model.addVars(model._locker_nodes, vtype=grb.GRB.BINARY, name="x")
        model._y = model.addVars(model._locker_nodes, vtype=grb.GRB.BINARY, name="y")
        model._p = model.addVars((1,2), model._districts, model._locker_nodes, lb=0.0, vtype=grb.GRB.CONTINUOUS, name="p")
        model._p0 = model.addVars(model._districts, lb=0.0, vtype=grb.GRB.CONTINUOUS, name="p0")

        for ll in model._locker_nodes:
            model._x[ll].start = 1 if ll in x_eq else 0
            model._y[ll].start = 1 if ll in y_eq else 0
        for dd in model._districts:
            model._p0[dd].start = 1 / (1 + sum(utilities[dd, llx] for llx in x_eq+y_eq))
            for ll in model._locker_nodes:
                model._p[1, dd, ll].start = utilities[dd, ll] / (1 + sum(utilities[dd, llx] for llx in x_eq+y_eq)) if ll in x_eq else 0
                model._p[2, dd, ll].start = utilities[dd, ll] / (1 + sum(utilities[dd, llx] for llx in x_eq+y_eq)) if ll in y_eq else 0

        if PoA_or_PoS == "PoS":
            model.setObjective(grb.quicksum(model._population_per_node[dd] * (model._p[1, dd, ll] + model._p[2, dd, ll]) for dd in model._districts for ll in model._locker_nodes), grb.GRB.MAXIMIZE)
        if PoA_or_PoS == "PoA":
            model.setObjective(grb.quicksum(model._population_per_node[dd] * (model._p[1, dd, ll] + model._p[2, dd, ll]) for dd in model._districts for ll in model._locker_nodes), grb.GRB.MINIMIZE)

        # # Budget Constraints
        model.addConstr(grb.quicksum(cost_ll * model._x[ll] for ll, cost_ll in model._locker_cost.items()) == model._budgets[0], "budget1")
        model.addConstr(grb.quicksum(cost_ll * model._y[ll] for ll, cost_ll in model._locker_cost.items()) == model._budgets[1], "budget2")

        # # First Margarida constraints

        model.addConstrs(model._p0[dd] + grb.quicksum(model._p[1, dd, ll] + model._p[2, dd, ll] for ll in model._locker_nodes) == 1  for dd in model._districts)
        model.addConstrs(model._p[1, dd, ll] <= model._upperbound[(dd,ll)] * model._x[ll] for dd in model._districts for ll in model._locker_nodes)
        model.addConstrs(model._p[2, dd, ll] <= model._upperbound[(dd,ll)] * model._y[ll] for dd in model._districts for ll in model._locker_nodes)
        # model.addConstrs(model._p[pp, dd, ll1] <= model._utilities[dd,ll1]/model._utilities[dd,ll2]*model._p[1, dd, ll2] + (1-model._x[ll2]) for pp in (1,2) for dd in model._districts for ll1 in model._locker_nodes for ll2 in model._locker_nodes if ll1!=ll2)  
        # model.addConstrs(model._p[pp, dd, ll1] <= model._utilities[dd,ll1]/model._utilities[dd,ll2]*model._p[2, dd, ll2] + (1-model._y[ll2]) for pp in (1,2) for dd in model._districts for ll1 in model._locker_nodes for ll2 in model._locker_nodes if ll1!=ll2)  
        model.addConstrs(model._p[1, dd, ll] >= model._utilities[dd,ll]*(model._p0[dd]+model._BigN[dd,ll]*(model._x[ll]-1)) for dd in model._districts for ll in model._locker_nodes)  
        model.addConstrs(model._p[2, dd, ll] >= model._utilities[dd,ll]*(model._p0[dd]+model._BigN[dd,ll]*(model._y[ll]-1)) for dd in model._districts for ll in model._locker_nodes)  
        model.addConstrs(model._p[pp, dd, ll] <= model._utilities[dd,ll]*model._p0[dd] for pp in (1,2) for dd in model._districts for ll in model._locker_nodes)  
        model.setParam('LazyConstraints', 1) #Add when using cbLazy
        # model.setParam('PreCrush', 1) #Add when using cbCut
        model.optimize(right_hand_cuts_callback)


        # # Margarida constraints from Dragotto
        # model._x = model.addVars(model._locker_nodes, vtype=grb.GRB.BINARY, name="x")
        # model._y = model.addVars(model._locker_nodes, vtype=grb.GRB.BINARY, name="y")
        # model._w = model.addVars((1,2), model._districts, model._locker_nodes, lb=0.0, vtype=grb.GRB.CONTINUOUS, name="p")
        # model._z = model.addVars(model._districts, lb=0.0, vtype=grb.GRB.CONTINUOUS, name="p0")

        # model.addConstr(grb.quicksum(cost_ll * model._x[ll] for ll, cost_ll in model._locker_cost.items()) <= model._budgets[0], "budget1")
        # model.addConstr(grb.quicksum(cost_ll * model._y[ll] for ll, cost_ll in model._locker_cost.items()) <= model._budgets[1], "budget2")
        # model._upperbound = {(pp,dd,ll) : 1/(1+model._utilities[dd,ll]+sum(sorted([model._utilities[dd,ll] for ll in model._locker_nodes])[:sum(model._budgets)-1])) for pp in [1,2] for dd in model._districts for ll in model._locker_nodes}
        # model.addConstrs(model._z[dd] +  grb.quicksum(model._utilities[dd,ll] * (model._w[1, dd, ll] + model._w[2, dd, ll]) for ll in model._locker_nodes) == 1  for dd in model._districts)
        # model.addConstrs(model._w[1, dd, ll] <= model._upperbound[(1,dd,ll)] * model._x[ll] for dd in model._districts for ll in model._locker_nodes)
        # model.addConstrs(model._w[2, dd, ll] <= model._upperbound[(2,dd,ll)] * model._y[ll] for dd in model._districts for ll in model._locker_nodes)
        # model.addConstrs(model._w[pp, dd, ll] <=  model._z[dd] for pp in [1,2] for dd in model._districts for ll in model._locker_nodes)
        # model.addConstrs(model._w[1, dd, ll] >= model._z[dd] + 1 * (model._x[ll] - 1) for dd in model._districts for ll in model._locker_nodes)
        # model.addConstrs(model._w[2, dd, ll] >= model._z[dd] + 1 * (model._y[ll] - 1) for dd in model._districts for ll in model._locker_nodes)
        # if PoA_or_PoS == "PoS":
        #     model.setObjective(grb.quicksum(model._population_per_node[dd] * model._utilities[dd,ll] * (model._w[1, dd, ll] + model._w[2, dd, ll]) for dd in model._districts for ll in model._locker_nodes), grb.GRB.MAXIMIZE)
        # elif PoA_or_PoS == "PoA":
        #     model.setObjective(grb.quicksum(model._population_per_node[dd] * model._utilities[dd,ll] * (model._w[1, dd, ll] + model._w[2, dd, ll]) for dd in model._districts for ll in model._locker_nodes), grb.GRB.MINIMIZE)
        # model.setParam('LazyConstraints', 1) #Add when using cbLazy
        # model.optimize(right_hand_cuts_callback_Dragotto)


        # ### McCormick envelopes
        # model._p0_lowerbound_x0y0=  {(dd,ll) : 1/(1+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes if tt!=ll], reverse=True)[:budgets[0]])+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes], reverse=True)[:budgets[1]])) for dd in model._districts for ll in model._locker_nodes}
        # model._p0_lowerbound_x1y0=  {(dd,ll) : 1/(1+model._utilities[dd, ll]+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes if tt!=ll], reverse=True)[:budgets[0]-1])+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes], reverse=True)[:budgets[1]])) for dd in model._districts for ll in model._locker_nodes}
        # model._p0_lowerbound_x0y1=  {(dd,ll) : 1/(1+model._utilities[dd, ll]+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes], reverse=True)[:budgets[0]])+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes if tt!=ll], reverse=True)[:budgets[1]-1])) for dd in model._districts for ll in model._locker_nodes}
        # model._p0_lowerbound_x1y1=  {(dd,ll) : 1/(1+2*model._utilities[dd, ll]+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes if tt!=ll], reverse=True)[:budgets[0]-1])+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes if tt!=ll], reverse=True)[:budgets[1]-1])) for dd in model._districts for ll in model._locker_nodes}
        # model._p0_upperbound_x0y0=  {(dd,ll) : 1/(1+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes if tt!=ll], reverse=False)[:budgets[0]])+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes], reverse=False)[:budgets[1]])) for dd in model._districts for ll in model._locker_nodes}
        # model._p0_upperbound_x1y0=  {(dd,ll) : 1/(1+model._utilities[dd, ll]+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes if tt!=ll], reverse=False)[:budgets[0]-1])+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes], reverse=False)[:budgets[1]])) for dd in model._districts for ll in model._locker_nodes}
        # model._p0_upperbound_x0y1=  {(dd,ll) : 1/(1+model._utilities[dd, ll]+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes], reverse=False)[:budgets[0]])+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes if tt!=ll], reverse=False)[:budgets[1]-1])) for dd in model._districts for ll in model._locker_nodes}
        # model._p0_upperbound_x1y1=  {(dd,ll) : 1/(1+2*model._utilities[dd, ll]+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes if tt!=ll], reverse=False)[:budgets[0]-1])+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes if tt!=ll], reverse=False)[:budgets[1]-1])) for dd in model._districts for ll in model._locker_nodes}
        # # model._p0_lowerbound_xeq1=  {(dd,ll) : model._utilities[dd, ll]/(1+model._utilities[dd, ll]+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes], reverse=True)[:sum(budgets)-1])) for dd in model._districts for ll in model._locker_nodes}
        # # model._p0_upperbound_xeq0=  {(dd,ll) : 1/(1+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes if tt!=ll], reverse=True)[:sum(budgets)])) for dd in model._districts for ll in model._locker_nodes}
        # # model._p0_upperbound_xeq1=  {(dd,ll) : model._utilities[dd, ll]/(1+model._utilities[dd, ll]+sum(sorted([model._utilities[dd, tt] for tt in model._locker_nodes], reverse=False)[:sum(budgets)-1])) for dd in model._districts for ll in model._locker_nodes}
        # model.addConstrs(model._p[1, dd, ll] >= model._p0_lowerbound_x0y1[(dd,ll)] * model._x[ll] for dd in model._districts for ll in model._locker_nodes)
        # model.addConstrs(model._p[1, dd, ll] <= model._p0_upperbound_xeq1[(dd,ll)] * model._x[ll] for dd in model._districts for ll in model._locker_nodes)
        # model.addConstrs(model._p[2, dd, ll] >= model._p0_lowerbound_xeq1[(dd,ll)] * model._y[ll] for dd in model._districts for ll in model._locker_nodes)
        # model.addConstrs(model._p[2, dd, ll] <= model._p0_upperbound_xeq1[(dd,ll)] * model._y[ll] for dd in model._districts for ll in model._locker_nodes)
        # model.addConstrs(model._p[1, dd, ll] >= model._p0[dd] - model._p0_upperbound_xeq0[(dd,ll)] * (1-model._x[ll]) for dd in model._districts for ll in model._locker_nodes)
        # model.addConstrs(model._p[1, dd, ll] <= model._p0[dd] - model._p0_lowerbound_xeq0[(dd,ll)] * (1-model._x[ll]) for dd in model._districts for ll in model._locker_nodes)
        # model.addConstrs(model._p[2, dd, ll] >= model._p0[dd] - model._p0_upperbound_xeq0[(dd,ll)] * (1-model._y[ll]) for dd in model._districts for ll in model._locker_nodes)
        # model.addConstrs(model._p[2, dd, ll] <= model._p0[dd] - model._p0_lowerbound_xeq0[(dd,ll)] * (1-model._y[ll]) for dd in model._districts for ll in model._locker_nodes)

        # Haase Constraints
        # model.addConstr(grb.quicksum(cost_ll * model._x[ll] for ll, cost_ll in model._locker_cost.items()) <= model._budgets[0], "budget1")
        # model.addConstr(grb.quicksum(cost_ll * model._y[ll] for ll, cost_ll in model._locker_cost.items()) <= model._budgets[1], "budget2")
        
        # model.addConstrs(model._p0[dd] + grb.quicksum(model._p[1, dd, ll] + model._p[2, dd, ll] for ll in model._locker_nodes) == 1  for dd in model._districts)
        # model.addConstrs(model._p[1, dd, ll] <= model._upperbound[(1,dd,ll)] * model._x[ll] for dd in model._districts for ll in model._locker_nodes)
        # model.addConstrs(model._p[2, dd, ll] <= model._upperbound[(2,dd,ll)] * model._y[ll] for dd in model._districts for ll in model._locker_nodes)

        # model.addConstrs(model._p[pp, dd, ll1] <= model._utilities[dd,ll1]/model._utilities[dd,ll2]*model._p[1, dd, ll2] + (1-model._x[ll2]) for pp in (1,2) for dd in model._districts for ll1 in model._locker_nodes for ll2 in model._locker_nodes if ll1!=ll2)  
        # model.addConstrs(model._p[pp, dd, ll1] <= model._utilities[dd,ll1]/model._utilities[dd,ll2]*model._p[2, dd, ll2] + (1-model._y[ll2]) for pp in (1,2) for dd in model._districts for ll1 in model._locker_nodes for ll2 in model._locker_nodes if ll1!=ll2)  
        # model.addConstrs(model._p[1, dd, ll] >= model._utilities[dd,ll]*(model._p0[dd]+1*(model._x[ll]-1)) for dd in model._districts for ll in model._locker_nodes)  
        # model.addConstrs(model._p[2, dd, ll] >= model._utilities[dd,ll]*(model._p0[dd]+1*(model._y[ll]-1)) for dd in model._districts for ll in model._locker_nodes)  
        # model.addConstrs(model._p[pp, dd, ll] <= model._utilities[dd,ll]*model._p0[dd] for pp in (1,2) for dd in model._districts for ll in model._locker_nodes)  

        ### Quadratic inspired constraints
        # model._t = model.addVars(model._districts, lb=-np.inf, ub=0, name="t")
        # model._z = model.addVars(model._districts, lb=0, name="z")
        # for dd in districts:
        #     model.addConstr(model._z[dd] == 1 + 
        #                     grb.quicksum(utilities[dd, ll] * model._x[ll] for ll in locker_nodes) + 
        #                     grb.quicksum(utilities[dd, ll] * model._y[ll] for ll in locker_nodes), f"z_{dd}")
        #     model.addConstr( - model._z[dd] * model._t[dd] >= 1, f"rsoc_{dd}") 

        # model.setObjective(grb.quicksum(model._population_per_node[dd] * (1 + model._t[dd]) for dd in districts), grb.GRB.MAXIMIZE)

        ### Altekin-Dasci-Karatas Second MILP
        # model._p0 = model.addVars(model._districts, vtype=grb.GRB.CONTINUOUS, lb=0.0, name="p0")
        # model._x = model.addVars(model._locker_nodes, vtype=grb.GRB.BINARY, name="y")
        # model._y = model.addVars(model._locker_nodes, vtype=grb.GRB.BINARY, name="y")
        # model._p = model.addVars([1,2], model._districts, model._locker_nodes, vtype=grb.GRB.CONTINUOUS, lb=0.0, name="p")

        # model.addConstrs(grb.quicksum(model._x[ll] for ll in model._locker_nodes) <= model._budgets[0])
        # model.addConstrs(grb.quicksum(model._y[ll] for ll in model._locker_nodes) <= model._budgets[1])
        # model.addConstrs(model._p0[dd] + grb.quicksum(model._utilities[dd, ll]*(model._p[1, dd, ll] + model._p[2, dd, ll]) for ll in model._locker_nodes) == 1 for dd in model._districts)
        # model.addConstrs(model._p[1, dd, ll] <= model._upperbound[(1,dd,ll)] * model._x[ll] for dd in model._districts for ll in model._locker_nodes)
        # model.addConstrs(model._p[2, dd, ll] <= model._upperbound[(2,dd,ll)] * model._y[ll] for dd in model._districts for ll in model._locker_nodes)
        # model.addConstrs(model._p[pp, dd, ll] <= model._p0[dd] for pp in [1,2] for dd in model._districts for ll in model._locker_nodes)
        # model.addConstrs(model._p[pp, dd, ll] >= model._p0[dd] - (1-model._y[pp,ll]) for pp in [1,2] for dd in model._districts for ll in model._locker_nodes)

        # model.setObjective(grb.quicksum(model._population_per_node[dd] * model._p0[dd] for dd in model._districts), grb.GRB.MINIMIZE)


        # model.setParam('LazyConstraints', 1) #Add when using cbLazy
        # model.setParam('PreCrush', 1) #Add when using cbCut
        # model.optimize(right_hand_cuts_callback)
        # model.optimize()

        ###No lazy but all constraints
        # for xx in list(product([0, 1], repeat=len(locker_nodes))):
        #     for yy in list(product([0, 1], repeat=len(locker_nodes))):
        #         if sum(xx) == budgets[0] and sum(yy) == budgets[1]:
        #             model.addConstr(grb.quicksum(model._population_per_node[dd] * model._p[1, dd, ll] for dd in model._districts for ll in model._locker_nodes) >= 
        #                     grb.quicksum(model._population_per_node[dd] * (MNL_function(model, xx, yy, dd, ll) + grb.quicksum(Deriv_MNL_function(model, xx, yy, dd, ll, loc_idx)*(model._y[loc_idx] - 1*(loc_idx in yy)) for loc_idx in model._locker_nodes)) 
        #                                 for dd in model._districts for ll in model._locker_nodes) )
        #             model.addConstr(grb.quicksum(model._population_per_node[dd] * model._p[2, dd, ll] for dd in model._districts for ll in model._locker_nodes) >= 
        #                     grb.quicksum(model._population_per_node[dd] * (MNL_function(model, yy, xx, dd, ll) + grb.quicksum(Deriv_MNL_function(model, yy, xx, dd, ll, loc_idx)*(model._x[loc_idx] - 1*(loc_idx in xx)) for loc_idx in model._locker_nodes)) 
        #                                 for dd in model._districts for ll in model._locker_nodes) )
        # model.optimize()
        ###        
        

        status = model.Status
        if status == grb.GRB.Status.OPTIMAL:
            pass
        elif status == grb.GRB.Status.INFEASIBLE:
            print("Model is infeasible")
            return ((),()),(np.NaN, np.NaN)
        elif status == grb.GRB.Status.UNBOUNDED:
            print("Model is unbounded")
        else:
            print(f"Optimization ended with status {status}")

        # social_payoff = 0.0
        x_lockers = [ll for ll in model._locker_nodes if model._x[ll].X > 0.5]
        y_lockers = [ll for ll in model._locker_nodes if model._y[ll].X > 0.5]

        payoff1 = payoff_per_location_decision(x_lockers, [y_lockers], model._population_per_node, model._utilities)
        payoff2 = payoff_per_location_decision(y_lockers, [x_lockers], model._population_per_node, model._utilities)

        if check_actual_equilibrium:
            location_actions = {player : list(combinations(locker_nodes, budgets[player])) for player in range(len(budgets))}
            if all(payoff1 >= payoff_per_location_decision(first_player_action, [y_lockers], model._population_per_node, model._utilities) for first_player_action in location_actions[0]) and all(payoff2 >= payoff_per_location_decision(second_player_action, [x_lockers], model._population_per_node, model._utilities) for second_player_action in location_actions[1]):
                print("Actual equilibrium found")
            else:
                print("Actual equilibrium NOT found")
        
    return (x_lockers, y_lockers), (payoff1, payoff2)

def find_best_equilibrium_and_stability_by_RSOC(population_per_node, utilities, locker_cost, budgets, method='reoptimize'):
    
    assert method in ['reoptimize', 'lazy'], "Method not recognized"

    districts = population_per_node.keys()
    locker_nodes = locker_cost.keys()
    equilibrium_found = False
    max_iteration = 100
    iteration = 0

    print("Looking for the best social payoff equilibrium")

    with grb.Env() as env, grb.Model(env=env) as model:

        model.setParam('MIPGap', 0)
        model.setParam('LazyConstraints', 0)
        model.setParam('OutputFlag', 0)  # Set silent mode
    
        # Variables
        model._x = model.addVars(locker_nodes, vtype=grb.GRB.BINARY, name="x")
        model._y = model.addVars(locker_nodes, vtype=grb.GRB.BINARY, name="y")
        model._t = model.addVars(districts, lb=-np.inf, ub=0, name="t")
        model._z = model.addVars(districts, lb=0, name="z")
        # model._forbidden_actions = []
        model._locker_nodes = locker_nodes
        model._locker_cost = locker_cost
        model._population_per_node = population_per_node
        model._utilities = utilities
        model._budgets = budgets

        model.setObjective(grb.quicksum(model._population_per_node[dd] * (1 + model._t[dd]) for dd in districts), grb.GRB.MAXIMIZE)

        # Constraints
        model.addConstr(grb.quicksum(cost_ll * model._x[ll] for ll, cost_ll in model._locker_cost.items()) <= model._budgets[0], "budget1")
        model.addConstr(grb.quicksum(cost_ll * model._y[ll] for ll, cost_ll in model._locker_cost.items()) <= model._budgets[1], "budget2")
        
        
        for dd in districts:
            model.addConstr(model._z[dd] == 1 + 
                            grb.quicksum(model._utilities[dd, ll] * model._x[ll] for ll in model._locker_nodes) + 
                            grb.quicksum(model._utilities[dd, ll] * model._y[ll] for ll in model._locker_nodes), f"z_{dd}")
            model.addConstr( - model._z[dd] * model._t[dd] >= 1, f"rsoc_{dd}") 

        if method == 'reoptimize':
            forbidden_actions = []	
            while not equilibrium_found and iteration < max_iteration:

                if len(forbidden_actions) > 0:
                    model.addConstr(grb.quicksum(model._x[ll] for ll in locker_nodes if ll not in forbidden_actions[-1][0]) + 
                                    grb.quicksum(1-model._x[ll] for ll in forbidden_actions[-1][0]) + 
                                    grb.quicksum(model._y[ll] for ll in locker_nodes if ll not in forbidden_actions[-1][1]) + 
                                    grb.quicksum(1-model._y[ll] for ll in forbidden_actions[-1][1]) >= 1, 
                                    f"forbidden_{forbidden_actions[-1]}")
                
                model.optimize()
                social_payoff = model.objVal
                x_lockers = [ll for ll in locker_nodes if model._x[ll].X > 0.5]
                y_lockers = [ll for ll in locker_nodes if model._y[ll].X > 0.5]

                _, x_iter_payoff = BestResponse_RSOC_optimization(y_lockers, population_per_node, utilities, locker_cost, budgets[0])
                _, y_iter_payoff = BestResponse_RSOC_optimization(x_lockers, population_per_node, utilities, locker_cost, budgets[1])

                x_social_payoff = payoff_per_location_decision(x_lockers, [y_lockers], model._population_per_node, model._utilities)
                y_social_payoff = payoff_per_location_decision(y_lockers, [x_lockers], model._population_per_node, model._utilities)

                if iteration % 1 == 0:
                    print(f"Iteration {iteration} - Social payoff: {social_payoff}, Sum of payoffs: {x_iter_payoff+y_iter_payoff}, x payoff: {x_iter_payoff}, y payoff: {y_iter_payoff}")
                    print(f"X lockers: {x_lockers}, Y lockers: {y_lockers}")
                if x_social_payoff > x_iter_payoff - 1e-4 and y_social_payoff > y_iter_payoff - 1e-4:
                    print("Best equilibrium found")
                    # print(f"Social payoff: {social_payoff}, Sum of payoffs: {x_iter_payoff+y_iter_payoff}, x payoff: {x_iter_payoff}, y payoff: {y_iter_payoff}")
                    equilibrium_found = True
                    return (x_lockers, y_lockers), (x_iter_payoff, y_iter_payoff)
                else:
                    print(f"Adding forbidden action: {x_lockers, y_lockers}")
                    forbidden_actions.append([x_lockers, y_lockers])
                    iteration += 1

            return (x_lockers, y_lockers), (round(x_iter_payoff, 3), round(y_iter_payoff, 3))
        
        elif method == 'lazy':

            model.setParam('LazyConstraints', 1)
            model.setParam('OutputFlag', 1) 
            model.optimize(nogood_callback)
    
            status = model.Status
            if status == grb.GRB.Status.OPTIMAL:
                pass
            elif status == grb.GRB.Status.INFEASIBLE:
                print("Model is infeasible")
            elif status == grb.GRB.Status.UNBOUNDED:
                print("Model is unbounded")
            else:
                print(f"Optimization ended with status {status}")

            # social_payoff = 0.0
            x_lockers = [ll for ll in locker_nodes if model._x[ll].X > 0.5]
            y_lockers = [ll for ll in locker_nodes if model._y[ll].X > 0.5]

            social_payoff = model.objVal

            _, x_iter_payoff = BestResponse_RSOC_optimization(y_lockers, population_per_node, utilities, locker_cost, budgets[0])
            _, y_iter_payoff = BestResponse_RSOC_optimization(x_lockers, population_per_node, utilities, locker_cost, budgets[1])

            print(f"Social payoff: {social_payoff}, Sum of payoffs: {x_iter_payoff+y_iter_payoff}, x payoff: {x_iter_payoff}, y payoff: {y_iter_payoff}")
            return (x_lockers, y_lockers), (x_iter_payoff, y_iter_payoff)

def plot_simulation_state(graph, current_actions, utilities, population_per_node, social_optimum_strategies=None, filename=None, gpd_shapes=False, show=True):
    
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    colors = ['red', 'blue', 'green', 'orange', 'brown', 'magenta', 'olive', 'cyan', 
                'purple',  'pink', 'green', 'lime', 'navy',
                'teal', 'maroon', 'aqua', 'fuchsia']
    
    number_of_players = len(current_actions)
    located_lockers = list(chain(*current_actions))

    probability_district_served_by_a_locker = [sum(utilities[district, locker] for locker in located_lockers) / (1 + sum(utilities[district, locker] for locker in located_lockers)) for district in graph.nodes]
    # populations = [round(float(data.get('node_population'))) * 2 for node, data in graph.nodes(data=True)]
    populations = [population_per_node[node] for node in graph.nodes]

    colors_per_node_with_players = {node: [colors[player] for player in range(number_of_players) if node in current_actions[player]] for node in graph.nodes() if node in located_lockers}
    
    ### Plot the graph
    fig, ax = plt.subplots()
    if gpd_shapes:
        import geopandas as gpd
        from matplotlib.colors import ListedColormap
        #os.path.dirname(current_folder)
        print("Printing district shapes")
        selected_districts = ["Binnenstad", "Witte Dame", "Bergen"]  # If you want to select a specific district, write the name here
        # custom_cmap = ListedColormap(["red", "green", "blue"])
        nbh_shapes = gpd.read_file(os.path.dirname(current_folder)+"/CompLLG_data/GeoNeighborhoods.geojson").rename(columns={"buurtnaam": "District"})
        nbh_shapes = nbh_shapes[nbh_shapes['District'].isin(selected_districts)]
        nbh_shapes.plot(ax=ax, cmap=ListedColormap(colors[:len(selected_districts)]), alpha=0.2)
        # nbh_shapes.plot(ax=ax, cmap='Paired', alpha=0.2)
    if len(current_actions) == 0 or sum(len(actions) for actions in current_actions) == 0:
        ox.plot_graph(graph, 
                        ax=ax,
                        edge_color='grey', 
                        bgcolor='white', 
                        node_color='black',
                        show=False, 
                        close=False)
    else:
        ox.plot_graph(graph,  
                        ax=ax,
                        edge_color='grey', 
                        bgcolor='white', 
                        node_color='black',
                        node_size=populations,
                        node_alpha=probability_district_served_by_a_locker,
                        show=False, 
                        close=False)
        
    limits_x_axis = ax.get_xlim()[1] - ax.get_xlim()[0]
    limits_y_axis = ax.get_ylim()[1] - ax.get_ylim()[0]
    rectangle_side_size = 2.5e-2

    x_side = rectangle_side_size * (ax.get_xlim()[1] - ax.get_xlim()[0]) * 1.15
    y_side = rectangle_side_size * (ax.get_ylim()[1] - ax.get_ylim()[0])
    for locker in [locker for locker, data in graph.nodes(data=True) if data.get('locker_possible') == 'locker' and locker not in located_lockers]:
        locker_position = (graph.nodes(data=True)[locker]['x'] - x_side/2, graph.nodes(data=True)[locker]['y'] - y_side/2)
        locker_rectangle = Rectangle(locker_position, width=x_side, height=y_side, facecolor='white', edgecolor='black')
        ax.add_patch(locker_rectangle)

    # Add a star for each locker in the social optimum
    if social_optimum_strategies is not None:
        import matplotlib.patheffects as path_effects
        SO_lockers = list(chain(*social_optimum_strategies))
        number_of_lockers_for_node_in_SO = Counter(SO_lockers)
        for locker in number_of_lockers_for_node_in_SO.keys():
            text = ax.text(graph.nodes(data=True)[locker]['x'], graph.nodes(data=True)[locker]['y']+y_side, '★' * number_of_lockers_for_node_in_SO[locker], fontsize=10, ha='center', va='center', color='gold', weight="bold")
            text.set_path_effects([
            path_effects.Stroke(linewidth=1, foreground='black'),
            path_effects.Normal()])

    # ### Add a star for each locker in the social optimum
    # if social_optimum_strategies is not None:
    #     import matplotlib.patheffects as path_effects
    #     for locker in list(chain(*social_optimum_strategies)):
    #         if locker in social_optimum_strategies[0] and locker in social_optimum_strategies[1]:
    #             # text = ax.text(graph.nodes(data=True)[locker]['x'], graph.nodes(data=True)[locker]['y']+y_side, '★★', fontsize=10, ha='center', va='center', color='gold', weight="bold")
    #             text = ax.text(graph.nodes(data=True)[locker]['x']-x_side/2, graph.nodes(data=True)[locker]['y']+y_side, '★', fontsize=10, ha='center', va='center', color='red', weight="bold")
    #             text.set_path_effects([
    #                     path_effects.Stroke(linewidth=1, foreground='black'),
    #                     path_effects.Normal()])
    #             text = ax.text(graph.nodes(data=True)[locker]['x']+x_side/2, graph.nodes(data=True)[locker]['y']+y_side, '★', fontsize=10, ha='center', va='center', color='blue', weight="bold")
    #         elif locker in social_optimum_strategies[0]:
    #             text = ax.text(graph.nodes(data=True)[locker]['x'], graph.nodes(data=True)[locker]['y']+y_side, '★', fontsize=10, ha='center', va='center', color='red', weight="bold")
    #         elif locker in social_optimum_strategies[1]:
    #             text = ax.text(graph.nodes(data=True)[locker]['x'], graph.nodes(data=True)[locker]['y']+y_side, '★', fontsize=10, ha='center', va='center', color='blue', weight="bold")
    #         text.set_path_effects([
    #         path_effects.Stroke(linewidth=1, foreground='black'),
    #         path_effects.Normal()])

    rectangle_increase = 1.5e-2
    if number_of_players == 1 or number_of_players > 2:
        for node, players in colors_per_node_with_players.items():
            for idx_player in range(len(players)-1,-1,-1):
                rectangle_side_size = 2.5e-2 + rectangle_increase * idx_player
                x_side = rectangle_side_size * limits_x_axis * 1.15
                y_side = rectangle_side_size * limits_y_axis
                locker_position = (graph.nodes(data=True)[node]['x'], graph.nodes(data=True)[node]['y'])  
                locker_position = (locker_position[0] - x_side/2, locker_position[1]- y_side/2)
                print(f"idx player: {idx_player}, Rectangle side size: {rectangle_side_size}, locker position: {locker_position}, x_side: {x_side}, y_side: {y_side}, facecolor: {players[idx_player]}")	
                locker_rectangle = Rectangle(locker_position, width=x_side, height=y_side, facecolor=players[idx_player], edgecolor='black', zorder = number_of_players-idx_player)
                ax.add_patch(locker_rectangle)
    if number_of_players == 2:
        for node, players in colors_per_node_with_players.items():
            if len(players) == 1:
                locker_position = (graph.nodes(data=True)[node]['x'] - x_side/2, graph.nodes(data=True)[node]['y'] - y_side/2)  
                locker_rectangle = Rectangle(locker_position, width=x_side, height=y_side, facecolor=players[0], edgecolor='black')
                ax.add_patch(locker_rectangle)
            if len(players) == 2:
                rectangle_side_size = 2.5e-2
                locker_position = (graph.nodes(data=True)[node]['x'] - x_side/2, graph.nodes(data=True)[node]['y'] - y_side/2)                  
                # Define the vertices of the two triangles
                triangle1 = [(locker_position[0], locker_position[1]), 
                            (locker_position[0] + x_side, locker_position[1]), 
                            (locker_position[0], locker_position[1] + y_side)]
                triangle2 = [(locker_position[0] + x_side, locker_position[1]), 
                            (locker_position[0] + x_side, locker_position[1] + y_side), 
                            (locker_position[0], locker_position[1] + y_side)]
                # # Create the two triangles with different colors
                # triangle1_patch = Polygon(triangle1, facecolor=players[0], edgecolor='black')
                # triangle2_patch = Polygon(triangle2, facecolor=players[1], edgecolor='black')
                # ax.add_patch(triangle1_patch)
                # ax.add_patch(triangle2_patch)

                # Create the two triangles with different colors
                triangle1_patch = Polygon(triangle1, facecolor=players[0], edgecolor='none')
                triangle2_patch = Polygon(triangle2, facecolor=players[1], edgecolor='none')

                # Add the triangles to the plot
                ax.add_patch(triangle1_patch)
                ax.add_patch(triangle2_patch)

                # Add black edges to the sides
                ax.plot([locker_position[0], locker_position[0] + x_side], [locker_position[1], locker_position[1]], color='black', lw=0.7)
                ax.plot([locker_position[0], locker_position[0]], [locker_position[1], locker_position[1] + y_side], color='black', lw=0.7)
                ax.plot([locker_position[0] + x_side, locker_position[0] + x_side], [locker_position[1], locker_position[1] + y_side], color='black', lw=0.7)
                ax.plot([locker_position[0], locker_position[0] + x_side], [locker_position[1] + y_side, locker_position[1] + y_side], color='black', lw=0.7)
        print()

    if filename:
        plt.savefig(filename, bbox_inches='tight')
        print(f"Color and size map created at {filename}")
    if show:
        plt.show()
    # plt.close()

def game_solver_by_RSOC(graph, pickle_path, location_actions, population_per_node, utilities, locker_cost, number_of_lockers_per_player, max_iterations, find_one_or_return_all, pictures_filename, analysis_filename):
    
    total_population = sum(population_per_node.values())
    info1_str = f"""
Maximal distance between locker and district: {max(all_pairs_distances[(district, locker)] for locker in nodes_with_locker_locations for district in population_per_node.keys())} 
Lockers per player: {number_of_lockers_per_player}
Total population: {total_population}
Number of intersections: {len(graph.nodes)} 
Number of locker locations: {len(nodes_with_locker_locations)}
Some utilities: {random.sample(list(utilities.items()), 5)}
Random seed: {RANDOM_SEED}
"""
        
    if pickle_path is not None and os.path.exists(pickle_path) and find_one_or_return_all == "all":
        print("Loading NEs from pickle")
        with open(pickle_path, "rb") as pickle_file:
            equilibria_actions_and_payoffs, number_of_lockers_per_player, utilities, population_per_node = joblib.load(pickle_file)
        computation_time = 0.0
    else:
        experiment_start_time = current_time()
        equilibria_actions_and_payoffs = find_equilibria_by_RSOC_for_all_initial_combinations(location_actions, population_per_node, utilities, max_iterations, find_one_or_return_all)
        computation_time = round((current_time() - experiment_start_time)/60.0, 2)
        pickle_path = os.path.dirname(analysis_filename) + f"/NEs_pickle_{solution_method}_{number_of_lockers_per_player[0]}_{number_of_lockers_per_player[1]}_beta_{beta}_alpha_mean_{alpha_mean}.pkl"
        with open(pickle_path, "wb") as pickle_file:
            joblib.dump((equilibria_actions_and_payoffs, number_of_lockers_per_player, utilities, population_per_node), pickle_file, compress=8)
    SO_action, SO_payoffs = find_social_optimum_by_RSOC(population_per_node, utilities, locker_cost, number_of_lockers_per_player)
    SO_payoff = sum(SO_payoffs)
    info1_str += f"Social optimum: {SO_action} with payoff {SO_payoff} ({round(SO_payoff/total_population*100,2)}% of the whole population)\n"
    if find_one_or_return_all == 'one':
        price_of_anarchy = SO_payoff / sum(payoff_per_location_decision(equilibria_actions_and_payoffs[0][0][player], [equilibria_actions_and_payoffs[0][0][1-player]], population_per_node, utilities) for player in [0,1])
        print(f"Price of Anarchy: {price_of_anarchy}")
        best_equilibrium_actions, best_equilibrium_payoffs  = find_best_equilibrium_and_stability_by_RSOC(population_per_node, utilities, locker_cost, number_of_lockers_per_player, method='reoptimize') 
        price_of_stability = "DIV_BY_0" if sum(best_equilibrium_payoffs) == 0.0 else SO_payoff / sum(best_equilibrium_payoffs)
        info_str = info1_str + f"""
Only one equilibrium has been searched
Computational time: {computation_time} minutes
Best social equilibrium is {best_equilibrium_actions} with payoffs {best_equilibrium_payoffs} whose sum is {round(sum(best_equilibrium_payoffs),2)} ({round(sum(best_equilibrium_payoffs)/total_population*100,2)}% of population) with Price of Stability: {price_of_stability}
Coincident lockers: {len(set(best_equilibrium_actions[0]).intersection(set(best_equilibrium_actions[1])))}
Distances between lockers: {[all_pairs_distances[(locker1, locker2)] for locker1, locker2 in combinations(best_equilibrium_actions[0]+best_equilibrium_actions[1], 2)]}"""
    elif find_one_or_return_all == 'all':  
        smallest_overall_payoff_equilibrium, largest_payoff_equilibrium = min(sum(x[1]) for x in equilibria_actions_and_payoffs), max(sum(x[1]) for x in equilibria_actions_and_payoffs)
        price_of_anarchy = "DIV_BY_0" if smallest_overall_payoff_equilibrium == 0.0 else SO_payoff / smallest_overall_payoff_equilibrium
        price_of_stability = "DIV_BY_0" if largest_payoff_equilibrium == 0.0 else SO_payoff / largest_payoff_equilibrium
        info_str = info1_str + f"""
{len(equilibria_actions_and_payoffs)} different equilibria have been found
Computational time: {computation_time} minutes
Smallest equilibrium payoff: {round(smallest_overall_payoff_equilibrium,2)} ({round(smallest_overall_payoff_equilibrium/total_population*100,2)}% of population) with Price of Anarchy: {price_of_anarchy}
Largest equilibrium payoff: {round(largest_payoff_equilibrium,2)} ({round(largest_payoff_equilibrium/total_population*100,2)}% of population) with Price of Stability: {price_of_stability}"""
        for idx, (equilibrium, payoff) in enumerate(equilibria_actions_and_payoffs):
            info_str += f"\nEquilibrium {idx}: {equilibrium} with payoffs {payoff} ({round(sum(payoff)/total_population*100, 2)}% of the total population)"
            info_str += f"\nCoincident lockers: {len(set(equilibrium[0]).intersection(set(equilibrium[1])))}"
            info_str += f"\nDistances between lockers: {[all_pairs_distances[(locker1, locker2)] for locker1, locker2 in combinations(equilibrium[0]+equilibrium[1], 2)]}\n"
    else:
        print("Find one or return all not recognized")
    print(info_str)
    with open(analysis_filename, "w") as text_file:
        text_file.write(info_str)

    if 'best_equilibrium_actions' in locals():
        print("Plotting the best equilibrium")
        analysis_state = best_equilibrium_actions
    else:
        analysis_state = random.choice(list(equilibria_actions_and_payoffs))[0]

    if not 'SO_action' in locals(): 
        SO_action = None
    plot_simulation_state(graph, analysis_state, utilities, population_per_node, social_optimum_strategies=SO_action, filename=pictures_filename, gpd_shapes=False , show=False)

def game_solver_by_enumeration(graph, pickle_path, location_actions, all_pairs_distances, population_per_node, utilities, number_of_lockers_per_player, find_one_or_return_all, pictures_filename, analysis_filename):
    if pickle_path is not None and os.path.exists(pickle_path):
        print("Loading NEs from pickle")
        analysis_folder = os.path.dirname(pickle_path)
        pictures_folder = os.path.dirname(pickle_path).replace("Analysis_Experiment", "Pictures_Experiment")
        print(f"Pictures_folder is {pictures_folder}\nAnalysis_folder is {analysis_folder}")
    else:
        pickle_path = analysis_filename.replace("analysis", "pkl").replace("txt","pkl")#analysis_folder + f"/NEs_pickle_{solution_method}_{number_of_lockers_per_player[0]}_{number_of_lockers_per_player[1]}_beta_{beta}_alpha_mean_{alpha_mean}.pkl"
    # pictures_filename = pictures_folder + f"/NEs_analysis_{solution_method}_beta_{beta}_alpha_mean_{alpha_mean}.pdf"
    # analysis_filename = analysis_folder + f"/NEs_analysis_{solution_method}_beta_{beta}_alpha_mean_{alpha_mean}.txt"
    total_population = sum(population_per_node.values())
    info1_str = f"""
Alpha mean: {find_float_after_word(analysis_filename, "alpha_mean_")}, Beta: {find_float_after_word(analysis_filename, "beta_")}
Number of lockers per player: {number_of_lockers_per_player}
Random seed: {RANDOM_SEED}
Maximal distance between locker and district: {max(all_pairs_distances[(district, locker)] for locker in nodes_with_locker_locations for district in population_per_node.keys())} 
Number of intersections: {len(graph.nodes)} 
Number of locker locations: {len(nodes_with_locker_locations)}
Some utilities: {random.sample(list(utilities.items()), 5)}
"""
    if os.path.exists(pickle_path) and find_one_or_return_all == "all":
        print("Loading NEs from pickle")
        with open(pickle_path, "rb") as pickle_file:
            equilibria_actions_and_payoffs, number_of_lockers_per_player, population_per_node, utilities = joblib.load(pickle_file)
        best_equilibrium_action, best_equilibrium_payoffs = max(equilibria_actions_and_payoffs, key=lambda x: sum(x[1]))
        computation_time = 0.0
    else:
        print("Enumerating actions and finding equilibria")
        ### Uncomment this
        start_enumeration_time = current_time()
        equilibria_actions_and_payoffs = find_equilibria_by_enumeration_for_two_players(location_actions, population_per_node, utilities, find_one_or_return_all)
        best_equilibrium_action, best_equilibrium_payoffs = max(equilibria_actions_and_payoffs, key=lambda x: sum(x[1]))
        # best_equilibrium_action, best_equilibrium_payoffs = Equilibrium_PoA_PoS_linearized_model(population_per_node, utilities, locker_cost, number_of_lockers_per_player, PoA_or_PoS="PoA")
        # equilibria_actions_and_payoffs = [(best_equilibrium_action, best_equilibrium_payoffs)]
        computation_time = round((current_time() - start_enumeration_time)/60.0, 2)
        with open(pickle_path, "wb") as pickle_file:
            joblib.dump((equilibria_actions_and_payoffs, number_of_lockers_per_player, population_per_node, utilities), pickle_file, compress=8)
    SO_action, SO_payoffs = find_social_optimum_by_RSOC(population_per_node, utilities, locker_cost, number_of_lockers_per_player)
    SO_payoff = sum(SO_payoffs)
    # print(f"Best equilibrium on linearized model: {best_equilibrium_action} with payoffs {best_equilibrium_payoffs}")
    info1_str += f"Social optimum: {SO_action} with payoff {round(SO_payoff,4)} ({round(SO_payoff/total_population*100,2)}% of population)\n"
    if find_one_or_return_all == 'one':
        price_of_anarchy = SO_payoff / sum(payoff_per_location_decision(equilibria_actions_and_payoffs[0][0][player], equilibria_actions_and_payoffs[0][0][1-player], population_per_node, utilities) for player in [0,1])
        print(f"Price of Anarchy: {price_of_anarchy}")
    elif find_one_or_return_all == 'all':  
        smallest_overall_payoff_equilibrium, largest_payoff_equilibrium = min(sum(x[1]) for x in equilibria_actions_and_payoffs), max(sum(x[1]) for x in equilibria_actions_and_payoffs)
        price_of_anarchy = SO_payoff / smallest_overall_payoff_equilibrium if smallest_overall_payoff_equilibrium != 0.0 else "DIV_BY_0" 
        if SO_payoff == 0.0:
            price_of_stability = "DIV_BY_0"
        elif Counter(best_equilibrium_action[0] + best_equilibrium_action[1])==Counter(SO_action[0] + SO_action[1]):
            price_of_stability = "Coincident"
            # print("Best equilibrium is coincident with the social optimum")
        else:
            price_of_stability = largest_payoff_equilibrium / SO_payoff
        info_str = info1_str + f"""
{len(equilibria_actions_and_payoffs)} different equilibria have been found
Computational time: {computation_time} minutes
Price of Anarchy: {price_of_anarchy}
Price of Stability: {price_of_stability}"""
        for idx, (equilibrium, payoff) in enumerate(equilibria_actions_and_payoffs):
            if sum(payoff) > 0:
                info_str += f"\nEquilibrium {idx}: {equilibrium} with payoff {payoff}  (sum is {round(sum(payoff), 2)}, i.e., {round(sum(payoff)/total_population*100, 2)}% of the total population, and corresponding PoA is {round(SO_payoff/sum(payoff),5)}, leading to a loss wrt the social optimum of {round((1-sum(payoff)/SO_payoff)*100,5)}%)"
                info_str += f"\nCoincident lockers: {len(set(equilibrium[0]).intersection(set(equilibrium[1])))}"
                info_str += f"\nDistances between lockers: {[all_pairs_distances[(locker1, locker2)] for locker1, locker2 in combinations(equilibrium[0]+equilibrium[1], 2)]}\n"
        print(info_str)
        with open(analysis_filename, "w") as text_file:
                text_file.write(info_str)
    else:
        print("Find one or return all not recognized")
    if len(equilibria_actions_and_payoffs) == 0:
        print("No Nash Equilibrium detected\n")
        return []
    # random_equilibrium = random.choice(equilibria_actions_and_payoffs)[0]
    if not 'SO_action' in locals(): 
        SO_action = None
    plot_simulation_state(graph, best_equilibrium_action, utilities, population_per_node, social_optimum_strategies=SO_action, filename=pictures_filename, show=False)
    largest_payoff_equilibrium = max(sum(x[1]) for x in equilibria_actions_and_payoffs)
    return largest_payoff_equilibrium/total_population, price_of_stability

def game_initializer_and_solver(graph, solution_method, location_actions, all_pairs_distances, population_per_node, alpha_mean_for_name, alpha, beta, locker_cost, number_of_lockers_per_player, max_iterations, find_one_or_return_all, pictures_folder, analysis_folder, pickle_upload=None):
    if pickle_upload:
        pickle_path = pickle_upload + f"/NEs_pkl_{solution_method}_{number_of_lockers_per_player[0]}_{number_of_lockers_per_player[1]}_beta_{beta}_alpha_mean_{alpha_mean_for_name}.pkl"
        analysis_folder = os.path.dirname(pickle_path)
        pictures_folder = os.path.dirname(pickle_path).replace("Analysis_Experiment", "Pictures_Experiment")
        pictures_filename = pictures_folder + f"/NEs_analysis_{solution_method}_{number_of_lockers_per_player[0]}_{number_of_lockers_per_player[1]}_beta_{beta}_alpha_mean_{alpha_mean_for_name}.pdf"
        analysis_filename = analysis_folder + f"/NEs_analysis_{solution_method}_{number_of_lockers_per_player[0]}_{number_of_lockers_per_player[1]}_beta_{beta}_alpha_mean_{alpha_mean_for_name}.txt"
        print(f"Uploading pickle from {pickle_path}")
        if os.path.exists(pickle_path):
            equilibria_actions_and_payoffs, number_of_lockers_per_player, population_per_node, utilities = joblib.load(pickle_path)
            print("Loaded from pickle")
        else:
            print("Pickle not found, running new")
            utilities = {(district, locker) : np.exp(alpha[district] - beta * all_pairs_distances[district, locker]) for district, locker in all_pairs_distances.keys()}
            # utilities = {(district, locker) : round(np.exp(alpha[district] - beta * all_pairs_distances[district, locker]), 1) for district, locker in all_pairs_distances.keys()}
        if solution_method == 'RSOC':
            game_solver_by_RSOC(graph, pickle_path, location_actions, population_per_node, utilities, locker_cost, number_of_lockers_per_player, max_iterations, find_one_or_return_all, pictures_filename, analysis_filename)
        elif solution_method == 'enumeration':
            covered_population, PoS = game_solver_by_enumeration(graph, pickle_path, location_actions, all_pairs_distances, population_per_node, utilities, number_of_lockers_per_player, find_one_or_return_all, pictures_filename, analysis_filename)
        else:
            print("Solution method not recognized")
    else:
        pictures_filename = pictures_folder + f"/NEs_analysis_{solution_method}_{number_of_lockers_per_player[0]}_{number_of_lockers_per_player[1]}_beta_{beta}_alpha_mean_{alpha_mean_for_name}.pdf"
        analysis_filename = analysis_folder + f"/NEs_analysis_{solution_method}_{number_of_lockers_per_player[0]}_{number_of_lockers_per_player[1]}_beta_{beta}_alpha_mean_{alpha_mean_for_name}.txt"
        pickle_path = analysis_folder + f"/NEs_pickle_{solution_method}_{number_of_lockers_per_player[0]}_{number_of_lockers_per_player[1]}_beta_{beta}_alpha_mean_{alpha_mean_for_name}.pkl"
        print("Computing utilities")
        utilities = {(district, locker) : np.exp(alpha[district] - beta * all_pairs_distances[district, locker]) for district, locker in all_pairs_distances.keys()}
        # utilities = {(district, locker) : round(np.exp(alpha[district] - beta * all_pairs_distances[district, locker]), 1) for district, locker in all_pairs_distances.keys()}
        print("utilities computed")
        if solution_method == 'RSOC':
            game_solver_by_RSOC(graph, None, location_actions, population_per_node, utilities, locker_cost, number_of_lockers_per_player, max_iterations, find_one_or_return_all, pictures_filename, analysis_filename)
        elif solution_method == 'enumeration':
            covered_population, PoS = game_solver_by_enumeration(graph, None, location_actions, all_pairs_distances, population_per_node, utilities, number_of_lockers_per_player, find_one_or_return_all, pictures_filename, analysis_filename)
        else:
            print("Solution method not recognized")
    return covered_population, PoS

def plot_and_info_from_pickle(graph, pickle_path, utilities, population_per_node, pictures_filename):
    if os.path.exists(pickle_path):
        results =joblib.load(pickle_path)
        with open(pickle_path, "rb") as pickle_file:
            # equilibria_actions_and_payoffs, number_of_lockers_per_player, utilities, population_per_node = joblib.load(pickle_file)
            results = joblib.load(pickle_file)
            if len(results) == 4:
                equilibria_actions_and_payoffs, _, utilities, population_per_node = results
            if len(results) == 2:
                equilibria_actions_and_payoffs, _ = results
        best_equilibrium, _ = max(equilibria_actions_and_payoffs, key=lambda x: sum(x[1]))
        plot_simulation_state(graph, best_equilibrium, utilities, population_per_node, filename=pictures_filename, gpd_shapes=False , show=False)
    else:
        print("Pickle not found")

def simulation_for_all_parameters(graph, solution_method, location_actions, all_pairs_distances, population_per_node, alpha_means, alpha_stds, betas, locker_cost, number_of_lockers_per_player, max_iterations, find_one_or_return_all, pictures_folder, analysis_folder, folder_to_work=None):
    alpha_ref = {district : np.random.normal(loc = 0, scale = 1.0) for district in population_per_node} #{district : 3 for district in graph.nodes}
    results_cover, results_PoS = dict(), dict()
    for beta in betas:
        for alpha_mean in alpha_means:
            for alpha_std in alpha_stds:
                alpha = {district : alpha_mean + alpha_ref[district] * alpha_std for district in population_per_node} #{district : 3 for district in graph.nodes}
                print(f"Alpha mean: {alpha_mean}, Beta: {beta}")
                results_cover[(alpha_mean, beta)], results_PoS[(alpha_mean, beta)] = game_initializer_and_solver(graph, solution_method, location_actions, all_pairs_distances, population_per_node, alpha_mean, alpha, beta, locker_cost, number_of_lockers_per_player, max_iterations, find_one_or_return_all, pictures_folder, analysis_folder, pickle_upload=folder_to_work)

    basic_naming = f"NEs_analysis_{solution_method}_{number_of_lockers_per_player[0]}_{number_of_lockers_per_player[1]}"
    if sum(number_of_lockers_per_player) > 0:
        with open(analysis_folder + f"/{basic_naming}_LaTeX_table.txt", "w") as table_file:
            table_file.write(LaTeX_table_generator(results_cover, results_PoS))
    LaTeX_subfigures = None

def LaTeX_table_generator(results_cover, results_PoS):

    alphas = sorted(set([alpha for alpha, _ in results_cover.keys()]))
    betas = sorted(set([beta for _, beta in results_cover.keys()]))

    table_str = f"""
\\begin{{table}}[h]
\centering
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{cl|{'c'*len(alphas)*2}|}}
\\cline{{3-{2+len(alphas)*2}}} 
\multicolumn{{2}}{{c|}}{{\multirow{{2}}{{*}}{{}}}} &
\multicolumn{{{len(alphas)*2}}}{{c|}}{{$\\alpha$}} \\\\ \\cline{{3-{2+len(alphas)*2}}} 
\multicolumn{{2}}{{c|}}{{}} &
"""
    for alpha in alphas[:-1]:
        table_str += f"\multicolumn{{2}}{{c|}}{{{alpha}}} & \n"
    table_str += f"""\multicolumn{{2}}{{c|}}{{{alphas[-1]}}} \\\\ \\hline
\multicolumn{{1}}{{|c|}}{{\multirow{{{len(betas)}}}{{*}}{{$\\beta$}}}} &
"""
    
    for beta in betas:
        if beta == betas[0]:
            table_str += f" {beta} & \n"
        else:
            table_str += f"\multicolumn{{1}}{{|l|}}{{}} & \n {beta} & \n"
        for alpha in alphas[:-1]:
            table_str += f"\multicolumn{{1}}{{c|}}{{{round(results_cover[(alpha, beta)]*100, 2)}\\%}} &\n\multicolumn{{1}}{{c|}}{{{round((1-results_PoS[(alpha, beta)])*100, 2) if results_PoS[(alpha, beta)] != 'Coincident' else '-'}\\%}} & \n"
        table_str += f"\multicolumn{{1}}{{c|}}{{{round(results_cover[(alphas[-1], beta)]*100, 2)}\\%}} &\n{round((1-results_PoS[(alphas[-1], beta)])*100, 2) if results_PoS[(alphas[-1], beta)] != 'Coincident' else '-'}\\% \\\\ "
        if beta == betas[-1]:
            table_str +="\\hline \n"
        else:
            table_str +=f"\\cline{{2-{2+len(alphas)*2}}} \n"

    table_str += f"""
\end{{tabular}}%
}}
\caption{{Use rate (first) and social optimal improvement with respect to the equilibrium (second) for proposed $\\alpha$ and $\\beta$.}}
\label{{tab:impact_alpha_beta}}
\end{{table}}
"""
    return table_str

def LaTex_subfigures_generator(basic_naming, results_cover, results_PoS):

    
    alphas = sorted(set([alpha for alpha, _ in results_cover.keys()]))
    betas = sorted(set([beta for _, beta in results_cover.keys()]))


    figure_str = "\\begin{{figure}}[h]\n\\centering\n"

    return figure_str


if __name__ == """__main__""":

    # Get the directory name of the current file
    current_file_path = os.path.abspath(__file__)
    current_folder = os.path.dirname(current_file_path)
    graph_path = os.path.dirname(current_folder) + "/CompLLG_data/eindhoven_with_districts_Binnenstad_Witte Dame_Bergen.graphml"#"/CompLLG_data/Eindhoven.graphml"#
    playing_style = 'sequential' # 'simultaneous' or 'sequential'
    solution_method = 'enumeration' # 'enumeration' or 'RSOC' or 'linearized'
    find_one_or_return_all = 'all' # 'one' or 'all'

    ### Define the parameters of the players: Players are 0, 1, ..., n_players-1
    number_of_lockers_per_player = [1,1]#{player: 2 for player in range(number_of_players)}
    max_iterations = 100
    number_of_players = len(number_of_lockers_per_player)
    alpha_means = [-1, 0, 1, 2, 3, 4]#[1]#
    alpha_stds = [0.0]
    betas = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 1.1e-2, 1.2e-2]#[3e-3]#

    today = datetime.today().strftime('%Y%m%d')
    extra_name = "ReCheckData"
    pictures_folder = os.path.dirname(current_folder) + f"/Pictures_Experiment_{extra_name}_{today}"
    analysis_folder = os.path.dirname(current_folder) + f"/Analysis_Experiment_{extra_name}_{today}"
    folder_to_work = None#os.path.dirname(current_folder)+"\\Analysis_Experiment_Enumeration_2vs2_20250303"# 

    ### Load the graph
    graph = ox.load_graphml(graph_path)
    if nx.is_directed(graph):
        graph = graph.to_undirected()

# Check whether the specified path exists or not
    if not os.path.exists(pictures_folder):
        # Create a new directory because it does not exist
        os.makedirs(pictures_folder)
        print(f"Directory {pictures_folder} created")
    else:
        print(f"Directory {pictures_folder} already exists")
    if not os.path.exists(analysis_folder):
        # Create a new directory because it does not exist
        os.makedirs(analysis_folder)
        print(f"Directory {analysis_folder} created")
    else:
        print(f"Directory {analysis_folder} already exists")


    # Find all connected components
    connected_components = list(nx.connected_components(graph))
    # Find the largest connected component
    largest_component = max(connected_components, key=len)
    # Create a subgraph containing only the largest connected component
    graph = graph.subgraph(largest_component).copy()
    all_nodes = list(graph.nodes)

    ### Get the nodes with possible lockers
    nodes_with_locker_locations = [node for node, data in graph.nodes(data=True) if data.get('locker_possible') == 'locker']
    random.shuffle(nodes_with_locker_locations)
    population_per_node = {node: round(float(data.get('node_population'))) if data.get('node_population') is not None else 0 for node, data in graph.nodes(data=True)}

    # Calculate shortest path lengths between district nodes and locker nodes
    print(f"Calculating shortest path lengths between {len(population_per_node)} district nodes and {len(nodes_with_locker_locations)} locker nodes")
    distance_pickle = graph_path[:-8]+".pkl"
    if os.path.exists(distance_pickle):
        print("Loading distances from pickle")
        with open(distance_pickle, "rb") as pickle_file:
            all_pairs_distances = joblib.load(distance_pickle)
    else:
        with tqdm_joblib(tqdm(desc="Progress", total=len(nodes_with_locker_locations))):
            all_pairs_distances = Parallel(n_jobs=N_JOBS, verbose=0)(delayed(distance_computation_for_locker)(graph, locker) for locker in nodes_with_locker_locations)   
        all_pairs_distances = dict(set().union(*all_pairs_distances))
        joblib.dump(all_pairs_distances, distance_pickle, compress = 3)
    print("Finished calculating shortest path lengths between district nodes and locker nodes")

    locker_cost = {node: 1 for node in nodes_with_locker_locations}

    ###TEMPORARY: SORT MOST PROFITABLE LOCATIONS
    # alpha = {district : 0 for district in population_per_node.keys()}
    # beta = 5e-3
    # utilities = {(district, locker) : np.exp(alpha[district] - beta * all_pairs_distances[district, locker]) for district in population_per_node.keys() for locker in nodes_with_locker_locations}
    # total_population = {locker: sum(population_per_node[district] * utilities[district, locker] for district in population_per_node.keys()) for locker in nodes_with_locker_locations}
    # sorted_lockers = sorted(total_population.items(), key=lambda x: x[1], reverse=True)
    # print(f"Most profitable lockers: {sorted_lockers}")
    # input("CODE BLOCKED")
    ### END TEMPORARY

    ### Enumerate the actions
    location_actions = {player : list(combinations(nodes_with_locker_locations, number_of_lockers_per_player[player])) for player in range(number_of_players)}
    # location_actions = {player : [] for player in range(number_of_players)}

    # if folder_to_work:
    # pkl_files = [f for f in os.listdir(folder_to_work) if '.pkl' in f]
    # for pkl_file in pkl_files:
    #     pickle_path = folder_to_work + "/" + pkl_file
    #     alpha_mean = find_first_float_after_substring(pkl_file, "alpha_mean_")
    #     alpha = {district : alpha_mean for district in population_per_node}
    #     beta = find_first_float_after_substring(pkl_file, "beta_")
    #     utilities = {(district, locker): np.exp(alpha[district] - beta * all_pairs_distances[locker][district]) for district in all_pairs_distances.keys() for locker in nodes_with_locker_locations}
    #     plot_and_info_from_pickle(graph, pickle_path, utilities, population_per_node, pictures_folder + f"/NEs_pickle_{solution_method}_beta_{beta}_alpha_mean_{alpha_mean}.pdf")

    # alpha_ref = {district : np.random.normal(loc = 0, scale = 1.0) for district in population_per_node} #{district : 3 for district in graph.nodes}
    # for beta in betas:
    #     for alpha_mean in alpha_means:
    #         for alpha_std in alpha_stds:
    #             alpha = {district : alpha_mean + alpha_ref[district] * alpha_std for district in population_per_node} #{district : 3 for district in graph.nodes}
    #             print(f"Alpha mean: {alpha_mean}, Beta: {beta}")
    #             game_initializer_and_solver(graph, solution_method, location_actions, all_pairs_distances, population_per_node, alpha, beta, locker_cost, number_of_lockers_per_player, max_iterations, find_one_or_return_all, pictures_folder, analysis_folder, pickle_upload=folder_to_work)

    simulation_for_all_parameters(graph, solution_method, location_actions, all_pairs_distances, population_per_node, alpha_means, alpha_stds, betas, locker_cost, number_of_lockers_per_player, max_iterations, find_one_or_return_all, pictures_folder, analysis_folder, folder_to_work=folder_to_work)
