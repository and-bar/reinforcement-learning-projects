import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import pandas as pd

vertical_dim = 20
horizontal_dim = 20
digit_for_representing_the_wall = 2


# Announsing terminal states
terminal_states = [(0, horizontal_dim - 1), (1, horizontal_dim - 1)]

# Creating walls in grid world  and first row and column for assigning starting policy letting free of walls
grid_world_ndarray = np.zeros([vertical_dim, horizontal_dim])
counter_walls = 0
while counter_walls <= int(vertical_dim*horizontal_dim*0.2):
    grid_world_ndarray[random.randint(0,vertical_dim-1) , random.randint(0,horizontal_dim-1)] = digit_for_representing_the_wall
    counter_walls += 1

# cleaning walls in start space and terminal spaces areas
grid_world_ndarray[0,horizontal_dim - 1] = 0
grid_world_ndarray[0,horizontal_dim - 2] = 0
grid_world_ndarray[1,horizontal_dim - 1] = 0
grid_world_ndarray[1,horizontal_dim - 2] = 0
grid_world_ndarray[vertical_dim-1,0] = 0

# Creating grid of V(S) of every state and populating it with zeros
V_of_S_for_every_state_ndarray = np.zeros([vertical_dim, horizontal_dim])

# Creating grid of rewards of every state
rewards_ndarray = np.zeros([vertical_dim, horizontal_dim])
rewards_ndarray[0,horizontal_dim - 1] = 100
rewards_ndarray[1,horizontal_dim - 1] = -1

def find_all_actions_of_the_state(tuple_state_for_searching):
    """
    Check for possible action up, left, down, right from current state
    If can make action adding tuple of coordinate of that action to the list
    of all posible actions.
    First check if action do not guide out of the grid
    Second check if action do not guide to the wall
    If action comply with this conditions add coordinates of the
    destination position of the grid to the list
    """
    list_of_tuple_actions = []
    
    # Action UP
    vertical_coord = tuple_state_for_searching[0] - 1    
    if (vertical_coord >= 0):
        if (grid_world_ndarray[vertical_coord, tuple_state_for_searching[1]] != digit_for_representing_the_wall):
            list_of_tuple_actions.append((vertical_coord, tuple_state_for_searching[1]))
    # Action RIGHT
    horizontal_coord = tuple_state_for_searching[1] + 1    
    if (horizontal_coord <= horizontal_dim - 1):
        if (grid_world_ndarray[tuple_state_for_searching[0], horizontal_coord] != digit_for_representing_the_wall):
            list_of_tuple_actions.append((tuple_state_for_searching[0], horizontal_coord))
    # Action DOWN
    vertical_coord = tuple_state_for_searching[0] + 1    
    if (vertical_coord <= vertical_dim - 1):
        if (grid_world_ndarray[vertical_coord, tuple_state_for_searching[1]] != digit_for_representing_the_wall):
            list_of_tuple_actions.append((vertical_coord, tuple_state_for_searching[1]))
    # Action LEFT
    horizontal_coord = tuple_state_for_searching[1] - 1    
    if (horizontal_coord >= 0):
        if (grid_world_ndarray[tuple_state_for_searching[0], horizontal_coord] != digit_for_representing_the_wall):
            list_of_tuple_actions.append((tuple_state_for_searching[0], horizontal_coord))
    
    return list_of_tuple_actions

# Executing Iterative Policy Evaluation
def iterative_policy_evaluation():
    """
    Calculating V of S for every state in state space
    """    
    min_value = 0.000000000001 # when traspasing this value to the less then stop while loop
    delta_max = min_value + 1
    gamma_value = 0.9
    while delta_max >= min_value:
        delta_max = 0
        for vertical_item in range(vertical_dim):
            for horizontal_item in range(horizontal_dim):
                # If its not a terminal state and its not the wall               
                if (not ((vertical_item, horizontal_item) in terminal_states)) and (grid_world_ndarray[vertical_item, horizontal_item] != digit_for_representing_the_wall):                    
                    action_list_tuple = find_all_actions_of_the_state((vertical_item, horizontal_item))
                    if len(action_list_tuple) != 0:
                        p_s_prime_r_given_s_and_a_for_for_every_s_prime = 1 / len(action_list_tuple) # conditioning equal distribution of all s' from s and a
                    else:
                        p_s_prime_r_given_s_and_a_for_for_every_s_prime = 1
                    V_of_S_k_plus_one = 0
                    for element_action in action_list_tuple:                    
                        V_of_S_k_plus_one = V_of_S_k_plus_one + p_s_prime_r_given_s_and_a_for_for_every_s_prime * (rewards_ndarray[element_action[0], element_action[1]] + gamma_value * V_of_S_for_every_state_ndarray[element_action[0], element_action[1]])
                    difference_VSk1_and_VSk = abs(V_of_S_for_every_state_ndarray[vertical_item, horizontal_item] - V_of_S_k_plus_one)
                    if difference_VSk1_and_VSk > delta_max:
                        delta_max = difference_VSk1_and_VSk
                    V_of_S_for_every_state_ndarray[vertical_item, horizontal_item] = V_of_S_k_plus_one

        if delta_max < min_value:
            break

iterative_policy_evaluation()

def policy_fill_the_grid_states_with_actions ():
    """
    Looping thru all states of the state space and finding the best action for every state
    """
    for state_vertical in range(vertical_dim):
        for state_horizontal in range(horizontal_dim):
            if (not ((state_vertical, state_horizontal) in terminal_states)) and (grid_world_ndarray[state_vertical, state_horizontal] != digit_for_representing_the_wall):
                grid_policy_states_ndarray[state_vertical, state_horizontal] = find_best_action_of_the_state((state_vertical, state_horizontal))

def find_best_action_of_the_state (tuple_state_for_searching):
    """
    As input coordenates of state and as output the best action of the state
    """
    dict_all_actions_and_its_q_values = {}
    # Action UP
    vertical_coord = tuple_state_for_searching[0] - 1    
    if (vertical_coord >= 0):
        if (grid_world_ndarray[vertical_coord, tuple_state_for_searching[1]] != digit_for_representing_the_wall):
             dict_all_actions_and_its_q_values[1] = V_of_S_for_every_state_ndarray[vertical_coord, tuple_state_for_searching[1]]
    # Action RIGHT
    horizontal_coord = tuple_state_for_searching[1] + 1    
    if (horizontal_coord <= horizontal_dim - 1):
        if (grid_world_ndarray[tuple_state_for_searching[0], horizontal_coord] != digit_for_representing_the_wall):
            dict_all_actions_and_its_q_values[2] = V_of_S_for_every_state_ndarray[tuple_state_for_searching[0], horizontal_coord]
    # Action DOWN
    vertical_coord = tuple_state_for_searching[0] + 1    
    if (vertical_coord <= vertical_dim - 1):
        if (grid_world_ndarray[vertical_coord, tuple_state_for_searching[1]] != digit_for_representing_the_wall):
            dict_all_actions_and_its_q_values[3] = V_of_S_for_every_state_ndarray[vertical_coord, tuple_state_for_searching[1]]
    # Action LEFT
    horizontal_coord = tuple_state_for_searching[1] - 1    
    if (horizontal_coord >= 0):
        if (grid_world_ndarray[tuple_state_for_searching[0], horizontal_coord] != digit_for_representing_the_wall):
            dict_all_actions_and_its_q_values[4] = V_of_S_for_every_state_ndarray[tuple_state_for_searching[0], horizontal_coord]
    
    if tuple_state_for_searching == (0, horizontal_dim - 2):
        return 2
    else:
        if len(dict_all_actions_and_its_q_values) != 0:
            return max(dict_all_actions_and_its_q_values, key=dict_all_actions_and_its_q_values.get)
        else:
            return 5

# announcing policy of allstates and will represent action UP by 1, action RIGHT by 2, action DOWN by 3, action LEFT by 4, no action by 5
grid_policy_states_ndarray = np.zeros((vertical_dim,horizontal_dim))
policy_fill_the_grid_states_with_actions()
# print(grid_policy_states_ndarray)
plt.figure(dpi=500, frameon=False)
plt.imshow(grid_policy_states_ndarray, interpolation='none')
plt.savefig('best_policy_max_q_of_every_state_in_grid_world.png')

def improvement_of_policy ():
    """
    Trying to improove the policy of all states
    """
    policy_improved_boolean = True
    for state_vertical in range(vertical_dim):
        for state_horizontal in range(horizontal_dim):
            if (not ((state_vertical, state_horizontal) in terminal_states)) and (grid_world_ndarray[state_vertical, state_horizontal] != digit_for_representing_the_wall):
                if grid_policy_states_ndarray[state_vertical, state_horizontal] != find_best_action_of_the_state((state_vertical, state_horizontal)):
                    grid_policy_states_ndarray[state_vertical, state_horizontal] = find_best_action_of_the_state((state_vertical, state_horizontal))
                else:
                    policy_improved_boolean = False
    return policy_improved_boolean

# Loop thru policy evaluation and policy improvement
policy_improved_boolean = True
while policy_improved_boolean == True:
    iterative_policy_evaluation()
    policy_improved_boolean = improvement_of_policy()
    if policy_improved_boolean == True:
        plt.imshow(grid_policy_states_ndarray, interpolation='none')
        plt.savefig('best_policy_max_q_of_every_state_in_grid_world.png')
        input("next")


x,y = np.meshgrid(np.arange(horizontal_dim), np.arange(vertical_dim))
v, u = np.zeros((vertical_dim,horizontal_dim)), np.zeros((vertical_dim,horizontal_dim))

# print("\nx\n")
# print(x)
# print("\ny\n")
# print(y)
# print("next")
# print("\nv\n")
# print(v)

for vertical_element in range(vertical_dim):
    for horizontal_element in range(horizontal_dim):
        action = grid_policy_states_ndarray[vertical_element, horizontal_element]
        if action == 1:
            v[vertical_dim - 1 - vertical_element, horizontal_element] = 0
            u[vertical_dim - 1 - vertical_element, horizontal_element] = 1
        elif action == 2:
            v[vertical_dim - 1 - vertical_element, horizontal_element] = 1
            u[vertical_dim - 1 - vertical_element, horizontal_element] = 0
        elif action == 3:
            v[vertical_dim - 1 - vertical_element, horizontal_element] = 0
            u[vertical_dim - 1 - vertical_element, horizontal_element] = -1
        elif action == 4:
            v[vertical_dim - 1 - vertical_element, horizontal_element] = -1
            u[vertical_dim - 1 - vertical_element, horizontal_element] = 0
                
fig, ax = plt.subplots()
q = ax.quiver(x,y,v,u)
plt.savefig('best_policy_max_q_of_every_state_in_grid_world_arrows.png', dpi=150)