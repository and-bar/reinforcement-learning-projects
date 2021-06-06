import numpy as np
import random
import matplotlib.pyplot as plt
import sys
import pandas as pd

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)

# Creating grid world represented by ndarray
# zeros agent can pass and ones it a wall
# Generate Grid world till terminal states : last horisontal to the right and first two vertical of the top not the walls


vertical_dim = 30
horizontal_dim = 30
digit_for_representing_the_wall = 2


# Announsing terminal states
terminal_states = [(0, horizontal_dim - 1), (1, horizontal_dim - 1)]

# Creating walls in grid world
grid_world_ndarray = np.zeros([vertical_dim, horizontal_dim])
counter_walls = 0
while counter_walls <= int(vertical_dim*horizontal_dim*0.1):
    grid_world_ndarray[random.randint(0,vertical_dim-1) , random.randint(0,horizontal_dim-1)] = digit_for_representing_the_wall
    counter_walls += 1

# cleaning walls in start space and terminal spaces areas
grid_world_ndarray[0,horizontal_dim - 1] = 0
grid_world_ndarray[1,horizontal_dim - 1] = 0
grid_world_ndarray[0,horizontal_dim - 2] = 0
grid_world_ndarray[1,horizontal_dim - 2] = 0
grid_world_ndarray[vertical_dim-1,0] = 0

# plt.imshow(grid_world_ndarray, interpolation='none')
# plt.savefig('gridworld.png')

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
min_value = 0.01 # when traspasing this value to the less then stop while loop
print(f"min_value {min_value}")
def iterative_policy_evaluation():
    """
    Calculating V of S for every state in state space
    """    
    global min_value
    
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


# dumping V of S of all states to excel
# V_of_S_for_every_state_df = pd.DataFrame(V_of_S_for_every_state_ndarray)
# filepath = 'V_of_S_for_every_state_ndarray.xlsx'
# V_of_S_for_every_state_df.to_excel(filepath, index=False)


def search_best_policy_as_best_action_of_every_state_of_the_policy():
    """
    Function get as input the coordinates of statrting state and
    searching the best policy and saves it and vusualizes it
    Returns True if reached terminal state
    """
    
    global min_value

    reached_terminal_state_boolean = True

    state_vert_coord = vertical_dim - 1
    state_horizont_coord  = 0
    
    grid_best_policy_ndrray[state_vert_coord, state_horizont_coord] = 1
    v_of_s_tuple = (state_vert_coord, state_horizont_coord)
    state_s1 = v_of_s_tuple
    state_s2 = 0
    state_s3 = 0

    actions_of_state_tuple = find_all_actions_of_the_state(v_of_s_tuple)
    while not (terminal_states[0] in actions_of_state_tuple):

        # print(f"possible actions {actions_of_state_tuple} for state {v_of_s_tuple}")
        v_of_s_tuple =  actions_of_state_tuple[0]
        for action in actions_of_state_tuple[1:]:
            if V_of_S_for_every_state_ndarray[action] > V_of_S_for_every_state_ndarray[v_of_s_tuple]:
                v_of_s_tuple = action
        grid_best_policy_ndrray[v_of_s_tuple] = 1

        # Check if the best action lead to the visited state previously, and thtat confirms internal loop between twostates with two actions
        if state_s2 == 0:
            state_s2 = v_of_s_tuple
        elif state_s3 == 0:
            state_s3 = v_of_s_tuple
        else:
            state_s1 = state_s2
            state_s2 = state_s3
            state_s3 = v_of_s_tuple
            # input('next')
        if state_s3 == state_s1:
            print(f'Looping between to states, cant proceed. S1 S2 S3 = {state_s1} {state_s2} {state_s3}')
            min_value = min_value / 10
            print(f"min_value {min_value}\n")
            reached_terminal_state_boolean = False
            break

        actions_of_state_tuple = find_all_actions_of_the_state(v_of_s_tuple)

    # Returns here True if reached terminal state
    return reached_terminal_state_boolean

founf_terminal_state = False
cycle_step = 0
while (founf_terminal_state == False) and (cycle_step < 50):
    iterative_policy_evaluation()
    print("finished Iterative policy evaluation")
    grid_best_policy_ndrray = np.zeros((vertical_dim,horizontal_dim))
    founf_terminal_state = search_best_policy_as_best_action_of_every_state_of_the_policy()
    cycle_step += 1

merged_grid_world_best_policy_for_visualising = grid_world_ndarray + grid_best_policy_ndrray
merged_grid_world_best_policy_for_visualising[terminal_states[0]] = 3
merged_grid_world_best_policy_for_visualising[terminal_states[1]] = 4
plt.imshow(merged_grid_world_best_policy_for_visualising, interpolation='none')
plt.savefig('policy_not_the_best_that_leads_to_exit_in_grid_world.png')
