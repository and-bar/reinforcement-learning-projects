import numpy as np
import random
import matplotlib.pyplot as plt

vertical_dim = 8
horizontal_dim = 8
digit_for_representing_the_wall = 9


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
grid_world_ndarray[vertical_dim-2,0] = 0
grid_world_ndarray[vertical_dim-1,1] = 0
grid_world_ndarray[vertical_dim-2,1] = 0

# Creating grid of V(S) of every state and populating it with zeros
v_of_s_for_every_state_ndarray = np.zeros([vertical_dim, horizontal_dim])

# Creating grid of rewards of every state
rewards_ndarray = np.random.randint(-1,0, [vertical_dim, horizontal_dim])
rewards_ndarray[0,horizontal_dim - 1] = 100
rewards_ndarray[1,horizontal_dim - 1] = -1

def find_all_actions_of_the_state(tuple_state_for_searching):
    """
    Check for possible action up, left, down, right from current state
    If can make action adding tuple of coordinates of that action to the list
    of all posible actions and adding its corresponding action.
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
            list_of_tuple_actions.append(((vertical_coord, tuple_state_for_searching[1]), 1))
    # Action RIGHT
    horizontal_coord = tuple_state_for_searching[1] + 1    
    if (horizontal_coord <= horizontal_dim - 1):
        if (grid_world_ndarray[tuple_state_for_searching[0], horizontal_coord] != digit_for_representing_the_wall):
            list_of_tuple_actions.append(((tuple_state_for_searching[0], horizontal_coord), 2))
    # Action DOWN
    vertical_coord = tuple_state_for_searching[0] + 1    
    if (vertical_coord <= vertical_dim - 1):
        if (grid_world_ndarray[vertical_coord, tuple_state_for_searching[1]] != digit_for_representing_the_wall):
            list_of_tuple_actions.append(((vertical_coord, tuple_state_for_searching[1]), 3))
    # Action LEFT
    horizontal_coord = tuple_state_for_searching[1] - 1    
    if (horizontal_coord >= 0):
        if (grid_world_ndarray[tuple_state_for_searching[0], horizontal_coord] != digit_for_representing_the_wall):
            list_of_tuple_actions.append(((tuple_state_for_searching[0], horizontal_coord), 4))
    
    return list_of_tuple_actions

grid_policy_states_ndarray = np.zeros((vertical_dim,horizontal_dim))

def print_policy_from_the_grid_of_action_states (name_of_the_file_png):
    """
    takes ndarray, get every digit from the cell that representing up or right or down or left
    and prints quiver table from pyplot
    """
    x,y = np.meshgrid(np.arange(horizontal_dim), np.arange(vertical_dim))
    v, u = np.zeros((vertical_dim,horizontal_dim)), np.zeros((vertical_dim,horizontal_dim))

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
    q = ax.quiver(x,y,v,u,angles='xy', scale_units='xy', scale=1)
    plt.savefig(name_of_the_file_png, dpi=100)

def get_random_action_from_state(vertical_coord, horizontal_coord):
    """
    Get random action from the given state
    """
    random_action_state_tuple = find_all_actions_of_the_state((vertical_coord, horizontal_coord))
    list_sequence_of_states_tupels = [element_seq[0] for element_seq in list_states_actions_of_initial_random_policy]
    random_action_state_tuple_for_loop = random_action_state_tuple.copy()
    for element in random_action_state_tuple_for_loop:
        if element[0] in list_sequence_of_states_tupels: # not to visit visited state
            random_action_state_tuple.remove(element)
    if len(random_action_state_tuple) != 0:
        random_action = random.choice(random_action_state_tuple)
        return random_action
    else:
        return None

n_of_steps_of_episode = 100

list_states_actions_of_initial_random_policy = []

def get_random_policy_from_initial_state(vertical_coord, horizontal_coord):
    """
    Get start state and build up random actions for every next state
    """
    step = 0
    previous_state = (-1, -1) # initial state out of the grid world
    while step <= n_of_steps_of_episode:
        next_action_state_pair_tuple = get_random_action_from_state(vertical_coord, horizontal_coord)
        if next_action_state_pair_tuple != None:
            list_states_actions_of_initial_random_policy.append(((vertical_coord, horizontal_coord), next_action_state_pair_tuple))
            grid_policy_states_ndarray[vertical_coord, horizontal_coord] = next_action_state_pair_tuple[1]
            vertical_coord = next_action_state_pair_tuple[0][0]
            horizontal_coord = next_action_state_pair_tuple[0][1]
            step += 1
        else:
            break
        if next_action_state_pair_tuple[0] in terminal_states:
            break

vertical_coord = vertical_dim - 1
horizontal_coord = 0

get_random_policy_from_initial_state(vertical_coord, horizontal_coord) 
print_policy_from_the_grid_of_action_states("MonteCarlostartingPolicy.png")
print(f"list_states_actions_of_initial_random_policy {list_states_actions_of_initial_random_policy}")

gamma = 0.9
g_return = 0
for state in list_states_actions_of_initial_random_policy[::-1]:
    v_of_s_for_every_state_ndarray[state[0]] = rewards_ndarray[state[1][0]] + gamma * g_return
    g_return += v_of_s_for_every_state_ndarray[state[0]]
    print(f"state {state} v_of_s_for_every_state_ndarray[state[0]] {v_of_s_for_every_state_ndarray[state[0]]} ")

print(grid_policy_states_ndarray)