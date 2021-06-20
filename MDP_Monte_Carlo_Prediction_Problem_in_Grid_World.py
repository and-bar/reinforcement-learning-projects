import numpy as np
import random
import matplotlib.pyplot as plt

vertical_dim = 5
horizontal_dim = 5
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
rewards_ndarray[0,horizontal_dim - 1] = 1
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
    # plt.imshow(grid_world_ndarray, interpolation='none')
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

list_states_actions_of_initial_random_policy = []

def get_random_policy_from_initial_state_and_find_returns(vertical_coord, horizontal_coord):
    """
    Get start state and build up random actions for every next state
    """
    step = 0
    global list_states_actions_of_initial_random_policy
    list_states_actions_of_initial_random_policy= []
    n_of_steps_of_episode = 300
    while step <= n_of_steps_of_episode:
        next_action_state_pair_tuple = get_random_action_from_state(vertical_coord, horizontal_coord)
        if next_action_state_pair_tuple != None:
            list_states_actions_of_initial_random_policy.append(((vertical_coord, horizontal_coord), next_action_state_pair_tuple))
            grid_policy_states_ndarray[vertical_coord, horizontal_coord] = next_action_state_pair_tuple[1]
            vertical_coord = next_action_state_pair_tuple[0][0]
            horizontal_coord = next_action_state_pair_tuple[0][1]
            step += 1
        else:
            #from this state can't make any valuable action, thou making random terminal action
            grid_policy_states_ndarray[vertical_coord, horizontal_coord] = random.choice([1,2,3,4])
            break
        if next_action_state_pair_tuple[0] in terminal_states:
            break
    gamma = 0.9
    g_return = 0
    for state in list_states_actions_of_initial_random_policy[::-1]:
        v_of_s_for_every_state_ndarray[state[0]] = rewards_ndarray[state[1][0]] + gamma * g_return
        g_return += v_of_s_for_every_state_ndarray[state[0]]

def create_random_policy_for_every_state ():
    """
    Creating random policy with random action for every state
    """
    for vertical_element in range(vertical_dim):
        for horizontal_element in range(horizontal_dim):
            if (vertical_element, horizontal_element) not in terminal_states:
                #creating random policy
                if (grid_policy_states_ndarray[vertical_element, horizontal_element] == 0) and ((vertical_element, horizontal_element) not in terminal_states) and grid_world_ndarray[vertical_element, horizontal_element] != digit_for_representing_the_wall:
                    get_random_policy_from_initial_state_and_find_returns(vertical_element, horizontal_element)


def evaluate_policy ():
    """
    Looping through every state-action of given random policy and seraching its avereged returns
    """
    treshold = 0.1
    global samples_of_returns_ndarray
    gamma = 0.9
    while True:
        delta_max = 0
        for vertical_cell in range(vertical_dim):
            for horizontal_cell in range(horizontal_dim):
                if (vertical_cell, horizontal_cell) in terminal_states:
                    continue
                elif grid_policy_states_ndarray[vertical_cell, horizontal_cell] == 1 and vertical_cell-1 >= 0:
                    s_prime_coord = (vertical_cell-1, horizontal_cell)
                elif grid_policy_states_ndarray[vertical_cell, horizontal_cell] == 2 and horizontal_cell + 1 <= horizontal_dim - 1:
                    s_prime_coord = (vertical_cell, horizontal_cell + 1)
                elif grid_policy_states_ndarray[vertical_cell, horizontal_cell] == 3 and vertical_cell + 1 <= vertical_dim - 1:
                    s_prime_coord = (vertical_cell + 1, horizontal_cell)
                elif grid_policy_states_ndarray[vertical_cell, horizontal_cell] == 4 and horizontal_cell - 1 >= 0:
                    s_prime_coord = (vertical_cell, horizontal_cell - 1)
                else:
                    s_prime_coord = (vertical_cell, horizontal_cell)
                samples_of_returns_ndarray[vertical_cell, horizontal_cell] += 1
                v_of_s_for_every_state_ndarray_old = v_of_s_for_every_state_ndarray[vertical_cell, horizontal_cell]
                return_g = rewards_ndarray[s_prime_coord] + gamma * v_of_s_for_every_state_ndarray[s_prime_coord]
                v_of_s_for_every_state_ndarray[vertical_cell, horizontal_cell] = v_of_s_for_every_state_ndarray_old + (return_g - v_of_s_for_every_state_ndarray_old)/samples_of_returns_ndarray[vertical_cell, horizontal_cell]
                absolute_difference_vsk_plus_1_and_vsk = abs(v_of_s_for_every_state_ndarray[vertical_cell, horizontal_cell] - v_of_s_for_every_state_ndarray_old)
                if delta_max < absolute_difference_vsk_plus_1_and_vsk:        
                    delta_max = absolute_difference_vsk_plus_1_and_vsk
        if delta_max <= treshold:
            print(f"delta_max {delta_max}")
            break

create_random_policy_for_every_state ()
print_policy_from_the_grid_of_action_states("MonteCarlostartingPolicy.png")

plt.clf() #clear figure
plt.imshow(grid_world_ndarray, interpolation='none')
plt.savefig('grid_world.png')

# ndarray with number of iterations of total returns for every state
samples_of_returns_ndarray = np.zeros([vertical_dim, horizontal_dim])

evaluate_policy()
print(v_of_s_for_every_state_ndarray)

print(grid_policy_states_ndarray)
print(samples_of_returns_ndarray)