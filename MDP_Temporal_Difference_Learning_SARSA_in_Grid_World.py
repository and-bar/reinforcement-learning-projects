"""
Implementation of Temporal Different Leraning algorithm with SARSA control in GridWorld in MDP framework
"""
import  time
import numpy as np
from PIL import Image
from numba import njit, prange
from numba.typed import List
from numpy import float32, random

def create_gridworld_ndarray_adding_walls_and_terminal_states(win_terminal_state, loose_terminal_state, starting_sate):
    """
    make numpy 2D array adding walls and terminal states
    representing actions and wall and terminal states:
    1 for up, 2 for right, 3 for down, 4 for left, 5 for wall, 6 for winner terminal state, 7 for loosing terminal state, 8 clear state, 9 start state
    
    """
    gridworld_ndarray = (np.arange(n_states_vertical * n_states_horizontal)).reshape((n_states_vertical, n_states_horizontal))
    gridworld_ndarray[:, :] = 8

    n_of_walls_total = (n_states_vertical*n_states_horizontal)*0.2 # here 0.2 is 20% of total occupancy by walls in gridworld
    n_of_walls = 0
    while n_of_walls <= n_of_walls_total:
        vert_coord = random.randint(0, n_states_vertical)
        hor_coord = random.randint(0, n_states_horizontal)
        if gridworld_ndarray[vert_coord, hor_coord] != 5:
            gridworld_ndarray[vert_coord, hor_coord] = 5
            n_of_walls+=1

    gridworld_ndarray[win_terminal_state[0], win_terminal_state[1]] = 6
    gridworld_ndarray[loose_terminal_state[0], loose_terminal_state[1]] = 7
    gridworld_ndarray[starting_sate[0], starting_sate[1]] = 9

    gridworld_ndarray[win_terminal_state[0], win_terminal_state[1] - 1] = 8 # clear left state from wall of terminal state
    gridworld_ndarray[win_terminal_state[0] + 1, win_terminal_state[1] - 1] = 8 # clear left, down state from wall of terminal state
    gridworld_ndarray[win_terminal_state[0] + 1, win_terminal_state[1]] = 8 # clear down state from wall of terminal state
    gridworld_ndarray[loose_terminal_state[0], loose_terminal_state[1] - 1] = 8 # clear left state from wall of terminal state
    gridworld_ndarray[loose_terminal_state[0] + 1, loose_terminal_state[1]] = 8 # clear down state from wall of terminal state
    gridworld_ndarray[loose_terminal_state[0] - 1, loose_terminal_state[1]] = 8 # clear up state from wall of terminal state
    gridworld_ndarray[starting_sate[0] - 1, starting_sate[1]] = 8 # clear down state from wall of starting state
    gridworld_ndarray[starting_sate[0], starting_sate[1] + 1] = 8 # clear down state from wall of starting state
    gridworld_ndarray[starting_sate[0] - 1, starting_sate[1] + 1] = 8 # clear down state from wall of starting state

    return gridworld_ndarray

def create_rewards_and_value_states__ndarray(win_terminal_state, win_terminal_state_reward, loose_terminal_state, loose_terminal_state_reward, starting_sate, starting_sate_reward, reward_of_each_state):
    """
    make 3d array with 5 layers, zero layer for rewards, 1st for Q(s,a) action 'up', 2nd for Q(s,a) action 'right',  3d for Q(s,a) action 'down', 4nd for Q(s,a) action 'left',
    5fth for number of returns collected for action 'up', 6th for number of returns collected for action 'right', 7th for number of returns collected for action 'down',
    8th for number of returns collected for action 'left'
    """
    rewards_plus_states_values_ndarray = (np.arange(9 * n_states_vertical * n_states_horizontal)).reshape((9, n_states_vertical, n_states_horizontal)) # , dtype = np.float32
    rewards_plus_states_values_ndarray[:,:,:] = 0 # instantiating with zeros for layers that contain n collected returns so far for actions and the rest of layers
    rewards_plus_states_values_ndarray[1:5,:,:] = -2.0
    rewards_plus_states_values_ndarray[0] = reward_of_each_state
    rewards_plus_states_values_ndarray[0,win_terminal_state[0], win_terminal_state[1]] = win_terminal_state_reward
    rewards_plus_states_values_ndarray[0,loose_terminal_state[0], loose_terminal_state[1]] = loose_terminal_state_reward
    rewards_plus_states_values_ndarray[0,starting_sate[0], starting_sate[1]] = starting_sate_reward
    rewards_plus_states_values_ndarray = rewards_plus_states_values_ndarray.astype(np.float32)
    
    return rewards_plus_states_values_ndarray

def make_arrows_images_as_2d_x_n_numpy_array(images_names_list):
    """
    Read image from file and convert it as grayscale and save to numpy ndarray as layer one of 7 each dimention represent one image of grayscale
    """
    ndarray_of_images = np.zeros([13, 11, 11])
    image_layer = 0
    
    for name in images_names_list:
        ndarray_of_images[image_layer, :, :] =  (np.array(Image.open(name).convert('L').getdata())).reshape([11,11]) # here 11 is a image size for arroes and walls
        image_layer += 1
    
    return ndarray_of_images

@njit
def find_all_actions_of_the_state(n_states_vertical, n_states_horizontal, state_for_searching, grid_world_ndarray):
    """
    Check for possible action up, left, down, right from current state
    If can make action adding tuple of coordinates of that action to the list
    of all posible actions and adding its corresponding action.
    First check if action do not guide out of the grid
    Second check if action do not guide to the wall
    If action comply with this conditions add coordinates of the
    destination position of the grid to the list
    """
    list_of_actions = List()
    vertical_coord = state_for_searching[0] - 1    
    # Action UP
    if (vertical_coord >= 0):
        if grid_world_ndarray[vertical_coord, state_for_searching[1]] in (1, 2, 3, 4, 6, 7, 8):
            
            action_up = List()
            action_up.append(vertical_coord)
            action_up.append(state_for_searching[1])
            action_up.append(1)
            list_of_actions.append(action_up)
    # Action RIGHT
    horizontal_coord = state_for_searching[1] + 1    
    if (horizontal_coord <= n_states_horizontal - 1):
        if grid_world_ndarray[state_for_searching[0], horizontal_coord] in (1, 2, 3, 4, 6, 7, 8):
            
            action_right = List()
            action_right.append(state_for_searching[0])
            action_right.append(horizontal_coord)
            action_right.append(2)
            list_of_actions.append(action_right)
            
    # Action DOWN
    vertical_coord = state_for_searching[0] + 1    
    if (vertical_coord <= n_states_vertical - 1):
        if grid_world_ndarray[vertical_coord, state_for_searching[1]] in (1, 2, 3, 4, 6, 7, 8):
            
            action_down = List()
            action_down.append(vertical_coord)
            action_down.append(state_for_searching[1])
            action_down.append(3)
            list_of_actions.append(action_down)
            
    # Action LEFT
    horizontal_coord = state_for_searching[1] - 1    
    if (horizontal_coord >= 0):
        if grid_world_ndarray[state_for_searching[0], horizontal_coord] in (1, 2, 3, 4, 6, 7,8):
            
            action_left = List()
            action_left.append(state_for_searching[0])
            action_left.append(horizontal_coord)
            action_left.append(4)
            list_of_actions.append(action_left)
            
    return list_of_actions

@njit
def change_for_improoved_policy_for_every_state_of_grid_world(n_states_vertical, n_states_horizontal, gridworld_ndarray, rewards_plus_qsa_values_ndarray):
    """
    look up for Q* for every state and change it for corresponded a* in gridworld
    """
    for vert_coorden in range(n_states_vertical):
        for horiz_coorden in range(n_states_horizontal):
            state_for_best_q_star_coordenates = List()
            state_for_best_q_star_coordenates.append(vert_coorden)
            state_for_best_q_star_coordenates.append(horiz_coorden)
            action_of_s_state = gridworld_ndarray[state_for_best_q_star_coordenates[0], state_for_best_q_star_coordenates[1]]
            if  action_of_s_state in (1,2,3,4):
                list_of_actions = find_all_actions_of_the_state(n_states_vertical, n_states_horizontal, state_for_best_q_star_coordenates, gridworld_ndarray)
                # structure of list_of_actions : [[0, 2, 2], [1, 1, 3]] first two coord of s' and and third action
                if len(list_of_actions) == 0:
                    continue
                action_from_s_to_s_prime = list_of_actions[0][2]
                best_qsa = rewards_plus_qsa_values_ndarray[list_of_actions[0][2], state_for_best_q_star_coordenates[0], state_for_best_q_star_coordenates[1]]
                list_of_actions.remove(list_of_actions[0])

                if len(list_of_actions) != 0:
                    for coord_of_s_prime_and_action_from_s in list_of_actions:
                        if rewards_plus_qsa_values_ndarray[coord_of_s_prime_and_action_from_s[2], state_for_best_q_star_coordenates[0], state_for_best_q_star_coordenates[1]] > best_qsa:
                            best_qsa = rewards_plus_qsa_values_ndarray[coord_of_s_prime_and_action_from_s[2], state_for_best_q_star_coordenates[0], state_for_best_q_star_coordenates[1]]
                            action_from_s_to_s_prime = coord_of_s_prime_and_action_from_s[2]
                
                gridworld_ndarray[state_for_best_q_star_coordenates[0], state_for_best_q_star_coordenates[1]] = action_from_s_to_s_prime

    return gridworld_ndarray

@njit
def select_action_from_state_by_epsilon_and_coord_of_next_state(n_states_vertical, n_states_horizontal, gridworld_ndarray, state_k_coord, rewards_plus_qsa_values_ndarray):
    """
    get action by epsilon for the state and coordenates of next state
    """
    list_all_possible_actions_from_the_state = find_all_actions_of_the_state(n_states_vertical, n_states_horizontal, state_k_coord, gridworld_ndarray)
    # print("list_all_possible_actions_from_the_state ", list_all_possible_actions_from_the_state)
    # structure of list_all_possible_actions_from_the_state: [[6, 0, 1], [7, 1, 2]]
    list_all_possible_actions_from_the_state_for_random = list_all_possible_actions_from_the_state.copy()

    if len(list_all_possible_actions_from_the_state) == 0:
        best_s_prime_and_a = List()
        best_s_prime_and_a.append(state_k_coord[0])
        best_s_prime_and_a.append(state_k_coord[1])
        best_s_prime_and_a.append(0)

    # action_from_s_to_s_prime = list_all_possible_actions_from_the_state[0][2]
    best_qsa = rewards_plus_qsa_values_ndarray[list_all_possible_actions_from_the_state[0][2], state_k_coord[0], state_k_coord[1]]
    best_s_prime_and_a = list_all_possible_actions_from_the_state[0]
    list_all_possible_actions_from_the_state.remove(list_all_possible_actions_from_the_state[0])

    if len(list_all_possible_actions_from_the_state) != 0:
        for coord_of_s_prime_and_action_from_s in list_all_possible_actions_from_the_state:
            if rewards_plus_qsa_values_ndarray[coord_of_s_prime_and_action_from_s[2], state_k_coord[0], state_k_coord[1]] > best_qsa:
                best_qsa = rewards_plus_qsa_values_ndarray[coord_of_s_prime_and_action_from_s[2], state_k_coord[0], state_k_coord[1]]
                best_s_prime_and_a = coord_of_s_prime_and_action_from_s

    epsilon = 0.15
    if random.random() < epsilon:
        # select random action
        if len(list_all_possible_actions_from_the_state_for_random) == 1:
            gridworld_ndarray[state_k_coord[0], state_k_coord[1]] = best_s_prime_and_a[2]
            
            return best_s_prime_and_a, gridworld_ndarray
        else:
            list_all_possible_actions_from_the_state_for_random = list_all_possible_actions_from_the_state_for_random.remove(best_s_prime_and_a)
            best_s_prime_and_a = list_all_possible_actions_from_the_state[random.randint(0, len(list_all_possible_actions_from_the_state))]
            gridworld_ndarray[state_k_coord[0], state_k_coord[1]] = best_s_prime_and_a[2]
            
            return best_s_prime_and_a, gridworld_ndarray
    else:
        # select best action from state wich lead to Q*(s,a)
        gridworld_ndarray[state_k_coord[0], state_k_coord[1]] = best_s_prime_and_a[2]

        return best_s_prime_and_a, gridworld_ndarray

@njit
def select_qsa_of_s_prime_by_epsilon_greedy_or_max_qsa(n_states_vertical, n_states_horizontal, gridworld_ndarray, s_prime_coord, rewards_plus_qsa_values_ndarray, epsilon_0_max_value_1):
    """
    get Q(s,a) value of s' by epsilob greedy or maximum Q(s,a)
    epsilon_0_max_value_1 -> structure "0" for epsilon greedy "1" for max value
    """
    list_all_possible_actions_from_the_s_prime = find_all_actions_of_the_state(n_states_vertical, n_states_horizontal, s_prime_coord, gridworld_ndarray)
    # structure of list_all_possible_actions_from_the_s_prime: [[coord, coord, action], [coord, coord, action]]
    list_all_possible_actions_from_the_s_prime_for_random = list_all_possible_actions_from_the_s_prime.copy()

    if len(list_all_possible_actions_from_the_s_prime) == 0:
        return 0

    best_qsa = rewards_plus_qsa_values_ndarray[list_all_possible_actions_from_the_s_prime[0][2], s_prime_coord[0], s_prime_coord[1]]
    best_s_prime_and_a = list_all_possible_actions_from_the_s_prime[0]
    list_all_possible_actions_from_the_s_prime.remove(list_all_possible_actions_from_the_s_prime[0])

    if len(list_all_possible_actions_from_the_s_prime) != 0:
        for coord_of_s_prime_and_action_from_s in list_all_possible_actions_from_the_s_prime:
            if rewards_plus_qsa_values_ndarray[coord_of_s_prime_and_action_from_s[2], s_prime_coord[0], s_prime_coord[1]] > best_qsa:
                best_qsa = rewards_plus_qsa_values_ndarray[coord_of_s_prime_and_action_from_s[2], s_prime_coord[0], s_prime_coord[1]]
                best_s_prime_and_a = coord_of_s_prime_and_action_from_s

    if epsilon_0_max_value_1 == 0:
        epsilon = 0.15
        if random.random() < epsilon:
            # select random action
            if len(list_all_possible_actions_from_the_s_prime_for_random) == 1:
                return rewards_plus_qsa_values_ndarray[best_s_prime_and_a[2], s_prime_coord[0], s_prime_coord[1]]
            else:
                list_all_possible_actions_from_the_s_prime_for_random = list_all_possible_actions_from_the_s_prime_for_random.remove(best_s_prime_and_a)
                best_s_prime_and_a = list_all_possible_actions_from_the_s_prime[random.randint(0, len(list_all_possible_actions_from_the_s_prime))]
                return rewards_plus_qsa_values_ndarray[best_s_prime_and_a[2], s_prime_coord[0], s_prime_coord[1]]
        else:
            # select best action from state wich lead to Q*(s,a)
            return rewards_plus_qsa_values_ndarray[best_s_prime_and_a[2], s_prime_coord[0], s_prime_coord[1]]

    if epsilon_0_max_value_1 == 1:
        return rewards_plus_qsa_values_ndarray[best_s_prime_and_a[2], s_prime_coord[0], s_prime_coord[1]]

@njit
def play_one_episode(n_states_vertical, n_states_horizontal, gridworld_ndarray, rewards_plus_qsa_values_ndarray, win_terminal_state, loose_terminal_state, s_state_coord):
    """
    play one episode of the game, update Q(s,a) for each visited state state
    s_prime_coord_and_action_from_s_to_s_prime -> structure [coord, coord, action]
    s_state_coord -> structure [coord, coord]
    """
    gamma = 0.9
    alfa = 0.1
    data_of_episode_list = List()
    played_from_start_state_and_reached_terminal_state = False
    s_prime_coord_and_action_from_s_to_s_prime, gridworld_ndarray = select_action_from_state_by_epsilon_and_coord_of_next_state(n_states_vertical, n_states_horizontal, gridworld_ndarray, s_state_coord, rewards_plus_qsa_values_ndarray)

    if s_prime_coord_and_action_from_s_to_s_prime == None:
        return rewards_plus_qsa_values_ndarray, gridworld_ndarray, played_from_start_state_and_reached_terminal_state, data_of_episode_list
    
    visited_states = List()
    visited_states.append(s_state_coord)

    while True:

        if  (s_prime_coord_and_action_from_s_to_s_prime[0:2] in visited_states) or (gridworld_ndarray[s_prime_coord_and_action_from_s_to_s_prime[0], s_prime_coord_and_action_from_s_to_s_prime[1]] == 5): # visited states or wall
            break

        data_of_episode = List()
        data_of_episode.append(s_state_coord[0])
        data_of_episode.append(s_state_coord[1])
        data_of_episode.append(s_prime_coord_and_action_from_s_to_s_prime[2])
        data_of_episode.append(s_prime_coord_and_action_from_s_to_s_prime[0])
        data_of_episode.append(s_prime_coord_and_action_from_s_to_s_prime[1])
        data_of_episode_list.append(data_of_episode)
        
        rewards_plus_qsa_values_ndarray[s_prime_coord_and_action_from_s_to_s_prime[2] + 4, s_state_coord[0],  s_state_coord[1]] += 1  # + 4 in line is for redirecting to the layer that corresponds to n returns collected so far for the corresponding action
        epsilon_0_max_value_1 = 0 # here using SARSA epsilon greedy for qsa and qs'a'
        q_s_prime_a_prime = select_qsa_of_s_prime_by_epsilon_greedy_or_max_qsa(n_states_vertical, n_states_horizontal, gridworld_ndarray, s_prime_coord_and_action_from_s_to_s_prime, rewards_plus_qsa_values_ndarray, epsilon_0_max_value_1)
        k_plus_1_qsa = rewards_plus_qsa_values_ndarray[s_prime_coord_and_action_from_s_to_s_prime[2], s_state_coord[0],  s_state_coord[1]] + alfa * (rewards_plus_qsa_values_ndarray[0, s_prime_coord_and_action_from_s_to_s_prime[0],  s_prime_coord_and_action_from_s_to_s_prime[1]] + gamma * q_s_prime_a_prime - rewards_plus_qsa_values_ndarray[s_prime_coord_and_action_from_s_to_s_prime[2], s_state_coord[0],  s_state_coord[1]])
        k_qsa = rewards_plus_qsa_values_ndarray[s_prime_coord_and_action_from_s_to_s_prime[2], s_state_coord[0],  s_state_coord[1]]
        rewards_plus_qsa_values_ndarray[s_prime_coord_and_action_from_s_to_s_prime[2], s_state_coord[0],  s_state_coord[1]] = k_plus_1_qsa
        # print(gridworld_ndarray)
        # print("\nvisited_states: ", visited_states)
        # print("s_state_coord: ", s_state_coord)
        # print("s_prime_coord_and_action_from_s_to_s_prime: ", s_prime_coord_and_action_from_s_to_s_prime)
        # print("k_qsa: ", k_qsa)
        # print("k_plus_1_qsa: ", k_plus_1_qsa, "\n")
        # print(rewards_plus_qsa_values_ndarray[0:5])
        # print(" ")

        if (s_prime_coord_and_action_from_s_to_s_prime[0:2] == win_terminal_state) or (s_prime_coord_and_action_from_s_to_s_prime[0:2] == loose_terminal_state):
            if s_prime_coord_and_action_from_s_to_s_prime[0:2] == win_terminal_state:
                played_from_start_state_and_reached_terminal_state = True
            break
            
        visited_states.append(s_prime_coord_and_action_from_s_to_s_prime[0:2])
        s_state_coord = s_prime_coord_and_action_from_s_to_s_prime[0:2]
        s_prime_coord_and_action_from_s_to_s_prime, gridworld_ndarray = select_action_from_state_by_epsilon_and_coord_of_next_state(n_states_vertical, n_states_horizontal, gridworld_ndarray, s_state_coord, rewards_plus_qsa_values_ndarray)
        # print("next action: ", s_prime_coord_and_action_from_s_to_s_prime,"\n")
    
    # input("played one episode, press for next move")

    return rewards_plus_qsa_values_ndarray, gridworld_ndarray, played_from_start_state_and_reached_terminal_state, data_of_episode_list

# @njit
def play_from_start_s_save_returns_select_best_q_s_a (rewards_plus_qsa_values_ndarray, gridworld_ndarray, win_terminal_state, loose_terminal_state, starting_sate):
    """
    Control problem of Monte Carlo
    """
    played_from_start_state_and_reached_terminal_state = False
    step = 0

    while played_from_start_state_and_reached_terminal_state == False:
        
        rewards_plus_qsa_values_ndarray, gridworld_ndarray, played_from_start_state_and_reached_terminal_state, data_of_episode_list = play_one_episode(n_states_vertical, n_states_horizontal, gridworld_ndarray, rewards_plus_qsa_values_ndarray, win_terminal_state, loose_terminal_state, starting_sate)

        if step%1 == 0:
            print("step: ", step)
            gridworld_ndarray_for_image = find_way_from_start_to_terminal_state(n_states_vertical, n_states_horizontal, gridworld_ndarray, starting_sate, win_terminal_state)
            grid_world_image_ndarray = create_ndarray_grid_world_image(n_states_vertical, n_states_horizontal, image_size, images_states_ndarray, gridworld_ndarray_for_image)
            save_image_to_file_from_ndarray(grid_world_image_ndarray, "grid_world_images/" + str(step) + ".png")

        if played_from_start_state_and_reached_terminal_state == False:    
            gridworld_ndarray = change_for_improoved_policy_for_every_state_of_grid_world(n_states_vertical, n_states_horizontal, gridworld_ndarray, rewards_plus_qsa_values_ndarray)
        
        step += 1

    gridworld_ndarray_for_image = find_way_from_start_to_terminal_state(n_states_vertical, n_states_horizontal, gridworld_ndarray, starting_sate, win_terminal_state)
    grid_world_image_ndarray = create_ndarray_grid_world_image(n_states_vertical, n_states_horizontal, image_size, images_states_ndarray, gridworld_ndarray_for_image)
    save_image_to_file_from_ndarray(grid_world_image_ndarray, "_grid_world_final_step.png")

    return rewards_plus_qsa_values_ndarray, gridworld_ndarray

@njit
def get_action_from_state_and_coord_of_next_state(n_states_vertical, n_states_horizontal, gridworld_ndarray, state_k_coord):
    """
    get action for the state and next state
    """
    action = gridworld_ndarray[state_k_coord[0], state_k_coord[1]]

    # Action UP
    if action == 1:
        if state_k_coord[0] - 1 >= 0:
            
            coord = List()
            coord.append(state_k_coord[0] - 1)
            coord.append(state_k_coord[1])
            coord.append(action)
            return  coord #((state_k_coord[0] - 1, state_k_coord[1]), action)
    # Action RIGHT
    elif action == 2:
        if state_k_coord[1] + 1 <= n_states_horizontal - 1:
            
            coord = List()
            coord.append(state_k_coord[0])
            coord.append(state_k_coord[1] + 1)
            coord.append(action)
            return coord #((state_k_coord[0], state_k_coord[1] + 1), action)
    # Action DOWN
    if action == 3:
        if state_k_coord[0] + 1 <= n_states_vertical -1 :
            
            coord = List()
            coord.append(state_k_coord[0] + 1)
            coord.append(state_k_coord[1])
            coord.append(action)
            return coord #((state_k_coord[0] + 1, state_k_coord[1]), action)
    # Action LEFT
    elif action == 4:
        if state_k_coord[1] - 1 >= 0:
            
            coord = List()
            coord.append(state_k_coord[0])
            coord.append(state_k_coord[1] - 1)
            coord.append(action)
            return coord #((state_k_coord[0], state_k_coord[1] - 1), action)
    else:
        
        coord = List()
        coord.append(state_k_coord[0])
        coord.append(state_k_coord[1])
        coord.append(action)
        return coord #(state_k_coord, action)

@njit
def find_way_from_start_to_terminal_state(n_states_vertical, n_states_horizontal, gridworld_ndarray, starting_state, win_terminal_state):
    """
    Find way out to the terminal state and save in gridworld numbers of arrows of way out
    """
    gridworld_ndarray_for_print = gridworld_ndarray.copy()
    list_of_visited_s_for_way_out = List()
    not_cycling_not_terminal_state = True
    while not_cycling_not_terminal_state:
        
        best_action_for_state = get_action_from_state_and_coord_of_next_state(n_states_vertical, n_states_horizontal, gridworld_ndarray, starting_state)
        
        for visited_state in list_of_visited_s_for_way_out:
            if visited_state == starting_state:
                not_cycling_not_terminal_state = False
                break
        if (starting_state == win_terminal_state):
            not_cycling_not_terminal_state = False

        if not_cycling_not_terminal_state:
            gridworld_ndarray_for_print[starting_state[0], starting_state[1]] = best_action_for_state[2] + 9
            list_of_visited_s_for_way_out.append(starting_state)
            starting_state = List()
            starting_state.append(best_action_for_state[0])
            starting_state.append(best_action_for_state[1])

    return gridworld_ndarray_for_print

@njit(parallel=True)
def create_ndarray_grid_world_image(n_states_vertical, n_states_horizontal, image_size, images_states_ndarray, gridworld_ndarray):
    """
    make ndarray of image from ndarray gridworld
    representing actions and wall and terminal states:
    1 for up, 2 for right, 3 for down, 4 for left, 5 for wall, 6 for winner terminal state, 7 for losing terminal state, 8 clear state, 9 start state
    10 for up way out, 11 for right way out, 12  for down way out, 13  for left way out
    """
    grid_world_image = np.arange(n_states_vertical*image_size*n_states_horizontal*image_size).reshape((n_states_vertical*image_size, n_states_horizontal*image_size))

    for vert_image in range(n_states_vertical):
        for horizontal_image in range(n_states_horizontal):
            if gridworld_ndarray[vert_image, horizontal_image] == 1:
                grid_world_image[vert_image*image_size : vert_image*image_size + image_size, horizontal_image*image_size : horizontal_image*image_size + image_size] = images_states_ndarray[0, :, :]
            elif gridworld_ndarray[vert_image, horizontal_image] == 2: 
                grid_world_image[vert_image*image_size : vert_image*image_size + image_size, horizontal_image*image_size : horizontal_image*image_size + image_size] = images_states_ndarray[1, :, :]
            elif gridworld_ndarray[vert_image, horizontal_image] == 3: 
                grid_world_image[vert_image*image_size : vert_image*image_size + image_size, horizontal_image*image_size : horizontal_image*image_size + image_size] = images_states_ndarray[2, :, :]
            elif gridworld_ndarray[vert_image, horizontal_image] == 4: 
                grid_world_image[vert_image*image_size : vert_image*image_size + image_size, horizontal_image*image_size : horizontal_image*image_size + image_size] = images_states_ndarray[3, :, :]
            elif gridworld_ndarray[vert_image, horizontal_image] == 5: 
                grid_world_image[vert_image*image_size : vert_image*image_size + image_size, horizontal_image*image_size : horizontal_image*image_size + image_size] = images_states_ndarray[4, :, :]
            elif gridworld_ndarray[vert_image, horizontal_image] == 6: 
                grid_world_image[vert_image*image_size : vert_image*image_size + image_size, horizontal_image*image_size : horizontal_image*image_size + image_size] = images_states_ndarray[5, :, :]
            elif gridworld_ndarray[vert_image, horizontal_image] == 7: 
                grid_world_image[vert_image*image_size : vert_image*image_size + image_size, horizontal_image*image_size : horizontal_image*image_size + image_size] = images_states_ndarray[6, :, :]
            elif gridworld_ndarray[vert_image, horizontal_image] == 8: 
                grid_world_image[vert_image*image_size : vert_image*image_size + image_size, horizontal_image*image_size : horizontal_image*image_size + image_size] = images_states_ndarray[7, :, :]
            elif gridworld_ndarray[vert_image, horizontal_image] == 9: 
                grid_world_image[vert_image*image_size : vert_image*image_size + image_size, horizontal_image*image_size : horizontal_image*image_size + image_size] = images_states_ndarray[8, :, :]
            elif gridworld_ndarray[vert_image, horizontal_image] == 10: 
                grid_world_image[vert_image*image_size : vert_image*image_size + image_size, horizontal_image*image_size : horizontal_image*image_size + image_size] = images_states_ndarray[9, :, :]
            elif gridworld_ndarray[vert_image, horizontal_image] == 11: 
                grid_world_image[vert_image*image_size : vert_image*image_size + image_size, horizontal_image*image_size : horizontal_image*image_size + image_size] = images_states_ndarray[10, :, :]
            elif gridworld_ndarray[vert_image, horizontal_image] == 12: 
                grid_world_image[vert_image*image_size : vert_image*image_size + image_size, horizontal_image*image_size : horizontal_image*image_size + image_size] = images_states_ndarray[11, :, :]
            elif gridworld_ndarray[vert_image, horizontal_image] == 13: 
                grid_world_image[vert_image*image_size : vert_image*image_size + image_size, horizontal_image*image_size : horizontal_image*image_size + image_size] = images_states_ndarray[12, :, :]

    return grid_world_image.astype(np.uint8)

def save_image_to_file_from_ndarray(numpy_ndarray, name_file):
    """
    Take as input an 2D Numpy array of integers between 0 and 255
    Converting  of grayscale values to a PIL image and saves it to .png
    """
    img = Image.fromarray(numpy_ndarray , 'L')
    img.save(name_file)

image_size = 11
n_states_vertical = 15
n_states_horizontal = 15
win_terminal_state = List() # position in gridworld
win_terminal_state.append(0)
win_terminal_state.append(n_states_vertical - 1)
loose_terminal_state = List()  # position in gridworld
loose_terminal_state.append(2)
loose_terminal_state.append(n_states_vertical - 1)
starting_sate = List() # position in gridworld
starting_sate.append(n_states_vertical - 1)
starting_sate.append(0)
win_terminal_state_reward = 100
loose_terminal_state_reward = -100
starting_sate_reward = -100
terminal_states = [win_terminal_state, loose_terminal_state]
reward_of_each_state = -0.01

gridworld_ndarray = create_gridworld_ndarray_adding_walls_and_terminal_states(win_terminal_state, loose_terminal_state, starting_sate)
rewards_plus_qsa_values_ndarray = create_rewards_and_value_states__ndarray(win_terminal_state, win_terminal_state_reward, loose_terminal_state, loose_terminal_state_reward, starting_sate, starting_sate_reward, reward_of_each_state)
images_states_ndarray = make_arrows_images_as_2d_x_n_numpy_array([r"arrow_up.png", r"arrow_right.png", r"arrow_down.png", r"arrow_left.png", r"wall.png", r"terminal_gain.png", r"terminal_loose.png", r"clear_state.png", r"start_state.png", r"arrow_up_final.png", r"arrow_right_final.png", r"arrow_down_final.png", r"arrow_left_final.png"])
rewards_plus_qsa_values_ndarray, gridworld_ndarray = play_from_start_s_save_returns_select_best_q_s_a (rewards_plus_qsa_values_ndarray, gridworld_ndarray, win_terminal_state, loose_terminal_state, starting_sate)
