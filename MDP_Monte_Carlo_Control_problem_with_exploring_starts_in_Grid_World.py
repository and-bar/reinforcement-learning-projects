# %%
import numpy as np
from PIL import Image
from numba import njit
from numpy import random

def save_image_to_file_from_ndarray(numpy_ndarray, name_file):
    """
    Take as input an 2D Numpy array of integers between 0 and 255
    Converting  of grayscale values to a PIL image and saves it to .png
    """
    img = Image.fromarray(numpy_ndarray , 'L')
    # img = Image.fromarray(np.uint8(numpy_ndarray * 255) , 'L')
    img.save(name_file)

@njit(parallel=True)
def create_ndarray_grid_world_image(n_states_vertical, n_states_horizontal, image_size, images_states_ndarray, gridworld_ndarray):
    """
    make ndarray of image from ndarray gridworld
    representing actions and wall and terminal states:
    1 for up, 2 for right, 3 for down, 4 for left, 5 for wall, 6 for winner terminal state, 7 for losing terminal state, 8 clear state, 9 start state
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

    return grid_world_image.astype(np.uint8)

def make_arrows_images_as_2d_x_n_numpy_array(images_names_list):
    """
    Read image from file and convert it as grayscale and save to numpy ndarray as layer one of 7 each dimention represent one image of grayscale
    """
    ndarray_of_images = np.zeros([9, 11, 11])
    image_layer = 0
    for name in images_names_list:
        ndarray_of_images[image_layer, :, :] =  (np.array(Image.open(name).convert('L').getdata())).reshape([11,11]) # here 11 is a image size for arroes and walls
        image_layer += 1
    return ndarray_of_images

@njit(parallel=True)
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
    gridworld_ndarray[win_terminal_state] = 6
    gridworld_ndarray[loose_terminal_state] = 7
    gridworld_ndarray[starting_sate] = 9

    gridworld_ndarray[win_terminal_state[0], win_terminal_state[1] - 1] = 8 # clear left state from wall of terminal state
    gridworld_ndarray[win_terminal_state[0] + 1, win_terminal_state[1]] = 8 # clear down state from wall of terminal state
    gridworld_ndarray[loose_terminal_state[0], loose_terminal_state[1] - 1] = 8 # clear left state from wall of terminal state
    gridworld_ndarray[loose_terminal_state[0] + 1, loose_terminal_state[1]] = 8 # clear down state from wall of terminal state
    gridworld_ndarray[loose_terminal_state[0] - 1, loose_terminal_state[1]] = 8 # clear down state from wall of terminal state
    gridworld_ndarray[starting_sate[0] - 1, starting_sate[1]] = 8 # clear down state from wall of starting state
    gridworld_ndarray[starting_sate[0], starting_sate[1] + 1] = 8 # clear down state from wall of starting state

    return gridworld_ndarray

@njit
def fill_grid_world_with_action_of_states(gridworld_ndarray):
    """
    fill states of gridworld with random actions: (1, 2, 3, 4)
    """
    for vert_cell in range(n_states_vertical):
        for horiz_cell in range(n_states_horizontal):
            if gridworld_ndarray[vert_cell, horiz_cell] == 8: #cell do not contain a wall
                gridworld_ndarray[vert_cell, horiz_cell] = random.randint(1,5)
    return gridworld_ndarray

def create_rewards_and_value_states__ndarray(win_terminal_state, win_terminal_state_reward, loose_terminal_state, loose_terminal_state_reward):
    """
    make 3d array with 5 layers, zero layer for rewards, 1st for Q(s,a) action 'up', 2nd for Q(s,a) action 'right',  3d for Q(s,a) action 'down', 4nd for Q(s,a) action 'left',
    5fth for number of returns collected for action 'up', 6th for number of returns collected for action 'right', 7th for number of returns collected for action 'down',
    8th for number of returns collected for action 'left'
    """
    rewards_plus_states_values_ndarray = (np.arange(9 * n_states_vertical * n_states_horizontal)).reshape((9, n_states_vertical, n_states_horizontal))
    rewards_plus_states_values_ndarray[:,:,:] = 0 # instantiating with zeros for layers that contain n collected returns so far for actions and the rest of layers
    rewards_plus_states_values_ndarray[0,win_terminal_state[0], win_terminal_state[1]] = win_terminal_state_reward
    rewards_plus_states_values_ndarray[0,loose_terminal_state[0], loose_terminal_state[1]] = loose_terminal_state_reward
    return rewards_plus_states_values_ndarray

@njit
def find_all_actions_of_the_state(n_states_vertical, n_states_horizontal, tuple_state_for_searching, grid_world_ndarray):
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
    vertical_coord = tuple_state_for_searching[0] - 1    
    # Action UP
    if (vertical_coord >= 0):
        if grid_world_ndarray[vertical_coord, tuple_state_for_searching[1]] in (1, 2, 3, 4, 6, 7):
            list_of_tuple_actions.append(((vertical_coord, tuple_state_for_searching[1]), 1))
    # Action RIGHT
    horizontal_coord = tuple_state_for_searching[1] + 1    
    if (horizontal_coord <= n_states_horizontal - 1):
        if grid_world_ndarray[tuple_state_for_searching[0], horizontal_coord] in (1, 2, 3, 4, 6, 7):
            list_of_tuple_actions.append(((tuple_state_for_searching[0], horizontal_coord), 2))
    # Action DOWN
    vertical_coord = tuple_state_for_searching[0] + 1    
    if (vertical_coord <= n_states_vertical - 1):
        if grid_world_ndarray[vertical_coord, tuple_state_for_searching[1]] in (1, 2, 3, 4, 6, 7):
            list_of_tuple_actions.append(((vertical_coord, tuple_state_for_searching[1]), 3))
    # Action LEFT
    horizontal_coord = tuple_state_for_searching[1] - 1    
    if (horizontal_coord >= 0):
        if grid_world_ndarray[tuple_state_for_searching[0], horizontal_coord] in (1, 2, 3, 4, 6, 7):
            list_of_tuple_actions.append(((tuple_state_for_searching[0], horizontal_coord), 4))

    return list_of_tuple_actions

@njit
def get_action_from_state_and_next_state(n_states_vertical, n_states_horizontal, gridworld_ndarray, state_k_coord):
    """
    get action for the state and next state
    """
    action = gridworld_ndarray[state_k_coord]

    # Action UP
    if action == 1:
        if state_k_coord[0] - 1 >= 0:
            return ((state_k_coord[0] - 1, state_k_coord[1]), action)
    # Action RIGHT
    elif action == 2:
        if state_k_coord[1] + 1 <= n_states_horizontal - 1:
            return ((state_k_coord[0], state_k_coord[1] + 1), action)
    # Action DOWN
    if action == 3:
        if state_k_coord[0] + 1 <= n_states_vertical -1 :
            return ((state_k_coord[0] + 1, state_k_coord[1]), action)
    # Action LEFT
    elif action == 4:
        if state_k_coord[1] - 1 >= 0:
            return ((state_k_coord[0], state_k_coord[1] - 1), action)
    else:
        return (state_k_coord, action)

def change_for_improoved_policy_for_every_state_of_grid_world(gridworld_ndarray, coord_of_states, rewards_plus_qsa_values_ndarray):
    """
    look up for Q* for every state and change it for corresponded a* in gridworld
    """
    for tuple_state_for_best_q_star in coord_of_states:
        list_of_tuple_actions = find_all_actions_of_the_state(n_states_vertical, n_states_horizontal, tuple_state_for_best_q_star, gridworld_ndarray)
        # print(f"list_of_tuple_actions : {list_of_tuple_actions}")
        # structure of list_of_tuple_actions : [((0, 2), 2), ((1, 1), 3)] action and its s'
        if len(list_of_tuple_actions) == 0:
            continue
        best_action = list_of_tuple_actions[0][1]
        best_qsa = rewards_plus_qsa_values_ndarray[list_of_tuple_actions[0][1], list_of_tuple_actions[0][0][0], list_of_tuple_actions[0][0][1]]
        # print(f"best_action : {best_action} best_qsa : {best_qsa}")
        list_of_tuple_actions.remove(list_of_tuple_actions[0])
        # print(f"removed firs action from all posible actions and it left: {list_of_tuple_actions}")
        for action in list_of_tuple_actions:
            if rewards_plus_qsa_values_ndarray[action[1], action[0][0], action[0][1]] > best_qsa:
                best_qsa = rewards_plus_qsa_values_ndarray[action[1], action[0][0], action[0][1]]
                best_action = action[1]
            # print(f"final best action: {best_action} and best qsa : {best_qsa}")
        gridworld_ndarray[tuple_state_for_best_q_star] = best_action

    return gridworld_ndarray
        
def play_from_random_s_a_save_returns_select_best_q_s_a (n_states_vertical, n_states_horizontal, rewards_plus_qsa_values_ndarray, gridworld_ndarray, terminal_states):
    """
    Control problem of Monte Carlo
    """
    gamma = 0.9
    coord_of_states = []
    for vert_state in range(n_states_vertical):
        for horiz_state in range(n_states_horizontal):
            if gridworld_ndarray[vert_state, horiz_state] in (1,2,3,4):
                coord_of_states.append((vert_state, horiz_state))
    
    for step in range(10000):
        
        state_k_coord = coord_of_states[random.randint(0, len(coord_of_states))] # get random state from list of all possible states
        all_actions_of_the_state = find_all_actions_of_the_state(n_states_vertical, n_states_horizontal, state_k_coord, gridworld_ndarray)
        action_of_state = all_actions_of_the_state[random.randint(0, len(all_actions_of_the_state))] # get random action from starting state
        data_of_episode_tuple = []
        terminal_state_bool = False
        visited_states = []
        visited_states.append(state_k_coord)
        
        while (not terminal_state_bool) and (action_of_state != None):
            if  (action_of_state[0] in visited_states) or (gridworld_ndarray[action_of_state[0]] == 5): # visited states or wall
                terminal_state_bool = True
            elif action_of_state[0] in terminal_states:
                data_of_episode_tuple.append((state_k_coord, action_of_state[1], action_of_state[0]))
                terminal_state_bool = True
            else:
                data_of_episode_tuple.append((state_k_coord, action_of_state[1], action_of_state[0]))
                visited_states.append(action_of_state[0])
                state_k_coord = action_of_state[0]
                action_of_state = get_action_from_state_and_next_state(n_states_vertical, n_states_horizontal, gridworld_ndarray, state_k_coord)
        g_return = 0        
        
        for state in data_of_episode_tuple[::-1]:
            g_return += rewards_plus_qsa_values_ndarray[0, state[2][0],  state[2][1]] + gamma * g_return
            rewards_plus_qsa_values_ndarray[state[1] + 4, state[0][0],  state[0][1]] += 1  # + 4 in line is for redirecting to the layer that corresponds to n returns collected so far for the action
            rewards_plus_qsa_values_ndarray[state[1], state[0][0],  state[0][1]] = rewards_plus_qsa_values_ndarray[state[1], state[0][0],  state[0][1]] + (g_return - rewards_plus_qsa_values_ndarray[state[1], state[0][0],  state[0][1]]) / rewards_plus_qsa_values_ndarray[state[1] + 4, state[0][0],  state[0][1]]
        gridworld_ndarray = change_for_improoved_policy_for_every_state_of_grid_world(gridworld_ndarray, coord_of_states, rewards_plus_qsa_values_ndarray)
        
        if step%100 == 0:
            grid_world_image_ndarray = create_ndarray_grid_world_image(n_states_vertical, n_states_horizontal, image_size, images_states_ndarray, gridworld_ndarray)
            save_image_to_file_from_ndarray(grid_world_image_ndarray, "_grid_world_"+ str(step) +".png")

    print(" show rewards of states ")
    print(rewards_plus_qsa_values_ndarray[0])
    print(f"qsa of up ")
    print(rewards_plus_qsa_values_ndarray[1])
    print(f"qsa of right ")
    print(rewards_plus_qsa_values_ndarray[2])
    print(f"qsa of down ")
    print(rewards_plus_qsa_values_ndarray[3])
    print(f"qsa of left ")
    print(rewards_plus_qsa_values_ndarray[4])
    print(f"teturns n collected for qsa up ")
    print(rewards_plus_qsa_values_ndarray[1+4])
    print(f"teturns n collected for qsa right ")
    print(rewards_plus_qsa_values_ndarray[2+4])
    print(f"teturns n collected for qsa down ")
    print(rewards_plus_qsa_values_ndarray[3+4])
    print(f"teturns n collected for qsa left ")
    print(rewards_plus_qsa_values_ndarray[4+4])

    return rewards_plus_qsa_values_ndarray, gridworld_ndarray

image_size = 11
n_states_vertical = 6
n_states_horizontal = 6
win_terminal_state = (0, n_states_vertical - 1) # position in gridworld
loose_terminal_state = (4, n_states_vertical - 1) # position in gridworld
starting_sate = (n_states_vertical - 1, 0) # position in gridworld
win_terminal_state_reward = 100
loose_terminal_state_reward = -100
terminal_states = (win_terminal_state, loose_terminal_state)

gridworld_ndarray = create_gridworld_ndarray_adding_walls_and_terminal_states(win_terminal_state, loose_terminal_state, starting_sate)
gridworld_ndarray = fill_grid_world_with_action_of_states(gridworld_ndarray)
rewards_plus_qsa_values_ndarray = create_rewards_and_value_states__ndarray(win_terminal_state, win_terminal_state_reward, loose_terminal_state, loose_terminal_state_reward)

# images_states_ndarray = make_arrows_images_as_2d_x_n_numpy_array([r"arrow_up.png", r"arrow_right.png", r"arrow_down.png", r"arrow_left.png", r"wall.png", r"terminal_gain.png", r"terminal_loose.png", r"clear_state.png", r"start_state.png"])
# grid_world_image_ndarray = create_ndarray_grid_world_image(n_states_vertical, n_states_horizontal, image_size, images_states_ndarray, gridworld_ndarray)
# save_image_to_file_from_ndarray(grid_world_image_ndarray, "_grid_world_00.png")

rewards_plus_qsa_values_ndarray, gridworld_ndarray = play_from_random_s_a_save_returns_select_best_q_s_a (n_states_vertical, n_states_horizontal, rewards_plus_qsa_values_ndarray, gridworld_ndarray, terminal_states)

grid_world_image_ndarray = create_ndarray_grid_world_image(n_states_vertical, n_states_horizontal, image_size, images_states_ndarray, gridworld_ndarray)
save_image_to_file_from_ndarray(grid_world_image_ndarray, "_grid_world_final.png")

# %%
