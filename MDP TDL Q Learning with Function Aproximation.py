"""
Implementation of Temporal Different Leraning algorithm with Q-Learning control with function approximation in GridWorld in MDP framework
"""
import numpy as np
from PIL import Image
from numba import njit
from numba.typed import List
from numpy import random
from numpy.lib.function_base import append
from sklearn.kernel_approximation import RBFSampler

def create_gridworld_ndarray_adding_walls_and_terminal_states(win_terminal_state, loose_terminal_state, starting_sate, n_states_vertical, n_states_horizontal):
  """
  make numpy 2D array adding walls and terminal states
  representing actions and wall and terminal states:
  1 for up, 2 for right, 3 for down, 4 for left, 5 for wall, 6 for winner terminal state, 7 for loosing terminal state, 8 clear state, 9 start state
  
  """
  gridworld_ndarray = (np.arange(n_states_vertical * n_states_horizontal)).reshape((n_states_vertical, n_states_horizontal))
  gridworld_ndarray[:, :] = 8

  n_of_walls_total = (n_states_vertical*n_states_horizontal) * 0.2 # here 0.2 is 20% of total occupancy by walls in gridworld
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

def find_all_states_of_gridworld (n_states_vertical, n_states_horizontal, gridworld_ndarray):
  all_states_of_gridworld = List()
  for vertical_coordinate in range(n_states_vertical):
    for horizontal_coordinate in range(n_states_horizontal):
      if gridworld_ndarray[vertical_coordinate, horizontal_coordinate] == 8:
        all_possible_actions_from_s = [[vertical_coordinate, horizontal_coordinate, action] for action in [action[2] for action in find_all_actions_of_the_state(n_states_vertical, n_states_horizontal, [vertical_coordinate, horizontal_coordinate], gridworld_ndarray)]]
        
        all_states_of_gridworld.extend([create_typed_numba_list(*state_a) for state_a in all_possible_actions_from_s])
  return all_states_of_gridworld

def create_rewards_ndarray(win_terminal_state, win_terminal_state_reward, loose_terminal_state, loose_terminal_state_reward, starting_sate, starting_sate_reward, 
                            reward_of_each_state, n_states_vertical, n_states_horizontal):
  """
  make ndarray for rewards
  """
  rewards_values_ndarray = (np.arange(n_states_vertical * n_states_horizontal, dtype = np.float32)).reshape((n_states_vertical, n_states_horizontal)) 
  rewards_values_ndarray[:] = reward_of_each_state
  rewards_values_ndarray[win_terminal_state[0], win_terminal_state[1]] = win_terminal_state_reward
  rewards_values_ndarray[loose_terminal_state[0], loose_terminal_state[1]] = loose_terminal_state_reward
  rewards_values_ndarray[starting_sate[0], starting_sate[1]] = starting_sate_reward

  return rewards_values_ndarray

@njit
def create_typed_numba_list (*input_list):
  """
  Takes as input list of elements and return typed numba list
  """
  list_numba_typed = List()
  for element in input_list:
    list_numba_typed.append(element)
  
  return list_numba_typed

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
  
  Structure to return like: [[s'y, s'x, action], [s'y, s'x, action]]  or 
  [[3, 0, 1], [4, 1, 2]] 
  
  """
  list_of_actions = List()
  # Action UP
  vertical_coord = state_for_searching[0] - 1    
  if (vertical_coord >= 0):
      if grid_world_ndarray[vertical_coord, state_for_searching[1]] in (1, 2, 3, 4, 6, 7, 8):
          
          list_of_actions.append(create_typed_numba_list(vertical_coord, state_for_searching[1], 1))
  # Action RIGHT
  horizontal_coord = state_for_searching[1] + 1    
  if (horizontal_coord <= n_states_horizontal - 1):
      if grid_world_ndarray[state_for_searching[0], horizontal_coord] in (1, 2, 3, 4, 6, 7, 8):
          
          list_of_actions.append(create_typed_numba_list(state_for_searching[0], horizontal_coord, 2))
          
  # Action DOWN
  vertical_coord = state_for_searching[0] + 1    
  if (vertical_coord <= n_states_vertical - 1):
      if grid_world_ndarray[vertical_coord, state_for_searching[1]] in (1, 2, 3, 4, 6, 7, 8):
          
          list_of_actions.append(create_typed_numba_list(vertical_coord, state_for_searching[1], 3))
          
  # Action LEFT
  horizontal_coord = state_for_searching[1] - 1    
  if (horizontal_coord >= 0):
      if grid_world_ndarray[state_for_searching[0], horizontal_coord] in (1, 2, 3, 4, 6, 7,8):
          
          list_of_actions.append(create_typed_numba_list(state_for_searching[0], horizontal_coord, 4))
          
  return list_of_actions

def make_arrows_images_as_2d_x_n_numpy_array(images_names_list):
    """
    Read image from file and convert it as grayscale and save to numpy ndarray as layer one of 7 each dimention represent one image of grayscale
    """
    ndarray_of_images = np.zeros([13, 11, 11])
    image_layer = 0
    for name in images_names_list:
        ndarray_of_images[image_layer, :, :] =  (np.array(Image.open(name).convert('L').getdata())).reshape([11,11]) 
        # here 11 is a image size for arroes and walls
        image_layer += 1
    return ndarray_of_images

def play_from_start_s_save_update_weights_of_the_model (rewards_ndarray, gridworld_ndarray, win_terminal_state, loose_terminal_state,
                                                        starting_sate, steps_of_episode, weights_of_the_model, rbfsample_model):
  """
  Control problem
  """
  played_from_start_state_and_reached_terminal_state = False
  step = 0

  while played_from_start_state_and_reached_terminal_state == False:
      
      weights_of_the_model, gridworld_ndarray, played_from_start_state_and_reached_terminal_state = play_one_episode(n_states_vertical, n_states_horizontal,
                            gridworld_ndarray, rewards_ndarray, win_terminal_state, loose_terminal_state, starting_sate, steps_of_episode, weights_of_the_model, rbfsample_model)
      
      step += 1

  return weights_of_the_model, gridworld_ndarray

def predicted_return_of_qsa_of_greedy_a_and_coord_of_s_prime(n_states_vertical, n_states_horizontal, gridworld_ndarray, s_state_coord, previous_state_coord_of_s, weights_of_the_model, rbfsample_model):
  """
  get action by epsilon for the state and coordinates of next state
  
  returns:  coordinates of s' and action from s to s' structure [s'y, s'x, action from s]
            predicted return of greedy qsa
  
  """
  list_all_possible_actions_from_the_state = find_all_actions_of_the_state(n_states_vertical, n_states_horizontal, s_state_coord, gridworld_ndarray)
  
  if len(list_all_possible_actions_from_the_state) > 1:
    # removing from all possible states, state previous of s if exist
    for action in list_all_possible_actions_from_the_state:
      if action[0:2] == previous_state_coord_of_s:
        list_all_possible_actions_from_the_state.remove(action)
        break
    if len(list_all_possible_actions_from_the_state) > 1:
      qsa_of_every_a = [[np.dot(weights_of_the_model.T, rbfsample_model.transform(np.array([*s_state_coord, action[2]]).reshape(1, -1))[0]), *action] for action in list_all_possible_actions_from_the_state]
      epsilon = 0.15
      if random.random() < epsilon:
        # select random action
        random_one_from_qsa_of_every_a = qsa_of_every_a[np.random.choice(len(qsa_of_every_a))]
        predicted_return_of_greedy_qsa = random_one_from_qsa_of_every_a[0]
        coord_s_prime_a = random_one_from_qsa_of_every_a[1:]
      else:
        # select best action from state wich lead to Q*(s,a)
        index_best_one_from_qsa_of_every_a = np.argmax(qsa[0] for qsa in qsa_of_every_a)
        predicted_return_of_greedy_qsa = qsa_of_every_a[index_best_one_from_qsa_of_every_a][0]
        coord_s_prime_a = qsa_of_every_a[index_best_one_from_qsa_of_every_a][1:]
    else:
      predicted_return_of_greedy_qsa = np.dot(weights_of_the_model.T, rbfsample_model.transform( np.array([ *s_state_coord, list_all_possible_actions_from_the_state[0][2] ]).reshape(1, -1) )[0])
      coord_s_prime_a = list_all_possible_actions_from_the_state[0]    
  
  elif len(list_all_possible_actions_from_the_state) == 1:
    predicted_return_of_greedy_qsa = np.dot(weights_of_the_model.T, rbfsample_model.transform( np.array([ *s_state_coord, list_all_possible_actions_from_the_state[0][2] ]).reshape(1, -1) )[0])
    coord_s_prime_a = list_all_possible_actions_from_the_state[0]

  return coord_s_prime_a, predicted_return_of_greedy_qsa

def play_one_episode(n_states_vertical, n_states_horizontal, gridworld_ndarray, rewards_ndarray, win_terminal_state, loose_terminal_state, s_state_coord,
                    steps_of_episode, weights_of_the_model, rbfsample_model):
  """
  play one episode of the game, update weights
  s_prime_coord_and_action_from_s_to_s_prime -> structure : [coord, coord, action]
  s_state_coord -> structure : [coord, coord]

  Every next state can step back to the previous state if its not unique step
  """
  gamma = 0.9
  alfa = 0.001
  played_from_start_state_and_reached_terminal_state = False
  visited_states = List()
  previous_state_coord_of_s = s_state_coord
  for step_of_loop in range(steps_of_episode):
    
    s_prime_coord_and_action_from_s_to_s_prime, predicted_return_of_greedy_qsa = predicted_return_of_qsa_of_greedy_a_and_coord_of_s_prime(n_states_vertical, n_states_horizontal,
                                                                    gridworld_ndarray, s_state_coord, previous_state_coord_of_s, weights_of_the_model, rbfsample_model)
    s_a_greedy = create_typed_numba_list(*s_state_coord, s_prime_coord_and_action_from_s_to_s_prime[2])
    
    s_prime_coordin = create_typed_numba_list(*s_prime_coord_and_action_from_s_to_s_prime[0:2]) # equaling types for next lines
    if ((s_a_greedy not in [s_a_visited for s_a_visited in visited_states]) and (  s_prime_coordin != win_terminal_state) and ( s_prime_coordin != loose_terminal_state )):
      
      q_s_prime_a_prime_of_every_a_prime = [[np.dot(weights_of_the_model.T, rbfsample_model.transform( np.array([ *s_state_coord[0:2], action[2] ]).reshape(1, -1) )[0])] for action in find_all_actions_of_the_state(n_states_vertical, n_states_horizontal, s_prime_coord_and_action_from_s_to_s_prime[:2], gridworld_ndarray)]
      true_value_y = rewards_ndarray[s_prime_coord_and_action_from_s_to_s_prime[0], s_prime_coord_and_action_from_s_to_s_prime[1]] + gamma * np.max(q_s_prime_a_prime_of_every_a_prime)
      weights_of_the_model = weights_of_the_model + alfa * (true_value_y - predicted_return_of_greedy_qsa) * rbfsample_model.transform( np.array(s_a_greedy).reshape(1, -1) )
      weights_of_the_model = weights_of_the_model.reshape(weights_of_the_model.shape[1])
      visited_states.append(create_typed_numba_list(s_a_greedy))
      
      gridworld_ndarray[s_a_greedy[0], s_a_greedy[1]] = s_a_greedy[2]
      
      # grid_world_image_ndarray = create_ndarray_grid_world_image(n_states_vertical, n_states_horizontal, image_size, images_states_ndarray, gridworld_ndarray)
      # save_image_to_file_from_ndarray(grid_world_image_ndarray, "grid_world_images/"+ str(step_of_loop) +" grid_image_next_step.png")
      
      previous_state_coord_of_s = s_a_greedy[:2]
      s_state_coord = s_prime_coord_and_action_from_s_to_s_prime[0:2]
      
    
    if (s_prime_coordin == win_terminal_state) or (s_prime_coordin == loose_terminal_state):
      true_value_y = rewards_ndarray[s_prime_coord_and_action_from_s_to_s_prime[0], s_prime_coord_and_action_from_s_to_s_prime[1]]
      weights_of_the_model = weights_of_the_model + alfa * (true_value_y - predicted_return_of_greedy_qsa) * rbfsample_model.transform( np.array(s_a_greedy).reshape(1, -1) )
      weights_of_the_model = weights_of_the_model.reshape(weights_of_the_model.shape[1])

      gridworld_ndarray[s_a_greedy[0], s_a_greedy[1]] = s_a_greedy[2]
      
      if s_prime_coordin == win_terminal_state:
        played_from_start_state_and_reached_terminal_state = True
      break

  return weights_of_the_model, gridworld_ndarray,  played_from_start_state_and_reached_terminal_state

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
n_states_vertical = 10
n_states_horizontal = 10
win_terminal_state = create_typed_numba_list(0, n_states_horizontal - 1)  # position in gridworld
loose_terminal_state = create_typed_numba_list(2, n_states_horizontal - 1)  # position in gridworld
starting_sate = create_typed_numba_list(n_states_vertical - 1, 0)  # position in gridworld
win_terminal_state_reward = 100
loose_terminal_state_reward = -100
starting_sate_reward = -100
terminal_states = [win_terminal_state, loose_terminal_state]
reward_of_each_state = -0.001
steps_of_episode = 1000 #for the second method here how many action of every next state will accomplish untill stop

gridworld_ndarray = create_gridworld_ndarray_adding_walls_and_terminal_states(win_terminal_state, loose_terminal_state,
                                                                                starting_sate, n_states_vertical, n_states_horizontal)
all_states_of_gridworld = find_all_states_of_gridworld (n_states_vertical, n_states_horizontal, gridworld_ndarray)
rbfsample_model = RBFSampler()
rbfsample_model.fit(all_states_of_gridworld)
weights_of_the_model = np.zeros(rbfsample_model.n_components)
rewards_ndarray = create_rewards_ndarray(win_terminal_state, win_terminal_state_reward, loose_terminal_state, loose_terminal_state_reward,
                                          starting_sate, starting_sate_reward, reward_of_each_state, n_states_vertical, n_states_horizontal)
images_states_ndarray = make_arrows_images_as_2d_x_n_numpy_array([r"arrow_up.png", r"arrow_right.png", r"arrow_down.png", r"arrow_left.png", r"wall.png",
                                                                  r"terminal_gain.png", r"terminal_loose.png", r"clear_state.png", r"start_state.png",
                                                                  r"arrow_up_final.png", r"arrow_right_final.png", r"arrow_down_final.png",
                                                                  r"arrow_left_final.png"])
for n_of_intent_of_finding_solution in range(1, 1000):
    weights_of_the_model, gridworld_ndarray = play_from_start_s_save_update_weights_of_the_model (rewards_ndarray,
      gridworld_ndarray, win_terminal_state, loose_terminal_state, starting_sate, steps_of_episode, weights_of_the_model, rbfsample_model)
    
    if n_of_intent_of_finding_solution%100 == 0:
        print(f"n_of_intent_of_finding_solution: {n_of_intent_of_finding_solution}")
        print(weights_of_the_model)
        
        grid_world_image_ndarray = create_ndarray_grid_world_image(n_states_vertical, n_states_horizontal, image_size, images_states_ndarray, gridworld_ndarray)
        save_image_to_file_from_ndarray(grid_world_image_ndarray, "grid_world_images/"+ str(n_of_intent_of_finding_solution) +" grid_image.png")
