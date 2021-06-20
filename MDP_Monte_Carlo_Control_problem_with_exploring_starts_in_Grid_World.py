# %%
import numpy as np
from PIL import Image
from numba import njit
# from numpy.core.numeric import ones
from numpy import random

# %%
def save_image_to_file_from_ndarray(numpy_ndarray, name_file):
    """
    Take as input an 2D Numpy array of integers between 0 and 255
    Converting  of grayscale values to a PIL image and saves it to .png
    """
    img = Image.fromarray(numpy_ndarray , 'L')
    # img = Image.fromarray(np.uint8(numpy_ndarray * 255) , 'L')
    img.save(name_file)

@njit(parallel=True)
def create_ndarray_grid_world_image(n_images_vertical, n_images_horizontal, image_size, images_states_ndarray, gridworld_ndarray):
    """
    make ndarray of image from ndarray gridworld
    representing actions and wall and terminal states:
    1 for up, 2 for right, 3 for down, 4 for left, 5 for wall, 6 for winner terminal state, 7 for losing terminal state, 8 clear state
    """
    grid_world_image = np.arange(n_images_vertical*image_size*n_images_horizontal*image_size).reshape((n_images_vertical*image_size, n_images_horizontal*image_size))

    for vert_image in range(n_images_vertical):
        for horizontal_image in range(n_images_horizontal):
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
    return grid_world_image.astype(np.uint8)

def make_arrows_images_as_2d_x_n_numpy_array(images_names_list):
    """
    Read image from file and convert it as grayscale and save to numpy ndarray as layer one of 7 each dimention represent one image of grayscale
    """
    ndarray_of_images = np.zeros([8, 11, 11])
    image_layer = 0
    for name in images_names_list:
        ndarray_of_images[image_layer, :, :] =  (np.array(Image.open(name).convert('L').getdata())).reshape([11,11]) # here 11 is a image size for arroes and walls
        image_layer += 1
    return ndarray_of_images

@njit(parallel=True)
def create_gridworld_ndarray_adding_walls_and_terminal_states(win_terminal_state, loose_terminal_state):
    """
    make numpy 2D array adding walls and terminal states
    representing actions and wall and terminal states:
    1 for up, 2 for right, 3 for down, 4 for left, 5 for wall, 6 for winner terminal state, 7 for losing terminal state, 8 clear state
    """
    gridworld_ndarray = (np.arange(n_images_vertical * n_images_horizontal)).reshape((n_images_vertical, n_images_horizontal))
    gridworld_ndarray[:, :] = 8

    n_of_walls_total = (n_images_vertical*n_images_horizontal)*0.2 # here 0.2 is 20% of total occupancy by walls in gridworld
    n_of_walls = 0
    while n_of_walls <= n_of_walls_total:
        vert_coord = random.randint(0, n_images_vertical - 1)
        hor_coord = random.randint(0, n_images_horizontal - 1)
        if gridworld_ndarray[vert_coord, hor_coord] != 5:
            gridworld_ndarray[vert_coord, hor_coord] = 5
            n_of_walls+=1
    gridworld_ndarray[win_terminal_state] = 6
    gridworld_ndarray[loose_terminal_state] = 7
    return gridworld_ndarray

# %%
@njit
def fill_grid_world_with_random_actions(gridworld_ndarray):
    """
    fill states of gridworld with random actions: (1, 2, 3, 4)
    """
    for vert_cell in range(n_images_vertical):
        for horiz_cell in range(n_images_horizontal):
            if gridworld_ndarray[vert_cell, horiz_cell] == 8: #cell do not contain a wall
                gridworld_ndarray[vert_cell, horiz_cell] = random.randint(1,5)
    return gridworld_ndarray

# %%
image_size = 11
n_images_vertical = 10
n_images_horizontal = 10
win_terminal_state = (0, n_images_vertical - 1)
loose_terminal_state = (9, n_images_vertical - 1)

# %%
gridworld_ndarray = create_gridworld_ndarray_adding_walls_and_terminal_states(win_terminal_state, loose_terminal_state)
gridworld_ndarray = fill_grid_world_with_random_actions(gridworld_ndarray)



# %%
images_states_ndarray = make_arrows_images_as_2d_x_n_numpy_array([r"arrow_up.png", r"arrow_right.png", r"arrow_down.png", r"arrow_left.png", r"wall.png", r"terminal_gain.png", r"terminal_loose.png", r"clear_state.png"])
grid_world_image_ndarray = create_ndarray_grid_world_image(n_images_vertical, n_images_horizontal, image_size, images_states_ndarray, gridworld_ndarray)
save_image_to_file_from_ndarray(grid_world_image_ndarray, "_grid_world.png")

# %%
