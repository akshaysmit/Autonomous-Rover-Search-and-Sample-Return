import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# This is used to detect rocks, applies a threshold above and below
def color_squeeze(img, rgb_d, rgb_u):
    # Create an empty array the same size in x and y as the image                                                                                                               
    # but just a single channel                                                                                                                                                 
    color_select = np.zeros_like(img[:,:,0])
                                                                                                                                       
    idx = (img[:,:,0] >= rgb_d[0]) & (img[:,:,0] <= rgb_u[0]) & (img[:,:,1] >= rgb_d[1]) & (img[:,:,1] <= rgb_u[1]) & \
    (img[:,:,2] >= rgb_d[2]) & (img[:,:,2] <= rgb_u[2])
    color_select[idx] = 1
    return color_select


# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img[:,:,0]), M, (img.shape[1], img.shape[0]))
    
    return warped, mask


# Define a function to cut off robot vision by range, default is 5m
def range_limit(xpix, ypix, range=50):
    dist = np.sqrt(xpix**2 + ypix**2)
    return xpix[dist < range], ypix[dist < range]

def perception_step(Rover):
    # Perform perception steps to update Rover()
    
    # NOTE: camera image is coming in Rover.img

    img = Rover.img
    
    # 1) Define source and destination points for perspective transform
    dst_size = 5
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                  [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                  ])

    # 2) Apply perspective transform
    warped, mask = perspect_transform(img, source, destination)

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    red_down = 120
    green_down = 110
    blue_down = 0
    rgb_down = (red_down, green_down, blue_down)

    red_up = 255
    green_up = 255
    blue_up = 50
    rgb_up = (red_up, green_up, blue_up)
    
    threshed = color_thresh(warped)
    squeezed = color_squeeze(warped, rgb_down, rgb_up)
    
    obs_map = np.absolute(np.float32(threshed)-1) * mask
    
    # 4) Update Rover.vision image displayed on left side of screen
    Rover.vision_image[:,:,2] = threshed * 255
    Rover.vision_image[:,:,0] = obs_map * 255
    
    # 5) Convert thresholded image pixel values to rover-centric coords and cut off 5m
    xpix, ypix = rover_coords(threshed)
    xobs, yobs = rover_coords(obs_map)

    Rover.nav_x = xpix
    Rover.nav_y = ypix
    
    xpix, ypix = range_limit(xpix, ypix)
    xobs, yobs = range_limit(xobs, yobs)

    # 6) Convert rover-centric pixel values to world coords
    scale = 2*dst_size
    world_size = Rover.worldmap.shape[0]
    
    navigable_x_world, navigable_y_world = pix_to_world(xpix, ypix, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
    obs_x_world, obs_y_world = pix_to_world(xobs, yobs, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
    
    # 7) Update worldmap if pitch and roll are small (to be displayed on right side of screen)
    if np.absolute(Rover.pitch) < 0.7 or np.absolute(Rover.pitch - 360) < 0.7:
        if np.absolute(Rover.roll) < 0.7 or np.absolute(Rover.roll - 360) < 0.7:
            Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 10
            Rover.worldmap[obs_y_world, obs_x_world, 0] += 1
            free_points = Rover.worldmap[:,:,2] > 0
            Rover.worldmap[free_points, 0] = 0

    # 8) Convert rover-centric pixels to polar coordinates
    dist, angles = to_polar_coords(xpix, ypix)
    Rover.nav_angles = angles

    # 9) Check for rocks and update data members if pitch and roll are small
    xrock, yrock = rover_coords(squeezed)
    xrock, yrock = range_limit(xrock, yrock)

    if len(xrock) > 0:
        rock_x_world, rock_y_world = pix_to_world(xrock, yrock, Rover.pos[0], Rover.pos[1], Rover.yaw, world_size, scale)
        Rover.rock_dist, Rover.rock_angles = to_polar_coords(xrock, yrock)
        if np.absolute(Rover.pitch)	< 0.7 or np.absolute(Rover.pitch - 360)	< 0.7:
            if np.absolute(Rover.roll) < 0.7 or np.absolute(Rover.roll - 360) < 0.7:
                Rover.worldmap[rock_y_world, rock_x_world, 1] = 255
    else:
        Rover.rock_dist = []
        Rover.rock_angles = []

    if squeezed.any():
        Rover.vision_image[:,:,1] = 255 * squeezed
    else:
        Rover.vision_image[:,:,1] = 0
    
    return Rover
