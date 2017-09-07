import numpy as np

def get_bearing(x_r, y_r, x_h, y_h, yaw):
    dy = y_r - y_h
    dx = x_r - x_h
    if dx == 0:
        if y_r > y_h:
            theta = 90
        else:
            theta = 270
    else:
        theta = np.arctan2(dy, dx) * 180/np.pi

    bearing = 180 + theta - yaw

    # Corrections that lead to smoother path planning
    if bearing > 180:
        bearing = bearing - 360
    elif bearing < -180:
        bearing = 360 + bearing
    elif np.absolute(bearing) == 360:
        bearing = 0
        
    return bearing

def calc_dist(pos1, pos2):
    return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1] - pos2[1])**2)

# Auxiliary function that implements careful forward motion 
def forward_aux(Rover):

    # If a rock is seen, then enter get_rock mode
    if Rover.mode != 'stuck' and len(Rover.rock_angles) >= Rover.rock_collect_threshold:
        Rover.throttle = 0
        Rover.brake = Rover.brake_set
        Rover.steer = 0
        Rover.mode = 'get_rock'
        return Rover

    # Get pixels in a vertical band ahead of the rover, 40 pixels wide
    mid_x = []
    mid_y = []

    for i in range(0, len(Rover.nav_x)):
        if Rover.nav_y[i] <= 20 and Rover.nav_y[i] >= -20:
            mid_x.append(Rover.nav_x[i])
            mid_y.append(Rover.nav_y[i])
    mid_x = np.array(mid_x)
    mid_y = np.array(mid_y)

        
    # Extremely low room to the side, then turn hard
    if len(mid_y[mid_y >= 0]) < Rover.stop_side:
        Rover.throttle = 0
        Rover.brake = 0
        Rover.steer = -15
        return Rover
    elif len(mid_y[mid_y < 0]) < Rover.stop_side:
        Rover.throttle = 0
        Rover.brake = 0
        Rover.steer = 15
        return Rover
    # Much more room to right, turn more gradually
    elif len(mid_y[mid_y < 0]) >= Rover.left_right_ratio * len(mid_y[mid_y >= 0]):
        Rover.steer = np.clip(np.mean(Rover.nav_angles[Rover.nav_angles < 0]) * 180/np.pi, -15, 15)
    # Much more room to left, turn more gradually
    elif len(mid_y[mid_y >= 0]) >= Rover.left_right_ratio * len(mid_y[mid_y < 0]):
        Rover.steer = np.clip(np.mean(Rover.nav_angles[Rover.nav_angles >= 0]) * 180/np.pi, -15, 15)
    # If on the way home, follow the bearing to home as closely as possible
    elif Rover.samples_collected == 6 and (Rover.pos[0] < 84 or Rover.pos[0] > 97 or Rover.pos[1] < 81 or Rover.pos[1] > 88):
        lo = np.percentile(Rover.nav_angles, 25)
        hi = np.percentile(Rover.nav_angles, 75)
        deg_angles = Rover.nav_angles[Rover.nav_angles < hi]
        deg_angles = deg_angles[deg_angles > lo] * 180/np.pi
        arr = np.absolute(deg_angles - get_bearing(Rover.pos[0], Rover.pos[1], Rover.init_pos[0], Rover.init_pos[1], Rover.yaw))
        idx = np.argmin(arr)
        Rover.steer = np.clip(deg_angles[idx], -15, 15)
    # Otherwise just follow the steer bias
    else:
        Rover.steer = np.clip(np.mean(Rover.nav_angles) * 180/np.pi + Rover.steer_bias, -15, 15)

    # Speed up if we're too slow
    if Rover.vel < Rover.max_vel:  
        Rover.throttle = Rover.throttle_set
    # Otherwise just coast
    else: 
        Rover.throttle = 0
    Rover.brake = 0
    
    return Rover
    
def decision_step(Rover):

    # Stop at the end of challenge
    if Rover.samples_collected == 6 and calc_dist(Rover.pos, Rover.init_pos) <= 2:
        Rover.throttle = 0
        Rover.brake = Rover.brake_set
        Rover.steer = 0
        return Rover
    
    # Adjustments for specific conditions
    if Rover.pos[0] >= 84 and Rover.pos[0] <= 97 and Rover.pos[1] >= 81 and Rover.pos[1] <= 88:
        Rover.steer_bias = 15
        Rover.max_vel = 0.5
    elif Rover.samples_collected == 6:
        Rover.steer_bias = 0
        Rover.max_vel = 1.5
    else:
        Rover.steer_bias = 10
        Rover.max_vel = 1.5
    Rover.throttle_set = 0.2
    
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:

        # Measure position in 10 seconds intervals
        if Rover.total_time - Rover.last_update_time >= 10:
            Rover.last_update_time = Rover.total_time
            # If rover didn't move 1m in 10 seconds and a pickup wasn't sent for 20 seconds, then enter stuck mode
            if calc_dist(Rover.pos, Rover.lastpos) <= 1 and Rover.total_time - Rover.last_stuck_time >= 10 and Rover.total_time - Rover.last_pickup_time >= 20:
                Rover.mode = 'stuck'
                Rover.stuck_start_pos = Rover.pos
                Rover.stuck_start_time = Rover.total_time
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.lastpos = Rover.pos
                return Rover 
            else:
                Rover.lastpos = Rover.pos

        # Record the last time steer angle was reasonably small
        if np.absolute(Rover.steer) <= 8:
            Rover.last_low_steer_time = Rover.total_time
        # If steer angle was too large for 10 seconds and there wasn't a pickup, stop, or stuck done recently, then enter circle mode
        if Rover.total_time - Rover.last_low_steer_time >= 10:
            if Rover.total_time - Rover.last_stop_time >= 10 and Rover.total_time - Rover.last_stuck_time >= 10 and Rover.total_time - Rover.last_pickup_time >= 20 and Rover.samples_collected < 6:
                Rover.throttle = 0
                Rover.brake = 10
                Rover.steer = 0
                Rover.circle_start_time = Rover.total_time
                Rover.mode = 'circle'
                return Rover
                
        # Check for Rover.mode status
        if Rover.mode == 'get_rock':
            # Once we collect the rock, enter forward mode
            if Rover.samples_collected > Rover.prev_samples_collected:
                Rover.last_pickup_time = Rover.total_time
                Rover.mode = 'forward'
                Rover.prev_samples_collected = Rover.samples_collected
            if Rover.picking_up:
                return Rover
            # Send pickup if near the rock
            if Rover.near_sample:
                Rover.send_pickup = True
                Rover.last_pickup_time = Rover.total_time
                return Rover
            Rover.brake = 0
            # Follow the closest rock pixel, this avoids confusion when two rocks can be seen
            if len(Rover.rock_angles) > Rover.rock_collect_threshold:
                idx = np.argmin(Rover.rock_dist)
                Rover.steer = np.clip(Rover.rock_angles[idx] * 180/np.pi, -15, 15)
            
        elif Rover.mode == 'circle':
            # Return to forward mode after ten seconds
            if Rover.total_time - Rover.circle_start_time >= 10:
                Rover.mode = 'forward'
                Rover.steer_bias = 10
            # For ten seconds, just follow the mean navigable angle
            else:
                Rover.steer_bias = 0
                Rover = forward_aux(Rover)
        
        elif Rover.mode == 'stuck':
            Rover.last_stuck_time = Rover.total_time
            # If rover moved 2 meters after at least 13 seconds, leave stuck mode
            if calc_dist(Rover.pos, Rover.stuck_start_pos) >= 2 and Rover.total_time - Rover.stuck_start_time >= 13:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.mode = 'forward'
                return Rover
            # First back up
            if np.floor(Rover.total_time - Rover.stuck_start_time) % 13 <= 5:
                Rover.throttle = -Rover.unstuck_throttle
                Rover.steer = 0
                Rover.brake = 0
            # Then stop
            elif np.floor(Rover.total_time - Rover.stuck_start_time) % 13 <= 6:
                Rover.throttle = 0
                Rover.steer = 0
                Rover.brake = Rover.brake_set
            # Turn to the direction with more navigable pixels
            elif np.floor(Rover.total_time - Rover.stuck_start_time) % 13 <= 7:
                Rover.throttle = 0
                if len(Rover.nav_y[Rover.nav_y >= 0]) > Rover.left_right_ratio * len(Rover.nav_y[Rover.nav_y < 0]):
                    Rover.steer = 15
                else:
                    Rover.steer = -15
                Rover.brake = 0
            # Then drive ahead aggressively with no steering bias
            else:
                Rover.steer_bias = 0
                Rover.throttle_set = Rover.unstuck_throttle
                Rover.max_vel = 2
                Rover = forward_aux(Rover)
                
        elif Rover.mode == 'forward': 
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                Rover = forward_aux(Rover)
            # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'stop'
            
        elif Rover.mode == 'stop':
            Rover.last_stop_time = Rover.total_time
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -15 
                # If we're stopped but see sufficient navigable terrain in front then go!
                if len(Rover.nav_angles) >= Rover.go_forward:
                    Rover.throttle = Rover.throttle_set
                    Rover.brake = 0
                    # Set steering to crawl along wall
                    mean_dir = np.mean(Rover.nav_angles)
                    Rover.steer = np.clip(mean_dir * 180/np.pi + Rover.steer_bias, -15, 15)
                    Rover.mode = 'forward'

            
    # Just to make the rover do something
    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    return Rover

