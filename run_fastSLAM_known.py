"""
Author: Peter Johnson 
Email: pjohnson@g.hmc.edu
Date of Creation: 07/31/20
Description:
    FastSLAM 2.0 implementation based on algorithm in Probabalistic Robotics
    by Sebastian Thrun et al. Dataset is from E205 State Estimation Course
    There is a single landmark, a post in an open field
    State to be estimated is pose (x, y, theta)
    Measurment is landmark range z_x, z_y
    Rao-Blackwellized Particle filter is used. PF for estimating pose and 
    each particle has an EKF for estimating landmark location. This 
    implementation will use known landmark correspondence. Goal is to build up 
    to unknown landmark correspondence
"""

import csv
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
import math
import os.path
import scipy as sp
from scipy.stats import norm, uniform, multivariate_normal
from statistics import stdev
from fastSLAM_particle import Particle


HEIGHT_THRESHOLD = 0.0                  # meters
GROUND_HEIGHT_THRESHOLD = -.4           # meters
dt = 0.1                                # timestep seconds
EARTH_RADIUS = 6.3781E6                 # meters
NUM_PARTICLES = 100                                    # 100 isnt quite good enough
# variances obtained from previous work with Accelerometer and LiDAR
VAR_THETA = 0.00058709
VAR_LIDAR = 0.0075**2 # this is actually range but works for x and y
VAR_MOTION = 9.18E-6
VAR_YAW = 5.8709E-4

# Covariance matrices to be computed once, ahead of tie
# Motion covariance
R_t = np.array([[VAR_MOTION,0,0],
                [0,VAR_MOTION,0],
                [0,0,VAR_YAW]],dtype=np.double)

R_t_inv = np.linalg.inv(R_t)

# Measurment covariance
Q_t = np.array([[VAR_LIDAR, 0],
               [0,  VAR_LIDAR]],dtype=np.double)

# Measurment function jacobian reltive to state
H_x = np.array([[ -1.0, 0, 0],
                [  0,-1.0, 0]],dtype = np.double)

H_x_T = np.transpose(H_x)


# Measurment function jacobian relative to map
H_m = np.array([[ 1.0, 0],
                [ 0, 1.0]],dtype = np.double)

H_m_T = np.transpose(H_m)

H_m_inv = np.linalg.inv(H_m)
H_m_inv_T = np.transpose(H_m_inv)
feat_init_cov = H_m_inv_T.dot(Q_t).dot(H_m_inv)

# Measurment function jacobian relative
# Default importance weight
p0 = 1/NUM_PARTICLES

PRINTING = False

global V #0.491 m/s


def load_data(filename):
    """Load data from the csv log

    Parameters:
    filename (str)  -- the name of the csv log

    Returns:
    data (dict)     -- the logged data with data categories as keys
                       and values list of np.doubles
    """
    is_filtered = False
    if os.path.isfile(filename + "_filtered.csv"):
        f = open(filename + "_filtered.csv")
        is_filtered = True
    else:
        f = open(filename + ".csv")

    file_reader = csv.reader(f, delimiter=',')

    # Load data into dictionary with headers as keys
    data = {}
    header = ["X", "Y", "Z", "Time Stamp", "Latitude", "Longitude",
              "Yaw", "Pitch", "Roll", "AccelX", "AccelY", "AccelZ"]
    for h in header:
        data[h] = []

    row_num = 0
    f_log = open("bad_data_log.txt", "w")
    for row in file_reader:
        for h, element in zip(header, row):
            # If got a bad value just use the previous value
            try:
                data[h].append(np.double(element))
            except ValueError:
                data[h].append(data[h][-1])
                f_log.write(str(row_num) + "\n")

        row_num += 1
    f.close()
    f_log.close()

    #Convert from CW degrees to CCW radians
    for i in range(0, len(data["Yaw"])):
        theta = data["Yaw"][i]
        theta = -theta*2*math.pi/360
        theta = wrap_to_pi(theta)
        data["Yaw"][i] = theta

    return data, is_filtered


def save_data(data, filename):
    """Save data from dictionary to csv

    Parameters:
    filename (str)  -- the name of the csv log
    data (dict)     -- data to log
    """
    header = ["X", "Y", "Z", "Time Stamp", "Latitude", "Longitude",
              "Yaw", "Pitch", "Roll", "AccelX", "AccelY", "AccelZ"]
    f = open(filename, "w")
    num_rows = len(data["X"])
    for i in range(num_rows):
        for h in header:
            f.write(str(data[h][i]) + ",")

        f.write("\n")

    f.close()


def filter_data(data):
    """Filter lidar points based on height and duplicate time stamp

    Parameters:
    data (dict)             -- unfilterd data

    Returns:
    filtered_data (dict)    -- filtered data
    """
    # Remove data that is not above a height threshold to remove
    # ground measurements and remove data below a certain height
    # to remove outliers like random birds in the Linde Field (fuck you birds)
    filter_idx = [idx for idx, ele in enumerate(data["Z"])
                  if ele > GROUND_HEIGHT_THRESHOLD and ele < HEIGHT_THRESHOLD]

    filtered_data = {}
    for key in data.keys():
        filtered_data[key] = [data[key][i] for i in filter_idx]

    # Remove data that at the same time stamp
    ts = filtered_data["Time Stamp"]
    filter_idx = [idx for idx in range(1, len(ts)) if ts[idx] != ts[idx-1]]
    for key in data.keys():
        filtered_data[key] = [filtered_data[key][i] for i in filter_idx]

    return filtered_data


def convert_gps_to_xy(lat_gps, lon_gps, lat_origin, lon_origin):
    """Convert gps coordinates to cartesian with equirectangular projection

    Parameters:
    lat_gps     (np.double)    -- latitude coordinate
    lon_gps     (np.double)    -- longitude coordinate
    lat_origin  (np.double)    -- latitude coordinate of your chosen origin
    lon_origin  (np.double)    -- longitude coordinate of your chosen origin

    Returns:
    x_gps (np.double)          -- the converted x coordinate
    y_gps (np.double)          -- the converted y coordinate
    """
    x_gps = EARTH_RADIUS*(math.pi/180.)*(lon_gps - lon_origin)*math.cos((math.pi/180.)*lat_origin)
    y_gps = EARTH_RADIUS*(math.pi/180.)*(lat_gps - lat_origin)

    return x_gps, y_gps


def wrap_to_pi(angle):
    """Wrap angle data in radians to [-pi, pi]
        Parameters:
        angle (np.double)   -- unwrapped angle

        Returns:
        angle (np.double)   -- wrapped angle
    """
    while angle >= math.pi:
        angle -= 2*math.pi
    while angle <= -math.pi:
        angle += 2*math.pi
    return angle


def propagate_state(p_i_t, u_t):
    """Propogate/predict the state based on chosen motion model
        Use the nonlinear function g(x_t_prev, u_t) to sample from distribution
        Assign a random velocity from bimodal distribution
        Either centered around 0 or the average speed
        This motion model forgoes the need for odometry

        Parameters:
        p_i_t (particle)     -- the particle to be propagated 
        u_t (np.array)       -- the current control input (really is odometry)

        Returns:
        x_hat_t (np.array)  -- the propagated state
    """
    #Destructure arrays
    x, y, theta = p_i_t.state
    #w = p_i_t.weight
    ux, uy, yaw = u_t

    Vel = V* np.random.uniform(0.89, 1.1)
    if (Vel < 0.9*V):
        Vel = 0 + np.random.normal(0,0.05)
    # There is a chance of staying still

    # Perturb the compass
    yaw += np.random.normal(0, np.sqrt(VAR_THETA))
    Vx = Vel*math.cos(yaw)
    Vy = Vel*math.sin(yaw)

    p_i_t.updateState(x+Vx*dt,y+Vy*dt,yaw)

    return p_i_t

def calc_inverse_Sensor(p_i_t,z_g_t):
    """Calculate the location of a feature given the measurment
        and the pose. 
        Parameters:
        z_g_t (np.array)  --measurment in global frame orientation
        x_t (np.array)  --pose

        Returns:
        mu_t(np.array) --the landmark location
    """

    x, y, theta = p_i_t.state
    z_x, z_y = z_g_t

    mu_t = np.array([x+z_x,
                     y+z_y],
                    dtype = np.double)
    return mu_t


def calc_meas_prediction(p_i_t):
    """Calculate predicted measurement based on the predicted state
        Implements the nonlinear measrument h function
        Parameters:
        p_i_t (particle)  -- the predicted particle

        Returns:
        z_bar_t (np.array)  -- the predicted measurement (x and y range)
    """
    x, y, theta = p_i_t.state

    # Orientation will not matter
    z_bar_t = np.array([p_i_t.feats[1]-x,
                        p_i_t.feats[2]-y],
                        dtype = np.double)

    return z_bar_t


def local_to_global(p_i_t, z_t):
    """Rotate the lidar x and y measurements from the lidar frame to the global frame orientation

       Parameters:
       x_bar_t (np.array)  -- the predicted state
       z_t     (np.array)  -- the measurement vector in the lidar frame

       Returns:
       z_global (np.array) -- global orientation measurment vector
    """
    x, y, theta = p_i_t.state

    zx, zy = z_t
    w_theta = wrap_to_pi(-theta+math.pi/2)

    z_global = np.array([zx*math.cos(w_theta) + zy*math.sin(w_theta),
                         -zx*math.sin(w_theta) + zy*math.cos(w_theta)],
                         dtype = np.double)

    return z_global

def prediction_step(P_prev, u_t, z_t):
    """Compute the prediction half of particle filter

        Parameters:
        P_prev (list of particles)  -- set of previous particles
        u_t (np.array)              -- the control input
        z_t (np.array)              -- the measurement

        Returns:
        P_pred  (list of particles) -- the set of predicted particles
        w_tot   (float)             -- the total weight of all the particles    
    """
    c_t = 1     # a made up correspondence variable 
    P_pred = []
    w_tot = 0

    # loop over all of the previous particles
    for p_prev in P_prev:
        # predict pose given previous particle, odometry + randomness (motion model)
        p_pred_t = propagate_state(p_prev, u_t)
        #print("Pred State: ", p_pred_t.state)
        # Mapping with observed feature
        # Globalize the measurment for each particle
        z_g_t = local_to_global(p_pred_t, z_t)
        #print("Map frame z: ", z_g_t)
        # measurement prediction
        z_bar_t = calc_meas_prediction(p_pred_t)
        #print("z_bar_t: ", z_bar_t)
        # measurment information 
        Q_j = Q_t+H_m.dot(p_pred_t.covs).dot(H_m_T)
        #print("Q_j: ", Q_j)
        Q_j_inv = np.linalg.inv(Q_j)
        #print("Q_j_inv: ", Q_j_inv)
        # Cov of proposal distribution
        sigma_x_j = np.linalg.inv(H_x_T.dot(Q_j_inv).dot(H_x)+R_t_inv)
        #print("sigma_x_j: ", sigma_x_j)
        # Mean of proposal distribution
        mu_x_j = sigma_x_j.dot(H_x_T).dot(Q_j_inv).dot(z_g_t-z_bar_t)+p_pred_t.state
        #print("mu_x_j: ", mu_x_j, mu_x_j.shape) 
        # Sample pose
        x_t = np.random.multivariate_normal(mu_x_j,sigma_x_j,1)
        #print("x_t: ", x_t, x_t.shape)
        #print(x_t[0,0], x_t[0,1], x_t[0,2])
        p_pred_t.updateState(x_t[0,0], x_t[0,1], x_t[0,2])
        # Predict measurment for sampled pose
        z_hat_t = calc_meas_prediction(p_pred_t)
        

        if(p_pred_t.observed(c_t)==True):
            
            # Kalman gain
            K = p_pred_t.covs.dot(H_m_T).dot(Q_j_inv)
            # update mean
            #print("feats: ", p_pred_t.feats[1:2])
            # np.array slicing is not inclusive of last index
            mu = p_pred_t.feats[1:3]+K.dot(z_g_t-z_hat_t)
            #print("mu: ", mu)
            # update map covariance
            sigma_j = (np.identity(2)-K.dot(H_m)).dot(p_pred_t.covs)
            # importance factor
            L = H_x.dot(R_t).dot(H_x_T)+H_m.dot(p_pred_t.covs).dot(H_m_T)+Q_t
            importance = multivariate_normal(z_hat_t,L)
            weight = importance.pdf(z_g_t)
            #print("Weight: ",weight)
            p_pred_t.updateFeat(c_t, mu, sigma_j, weight)

        else: # Never seen before
            #print("New observed feature")
            mu = calc_inverse_Sensor(p_pred_t,z_g_t)
            p_pred_t.initFeat(c_t, mu, feat_init_cov,p0) 

        # We always see the one feature, no need for a case with negative information
        
        # find particle's weight using wt = P(zt | xt)
        w_t = p_pred_t.weight
        w_tot += w_t
        # add new particle to the current belief
        P_pred.append(p_pred_t)
        #print("State: ", p_pred_t.state)
        #print("Feat: ", p_pred_t.feats)
        #print("Weight: ", w_t)

    return [P_pred, w_tot]


def correction_step(P_pred, w_tot):
    """Compute the correction portion of particle filter
        Resample based on the weights of the particles

        Parameters:
        P_pred    (list of np.array)  -- the predicted particles of time t
        w_tot     (np.double)             -- the sum of all the particle weights

        Returns:
        P_corr    (list of np.array)  -- the corrected particles of time t
    """
    if (PRINTING):
        print("RESAMPLING")
        for p in P_pred:
            print("x: ", p[1], "   y: ", p[3], "   weight: ", p[5])

    P_corr = []

    p0 = P_pred[0]
    w0 = p0.weight 
    # resampling algorithm
    for p in P_pred:
        r = np.random.uniform(0, 1)*w_tot
        j = 0
        wsum = w0
        while (wsum < r):
            j += 1
            if (j == NUM_PARTICLES-1):
                break
            p_j = P_pred[j]
            w_j = p_j.weight
            wsum += w_j

        p_c = P_pred[j]
        #print(p_c)
        P_corr.append(p_c)

    return P_corr


def distance(x1,y1,x2,y2):
    """Compute the distance between two points
            Parameters:
            x1 (float)      --x coordinate of first point
            y1 (float)      --y coordinate of first point
            x2 (float)      --x coordinate of second point
            y2 (float)      --y coordinate of second point

            Returns:
            dist (float)    --euclidean distance
    """
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def simple_clustering(P_t):
    """ O(N) Clustering algorithm to find highest weighted particle
            as the centroid of a single cluster. Picks state to be plotted
            Parameters:
            P_t (array)     --the set of particles

            Returns:
            best_particle (particle)        --the highest weighted particle 
    """
    highest_weight = 0;
    best_particle = P_t[0];
    for p in P_t:
        if p.weight > highest_weight:
            highest_weight = p.weight
            best_particle = p
    return best_particle


def path_rmse(state_estimates):
    """ Computes the RMSE error of the distance at each time step from the expected path

        Parameters:
        x_estimate      (np.array)    -- array  of state estimates

        Returns:
        rmse              (np.double)          -- rmse
        residuals         (np.array)      -- array of residuals
    """
    x_est = state_estimates[1][:]
    y_est = state_estimates[3][:]
    sqerrors = []
    errors = []
    residuals = []
    rmse_time = []

    #resid = measured - predicted by segments

    for i in range(len(x_est)):
        if (x_est[i]<0 and y_est[i]>0):
            #Upper left corner
            resid = distance(x_est[i], y_est[i], 0,0)
            sqerror = resid**2

        elif (x_est[i]>10 and y_est[i]>0):
            #Upper right coner
            resid = distance(x_est[i], y_est[i], 10,0)
            sqerror = resid**2

        elif (x_est[i]>10 and y_est[i]<-10):
            #Lower right coner
            resid = distance(x_est[i], y_est[i], 10,-10)
            sqerror = resid**2

        elif (x_est[i]<0 and y_est[i]<-10):
            #Lower right coner
            resid = distance(x_est[i], y_est[i], 0,-10)
            sqerror = resid**2

        else:
            #General case
            r1 = (y_est[i] - 0)
            r2 = (x_est[i] - 10)
            r3 = (y_est[i] - (-10))
            r4 = (x_est[i] -0)
            resid = min(abs(r1),abs(r2),abs(r3),abs(r4))

        residuals.append(resid) #residuals are basically cte
        sqerrors.append(resid**2)
        errors.append(abs(resid))
        mse = np.mean(sqerrors)
        rmse = math.sqrt(mse)
        rmse_time.append(rmse)

    mean_error = np.mean(errors)
    mse = np.mean(sqerrors)
    rmse = math.sqrt(mse)

    return rmse, residuals, mean_error, rmse_time

def find_sigma(data_set):
    """Finds the standard deviation, sigma, of a 1D data_set

        Parameters:
        data_set (list)    -- data set to find the sigma of

        Returns:
        sigma (float)      -- the standard deviation of data_set
    """
    mu = np.mean(data_set)
    sigma = stdev(data_set, mu)
    return sigma

def main():
    """Run FastSLAM on logged data from IMU and LiDAR moving in a box formation around a landmark"""

    #np.random.seed(28)

    filepath = ""
    filename =  "2020_2_26__16_59_7" #"2020_2_26__17_21_59"
    data, is_filtered = load_data(filepath + filename)

    # Save filtered data so don't have to process unfiltered data everytime
    if not is_filtered:
        f_data = filter_data(data)
        save_data(f_data, filepath+filename+"_filtered.csv")

    # Load data into variables
    x_lidar = data["X"]
    y_lidar = data["Y"]
    z_lidar = data["Z"]
    time_stamps = data["Time Stamp"]
    lat_gps = data["Latitude"]
    lon_gps = data["Longitude"]
    yaw_lidar = data["Yaw"]
    pitch_lidar = data["Pitch"]
    roll_lidar = data["Roll"]
    x_ddot = data["AccelX"]
    y_ddot = data["AccelY"]

    # Find variances by looking at a constant measurement
    VAR_AX = find_sigma(x_ddot[0:79])**2
    VAR_AY = find_sigma(y_ddot[0:79])**2
    # print(VAR_AX)
    # print(VAR_AY)

    lat_origin = lat_gps[0]
    lon_origin = lon_gps[0]

    #Compute avg velocity for use in motion model
    global V
    V = 4*10/(len(time_stamps)*dt)
    print("V: ", V)


    #  Initialize filter
    P_prev_t = []
    for i in range(0,NUM_PARTICLES):
        randx = np.random.uniform(-5,15)
        randy = np.random.uniform(-15,5)
        randtheta = np.random.uniform(-math.pi,math.pi)
        # Start in random location
        #p = Particle(randx, randy, randtheta, p0)
        # Start in the NW corner
        p = Particle(0,0,0,1/NUM_PARTICLES)
        P_prev_t.append(p)

    #allocate
    gps_estimates = np.empty((2, len(time_stamps)))
    centroids_logged = np.empty((3, len(time_stamps)))
    wt_logged = np.empty((len(time_stamps)))

    #Expected path
    pathx = [0,10,10,0,0]
    pathy = [0,0,-10,-10,0]

    #Initialize animated plot
    plt.figure(1)
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    plt.axis([-15, 15, -17, 15])
    ax.set_ylabel("Y position (Global Frame, m)")
    ax.set_xlabel("X position (Global Frame, m)")
    ax.legend(["Expected Path", "Estimated Position", "GPS Position"], loc='center right')


    #  Run filter over data
    for t, _ in enumerate(time_stamps):
    #for t in range(200):
        print(t)
        x_gps, y_gps = convert_gps_to_xy(lat_gps[t],lon_gps[t],
                                        lat_origin,lon_origin)
        gps_estimates[:,t] = np.array([x_gps, y_gps])
        

        # plt.scatter(x_gps, y_gps, c='b', marker='.')
        centroid = simple_clustering(P_prev_t)
        # for c in centroids:
        #     plt.scatter(c[1],c[3], c='k', marker='.')
        centroids_logged[:,t] = centroid.state.reshape((3,))
        wt_logged[t]=centroid.weight
        # Print all particles
        if ( t>0):
            print("Feat: ", centroid.feats)
        
        if (t % 50 == 0):
            print("State: ", centroid.state)
            print("Feat: ", centroid.feats)
            print("Covs: ", centroid.covs)
            print("Weight: ", centroid.weight)
            for p in P_prev_t:
                x,y,_=p.state
                ax.scatter(x, y, c='r', marker='.')

        if (PRINTING):
            print("Time Step: %d", t)
            print("x: ", centroids[1], "   y: ", centroids[3], "   theta: ", centroids[4])

        # Get control input
        u_t = np.array([x_ddot[t],
                        y_ddot[t],
                        yaw_lidar[t]])
        #print("u_t: ", u_t.shape)

        # Get measurement
        z_t = np.array([x_lidar[t], y_lidar[t]])
        #print("z_t: ", z_t.shape)

        # Prediction Step
        P_pred_t,  w_tot = prediction_step(P_prev_t, u_t, z_t)
        
        # Correction Step
        P_t = correction_step(P_pred_t, w_tot)

        #  For clarity sake/teaching purposes, we explicitly update t->(t-1)
        P_prev_t = P_t


    plt.scatter(gps_estimates[0][:], gps_estimates[1][:], marker='.')
    plt.plot(centroids_logged[0][:],centroids_logged[1][:], c='k', linestyle='--')
    plt.plot(pathx, pathy)
    plt.legend([ "PF Estimate", "Expected Path", "Particles", "GPS Measurements"], loc="lower center", ncol=2)
    plt.xlabel("Global X (m)")
    plt.ylabel("Global Y (m)")
    plt.show()

    with open('1000particles.csv', 'w', newline='') as csvfile:
        fieldnames = ['times', 'x','y', 'theta', 'w']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for t, _ in enumerate(time_stamps):
            writer.writerow({'times': time_stamps[t], 'x': centroids_logged[0][t], 'y': centroids_logged[1][t], \
             'theta': centroids_logged[2][t], 'w': wt_logged[t]})

    print("Done plotting, exiting")
    return 0


if __name__ == "__main__":
    main()
