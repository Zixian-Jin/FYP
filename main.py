import numpy as np
import pickle
import sys
import re
import itertools
import time

sys.path.append('E:/FYP/deep-autoencoders-cartography/python')
sys.path.insert(1, './gsim/')
from Generators.map_generator import MapGenerator
from Generators.gudmundson_map_generator import GudmundsonMapGenerator
from Generators.insite_map_generator import InsiteMapGenerator
from utils.communications import db_to_natural, natural_to_db, natural_to_dbm, db_to_dbm, dbm_to_db, dbm_to_natural
from Samplers.map_sampler import MapSampler
from Estimators.knn_estimator import KNNEstimator
from Estimators.kernel_rr_estimator import KernelRidgeRegrEstimator
from Estimators.group_lasso_multiker import GroupLassoMKEstimator
from Estimators.gaussian_proc_regr import GaussianProcessRegrEstimator
from Estimators.matrix_compl_estimator import MatrixComplEstimator
from Estimators.bem_centralized_lasso import BemCentralizedLassoKEstimator
from Simulators.simulator import Simulator
from Estimators.autoencoder_estimator import AutoEncoderEstimator, get_layer_activations
import matplotlib
from numpy import linalg as npla
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 15})
import gsim
from gsim.gfigure import GFigure
import os



########################################################################
# Select experiment file:
from Experiments.autoencoder_experiments import ExperimentSet

########################################################################


def calculate_map_weights(t_map, m_meta_map):
    ''' This function corresponds to the Fig. 1 in the final report.
    t_map: the discrete radio map (size: 32*32)
    m_meta_map: the terrian map, containing building locations (size: 32*32)
    '''    

    # calculate map weights with different window sizes
    map_weights_1 = determine_map_weights(t_map, 2)
    map_weights_2 = determine_map_weights(t_map, 3) 
    map_weights_3 = determine_map_weights(t_map, 5) 
    plot_map_weights(map_weights_1, 2)
    plot_map_weights(map_weights_2, 3)
    plot_map_weights(map_weights_3, 5)
    plt.show()
    return map_weights_2  # window size = 3 is accepted, use this weights for map sampling


def map_weighted_sampling(t_map, m_meta_map, testing_sampler, map_generator, map_weights):
    ''' This function corresponds to the Fig. 2 in the final report.
    t_map: the discrete radio map (size: 32*32)
    m_meta_map: the terrian map, containing building locations (size: 32*32)
    '''    
    t_sampled_maps = []

    # Unweighted sampling
    testing_sampler.v_sampling_factor = 0.2  # sample size = sampling_factor * total number of data points
    t_sampled_map_1, m_mask_1 = testing_sampler.sample_map(t_map, m_meta_map)
    t_sampled_maps.append((t_sampled_map_1, m_mask_1))
    plot_sampled_map(map_generator.x_length, map_generator.y_length, t_sampled_map_1, "Sampled Map (Ns = 200, Epsilon = 0.0")
    
    # Weighted sampling, epsilon=0.4
    testing_sampler.v_sampling_factor = 0.2
    t_sampled_map_2, m_mask_2 = testing_sampler.sample_map(t_map, m_meta_map, weights=map_weights, epsilon=1)
    t_sampled_maps.append((t_sampled_map_1, m_mask_1))
    plot_sampled_map(map_generator.x_length, map_generator.y_length, t_sampled_map_2, "Sampled Map (Ns = 200, Epsilon = 0.4)")

    # Weighted sampling, epsilon=0.8
    testing_sampler.v_sampling_factor = 0.2
    t_sampled_map_3, m_mask_3 = testing_sampler.sample_map(t_map, m_meta_map, weights=map_weights, epsilon=0.6)
    t_sampled_maps.append((t_sampled_map_1, m_mask_1))
    plot_sampled_map(map_generator.x_length, map_generator.y_length, t_sampled_map_3, "Sampled Map (Ns = 200, Epsilon = 0.8)")
    plt.show()
    return t_sampled_maps


def map_reconstruction(t_sampled_maps, t_map, m_meta_map, map_generator):
    ''' This function corresponds to the Fig. 3 in the final report.
    '''    

    estimator2 = KNNEstimator(map_generator.x_length,
                        map_generator.y_length)

    
    estimated_map_1 = estimator2.estimate_map(t_sampled_maps[0][0], t_sampled_maps[0][1], m_meta_map)
    estimated_map_2 = estimator2.estimate_map(t_sampled_maps[1][0], t_sampled_maps[1][1], m_meta_map)
    estimated_map_3 = estimator2.estimate_map(t_sampled_maps[2][0], t_sampled_maps[2][1], m_meta_map)
    plot_map(map_generator.x_length, map_generator.y_length, t_map, m_meta_map, "True Map")
    plot_map(map_generator.x_length, map_generator.y_length, estimated_map_1, m_meta_map, "Recovered Map (Ns = 200, Epsilon = 0.0)")
    plot_map(map_generator.x_length, map_generator.y_length, estimated_map_2, m_meta_map, "Recovered Map (Ns = 200, Epsilon = 0.4)")
    plot_map(map_generator.x_length, map_generator.y_length, estimated_map_3, m_meta_map, "Recovered Map (Ns = 200, Epsilon = 0.6)")
    plt.show()
    return



def plot_reconstruction(x_len,
                        y_len,
                        l_true_map,
                        l_sampled_maps,
                        l_masks,
                        realization_sampl_fac,
                        meta_data,
                        l_reconstructed_maps,
                        exp_num):
    # sim_real_maps=False):
    # Computes and prints the error
    vec_meta = meta_data.flatten()
    vec_map = l_true_map[0].flatten()
    vec_est_map = l_reconstructed_maps[0].flatten()
    err = np.sqrt((npla.norm((1 - vec_meta) * (vec_map - vec_est_map))) ** 2 / len(np.where(vec_meta == 0)[0]))
    print('The realization error is %.5f' % err)

    # Set plot_in_db  to True if the map entries are in natural units and have to be displayed in dB

    for in_truemap in range(len(l_true_map)):
        for ind_1 in range(l_true_map[in_truemap].shape[0]):
            for ind_2 in range(l_true_map[in_truemap].shape[1]):
                if meta_data[ind_1][ind_2] == 1:
                    l_true_map[in_truemap][ind_1][ind_2] = 'NaN'

    for in_reconsmap in range(len(l_reconstructed_maps)):
        for ind_1 in range(l_reconstructed_maps[in_reconsmap].shape[0]):
            for ind_2 in range(l_reconstructed_maps[in_reconsmap].shape[1]):
                if meta_data[ind_1][ind_2] == 1:
                    l_reconstructed_maps[in_reconsmap][ind_1][ind_2] = 'NaN'

    # tr_map = ax1.imshow(db_to_dbm(l_true_map[0]),
    #                     interpolation='bilinear',
    #                     extent=(0, x_len, 0, y_len),
    #                     cmap='jet',
    #                     origin='lower',
    #                     vmin=v_min,
    #                     vmax=v_max)  #

    for in_samplmap in range(len(l_sampled_maps)):
        for ind_1 in range(l_sampled_maps[in_samplmap].shape[0]):
            for ind_2 in range(l_sampled_maps[in_samplmap].shape[1]):
                if l_masks[in_samplmap][ind_1][ind_2] == 0 or meta_data[ind_1][ind_2] == 1:
                    l_sampled_maps[in_samplmap][ind_1][ind_2] = 'NaN'

    fig1 = plt.figure(figsize=(15, 5))
    fig1.subplots_adjust(hspace=0.7, wspace=0.4)
    n_rows = len(l_true_map)
    n_cols = len(l_sampled_maps) + len(l_reconstructed_maps) + 1
    v_min = -60
    v_max = -30
    tr_im_col = []
    for ind_row in range(n_rows):
        ax = fig1.add_subplot(n_rows, n_cols, 1)
        im_tr = ax.imshow(db_to_dbm(l_true_map[ind_row][:, :]),
                        interpolation='bilinear',
                        extent=(0, x_len, 0, y_len),
                        cmap='jet',
                        origin='lower',
                        vmin=v_min,
                        vmax=v_max)
        tr_im_col = im_tr
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_title('True map')

    for ind_col in range(len(l_sampled_maps)):
        ax = fig1.add_subplot(n_rows, n_cols, ind_col + 2)
        im = ax.imshow(db_to_dbm(l_sampled_maps[ind_col][:, :]),
                        extent=(0, x_len, 0, y_len),
                        cmap='jet',
                        origin='lower',
                        vmin=v_min,
                        vmax=v_max)
        ax.set_xlabel('x [m]')
        ax.set_title('Sampled map \n' + r'$\vert \Omega \vert $=%d' % (np.rint(len(np.where(vec_meta == 0)[0]) *
                                                                            realization_sampl_fac[0])))

    for ind_col in range(len(l_reconstructed_maps)):
        ax = fig1.add_subplot(n_rows, n_cols, ind_col + len(l_sampled_maps) + 2)
        im = ax.imshow(db_to_dbm(l_reconstructed_maps[ind_col][:, :]),
                        interpolation='bilinear',
                        extent=(0, x_len, 0, y_len),
                        cmap='jet',
                        origin='lower',
                        vmin=v_min,
                        vmax=v_max)
        ax.set_xlabel('x [m]')
        ax.set_title('Reconstructed map \n' + r'$\vert \Omega \vert $=%d' % (np.rint(len(np.where(vec_meta == 0)[0]) *
                                                                            realization_sampl_fac[ind_col])))

    fig1.subplots_adjust(right=0.85)
    cbar_ax = fig1.add_axes([0.88, 0.28, 0.02, 0.43])
    fig1.colorbar(tr_im_col, cax=cbar_ax, label='dBm')

    plt.show()  # (block=False)
    # plt.pause(10)
    fig1.savefig(
        'output/autoencoder_experiments/savedResults/True_Sampled_and_Rec_maps%d.pdf'
        % exp_num)
    return

def plot_map(x_len, y_len, map_in, meta_data, message):
    #plt.switch_backend('agg')
    fig1 = plt.figure(figsize=(15, 5))
    fig1.subplots_adjust(hspace=0.7, wspace=0.4)
    # n_rows = len(l_true_map)
    # n_cols = len(l_sampled_maps) + len(l_reconstructed_maps) + 1
    v_min = -60
    v_max = -30
    tr_im_col = []
    for ind_1 in range(map_in.shape[0]):
        for ind_2 in range(map_in.shape[1]):
            if meta_data[ind_1][ind_2] == 1:
                map_in[ind_1][ind_2] = 'NaN'

    # ax = fig1.add_subplot(n_rows, n_cols, ind_col + len(l_sampled_maps) + 2)
    ax1 = fig1.add_subplot(111)
    im = ax1.imshow(db_to_dbm(map_in),
                    interpolation='bilinear',
                    extent=(0, x_len, 0, y_len),
                    cmap='jet',
                    origin='lower',
                    vmin=v_min,
                    vmax=v_max)
    ax1.set_xlabel('x [m]')
    ax1.set_title(message)
    # fig1.set_title('Reconstructed map \n' + r'$\vert \Omega \vert $=%d' % (np.rint(len(np.where(vec_meta == 0)[0]) *
    #                                                                     realization_sampl_fac[ind_col])))

    # fig1.subplots_adjust(right=0.85)
    # cbar_ax = fig1.add_axes([0.88, 0.28, 0.02, 0.43])
    # fig1.colorbar(tr_im_col, cax=cbar_ax, label='dBm')

    # plt.show()  # (block=False)
    return

def plot_sampled_map(x_len, y_len, sampled_map_in, message):
        #plt.switch_backend('agg')
    fig1 = plt.figure(figsize=(15, 5))
    fig1.subplots_adjust(hspace=0.7, wspace=0.4)
    # n_rows = len(l_true_map)
    # n_cols = len(l_sampled_maps) + len(l_reconstructed_maps) + 1
    v_min = -60
    v_max = -30
    tr_im_col = []

    for ind_1 in range(sampled_map_in.shape[0]):
        for ind_2 in range(sampled_map_in.shape[1]):
            if sampled_map_in[ind_1][ind_2] == 0:
                sampled_map_in[ind_1][ind_2] = 'NaN'
    # ax = fig1.add_subplot(n_rows, n_cols, ind_col + len(l_sampled_maps) + 2)
    ax1 = fig1.add_subplot(111)

    im = ax1.imshow(db_to_dbm(sampled_map_in),
                extent=(0, x_len, 0, y_len),
                cmap='jet',
                origin='lower',
                vmin=v_min,
                vmax=v_max)
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_title(message)
    # plt.show()

def plot_map_weights(map_weights_in, window_size):
    #plt.switch_backend('agg')
    fig1 = plt.figure(figsize=(15, 5))
    fig1.subplots_adjust(hspace=0.7, wspace=0.4)
    # n_rows = len(l_true_map)
    # n_cols = len(l_sampled_maps) + len(l_reconstructed_maps) + 1
    v_min = -60
    v_max = -30
    tr_im_col = []

    # ax = fig1.add_subplot(n_rows, n_cols, ind_col + len(l_sampled_maps) + 2)
    ax1 = fig1.add_subplot(111)
    im = ax1.imshow(map_weights_in, origin='lower')
    ax1.set_xlabel('x [m]')
    ax1.set_title('Map Weights (Window Size = %d)'%window_size)
    # plt.show()    

def determine_map_weights(map_in, window_size, thres=None):
    x_len = map_in.shape[0]
    y_len = map_in.shape[1]
    map_weights = np.zeros((x_len, y_len, 1))
    for row in range(x_len):
        for col in range(y_len):
            var = getAdjacentVar(map_in, row, col, window_size)
            if thres == None:
                weight = var
            else:
                thres_low, thres_high = thres
                if var < thres_low:
                    weight = 0.1
                elif var < thres_high:
                    weight = 0.2
                else:
                    weight = 0.8
            map_weights[row][col][0] = weight
    return map_weights

def getAdjacentVar(map_in, row, col, window_size):
    x_len = map_in.shape[0]
    y_len = map_in.shape[1]

    adjacent_entries = [map_in[row][col][0]]

    for row_delta in range(-window_size, window_size+1):
        cur_row = row + row_delta
        for col_delta in range(-window_size, window_size+1):
            cur_col = col + col_delta
            if (cur_row >= 0 and cur_row < x_len ) and (cur_col >= 0 and cur_col < y_len):
                adjacent_entries.append(map_in[cur_row][cur_col][0])
    # calculate the variance of the adjacent entries
    adjacent_entries = np.array(adjacent_entries)
    return np.var(adjacent_entries)

    
if __name__ == "__main__":
    # Generator
    map_generator = InsiteMapGenerator(
        l_file_num=np.arange(50, 52)  # the list is the interval [start, stop)
    )

    # Sampler
    testing_sampler = MapSampler(v_sampling_factor=[0.05, 0.2], std_noise=1)


    t_map, m_meta_map, t_ch_power = map_generator.generate()

    # Implement Fig. 1 of the final report: calculate map weights with different window size
    map_weights = calculate_map_weights(t_map, m_meta_map)

    # Implement Fig. 2 of the final report: sample the map with different sampling factor (epsilon)
    sampled_maps = map_weighted_sampling(t_map, m_meta_map, testing_sampler, map_generator, map_weights)

    # Implement Fig. 3 of the final report: reconstruct the sampeld maps   
    map_reconstruction(sampled_maps, t_map, m_meta_map, map_generator)
