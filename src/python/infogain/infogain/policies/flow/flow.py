# implementing motion flow in a matrix -- complete with test cases!

from enum import IntEnum
from math import floor
import matplotlib.pyplot as plt
import numpy as np
from timeit import timeit

from numpy.core.shape_base import block

def arrange_by_index( A, x, y ):
    res = np.zeros_like(A)
    H,W = A.shape

    for r in range(H):
        for c in range(W):
            yi = y[r,c]
            xi = x[r,c]
            res[yi,xi] += A[r,c]
    return res

def flow( probability_grid, velocity_grid, scale=1, timesteps=1, dt=0.1, mode='nearest' ):

        # pad the grids to allow motion to flow off the field
        PAD = 1

        # prep output
        flow_prob = [ (probability_grid, velocity_grid) ]

        probability_grid = np.pad(probability_grid, PAD, mode='constant')
        velocity_pad = ((PAD, ), (PAD, ),(0,))
        velocity_grid = np.pad(velocity_grid, velocity_pad, mode='constant', constant_values=0) 

        # mask off the static elements
        probability_grid = probability_grid * (np.sum( velocity_grid, axis=2 ) != 0).astype(float)

        if mode == 'nearest':
            def map_fn(mo, int_mo): return ((int_mo - mo) > 0.5).astype(float)
        else:  # 'bilinear'
            def map_fn(mo, int_mo): return (int_mo - mo)

        H,W = probability_grid.shape

        for t in range(timesteps):
            k_prob_11 = np.zeros_like(probability_grid)
            k_prob_12 = np.zeros_like(probability_grid)
            k_prob_21 = np.zeros_like(probability_grid)
            k_prob_22 = np.zeros_like(probability_grid)

            # keep a separate grid to store the updated velocities.  For now, we'll assume an 
            # equal average of those that end up in this cell
            k_v = np.zeros_like(velocity_grid)
            k_v_count = np.zeros_like(probability_grid)


            xx, yy = np.meshgrid(np.arange(W), np.arange(H))
            nx = xx + velocity_grid[:, :, 0] * dt / scale
            ny = yy + velocity_grid[:, :, 1] * dt / scale

            x1 = np.floor(nx)
            y1 = np.floor(ny)
            x2 = x1 + 1.
            y2 = y1 + 1.

            x1prop = map_fn(nx, x2)
            y1prop = map_fn(ny, y2)

            x2prop = 1. - x1prop
            y2prop = 1. - y1prop

            p11 = y1prop * x1prop
            p12 = y1prop * x2prop
            p21 = y2prop * x1prop
            p22 = y2prop * x2prop

            # after proportioning the blame, clip the indices so they
            # still fit in the grid -- this is why the edges are padded
            x1 = np.clip(x1, 0, W-1).astype(int)
            x2 = np.clip(x2, 0, W-1).astype(int)
            y1 = np.clip(y1, 0, H-1).astype(int)
            y2 = np.clip(y2, 0, H-1).astype(int)

            k_prob_11 = arrange_by_index(probability_grid * p11, x1, y1 )
            k_prob_12 = arrange_by_index(probability_grid * p12, x2, y1 )
            k_prob_21 = arrange_by_index(probability_grid * p21, x1, y2 )
            k_prob_22 = arrange_by_index(probability_grid * p22, x2, y2 )

            # k_prob_11[y1, x1] += probability_grid * p11
            # k_prob_12[y1, x2] += probability_grid * p12
            # k_prob_21[y2, x1] += probability_grid * p21
            # k_prob_22[y2, x2] += probability_grid * p22

            k_v[:,:,0] += arrange_by_index(velocity_grid[:,:,0] * (p11 > 0), x1, y1 )
            k_v[:,:,0] += arrange_by_index(velocity_grid[:,:,0] * (p12 > 0), x2, y1 )
            k_v[:,:,0] += arrange_by_index(velocity_grid[:,:,0] * (p21 > 0), x1, y2 )
            k_v[:,:,0] += arrange_by_index(velocity_grid[:,:,0] * (p22 > 0), x2, y2 )
            
            # k_v[y1, x1, 0] += velocity_grid[:,:,0] * (p11 > 0)
            # k_v[y1, x2, 0] += velocity_grid[:,:,0] * (p12 > 0)
            # k_v[y2, x1, 0] += velocity_grid[:,:,0] * (p21 > 0)
            # k_v[y2, x2, 0] += velocity_grid[:,:,0] * (p22 > 0)

            k_v[:,:,1] += arrange_by_index(velocity_grid[:,:,1] * (p11 > 0), x1, y1 )
            k_v[:,:,1] += arrange_by_index(velocity_grid[:,:,1] * (p12 > 0), x2, y1 )
            k_v[:,:,1] += arrange_by_index(velocity_grid[:,:,1] * (p21 > 0), x1, y2 )
            k_v[:,:,1] += arrange_by_index(velocity_grid[:,:,1] * (p22 > 0), x2, y2 )

            # k_v[y1, x1, 1] += velocity_grid[:,:,1] * (p11 > 0)
            # k_v[y1, x2, 1] += velocity_grid[:,:,1] * (p12 > 0)
            # k_v[y2, x1, 1] += velocity_grid[:,:,1] * (p21 > 0)
            # k_v[y2, x2, 1] += velocity_grid[:,:,1] * (p22 > 0)

            k_v_count += arrange_by_index(np.logical_or(velocity_grid[:,:,0] * (p11 > 0), velocity_grid[:,:,1] * (p11 > 0)), x1, y1 )
            k_v_count += arrange_by_index(np.logical_or(velocity_grid[:,:,0] * (p12 > 0), velocity_grid[:,:,1] * (p12 > 0)), x2, y1 )
            k_v_count += arrange_by_index(np.logical_or(velocity_grid[:,:,0] * (p21 > 0), velocity_grid[:,:,1] * (p21 > 0)), x1, y2)
            k_v_count += arrange_by_index(np.logical_or(velocity_grid[:,:,0] * (p22 > 0), velocity_grid[:,:,1] * (p22 > 0)), x2, y2)

            # probability of occupancy is 1-prob that no cell moved here... first calculate the prob
            # that no cell moved here
            next_prob = (1 - k_prob_11)*(1 - k_prob_12)*(1 - k_prob_21)*(1 - k_prob_22)

            # then reverse the probabilistic sense to get prob of flow for all 'j' cells and clip to
            # ensure we remain in bounds
            next_prob = np.clip(1 - next_prob, 0, 1)

            k_v[:,:,0] = np.divide( k_v[:,:,0], k_v_count, out=np.zeros_like(k_v_count), where=(k_v_count!=0))
            k_v[:,:,1] = np.divide( k_v[:,:,1], k_v_count, out=np.zeros_like(k_v_count), where=(k_v_count!=0))

            flow_prob.append( (next_prob[PAD:-PAD, PAD:-PAD], k_v[PAD:-PAD, PAD:-PAD, :]))

            # update for the next round
            probability_grid = next_prob
            velocity_grid = k_v

        # return the final result
        return flow_prob



def main():
    T = 5
    H = 5
    W = 5
    prob = np.zeros((H, W))
    prob[1:3, 1:3] = 1
    # prob[7:9, 7:9] = 1

    motion = np.zeros((H, W, 2))
    motion[1:3,1:3, 0] = 0.1
    # motion[7:9, 7:9, 1] = -0.75

    fgg = flow( prob, motion, scale=0.01, timesteps=T, dt=0.1, mode='nearest')

    fig, ax = plt.subplots(3, T+1)
    for i in range(T+1):
        ax[0, i].imshow(np.flipud(fgg[i][0][:, :]))
        ax[1, i].imshow(np.flipud(fgg[i][1][:, :, 0]))
        ax[2, i].imshow(np.flipud(fgg[i][1][:, :, 1]))

    plt.show(block=True)



if __name__ == '__main__':
    main()
