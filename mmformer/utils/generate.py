import os
import copy
import numpy as np

####flair, t1ce, t1, t2
def generate_snapshot(x, H, W, T, output, target_cpu, gap_width=2):
    f1 = x[0, :H, :W, np.newaxis, :T]
    f1 = ((f1 - np.min(f1)) * 255.0 / (np.max(f1) - np.min(f1))).astype(np.uint8)
    f1 = np.tile(f1, (1, 1, 3, 1))

    f2 = x[1, :H, :W, np.newaxis, :T]
    f2 = ((f2 - np.min(f2)) * 255.0 / (np.max(f2) - np.min(f2))).astype(np.uint8)
    f2 = np.tile(f2, (1, 1, 3, 1))

    f3 = x[2, :H, :W, np.newaxis, :T]
    f3 = ((f3 - np.min(f3)) * 255.0 / (np.max(f3) - np.min(f3))).astype(np.uint8)
    f3 = np.tile(f3, (1, 1, 3, 1))

    f4 = x[3, :H, :W, np.newaxis, :T]
    f4 = ((f4 - np.min(f4)) * 255.0 / (np.max(f4) - np.min(f4))).astype(np.uint8)
    f4 = np.tile(f4, (1, 1, 3, 1))

    Snapshot_img2 = np.zeros(shape=(H, H*4+gap_width*3, 3, T), dtype=np.uint8)
    Snapshot_img2[:,W:W+gap_width,:] = 255
    Snapshot_img2[:,2*W+gap_width:2*W+2*gap_width, :] = 255
    Snapshot_img2[:,3*W+2*gap_width:3*W+3*gap_width, :] = 255

    empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
    empty_fig[np.where(output == 1)] = 255
    Snapshot_img2[:,:W,0,:] = empty_fig
    empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
    empty_fig[np.where(target_cpu == 1)] = 255
    Snapshot_img2[:, W+gap_width:2*W+gap_width, 0, :] = empty_fig
    empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
    empty_fig[np.where((target_cpu == 1) * (output != 1))] = 255
    Snapshot_img2[:, 2*W+2*gap_width:3*W+2*gap_width, 0, :] = empty_fig
    empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
    empty_fig[np.where((target_cpu != 1) * (output == 1))] = 255
    Snapshot_img2[:, 3*W+3*gap_width:4*W+3*gap_width, 0, :] = empty_fig

    empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
    empty_fig[np.where(output == 2)] = 255
    Snapshot_img2[:,:W,1,:] = empty_fig
    empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
    empty_fig[np.where(target_cpu == 2)] = 255
    Snapshot_img2[:, W+gap_width:2*W+gap_width, 1, :] = empty_fig
    empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
    empty_fig[np.where((target_cpu == 2) * (output != 2))] = 255
    Snapshot_img2[:, 2*W+2*gap_width:3*W+2*gap_width, 1, :] = empty_fig
    empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
    empty_fig[np.where((target_cpu != 2) * (output == 2))] = 255
    Snapshot_img2[:, 3*W+3*gap_width:4*W+3*gap_width, 1, :] = empty_fig

    empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
    empty_fig[np.where(output == 3)] = 255
    Snapshot_img2[:,:W,2,:] = empty_fig
    empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
    empty_fig[np.where(target_cpu == 3)] = 255
    Snapshot_img2[:, W+gap_width:2*W+gap_width, 2, :] = empty_fig
    empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
    empty_fig[np.where((target_cpu == 3) * (output != 3))] = 255
    Snapshot_img2[:, 2*W+2*gap_width:3*W+2*gap_width, 2, :] = empty_fig
    empty_fig = np.zeros(shape=(H, W, T), dtype=np.uint8)
    empty_fig[np.where((target_cpu != 3) * (output == 3))] = 255
    Snapshot_img2[:, 3*W+3*gap_width:4*W+3*gap_width, 2, :] = empty_fig

    gap_horizon = np.ones(shape=(gap_width, W*4+3*gap_width, 3, T), dtype=np.uint8) * 255
    gap_vetical = np.ones(shape=(H, gap_width, 3, T), dtype=np.uint8) * 255

    Snapshot_img1 = np.concatenate((f1, gap_vetical, f2, gap_vetical, f3, gap_vetical, f4), axis=1)
    Snapshot_img = np.concatenate((Snapshot_img1, gap_horizon, Snapshot_img2), axis=0)

    return Snapshot_img
