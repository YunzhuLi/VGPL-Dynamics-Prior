from pytorch3d.ops import sample_farthest_points
import numpy as np
from time import time
import torch
import os
import h5py
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int)
    parser.add_argument("--trajf", type=str)
    parser.add_argument("--n_trajs", type=int)
    parser.add_argument("--horizon", type=int, default=250)
    args = parser.parse_args()
    for i in range(700, 800):
        # read in all the particles info for given trajectory
        a = time()
        try:
            traj = np.load(f"{args.trajf}/{i}/x.npy", mmap_mode="r") # (T, N, 3)
        except Exception as e:
            print(e)
        # consider only the particle positions at the last timestep for fps
        points = traj[-1].copy()
        points = torch.from_numpy(points).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        sampled_points, sampled_indices = sample_farthest_points(points, K=args.k)
        sampled_points = sampled_points.squeeze(0)
        sampled_indices = sampled_indices.squeeze(0).cpu().numpy()
        # traj_subsampled = traj[:, sampled_indices]
        # np.save(f"{args.trajf}/{i}/fps.npy", sampled_indices)
        # np.save(f"{args.trajf}/{i}/x_lite.npy", traj_subsampled)
        np.save(f"{args.trajf}/{i}/x_t.npy", traj.transpose((1, 0, 2)))
        print(f"Conducted FPS for trajectory {i}!", time() - a, "seconds!")



if __name__ == "__main__":
    main()