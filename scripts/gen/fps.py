from pytorch3d.ops import sample_farthest_points
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
    for i in range(args.n_trajs):
        # read in the particle info at the last time step
        f = h5py.File(f"{args.trajf}/{i}/{args.horizon - 1}.hdf5", "r")
        points = f["x"][:]
        points = torch.from_numpy(points).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        sampled_points, sampled_indices = sample_farthest_points(points, K=args.k)
        sampled_points = sampled_points.squeeze(0)
        sampled_indices = sampled_indices.squeeze(0)
        stat = h5py.File(f"{args.trajf}/{i}/stat.hdf5", "a")
        try:
            stat.create_dataset("sampled_points", data=sampled_points.cpu().numpy(), dtype='float32', compression="gzip", chunks=True, compression_opts=9)
            stat.create_dataset("sampled_indices", data=sampled_indices.cpu().numpy(), dtype='int32', compression="gzip", chunks=True, compression_opts=9)
        except ValueError:
            pass
        stat.close()
        print(f"Conducted FPS for trajectory {i}!")
        f.close()



if __name__ == "__main__":
    main()