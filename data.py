from multiprocessing import Pool, shared_memory
import os
import cv2
import time
import h5py
import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from pytorch3d.ops import sample_farthest_points
import numpy as np
import scipy.misc
import scipy.spatial as spatial
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset

from utils import rand_int, rand_float

### from DPI


def store_data(data_names, data, path):
    hf = h5py.File(path, "w")
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_data(data_names, path, idxs=None, is_fluidlab=False):
    if is_fluidlab:
        pass
    else:
        hf = h5py.File(path, "r")
        data = []
        for i in range(len(data_names)):
            d = np.array(hf.get(data_names[i]))
            data.append(d)
        if idxs is not None:
            data[0] = data[0][idxs]
        hf.close()
    return data


def combine_stat(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt(
        (
            std_0**2 * n_0
            + std_1**2 * n_1
            + (mean_0 - mean) ** 2 * n_0
            + (mean_1 - mean) ** 2 * n_1
        )
        / (n_0 + n_1)
    )
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)


def init_stat(dim):
    # mean, std, count
    return np.zeros((dim, 3))


def normalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            stat[i][stat[i][:, 1] == 0, 1] = 1.0
            s = Variable(torch.FloatTensor(stat[i]).cuda())

            stat_dim = stat[i].shape[0]
            n_rep = int(data[i].size(1) / stat_dim)
            data[i] = data[i].view(-1, n_rep, stat_dim)

            data[i] = (data[i] - s[:, 0]) / s[:, 1]

            data[i] = data[i].view(-1, n_rep * stat_dim)

    else:
        for i in range(len(stat)):
            stat[i][stat[i][:, 1] == 0, 1] = 1.0

            stat_dim = stat[i].shape[0]
            n_rep = int(data[i].shape[1] / stat_dim)
            data[i] = data[i].reshape((-1, n_rep, stat_dim))

            data[i] = (data[i] - stat[i][:, 0]) / stat[i][:, 1]

            data[i] = data[i].reshape((-1, n_rep * stat_dim))

    return data


def denormalize(data, stat, var=False):
    if var:
        for i in range(len(stat)):
            s = Variable(torch.FloatTensor(stat[i]).cuda())
            data[i] = data[i] * s[:, 1] + s[:, 0]
    else:
        for i in range(len(stat)):
            data[i] = data[i] * stat[i][:, 1] + stat[i][:, 0]

    return data


def calc_rigid_transform(XX, YY):
    X = XX.copy().T
    Y = YY.copy().T

    mean_X = np.mean(X, 1, keepdims=True)
    mean_Y = np.mean(Y, 1, keepdims=True)
    X = X - mean_X
    Y = Y - mean_Y
    C = np.dot(X, Y.T)
    U, S, Vt = np.linalg.svd(C)
    D = np.eye(3)
    D[2, 2] = np.linalg.det(np.dot(Vt.T, U.T))
    R = np.dot(Vt.T, np.dot(D, U.T))
    T = mean_Y - np.dot(R, mean_X)

    """
    YY_fitted = (np.dot(R, XX.T) + T).T
    print("MSE fit", np.mean(np.square(YY_fitted - YY)))
    """

    return R, T


def normalize_scene_param(scene_params, param_idx, param_range, norm_range=(-1, 1)):
    normalized = np.copy(scene_params[:, param_idx])
    low, high = param_range
    if low == high:
        return normalized
    nlow, nhigh = norm_range
    normalized = nlow + (normalized - low) * (nhigh - nlow) / (high - low)
    return normalized


def gen_PyFleX(info):
    env, env_idx = info["env"], info["env_idx"]
    thread_idx, data_dir, data_names = (
        info["thread_idx"],
        info["data_dir"],
        info["data_names"],
    )
    n_rollout, time_step = info["n_rollout"], info["time_step"]
    shape_state_dim, dt = info["shape_state_dim"], info["dt"]

    gen_vision = info["gen_vision"]
    vision_dir, vis_width, vis_height = (
        info["vision_dir"],
        info["vis_width"],
        info["vis_height"],
    )

    np.random.seed(round(time.time() * 1000 + thread_idx) % 2**32)

    # positions
    stats = [init_stat(3)]

    import pyflex

    pyflex.init()

    for i in range(n_rollout):
        if i % 10 == 0:
            print("%d / %d" % (i, n_rollout))

        rollout_idx = thread_idx * n_rollout + i
        rollout_dir = os.path.join(data_dir, str(rollout_idx))
        os.system("mkdir -p " + rollout_dir)

        if env == "RigidFall":
            g_low, g_high = info["physics_param_range"]
            gravity = rand_float(g_low, g_high)
            print(
                "Generated RigidFall rollout {} with gravity {} from range {} ~ {}".format(
                    i, gravity, g_low, g_high
                )
            )

            n_instance = 3
            draw_mesh = 1
            scene_params = np.zeros(n_instance * 3 + 3)
            scene_params[0] = n_instance
            scene_params[1] = gravity
            scene_params[-1] = draw_mesh

            low_bound = 0.09
            for j in range(n_instance):
                x = rand_float(0.0, 0.1)
                y = rand_float(low_bound, low_bound + 0.01)
                z = rand_float(0.0, 0.1)

                scene_params[j * 3 + 2] = x
                scene_params[j * 3 + 3] = y
                scene_params[j * 3 + 4] = z

                low_bound += 0.21

            pyflex.set_scene(env_idx, scene_params, thread_idx)
            pyflex.set_camPos(np.array([0.2, 0.875, 2.0]))

            n_particles = pyflex.get_n_particles()
            n_shapes = 1  # the floor

            positions = np.zeros(
                (time_step, n_particles + n_shapes, 3), dtype=np.float32
            )
            shape_quats = np.zeros((time_step, n_shapes, 4), dtype=np.float32)

            for j in range(time_step):
                positions[j, :n_particles] = pyflex.get_positions().reshape(-1, 4)[
                    :, :3
                ]

                ref_positions = positions[0]

                for k in range(n_instance):
                    XX = ref_positions[64 * k : 64 * (k + 1)]
                    YY = positions[j, 64 * k : 64 * (k + 1)]

                    X = XX.copy().T
                    Y = YY.copy().T

                    mean_X = np.mean(X, 1, keepdims=True)
                    mean_Y = np.mean(Y, 1, keepdims=True)
                    X = X - mean_X
                    Y = Y - mean_Y
                    C = np.dot(X, Y.T)
                    U, S, Vt = np.linalg.svd(C)
                    D = np.eye(3)
                    D[2, 2] = np.linalg.det(np.dot(Vt.T, U.T))
                    R = np.dot(Vt.T, np.dot(D, U.T))
                    t = mean_Y - np.dot(R, mean_X)

                    YY_fitted = (np.dot(R, XX.T) + t).T
                    # print("MSE fit", np.mean(np.square(YY_fitted - YY)))

                    positions[j, 64 * k : 64 * (k + 1)] = YY_fitted

                if gen_vision:
                    pyflex.step(
                        capture=True, path=os.path.join(rollout_dir, str(j) + ".tga")
                    )
                else:
                    pyflex.step()

                data = [positions[j], shape_quats[j], scene_params]
                store_data(data_names, data, os.path.join(rollout_dir, str(j) + ".h5"))

            if gen_vision:
                images = np.zeros((time_step, vis_height, vis_width, 3), dtype=np.uint8)
                for j in range(time_step):
                    img_path = os.path.join(rollout_dir, str(j) + ".tga")
                    img = scipy.misc.imread(img_path)[:, :, :3][:, :, ::-1]
                    img = cv2.resize(
                        img, (vis_width, vis_height), interpolation=cv2.INTER_AREA
                    )
                    images[j] = img
                    os.system("rm " + img_path)

                store_data(
                    ["positions", "images", "scene_params"],
                    [positions, images, scene_params],
                    os.path.join(vision_dir, str(rollout_idx) + ".h5"),
                )

        elif env == "MassRope":
            s_low, s_high = info["physics_param_range"]
            stiffness = rand_float(s_low, s_high)
            print(
                "Generated MassRope rollout {} with gravity {} from range {} ~ {}".format(
                    i, stiffness, s_low, s_high
                )
            )

            x = 0.0
            y = 1.0
            z = 0.0
            length = 0.7
            draw_mesh = 1.0

            scene_params = np.array([x, y, z, length, stiffness, draw_mesh])

            pyflex.set_scene(env_idx, scene_params, 0)
            pyflex.set_camPos(np.array([0.13, 2.0, 3.2]))

            action = np.zeros(3)

            # the last particle is the pin, regarded as shape
            n_particles = pyflex.get_n_particles() - 1
            n_shapes = 1  # the mass at the top of the rope

            positions = np.zeros(
                (time_step + 1, n_particles + n_shapes, 3), dtype=np.float32
            )
            shape_quats = np.zeros((time_step + 1, n_shapes, 4), dtype=np.float32)

            action = np.zeros(3)
            for j in range(time_step + 1):
                positions[j] = pyflex.get_positions().reshape(-1, 4)[:, :3]
                if j >= 1:
                    # append the action (position of the pin) to the previous time step
                    positions[j - 1, -1, :] = positions[j, -1, :]

                ref_positions = positions[0]

                # apply rigid projection to the rigid object
                # cube: [0, 81)
                # rope: [81, 95)
                # pin: [95, 96)
                XX = ref_positions[:81]
                YY = positions[j, :81]

                X = XX.copy().T
                Y = YY.copy().T

                mean_X = np.mean(X, 1, keepdims=True)
                mean_Y = np.mean(Y, 1, keepdims=True)
                X = X - mean_X
                Y = Y - mean_Y
                C = np.dot(X, Y.T)
                U, S, Vt = np.linalg.svd(C)
                D = np.eye(3)
                D[2, 2] = np.linalg.det(np.dot(Vt.T, U.T))
                R = np.dot(Vt.T, np.dot(D, U.T))
                t = mean_Y - np.dot(R, mean_X)

                YY_fitted = (np.dot(R, XX.T) + t).T

                positions[j, :81] = YY_fitted

                scale = 0.1
                action[0] += rand_float(-scale, scale) - positions[j, -1, 0] * 0.1
                action[2] += rand_float(-scale, scale) - positions[j, -1, 2] * 0.1

                if gen_vision:
                    pyflex.step(
                        action * dt,
                        capture=True,
                        path=os.path.join(rollout_dir, str(j) + ".tga"),
                    )
                else:
                    pyflex.step(action * dt)

                if j >= 1:
                    data = [positions[j - 1], shape_quats[j - 1], scene_params]
                    store_data(
                        data_names, data, os.path.join(rollout_dir, str(j - 1) + ".h5")
                    )

            if gen_vision:
                images = np.zeros((time_step, vis_height, vis_width, 3), dtype=np.uint8)
                for j in range(time_step):
                    img_path = os.path.join(rollout_dir, str(j) + ".tga")
                    img = scipy.misc.imread(img_path)[:, :, :3][:, :, ::-1]
                    img = cv2.resize(
                        img, (vis_width, vis_height), interpolation=cv2.INTER_AREA
                    )
                    images[j] = img
                    os.system("rm " + img_path)

                store_data(
                    ["positions", "images", "scene_params"],
                    [positions, images, scene_params],
                    os.path.join(vision_dir, str(rollout_idx) + ".h5"),
                )

        else:
            raise AssertionError("Unsupported env")

        # change dtype for more accurate stat calculation
        # only normalize positions
        datas = [positions[:time_step].astype(np.float64)]

        for j in range(len(stats)):
            stat = init_stat(stats[j].shape[0])
            stat[:, 0] = np.mean(datas[j], axis=(0, 1))[:]
            stat[:, 1] = np.std(datas[j], axis=(0, 1))[:]
            stat[:, 2] = datas[j].shape[0] * datas[j].shape[1]
            stats[j] = combine_stat(stats[j], stat)

    pyflex.clean()

    return stats


def axisEqual3D(ax):
    extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)


def visualize_neighbors(anchors, queries, idx, neighbors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(queries[idx, 0], queries[idx, 1], queries[idx, 2], c="g", s=80)
    ax.scatter(
        anchors[neighbors, 0], anchors[neighbors, 1], anchors[neighbors, 2], c="r", s=80
    )
    ax.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], alpha=0.2)
    axisEqual3D(ax)

    plt.show()


def find_relations_neighbor(pos, query_idx, anchor_idx, radius, order, var=False):
    if np.sum(anchor_idx) == 0:
        return []

    point_tree = spatial.cKDTree(pos[anchor_idx])
    neighbors = point_tree.query_ball_point(pos[query_idx], radius, p=order)

    """
    for i in range(len(neighbors)):
        visualize_neighbors(pos[anchor_idx], pos[query_idx], i, neighbors[i])
    """

    relations = []
    for i in range(len(neighbors)):
        count_neighbors = len(neighbors[i])
        if count_neighbors == 0:
            continue

        receiver = np.ones(count_neighbors, dtype=int) * query_idx[i]
        sender = np.array(anchor_idx[neighbors[i]])

        # receiver, sender, relation_type
        relations.append(np.stack([receiver, sender], axis=1))

    return relations


def find_k_relations_neighbor(
    k, positions, query_idx, anchor_idx, radius, order, var=False
):
    """
    Same as find_relations_neighbor except that each point is only connected to the k nearest neighbors

    For each particle, only take the first min_neighbor neighbors, where
    min_neighbor = minimum number of neighbors among all particle's numbers of neighbors
    """
    if np.sum(anchor_idx) == 0:
        return []

    pos = positions.data.cpu().numpy() if var else positions

    point_tree = spatial.cKDTree(pos[anchor_idx])
    neighbors = point_tree.query_ball_point(pos[query_idx], radius, p=order)

    """
    for i in range(len(neighbors)):
        visualize_neighbors(pos[anchor_idx], pos[query_idx], i, neighbors[i])
    """

    relations = []
    min_neighbors = None
    for i in range(len(neighbors)):
        if min_neighbors is None:
            min_neighbors = len(neighbors[i])
        elif len(neighbors[i]) < min_neighbors:
            min_neighbors = len(neighbors[i])
        else:
            pass

    for i in range(len(neighbors)):
        receiver = np.ones(min_neighbors, dtype=int) * query_idx[i]
        sender = np.array(anchor_idx[neighbors[i][:min_neighbors]])

        # receiver, sender, relation_type
        relations.append(np.stack([receiver, sender], axis=1))

    return relations


def get_scene_info(data):
    """
    A subset of prepare_input() just to get number of particles
    for initialization of grouping
    """
    positions, shape_quats, scene_params = data
    n_shapes = shape_quats.shape[0]
    count_nodes = positions.shape[0]
    n_particles = count_nodes - n_shapes

    return n_particles, n_shapes, scene_params


from time import time
def worker_func(args):
    idx, data_path, k, state_dim = args
    particles_path = os.path.join(data_path, str(idx), "x_t.npy")
    indices_path = os.path.join(data_path, str(idx), "fps.npy")
    quat_path = os.path.join(data_path, str(idx), "quat.npy")
    indices_i = np.load(indices_path, mmap_mode="r+")
    loaded_particles = np.load(particles_path, mmap_mode="r+")
    particles_i = np.concatenate((loaded_particles[indices_i], np.expand_dims(loaded_particles[-1], 0)), axis=0)
    quats_i = np.load(quat_path)
    # Access the existing shared memory
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    existing_shm_indices = shared_memory.SharedMemory(name=shm_name_indices)
    existing_shm_quats = shared_memory.SharedMemory(name=shm_name_quats)
    np_array = np.ndarray(
        (n_rollout, k + 1, time_step, state_dim), dtype="float", buffer=existing_shm.buf
    )
    np_array_indices = np.ndarray(
        (n_rollout, k), dtype=int, buffer=existing_shm_indices.buf
    )
    np_array_quats = np.ndarray(
        (n_rollout, time_step, 4), dtype='float64', buffer=existing_shm_quats.buf
    )

    # Write data to shared memory
    np_array[idx] = particles_i
    np_array_indices[idx] = indices_i
    np_array_quats[idx] = quats_i
    existing_shm_indices.close()
    existing_shm.close()
    existing_shm_quats.close()


def get_scene_info_fluidlab(args, data_path, phase):
    global shm_name, shm_name_indices, n_rollout, time_step
    global shm_name_quats
    n_rollout, time_step = args.n_rollout, args.time_step
    if phase == "valid":
        n_rollout = args.n_rollout_valid
    else:
        n_rollout = args.n_rollout
    k, state_dim = args.k, args.state_dim
    particles = np.zeros((n_rollout, k + 1, time_step, state_dim))
    sampled_indices = np.zeros((n_rollout, k), dtype=int)
    quats = np.zeros((n_rollout, time_step, 4))
    shm = shared_memory.SharedMemory(create=True, size=particles.nbytes)
    shm_indices = shared_memory.SharedMemory(create=True, size=sampled_indices.nbytes)
    shm_quats = shared_memory.SharedMemory(create=True, size=quats.nbytes)
    shm_name = shm.name  # Save shared memory name globally
    shm_name_indices = shm_indices.name
    shm_name_quats = shm_quats.name
    np_array = np.ndarray(
        (n_rollout, k + 1, time_step, state_dim), dtype="float64", buffer=shm.buf
    )
    np_array_indices= np.ndarray(
        (n_rollout, k), dtype=int, buffer=shm_indices.buf
    )
    np_array_quats = np.ndarray(
        (n_rollout, time_step, 4), dtype='float64', buffer=shm_quats.buf
    )
    np_array_indices[:] = sampled_indices[:]
    np_array[:] = particles[:]
    np_array_quats[:] = quats[:]

    a = time()
    with Pool(10) as pool:
        pool.map(worker_func, [(i, data_path, k, state_dim) for i in range(n_rollout)])
    print(time() - a, "seconds")
    # Load scene params
    params_path = os.path.join(data_path, "0", "stat.npy")
    scene_params = np.load(params_path, mmap_mode="r")
    scene_params = np.expand_dims(scene_params.copy(), 0)
    scene_params = np.squeeze(scene_params[:, np_array_indices], 0)
    # Close and unlink shared memory
    particles = np.copy(np_array.transpose(0, 2, 1, 3)[:])
    quats = np.copy(np_array_quats[:])
    shm.close()
    shm.unlink()
    shm_indices.close()
    shm_indices.unlink()
    shm_quats.close()
    shm_quats.unlink()
    n_shapes = 1
    return particles, n_shapes, scene_params, quats


def load_data_fluidlab(args, data_path):
    particles_path = os.path.join(data_path, "x_t.npy") 
    indices_path = os.path.join(data_path, "fps.npy")
    indices = np.load(indices_path, mmap_mode="r+")
    params_path = os.path.join(data_path, "stat.npy")
    particles = np.load(particles_path, mmap_mode="r+")
    particles = particles[indices]
    pos_boundary = np.array([[0.5, 0.75, 0.5]])
    particles = np.transpose(particles, (1, 0, 2))
    n_particle = particles.shape[1]
    pos_boundary = np.tile(pos_boundary, (args.time_step, 1, 1) )
    particles = np.concatenate((particles, pos_boundary), axis=1)
    scene_params = np.load(params_path, mmap_mode="r+")[indices]
    scene_params = np.expand_dims(scene_params, 0)
    n_shapes = 1
    return particles, n_particle, n_shapes, scene_params, indices.copy()

def rotation_matrix_from_quaternion(params):
        # params: (B * n_instance) x 4
        # w, x, y, z

        one = torch.ones(1, 1)
        zero = torch.zeros(1, 1)
        use_gpu = True
        if use_gpu:
            one = one.cuda()
            zero = zero.cuda()

        # multiply the rotation matrix from the right-hand side
        # the matrix should be the transpose of the conventional one

        # Reference
        # http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm

        params = params / torch.norm(params, dim=1, keepdim=True)
        w, x, y, z = \
                params[:, 0].view(-1, 1, 1), params[:, 1].view(-1, 1, 1), \
                params[:, 2].view(-1, 1, 1), params[:, 3].view(-1, 1, 1)

        rot = torch.cat((
            torch.cat((one - y * y * 2 - z * z * 2, x * y * 2 + z * w * 2, x * z * 2 - y * w * 2), 2),
            torch.cat((x * y * 2 - z * w * 2, one - x * x * 2 - z * z * 2, y * z * 2 + x * w * 2), 2),
            torch.cat((x * z * 2 + y * w * 2, y * z * 2 - x * w * 2, one - x * x * 2 - y * y * 2), 2)), 1)

        # rot: (B * n_instance) x 3 x 3
        return rot
    
def get_env_group(args, n_particles, scene_params, use_gpu=False):
    # n_particles (int)
    # scene_params: B x param_dim
    B = scene_params.shape[0]
    if not torch.is_tensor(scene_params):
        scene_params = torch.FloatTensor(scene_params)
    p_rigid = torch.zeros(B, args.n_instance)
    p_instance = torch.zeros(B, n_particles, args.n_instance)
    physics_param = torch.zeros(B, n_particles)

    if args.env == "RigidFall":
        norm_g = normalize_scene_param(scene_params, 1, args.physics_param_range)
        physics_param[:] = torch.FloatTensor(norm_g).view(B, 1)

        p_rigid[:] = 1

        for i in range(args.n_instance):
            p_instance[:, 64 * i : 64 * (i + 1), i] = 1

    elif args.env == "MassRope":
        norm_stiff = normalize_scene_param(scene_params, 4, args.physics_param_range)
        physics_param[:] = torch.FloatTensor(norm_stiff).view(B, 1)

        n_rigid_particle = 81

        p_rigid[:, 0] = 1
        p_instance[:, :n_rigid_particle, 0] = 1
        p_instance[:, n_rigid_particle:, 1] = 1

    elif args.env in ["LatteArt", "Pouring"]:
        p_rigid[:] = 0
        p_instance[:, :, 0] = scene_params[:, :n_particles] == 0
        p_instance[:, :, 1] = 1 - p_instance[:, :, 0]
    else:
        raise AssertionError("Unsupported env")

    if use_gpu:
        p_rigid = p_rigid.cuda()
        p_instance = p_instance.cuda()
        physics_param = physics_param.cuda()

    # p_rigid: B x n_instance
    # p_instance: B x n_p x n_instance
    # physics_param: B x n_p
    return [p_rigid, p_instance, physics_param]


def prepare_input(positions, n_particle, n_shape, args, bottom_cup_xyz, var=False, shape_quat=None):
    # positions: (n_p + n_s) x 3

    verbose = args.verbose_data

    count_nodes = n_particle + n_shape

    if verbose:
        print("prepare_input::positions", positions.shape)
        print("prepare_input::n_particle", n_particle)
        print("prepare_input::n_shape", n_shape)

    ### object attributes
    attr = np.zeros((count_nodes, args.attr_dim))

    ##### add env specific graph components
    rels = []
    if args.env == "RigidFall":
        # object attr:
        # [particle, floor]
        attr[n_particle, 1] = 1
        pos = positions.data.cpu().numpy() if var else positions

        # conncetion between floor and particles when they are close enough
        dis = pos[:n_particle, 1] - pos[n_particle, 1]
        nodes = np.nonzero(dis < args.neighbor_radius)[0]

        """
        if verbose:
            visualize_neighbors(pos, pos, 0, nodes)
            print(np.sort(dis)[:10])
        """

        floor = np.ones(nodes.shape[0], dtype=int) * n_particle
        rels += [np.stack([nodes, floor], axis=1)]

    elif args.env == "MassRope":
        pos = positions.data.cpu().numpy() if var else positions
        dis = np.sqrt(np.sum((pos[n_particle] - pos[:n_particle]) ** 2, 1))
        nodes = np.nonzero(dis < args.neighbor_radius)[0]

        """
        if verbose:
            visualize_neighbors(pos, pos, 0, nodes)
            print(np.sort(dis)[:10])
        """

        pin = np.ones(nodes.shape[0], dtype=int) * n_particle
        rels += [np.stack([nodes, pin], axis=1)]

    elif args.env == "LatteArt":
        pos = (
            positions.cpu().numpy()[:n_particle]
            if torch.is_tensor(positions)
            else positions[:n_particle]
        )
        # hard-code the position of the cup boundary (treat it as a particle at the center)
        pos_boundary = np.array([[0.5, 0.75, 0.5]])
        positions = np.concatenate((pos, pos_boundary), axis=0)
        dis = np.sqrt(np.sum((pos_boundary - pos) ** 2, 1))
        nodes = np.nonzero(dis < args.neighbor_radius)[0]
        cup = np.ones(nodes.shape[0], dtype=int) * n_particle
        rels += [np.stack([nodes, cup], axis=1)]

    elif args.env == "Pouring":
        pos = (
            positions.cpu().numpy()[:n_particle]
            if torch.is_tensor(positions)
            else positions[:n_particle]
        )
        # shape_quat = np.expand_dims(shape_quat, 0)
        # shape_quat = torch.from_numpy(shape_quat).to(torch.float64).to("cuda")
        # rot_matrix = rotation_matrix_from_quaternion(shape_quat)
        # bottom_cup_xyz = (bottom_cup_xyz @ rot_matrix).squeeze(0)
        # positions[-1] = bottom_cup_xyz.cpu().numpy()
        # dis = np.sqrt(np.sum((positions[n_particle] - positions[:n_particle]) ** 2, 1))
        # nodes = np.nonzero(dis < args.neighbor_radius)[0]
        # cup = np.ones(nodes.shape[0], dtype=int) * n_particle
        # rels += [np.stack([nodes, cup], axis=1)]
        
    else:
        AssertionError("Unsupported env %s" % args.env)

    ##### add relations between leaf particles

    if args.env in ["RigidFall", "MassRope", "LatteArt", "Pouring"]:
        queries = np.arange(n_particle)
        anchors = np.arange(n_particle)
    
    rels += find_relations_neighbor(pos, queries, anchors, args.neighbor_radius, 2, var)
    # rels += find_k_relations_neighbor(args.neighbor_k, pos, queries, anchors, args.neighbor_radius, 2, var)

    if len(rels) > 0:
        rels = np.concatenate(rels, 0)

    if verbose:
        print("Relations neighbor", rels.shape)

    n_rel = rels.shape[0]
    Rr = torch.zeros(n_rel, n_particle + n_shape)
    Rs = torch.zeros(n_rel, n_particle + n_shape)
    Rr[np.arange(n_rel), rels[:, 0]] = 1
    Rs[np.arange(n_rel), rels[:, 1]] = 1

    if verbose:
        print("Object attr:", np.sum(attr, axis=0))
        print("Particle attr:", np.sum(attr[:n_particle], axis=0))
        print("Shape attr:", np.sum(attr[n_particle : n_particle + n_shape], axis=0))

    if verbose:
        print("Particle positions stats")
        print("  Shape", positions.shape)
        print("  Min", np.min(positions[:n_particle], 0))
        print("  Max", np.max(positions[:n_particle], 0))
        print("  Mean", np.mean(positions[:n_particle], 0))
        print("  Std", np.std(positions[:n_particle], 0))

    if var:
        particle = positions
    else:
        particle = torch.FloatTensor(positions)

    if verbose:
        for i in range(count_nodes - 1):
            if np.sum(np.abs(attr[i] - attr[i + 1])) > 1e-6:
                print(i, attr[i], attr[i + 1])

    attr = torch.FloatTensor(attr)
    assert attr.size(0) == count_nodes
    assert attr.size(1) == args.attr_dim

    # attr: (n_p + n_s) x attr_dim
    # particle (unnormalized): (n_p + n_s) x state_dim
    # Rr, Rs: n_rel x (n_p + n_s)
    return attr, particle, Rr, Rs, bottom_cup_xyz


class FluidLabDataset(Dataset):
    def __init__(self, args, phase, K=300):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.phase = phase
        args.k = self.K = K
        self.data_dir = os.path.join(self.args.dataf, phase)
        self.vision_dir = self.data_dir + "_vision"
        self.bottom_cup_xyz = torch.Tensor([0.5, 0.05, 0.5]).to(torch.float64).to("cuda")
        self.stat_path = os.path.join(self.args.dataf, "stat.h5")
        self.all_particles, self.n_shape, self.all_scene_params, self.all_shape_quats = get_scene_info_fluidlab(
            args, self.data_dir, phase
        )
        self.n_particle = self.K
        if args.gen_data:
            os.system("mkdir -p " + self.data_dir)
        if args.gen_vision:
            os.system("mkdir -p " + self.vision_dir)
        if args.env in ["LatteArt", "Pouring"]:
            self.data_names = ["x"]
        else:
            raise AssertionError("Unsupported env")
        ratio = self.args.train_valid_ratio
        if phase == "train":
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase == "valid":
            self.n_rollout = args.n_rollout_valid
        else:
            raise AssertionError("Unknown phase")
        print(f"phase: {phase} self.n_rollout: {self.n_rollout}")

    def __len__(self):
        """
        Each data point is consisted of a whole trajectory
        """
        args = self.args
        return self.n_rollout * (args.time_step - args.sequence_length + 1)

    def load_data(self, name):
        pass

    def __getitem__(self, idx):
        """
        Load a trajectory of length sequence_length
        """
        args = self.args

        offset = args.time_step - args.sequence_length + 1
        idx_rollout = idx // offset
        st_idx = idx % offset
        ed_idx = st_idx + args.sequence_length
        if args.stage in ["dy"]:
            # load ground truth data
            attrs, particles, Rrs, Rss = [], [], [], []
            max_n_rel = 0
            scene_params = self.all_scene_params[idx_rollout]
            for t in range(st_idx, ed_idx):
                # load data
                points = self.all_particles[idx_rollout, t] 
                shape_quat = self.all_shape_quats[idx_rollout,t]
                attr, particle, Rr, Rs, self.bottom_cup_xyz = prepare_input(
                    points, self.n_particle, self.n_shape, self.args, self.bottom_cup_xyz, shape_quat=shape_quat,
                )

                max_n_rel = max(max_n_rel, Rr.size(0))

                attrs.append(attr)
                particles.append(particle.numpy())
                Rrs.append(Rr)
                Rss.append(Rs)
        """
        add augmentation
        """
        if args.stage in ["dy"]:
            for t in range(args.sequence_length):
                if t == args.n_his - 1:
                    # set anchor for transforming rigid objects
                    particle_anchor = particles[t].copy()

                if t < args.n_his:
                    # add noise to observation frames - idx smaller than n_his
                    noise = (
                        np.random.randn(self.n_particle, 3) * args.std_d * args.augment_ratio
                    )
                    particles[t][:self.n_particle] += noise
        else:
            AssertionError("Unknown stage %s" % args.stage)

        # attr: (n_p + n_s) x attr_dim
        # particles (unnormalized): seq_length x (n_p + n_s) x state_dim
        # scene_params: param_dim
        attr = torch.FloatTensor(attrs[0])
        particles = torch.FloatTensor(np.stack(particles))
        scene_params = torch.FloatTensor(scene_params)

        # pad the relation set
        # Rr, Rs: seq_length x n_rel x (n_p + n_s)
        if args.stage in ["dy"]:
            for i in range(len(Rrs)):
                Rr, Rs = Rrs[i], Rss[i]
                Rr = torch.cat(
                    [Rr, torch.zeros(max_n_rel - Rr.size(0), self.n_particle + self.n_shape)], 0
                )
                Rs = torch.cat(
                    [Rs, torch.zeros(max_n_rel - Rs.size(0), self.n_particle + self.n_shape)], 0
                )
                Rrs[i], Rss[i] = Rr, Rs
            Rr = torch.FloatTensor(np.stack(Rrs))
            Rs = torch.FloatTensor(np.stack(Rss))

        if args.stage in ["dy"]:
            return attr, particles, self.n_particle, self.n_shape, scene_params, Rr, Rs


class PhysicsFleXDataset(Dataset):
    def __init__(self, args, phase):
        self.args = args
        self.phase = phase
        self.data_dir = os.path.join(self.args.dataf, phase)
        self.vision_dir = self.data_dir + "_vision"
        self.stat_path = os.path.join(self.args.dataf, "stat.h5")

        if args.gen_data:
            os.system("mkdir -p " + self.data_dir)
        if args.gen_vision:
            os.system("mkdir -p " + self.vision_dir)

        if args.env in ["RigidFall", "MassRope"]:
            self.data_names = ["positions", "shape_quats", "scene_params"]
        else:
            raise AssertionError("Unsupported env")

        ratio = self.args.train_valid_ratio
        if phase == "train":
            self.n_rollout = int(self.args.n_rollout * ratio)
        elif phase == "valid":
            self.n_rollout = self.args.n_rollout - int(self.args.n_rollout * ratio)
        else:
            raise AssertionError("Unknown phase")

    def __len__(self):
        """
        Each data point is consisted of a whole trajectory
        """
        args = self.args
        return self.n_rollout * (args.time_step - args.sequence_length + 1)

    def load_data(self, name):
        print("Loading stat from %s ..." % self.stat_path)
        self.stat = load_data(self.data_names[:1], self.stat_path)

    def gen_data(self, name):
        # if the data hasn't been generated, generate the data
        print(
            "Generating data ... n_rollout=%d, time_step=%d"
            % (self.n_rollout, self.args.time_step)
        )

        infos = []
        for i in range(self.args.num_workers):
            info = {
                "env": self.args.env,
                "thread_idx": i,
                "data_dir": self.data_dir,
                "data_names": self.data_names,
                "n_rollout": self.n_rollout // self.args.num_workers,
                "time_step": self.args.time_step,
                "dt": self.args.dt,
                "shape_state_dim": self.args.shape_state_dim,
                "physics_param_range": self.args.physics_param_range,
                "gen_vision": self.args.gen_vision,
                "vision_dir": self.vision_dir,
                "vis_width": self.args.vis_width,
                "vis_height": self.args.vis_height,
            }

            if self.args.env == "RigidFall":
                info["env_idx"] = 3
            elif self.args.env == "MassRope":
                info["env_idx"] = 9
            else:
                raise AssertionError("Unsupported env")

            infos.append(info)

        cores = self.args.num_workers
        pool = mp.Pool(processes=cores)
        data = pool.map(gen_PyFleX, infos)

        print("Training data generated, warpping up stats ...")

        if self.phase == "train" and self.args.gen_stat:
            # positions [x, y, z]
            self.stat = [init_stat(3)]
            for i in range(len(data)):
                for j in range(len(self.stat)):
                    self.stat[j] = combine_stat(self.stat[j], data[i][j])
            store_data(self.data_names[:1], self.stat, self.stat_path)
        else:
            print("Loading stat from %s ..." % self.stat_path)
            self.stat = load_data(self.data_names[:1], self.stat_path)

    def __getitem__(self, idx):
        """
        Load a trajectory of length sequence_length
        """
        args = self.args

        offset = args.time_step - args.sequence_length + 1

        idx_rollout = idx // offset
        st_idx = idx % offset
        ed_idx = st_idx + args.sequence_length

        if args.stage in ["dy"]:
            # load ground truth data
            attrs, particles, Rrs, Rss = [], [], [], []
            max_n_rel = 0
            for t in range(st_idx, ed_idx):
                # load data
                data_path = os.path.join(
                    self.data_dir, str(idx_rollout), str(t) + ".h5"
                )
                data = load_data(self.data_names, data_path)

                # load scene param
                if t == st_idx:
                    n_particle, n_shape, scene_params = get_scene_info(data)

                # attr: (n_p + n_s) x attr_dim
                # particle (unnormalized): (n_p + n_s) x state_dim
                # Rr, Rs: n_rel x (n_p + n_s)
                attr, particle, Rr, Rs = prepare_input(
                    data[0], n_particle, n_shape, self.args
                )

                max_n_rel = max(max_n_rel, Rr.size(0))

                attrs.append(attr)
                particles.append(particle.numpy())
                Rrs.append(Rr)
                Rss.append(Rs)
        """
        add augmentation
        """
        if args.stage in ["dy"]:
            for t in range(args.sequence_length):
                if t == args.n_his - 1:
                    # set anchor for transforming rigid objects
                    particle_anchor = particles[t].copy()

                if t < args.n_his:
                    # add noise to observation frames - idx smaller than n_his
                    noise = (
                        np.random.randn(n_particle, 3) * args.std_d * args.augment_ratio
                    )
                    particles[t][:n_particle] += noise

                else:
                    # for augmenting rigid object,
                    # make sure the rigid transformation is the same before and after augmentation
                    if args.env == "RigidFall":
                        for k in range(args.n_instance):
                            XX = particle_anchor[64 * k : 64 * (k + 1)]
                            XX_noise = particles[args.n_his - 1][64 * k : 64 * (k + 1)]

                            YY = particles[t][64 * k : 64 * (k + 1)]

                            R, T = calc_rigid_transform(XX, YY)

                            particles[t][64 * k : 64 * (k + 1)] = (
                                np.dot(R, XX_noise.T) + T
                            ).T

                            """
                            # checking the correctness of the implementation
                            YY_noise = particles[t][64*k:64*(k+1)]
                            RR, TT = calc_rigid_transform(XX_noise, YY_noise)
                            print(R, T)
                            print(RR, TT)
                            """

                    elif args.env == "MassRope":
                        n_rigid_particle = 81

                        XX = particle_anchor[:n_rigid_particle]
                        XX_noise = particles[args.n_his - 1][:n_rigid_particle]
                        YY = particles[t][:n_rigid_particle]

                        R, T = calc_rigid_transform(XX, YY)

                        particles[t][:n_rigid_particle] = (np.dot(R, XX_noise.T) + T).T

                        """
                        # checking the correctness of the implementation
                        YY_noise = particles[t][:n_rigid_particle]
                        RR, TT = calc_rigid_transform(XX_noise, YY_noise)
                        print(R, T)
                        print(RR, TT)
                        """

        else:
            AssertionError("Unknown stage %s" % args.stage)

        # attr: (n_p + n_s) x attr_dim
        # particles (unnormalized): seq_length x (n_p + n_s) x state_dim
        # scene_params: param_dim
        attr = torch.FloatTensor(attrs[0])
        particles = torch.FloatTensor(np.stack(particles))
        scene_params = torch.FloatTensor(scene_params)

        # pad the relation set
        # Rr, Rs: seq_length x n_rel x (n_p + n_s)
        if args.stage in ["dy"]:
            for i in range(len(Rrs)):
                Rr, Rs = Rrs[i], Rss[i]
                Rr = torch.cat(
                    [Rr, torch.zeros(max_n_rel - Rr.size(0), n_particle + n_shape)], 0
                )
                Rs = torch.cat(
                    [Rs, torch.zeros(max_n_rel - Rs.size(0), n_particle + n_shape)], 0
                )
                Rrs[i], Rss[i] = Rr, Rs
            Rr = torch.FloatTensor(np.stack(Rrs))
            Rs = torch.FloatTensor(np.stack(Rss))

        if args.stage in ["dy"]:
            return attr, particles, n_particle, n_shape, scene_params, Rr, Rs
