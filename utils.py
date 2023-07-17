from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable


def my_collate(batch):
    len_batch = len(batch[0])
    len_rel = 2

    ret = []
    for i in range(len_batch - len_rel):
        d = [item[i] for item in batch]
        if isinstance(d[0], int):
            d = torch.LongTensor(d)
        else:
            d = torch.FloatTensor(torch.stack(d))
        ret.append(d)

    # processing relations
    # R: B x seq_length x n_rel x (n_p + n_s)
    for i in range(len_rel):
        R = [item[-len_rel + i] for item in batch]
        max_n_rel = 0
        seq_length, _, N = R[0].size()
        for j in range(len(R)):
            max_n_rel = max(max_n_rel, R[j].size(1))
        for j in range(len(R)):
            r = R[j]
            r = torch.cat([r, torch.zeros(seq_length, max_n_rel - r.size(1), N)], 1)
            R[j] = r

        R = torch.FloatTensor(torch.stack(R))

        ret.append(R)

    return tuple(ret)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()

    def close(self):
        self.__del__()


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_gradient(step):
    def hook(grad):
        print(step, torch.mean(grad, 1)[:4])
    return hook


def add_log(fn, content, is_append=True):
    if is_append:
        with open(fn, "a+") as f:
            f.write(content)
    else:
        with open(fn, "w+") as f:
            f.write(content)


def rand_int(lo, hi):
    return np.random.randint(lo, hi)


def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_var(tensor, use_gpu, requires_grad=False):
    if use_gpu:
        return Variable(torch.FloatTensor(tensor).cuda(),
                        requires_grad=requires_grad)
    else:
        return Variable(torch.FloatTensor(tensor),
                        requires_grad=requires_grad)


def make_graph(log, title, args):
    """make a loss graph"""
    plt.plot(log)
    plt.xlabel('iter')
    plt.ylabel('loss')

    title + '_loss_graph'
    plt.title(title)
    plt.savefig(os.path.join(args.logf, title + '.png'))
    plt.close()


def get_color_from_prob(prob, colors):
    # there's only one instance
    if len(colors) == 1:
        return colors[0] * prob
    elif len(prob) == 1:
        return colors * prob[0]
    else:
        res = np.zeros(4)
        for i in range(len(prob)):
            res += prob[i] * colors[i]
        return res


def create_instance_colors(n):
    # TODO: come up with a better way to initialize instance colors
    return np.array([
        [1., 0., 0., 1.],
        [0., 1., 0., 1.],
        [0., 0., 1., 1.],
        [1., 1., 0., 1.],
        [1., 0., 1., 1.]])[:n]


def convert_groups_to_colors(group, n_particles, n_rigid_instances, instance_colors, env=None):
    """
    Convert grouping to RGB colors of shape (n_particles, 4)
    :param grouping: [p_rigid, p_instance, physics_param]
    :return: RGB values that can be set as color densities
    """
    # p_rigid: n_instance
    # p_instance: n_p x n_instance
    p_rigid, p_instance = group[:2]

    p = p_instance

    colors = np.empty((n_particles, 4))

    for i in range(n_particles):
        colors[i] = get_color_from_prob(p[i], instance_colors)

    # print("colors", colors)
    return colors


def visualize_point_clouds(point_clouds, c=['b', 'r'], view=None, store=False, store_path=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')

    frame = plt.gca()
    frame.axes.xaxis.set_ticklabels([])
    frame.axes.yaxis.set_ticklabels([])
    frame.axes.zaxis.set_ticklabels([])

    for i in range(len(point_clouds)):
        points = point_clouds[i]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=c[i], s=10, alpha=0.3)

    X, Y, Z = point_clouds[0][:, 0], point_clouds[0][:, 1], point_clouds[0][:, 2]

    max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    ax.grid(False)
    plt.show()

    if view is None:
        view = 0, 0
    ax.view_init(view[0], view[1])
    plt.draw()

    # plt.pause(5)

    if store:
        os.system('mkdir -p ' + store_path)
        fig.savefig(os.path.join(store_path, "vis.png"), bbox_inches='tight')

    '''
    for angle in range(0, 360, 2):
        ax.view_init(90, angle)
        plt.draw()
        # plt.pause(.001)

        if store:
            if angle % 100 == 0:
                print("Saving frame %d / %d" % (angle, 360))

            os.system('mkdir -p ' + store_path)
            fig.savefig(os.path.join(store_path, "%d.png" % angle), bbox_inches='tight')
    '''


def quatFromAxisAngle(axis, angle):
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat


def quatFromAxisAngle_var(axis, angle):
    axis /= torch.norm(axis)

    half = angle * 0.5
    w = torch.cos(half)

    sin_theta_over_two = torch.sin(half)
    axis *= sin_theta_over_two

    quat = torch.cat([axis, w])
    # print("quat size", quat.size())

    return quat


class ChamferLoss(torch.nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def chamfer_distance(self, x, y):
        # x: [N, D]
        # y: [M, D]
        x = x.repeat(y.size(0), 1, 1)  # x: [M, N, D]
        x = x.transpose(0, 1)  # x: [N, M, D]
        y = y.repeat(x.size(0), 1, 1)  # y: [N, M, D]
        dis = torch.norm(torch.add(x, -y), 2, dim=2)  # dis: [N, M]
        dis_xy = torch.mean(torch.min(dis, dim=1)[0])  # dis_xy: mean over N
        dis_yx = torch.mean(torch.min(dis, dim=0)[0])  # dis_yx: mean over M

        return dis_xy + dis_yx

    def __call__(self, pred, label):
        return self.chamfer_distance(pred, label)


def get_l2_loss(g):
    num_particles = len(g)
    return torch.norm(num_particles - torch.norm(g, dim=1, keepdim=True))

def render_vispy():
    ### render in VisPy
    import vispy.scene
    from vispy import app
    from vispy.visuals import transforms

    particle_size = 0.01
    border = 0.025
    height = 1.3
    y_rotate_deg = -45.0


    def y_rotate(obj, deg=y_rotate_deg):
        tr = vispy.visuals.transforms.MatrixTransform()
        tr.rotate(deg, (0, 1, 0))
        obj.transform = tr

    def add_floor(v):
        # add floor
        floor_length = 3.0
        w, h, d = floor_length, floor_length, border
        b1 = vispy.scene.visuals.Box(width=w, height=h, depth=d, color=[0.8, 0.8, 0.8, 1], edge_color='black')
        y_rotate(b1)
        v.add(b1)

        # adjust position of box
        mesh_b1 = b1.mesh.mesh_data
        v1 = mesh_b1.get_vertices()
        c1 = np.array([0., -particle_size - border, 0.], dtype=np.float32)
        mesh_b1.set_vertices(np.add(v1, c1))

        mesh_border_b1 = b1.border.mesh_data
        vv1 = mesh_border_b1.get_vertices()
        cc1 = np.array([0., -particle_size - border, 0.], dtype=np.float32)
        mesh_border_b1.set_vertices(np.add(vv1, cc1))

    def update_box_states(boxes, last_states, curr_states):
        v = curr_states[0] - last_states[0]
        if args.verbose_data:
            print("box states:", last_states, curr_states)
            print("box velocity:", v)

        tr = vispy.visuals.transforms.MatrixTransform()
        tr.rotate(y_rotate_deg, (0, 1, 0))

        for i, box in enumerate(boxes):
            # use v to update box translation
            trans = (curr_states[i][0], curr_states[i][1], curr_states[i][2])
            box.transform = tr * vispy.visuals.transforms.STTransform(translate=trans)

    def translate_box(b, x, y, z):
        mesh_b = b.mesh.mesh_data
        v = mesh_b.get_vertices()
        c = np.array([x, y, z], dtype=np.float32)
        mesh_b.set_vertices(np.add(v, c))

        mesh_border_b = b.border.mesh_data
        vv = mesh_border_b.get_vertices()
        cc = np.array([x, y, z], dtype=np.float32)
        mesh_border_b.set_vertices(np.add(vv, cc))

    def add_box(v, w=0.1, h=0.1, d=0.1, x=0.0, y=0.0, z=0.0):
        """
        Add a box object to the scene view
        :param v: view to which the box should be added
        :param w: width
        :param h: height
        :param d: depth
        :param x: x center
        :param y: y center
        :param z: z center
        :return: None
        """
        # render background box
        b = vispy.scene.visuals.Box(width=w, height=h, depth=d, color=[0.8, 0.8, 0.8, 1], edge_color='black')
        y_rotate(b)
        v.add(b)

        # adjust position of box
        translate_box(b, x, y, z)

        return b

    def calc_box_init(x, z):
        boxes = []

        # floor
        boxes.append([x, z, border, 0., -particle_size / 2, 0.])

        # left wall
        boxes.append([border, z, (height + border), -particle_size / 2, 0., 0.])

        # right wall
        boxes.append([border, z, (height + border), particle_size / 2, 0., 0.])

        # back wall
        boxes.append([(x + border * 2), border, (height + border)])

        # front wall (disabled when colored)
        # boxes.append([(x + border * 2), border, (height + border)])

        return boxes

    def add_container(v, box_x, box_z):
        boxes = calc_box_init(box_x, box_z)
        visuals = []
        for b in boxes:
            if len(b) == 3:
                visual = add_box(v, b[0], b[1], b[2])
            elif len(b) == 6:
                visual = add_box(v, b[0], b[1], b[2], b[3], b[4], b[5])
            else:
                raise AssertionError("Input should be either length 3 or length 6")
            visuals.append(visual)
        return visuals


    c = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')
    view = c.central_widget.add_view()

    if args.env == 'RigidFall':
        view.camera = vispy.scene.cameras.TurntableCamera(fov=50, azimuth=45, elevation=20, distance=2, up='+y')
        # set instance colors
        instance_colors = create_instance_colors(args.n_instance)

        # render floor
        add_floor(view)

    if args.env == 'MassRope':
        view.camera = vispy.scene.cameras.TurntableCamera(fov=30, azimuth=0, elevation=20, distance=8, up='+y')

        # set instance colors
        n_string_particles = 15
        instance_colors = create_instance_colors(args.n_instance)

        # render floor
        add_floor(view)


    # render particles
    p1 = vispy.scene.visuals.Markers()
    p1.antialias = 0  # remove white edge

    y_rotate(p1)

    view.add(p1)

    # set animation
    t_step = 0


    '''
    set up data for rendering
    '''
    #0 - p_pred: seq_length x n_p x 3
    #1 - p_gt: seq_length x n_p x 3
    #2 - s_gt: seq_length x n_s x 3
    print('p_pred', p_pred.shape)
    print('p_gt', p_gt.shape)
    print('s_gt', s_gt.shape)

    # create directory to save images if not exist
    vispy_dir = args.evalf + "/vispy"
    os.system('mkdir -p ' + vispy_dir)


    def update(event):
        global p1
        global t_step
        global colors

        if t_step < vis_length:
            if t_step == 0:
                print("Rendering ground truth")

            t_actual = t_step

            colors = convert_groups_to_colors(
                group_gt, n_particle, args.n_instance,
                instance_colors=instance_colors, env=args.env)

            colors = np.clip(colors, 0., 1.)

            p1.set_data(p_gt[t_actual, :n_particle], edge_color='black', face_color=colors)

            # render for ground truth
            img = c.render()
            img_path = os.path.join(vispy_dir, "gt_{}_{}.png".format(str(idx_episode), str(t_actual)))
            vispy.io.write_png(img_path, img)


        elif vis_length <= t_step < vis_length * 2:
            if t_step == vis_length:
                print("Rendering prediction result")

            t_actual = t_step - vis_length

            colors = convert_groups_to_colors(
                group_gt, n_particle, args.n_instance,
                instance_colors=instance_colors, env=args.env)

            colors = np.clip(colors, 0., 1.)

            p1.set_data(p_pred[t_actual, :n_particle], edge_color='black', face_color=colors)

            # render for perception result
            img = c.render()
            img_path = os.path.join(vispy_dir, "pred_{}_{}.png".format(str(idx_episode), str(t_actual)))
            vispy.io.write_png(img_path, img)

        else:
            # discarded frames
            pass

        # time forward
        t_step += 1


    # start animation
    timer = app.Timer()
    timer.connect(update)
    timer.start(interval=1. / 60., iterations=vis_length * 2)

    c.show()
    app.run()

    # render video for evaluating grouping result
    if args.stage in ['dy']:
        print("Render video for dynamics prediction")

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(
            os.path.join(args.evalf, 'vid_%d_vispy.avi' % (idx_episode)),
            fourcc, 20, (800 * 2, 600))

        for step in range(vis_length):
            gt_path = os.path.join(args.evalf, 'vispy', 'gt_%d_%d.png' % (idx_episode, step))
            pred_path = os.path.join(args.evalf, 'vispy', 'pred_%d_%d.png' % (idx_episode, step))

            gt = cv2.imread(gt_path)
            pred = cv2.imread(pred_path)

            frame = np.zeros((600, 800 * 2, 3), dtype=np.uint8)
            frame[:, :800] = gt
            frame[:, 800:] = pred

            out.write(frame)

        out.release()
def render_fluidlab():
    