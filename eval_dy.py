import argparse
import copy
import os
import time
import cv2

import numpy as np
import scipy.misc
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from config import gen_args
from data import load_data, get_scene_info, normalize_scene_param
from data import get_env_group, prepare_input, denormalize
from models import Model
from utils import add_log, convert_groups_to_colors
from utils import create_instance_colors, set_seed, Tee, count_parameters

import matplotlib.pyplot as plt


args = gen_args()
set_seed(args.random_seed)

os.system('mkdir -p ' + args.evalf)
os.system('mkdir -p ' + os.path.join(args.evalf, 'render'))

tee = Tee(os.path.join(args.evalf, 'eval.log'), 'w')


### evaluating

data_names = args.data_names

use_gpu = torch.cuda.is_available()

# create model and load weights
model = Model(args, use_gpu)
print("model_kp #params: %d" % count_parameters(model))

if args.eval_epoch < 0:
    model_name = 'net_best.pth'
else:
    model_name = 'net_epoch_%d_iter_%d.pth' % (args.eval_epoch, args.eval_iter)

model_path = os.path.join(args.outf, model_name)
print("Loading network from %s" % model_path)

if args.stage == 'dy':
    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()
    # only load parameters in dynamics_predictor
    pretrained_dict = {
        k: v for k, v in pretrained_dict.items() \
        if 'dynamics_predictor' in k and k in model_dict}
    model.load_state_dict(pretrained_dict, strict=False)

else:
    AssertionError("Unsupported stage %s, using other evaluation scripts" % args.stage)

model.eval()


if use_gpu:
    model = model.cuda()


infos = np.arange(10)

for idx_episode in range(len(infos)):

    print("Rollout %d / %d" % (idx_episode, len(infos)))

    B = 1
    n_particle, n_shape = 0, 0

    # ground truth
    datas = []
    p_gt = []
    s_gt = []
    for step in range(args.time_step):
        data_path = os.path.join(args.dataf, 'valid', str(infos[idx_episode]), str(step) + '.h5')

        data = load_data(data_names, data_path)

        if n_particle == 0 and n_shape == 0:
            n_particle, n_shape, scene_params = get_scene_info(data)
            scene_params = torch.FloatTensor(scene_params).unsqueeze(0)

        if args.verbose_data:
            print("n_particle", n_particle)
            print("n_shape", n_shape)

        datas.append(data)

        p_gt.append(data[0])
        s_gt.append(data[1])

    # p_gt: time_step x N x state_dim
    # s_gt: time_step x n_s x 4
    p_gt = torch.FloatTensor(np.stack(p_gt))
    s_gt = torch.FloatTensor(np.stack(s_gt))
    p_pred = torch.zeros(args.time_step, n_particle + n_shape, args.state_dim)

    # initialize particle grouping
    group_gt = get_env_group(args, n_particle, scene_params, use_gpu=use_gpu)

    print('scene_params:', group_gt[-1][0, 0].item())

    # memory: B x mem_nlayer x (n_particle + n_shape) x nf_memory
    # for now, only used as a placeholder
    memory_init = model.init_memory(B, n_particle + n_shape)

    # model rollout
    loss = 0.
    loss_raw = 0.
    loss_counter = 0.
    st_idx = args.n_his
    ed_idx = args.sequence_length

    with torch.set_grad_enabled(False):

        for step_id in range(st_idx, ed_idx):

            if step_id == st_idx:
                # state_cur (unnormalized): n_his x (n_p + n_s) x state_dim
                state_cur = p_gt[step_id - args.n_his:step_id].cuda()

            if step_id % 50 == 0:
                print("Step %d / %d" % (step_id, ed_idx))

            # attr: (n_p + n_s) x attr_dim
            # Rr_cur, Rs_cur: n_rel x (n_p + n_s)
            # state_cur (unnormalized): n_his x (n_p + n_s) x state_dim
            attr, _, Rr_cur, Rs_cur = prepare_input(state_cur[-1].cpu().numpy(), n_particle, n_shape, args)

            if use_gpu:
                attr = attr.cuda()
                Rr_cur = Rr_cur.cuda()
                Rs_cur = Rs_cur.cuda()

            # t
            st_time = time.time()

            # unsqueeze the batch dimension
            # attr: B x (n_p + n_s) x attr_dim
            # Rr_cur, Rs_cur: B x n_rel x (n_p + n_s)
            # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
            attr = attr.unsqueeze(0)
            Rr_cur = Rr_cur.unsqueeze(0)
            Rs_cur = Rs_cur.unsqueeze(0)
            state_cur = state_cur.unsqueeze(0)

            if args.stage in ['dy']:
                inputs = [attr, state_cur, Rr_cur, Rs_cur, memory_init, group_gt]

            # pred_pos (unnormalized): B x n_p x state_dim
            # pred_motion_norm (normalized): B x n_p x state_dim
            pred_pos, pred_motion_norm = model.predict_dynamics(inputs)

            # concatenate the state of the shapes
            # pred_pos (unnormalized): B x (n_p + n_s) x state_dim
            gt_pos = p_gt[step_id].cuda().unsqueeze(0)
            pred_pos = torch.cat([pred_pos, gt_pos[:, n_particle:]], 1)

            # gt_motion_norm (normalized): B x (n_p + n_s) x state_dim
            # pred_motion_norm (normalized): B x (n_p + n_s) x state_dim
            gt_motion = (p_gt[step_id] - p_gt[step_id - 1]).cuda().unsqueeze(0)
            mean_d, std_d = model.stat[2:]
            gt_motion_norm = (gt_motion - mean_d) / std_d
            pred_motion_norm = torch.cat([pred_motion_norm, gt_motion_norm[:, n_particle:]], 1)

            loss_cur = F.l1_loss(pred_motion_norm[:, :n_particle], gt_motion_norm[:, :n_particle])
            loss_cur_raw = F.l1_loss(pred_pos, gt_pos)

            loss += loss_cur
            loss_raw += loss_cur_raw
            loss_counter += 1

            # state_cur (unnormalized): B x n_his x (n_p + n_s) x state_dim
            state_cur = torch.cat([state_cur[:, 1:], pred_pos.unsqueeze(1)], 1)
            state_cur = state_cur.detach()[0]

            # record the prediction
            p_pred[step_id] = state_cur[-1].detach().cpu()


    '''
    print loss
    '''
    loss /= loss_counter
    loss_raw /= loss_counter
    print("loss: %.6f, loss_raw: %.10f" % (loss.item(), loss_raw.item()))


    '''
    visualization
    '''
    group_gt = [d.data.cpu().numpy()[0, ...] for d in group_gt]
    p_pred = p_pred.numpy()[st_idx:ed_idx]
    p_gt = p_gt.numpy()[st_idx:ed_idx]
    s_gt = s_gt.numpy()[st_idx:ed_idx]
    vis_length = ed_idx - st_idx

    if args.vispy:

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

