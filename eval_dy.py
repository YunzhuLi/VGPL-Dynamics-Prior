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
from utils import add_log, convert_groups_to_colors, render_vispy, render_fluidlab
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
                state_cur = p_gt[step_id - args.n_his:step_id]
                if use_gpu:
                    state_cur = state_cur.cuda()

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
            gt_pos = p_gt[step_id].unsqueeze(0)
            if use_gpu:
                gt_pos = gt_pos.cuda()
            pred_pos = torch.cat([pred_pos, gt_pos[:, n_particle:]], 1)

            # gt_motion_norm (normalized): B x (n_p + n_s) x state_dim
            # pred_motion_norm (normalized): B x (n_p + n_s) x state_dim
            gt_motion = (p_gt[step_id] - p_gt[step_id - 1]).unsqueeze(0)
            if use_gpu:
                gt_motion = gt_motion.cuda()
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
        render_vispy()
    elif args.fluidlab:
        render_fluidlab()