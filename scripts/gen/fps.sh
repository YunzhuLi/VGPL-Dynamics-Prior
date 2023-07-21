CONDA_ENV=vgpl_new
K=300
N_TRAJS_VALID=10
N_TRAJS_TRAIN=100
TRAJF_TRAIN=data/data_Pouring/train
TRAJF_VALID=data/data_Pouring/valid
python scripts/gen/fps.py --k ${K} --n_trajs ${N_TRAJS_VALID} --trajf ${TRAJF_VALID}
