CONDA_ENV=vgpl_new
K=300
N_TRAJS_VALID=10
N_TRAJS_TRAIN=1000
TRAJF_TRAIN=data/data_LatteArt/train
TRAJF_VALID=data/data_LatteArt/valid
python scripts/gen/fps.py --k ${K} --n_trajs ${N_TRAJS_TRAIN} --trajf ${TRAJF_TRAIN}
