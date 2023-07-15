CONDA_ENV=vgpl_new
K=300
N_TRAJS=1000
TRAJF=data/data_LatteArt/train
python scripts/gen/fps.py --k ${K} --n_trajs ${N_TRAJS} --trajf ${TRAJF}