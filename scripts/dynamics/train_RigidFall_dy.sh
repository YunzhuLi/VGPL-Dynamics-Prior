CUDA_VISIBLE_DEVICES=0		\
python train.py 		\
	--env RigidFall 	\
	--stage dy		\
	--gen_data 0 		\
	--gen_stat 0		\
	--gen_vision 0		\
	--num_workers 10 	\
	--resume 0		\
	--resume_epoch 0	\
	--resume_iter 0 	\
	--lr 0.0001		\
	--optimizer Adam 	\
	--batch_size 4		\
	--n_his 4		\
	--sequence_length 5	\
	--augment 0.05		\
	--verbose_data 0 	\
	--verbose_model 0	\
	--log_per_iter 100	\
	--ckp_per_iter 5000	\
	--eval 0

