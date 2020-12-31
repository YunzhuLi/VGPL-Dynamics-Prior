# VGPL-Dynamics-Prior

PyTorch implementation for the dynamics prior (i.e., dynamics module) of the Visually Grounded Physics Learner (VGPL). Given the current state of the system in the form of particles, the dynamics prior learns to predict their future movements. Please see the following paper for more details.

**Visual Grounding of Learned Physical Models**

Yunzhu Li, Toru Lin*, Kexin Yi*, Daniel M. Bear, Daniel L. K. Yamins, Jiajun Wu, Joshua B. Tenenbaum, and Antonio Torralba

**ICML 2020**
[[website]](http://visual-physics-grounding.csail.mit.edu/) [[paper]](https://arxiv.org/abs/2004.13664) [[video]](https://www.youtube.com/watch?v=P_LrG0lzc-0&feature=youtu.be)


## Demo

Rollout from our learned model

![](imgs/MassRope.gif)  ![](imgs/RigidFall.gif)


## Evaluate the trained model on the demo validation data

Go to the project folder, and type one of the following commands (assuming you have all the dependencies installed):

    bash scripts/dynamics/eval_RigidFall_dy.sh
    bash scripts/dynamics/eval_MassRope_dy.sh
    
## Train the model

Download the training data from the following links, and put them in the `data` folder.

- MassRope [[DropBox]](https://www.dropbox.com/s/mqc87hwo9sdubnu/data_MassRope.zip?dl=0) (1.14 GB)
- RigidFall [[DropBox]](https://www.dropbox.com/s/hra0okrkg99h0bb/data_RigidFall.zip?dl=0) (2.9 GB)

Type the following commands for training the model:

    bash scripts/dynamics/train_RigidFall_dy.sh
    bash scripts/dynamics/train_MassRope_dy.sh
    
## Difference between this repo and [DPI-Net](https://github.com/YunzhuLi/DPI-Net)

- We optimized the training procedure by injecting noise to the particle locations, which will lead to more robust long-term rollout performance. The option `--augment_ratio` controls the scale of the noise augmentation.
- We also padded the input data to allow training with a batch size larger than 1.
- Instead of relying on [PyFleX](https://github.com/YunzhuLi/PyFleX), we added support for using [VisPy](http://vispy.org/) to visualize the predicted trajectories, which may be easier to run for the users who do not have the appropriate CUDA support.


## Citing VGPL

If you find this codebase useful in your research, please consider citing:

    @inproceedings{li2020visual,
        Title={Visual Grounding of Learned Physical Models},
        Author={Li, Yunzhu and Lin, Toru and Yi, Kexin and Bear, Daniel and Yamins, Daniel L.K. and Wu, Jiajun and Tenenbaum, Joshua B. and Torralba, Antonio},
        Booktitle={ICML},
        Year={2020}
    }

    @inproceedings{li2019learning,
        Title={Learning Particle Dynamics for Manipulating Rigid Bodies, Deformable Objects, and Fluids},
        Author={Li, Yunzhu and Wu, Jiajun and Tedrake, Russ and Tenenbaum, Joshua B and Torralba, Antonio},
        Booktitle={ICLR},
        Year={2019}
    }
