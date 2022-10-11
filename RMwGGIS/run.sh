############### Original generalized ratio matching

### 2spirals
CUDA_VISIBLE_DEVICES=0 python main_rm_synthetic.py --data_name=2spirals --save_dir=./results_new_mmd/RM/2spirals-swish-lr5e-4-wd1e-4-gradclip1 --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --wd=1e-4

### circles
CUDA_VISIBLE_DEVICES=0 python main_rm_synthetic.py --data_name=circles --save_dir=./results_new_mmd/RM/circles-swish-lr5e-4-wd1e-4-gradclip1 --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --wd=1e-4

### moons
CUDA_VISIBLE_DEVICES=0 python main_rm_synthetic.py --data_name=moons --save_dir=./results_new_mmd/RM/moons-swish-lr5e-4-wd1e-4-gradclip1 --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --wd=1e-4

### swissroll
CUDA_VISIBLE_DEVICES=0 python main_rm_synthetic.py --data_name=swissroll --save_dir=./results_new_mmd/RM/swissroll-swish-lr5e-4-wd1e-4-gradclip1 --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --wd=1e-4

### checkerboard
CUDA_VISIBLE_DEVICES=0 python main_rm_synthetic.py --data_name=checkerboard --save_dir=./results_new_mmd/RM/checkerboard-swish-lr1e-4-wd1e-4-gradclip1 --lr=1e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --wd=1e-4

### 8gaussians
CUDA_VISIBLE_DEVICES=0 python main_rm_synthetic.py --data_name=8gaussians --save_dir=./results_new_mmd/RM/8gaussians-swish-lr5e-4-wd1e-4-gradclip1 --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --wd=1e-4

### pinwheel
CUDA_VISIBLE_DEVICES=0 python main_rm_synthetic.py --data_name=pinwheel --save_dir=./results_new_mmd/RM/pinwheel-swish-lr5e-4-wd1e-4-gradclip1 --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --wd=1e-4

############### RMwRAND

### 2spirals
CUDA_VISIBLE_DEVICES=0 python main_randn_synthetic.py --data_name=2spirals --save_dir=./results_new_mmd/randn/2spirals-swish-lr1e-4-wd1e-4-gradclip1-unbiased-dim32 --lr=1e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=True --wd=1e-4 --mmd=True

### circles
CUDA_VISIBLE_DEVICES=0 python main_randn_synthetic.py --data_name=circles --save_dir=./results_new_mmd/randn/circles-swish-lr1e-4-wd1e-4-gradclip1-unbiased-dim32 --lr=1e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=True --wd=1e-4 --mmd=True

### moons
CUDA_VISIBLE_DEVICES=0 python main_randn_synthetic.py --data_name=moons --save_dir=./results_new_mmd/randn/moons-swish-lr1e-4-wd1e-4-gradclip1-unbiased-dim32 --lr=1e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=True --wd=1e-4 --mmd=True

### swissroll
CUDA_VISIBLE_DEVICES=0 python main_randn_synthetic.py --data_name=swissroll --save_dir=./results_new_mmd/randn/swissroll-swish-lr1e-4-wd1e-4-gradclip1-unbiased-dim32 --lr=1e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=True --wd=1e-4 --mmd=True

### checkerboard
CUDA_VISIBLE_DEVICES=0 python main_randn_synthetic.py --data_name=checkerboard --save_dir=./results_new_mmd/randn/checkerboard-swish-lr1e-4-wd1e-4-gradclip1-unbiased-dim32 --lr=1e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=True --wd=1e-4 --mmd=True

### 8gaussians
CUDA_VISIBLE_DEVICES=0 python main_randn_synthetic.py --data_name=8gaussians --save_dir=./results_new_mmd/randn/8gaussians-swish-lr1e-4-wd1e-4-gradclip1-unbiased-dim32 --lr=1e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=True --wd=1e-4 --mmd=True

### pinwheel
CUDA_VISIBLE_DEVICES=0 python main_randn_synthetic.py --data_name=pinwheel --save_dir=./results_new_mmd/randn/pinwheel-swish-lr1e-4-wd1e-4-gradclip1-unbiased-dim32 --lr=1e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=True --wd=1e-4 --mmd=True




############### Our proposal: ratio matching with Gradient-Guided Importance Sampling

###### [Basic][Unbiased] 

### 2spirals
CUDA_VISIBLE_DEVICES=0 python main_ggis_synthetic.py --data_name=2spirals --save_dir=./results_new_mmd/GGIS/2spirals-swish-lr5e-4-wd1e-4-gradclip1-unbiased --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=True --wd=1e-4

### circles
CUDA_VISIBLE_DEVICES=0 python main_ggis_synthetic.py --data_name=circles --save_dir=./results_new_mmd/GGIS/circles-swish-lr5e-4-wd1e-4-gradclip1-unbiased --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=True --wd=1e-4

### moons
CUDA_VISIBLE_DEVICES=0 python main_ggis_synthetic.py --data_name=moons --save_dir=./results_new_mmd/GGIS/moons-swish-lr5e-4-wd1e-4-gradclip1-unbiased --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=True --wd=1e-4

### swissroll
CUDA_VISIBLE_DEVICES=0 python main_ggis_synthetic.py --data_name=swissroll --save_dir=./results_new_mmd/GGIS/swissroll-swish-lr5e-4-wd1e-4-gradclip1-unbiased --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=True --wd=1e-4

### checkerboard
CUDA_VISIBLE_DEVICES=0 python main_ggis_synthetic.py --data_name=checkerboard --save_dir=./results_new_mmd/GGIS/checkerboard-swish-lr1e-4-wd1e-4-gradclip1-unbiased --lr=1e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=True --wd=1e-4

### 8gaussians
CUDA_VISIBLE_DEVICES=0 python main_ggis_synthetic.py --data_name=8gaussians --save_dir=./results_new_mmd/GGIS/8gaussians-swish-lr5e-4-wd1e-4-gradclip1-unbiased --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=True --wd=1e-4

### pinwheel
CUDA_VISIBLE_DEVICES=0 python main_ggis_synthetic.py --data_name=pinwheel --save_dir=./results_new_mmd/GGIS/pinwheel-swish-lr5e-4-wd1e-4-gradclip1-unbiased --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=True --wd=1e-4


###### [Advanced][Biased]

### 2spirals
CUDA_VISIBLE_DEVICES=0 python main_ggis_synthetic.py --data_name=2spirals --save_dir=./results_new_mmd/GGIS/2spirals-swish-lr5e-4-wd1e-4-gradclip1-biased --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=False --wd=1e-4

### circles
CUDA_VISIBLE_DEVICES=0 python main_ggis_synthetic.py --data_name=circles --save_dir=./results_new_mmd/GGIS/circles-swish-lr5e-4-wd1e-4-gradclip1-biased --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=False --wd=1e-4

### moons
CUDA_VISIBLE_DEVICES=0 python main_ggis_synthetic.py --data_name=moons --save_dir=./results_new_mmd/GGIS/moons-swish-lr5e-4-wd1e-4-gradclip1-biased --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=False --wd=1e-4

### swissroll
CUDA_VISIBLE_DEVICES=0 python main_ggis_synthetic.py --data_name=swissroll --save_dir=./results_new_mmd/GGIS/swissroll-swish-lr5e-4-wd1e-4-gradclip1-biased --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=False --wd=1e-4

### checkerboard
CUDA_VISIBLE_DEVICES=0 python main_ggis_synthetic.py --data_name=checkerboard --save_dir=./results_new_mmd/GGIS/checkerboard-swish-lr1e-4-wd1e-4-gradclip1-biased --lr=1e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=False --wd=1e-4

### 8gaussians
CUDA_VISIBLE_DEVICES=0 python main_ggis_synthetic.py --data_name=8gaussians --save_dir=./results_new_mmd/GGIS/8gaussians-swish-lr5e-4-wd1e-4-gradclip1-biased --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=False --wd=1e-4

### pinwheel
CUDA_VISIBLE_DEVICES=0 python main_ggis_synthetic.py --data_name=pinwheel --save_dir=./results_new_mmd/GGIS/pinwheel-swish-lr5e-4-wd1e-4-gradclip1-biased --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=1000 --plot=True --grad_clip=1 --unbiased=False --wd=1e-4









### Experiments on convergency speed dim256
# Note: We should use cpu when measuring MMD due to the memory issue.

### RM
CUDA_VISIBLE_DEVICES=0 python main_rm_synthetic.py --data_name=2spirals --save_dir=./results_new_mmd/RM/2spirals-swish-lr5e-4-wd1e-4-gradclip1-dim256 --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=3000 --plot=True --grad_clip=1 --wd=1e-4 --discrete_dim=256 --batch_size=1024 --mmd=False

CUDA_VISIBLE_DEVICES=0 python main_rm_synthetic.py --data_name=moons --save_dir=./results_new_mmd/RM/moons-swish-lr5e-4-wd1e-4-gradclip1-dim256 --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=3000 --plot=True --grad_clip=1 --wd=1e-4 --discrete_dim=256 --batch_size=1024 --mmd=False

### [Unbiased]
CUDA_VISIBLE_DEVICES=0 python main_ggis_synthetic.py --data_name=2spirals --save_dir=./results_new_mmd/GGIS/2spirals-swish-lr5e-4-wd1e-4-gradclip1-unbiased-dim256 --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=3000 --plot=True --grad_clip=1 --unbiased=True --wd=1e-4 --discrete_dim=256 --batch_size=1024 --mmd=False

CUDA_VISIBLE_DEVICES=0 python main_ggis_synthetic.py --data_name=moons --save_dir=./results_new_mmd/GGIS/moons-swish-lr5e-4-wd1e-4-gradclip1-unbiased-dim256 --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=3000 --plot=True --grad_clip=1 --unbiased=True --wd=1e-4 --discrete_dim=256 --batch_size=1024 --mmd=False

### [Biased]
CUDA_VISIBLE_DEVICES=0 python main_ggis_synthetic.py --data_name=2spirals --save_dir=./results_new_mmd/GGIS/2spirals-swish-lr3e-4-wd1e-4-gradclip1-biased-dim256 --lr=3e-4 --activation=swish --save_interval=50 --max_epochs=3000 --plot=True --grad_clip=1 --unbiased=False --wd=1e-4 --discrete_dim=256 --batch_size=1024 --mmd=False

CUDA_VISIBLE_DEVICES=0 python main_ggis_synthetic.py --data_name=moons --save_dir=./results_new_mmd/GGIS/moons-swish-lr5e-4-wd1e-4-gradclip1-biased-dim256 --lr=5e-4 --activation=swish --save_interval=50 --max_epochs=3000 --plot=True --grad_clip=1 --unbiased=False --wd=1e-4 --discrete_dim=256 --batch_size=1024 --mmd=False

### randn
CUDA_VISIBLE_DEVICES=0 python main_randn_synthetic.py --data_name=2spirals --save_dir=./results_new_mmd/randn/2spirals-swish-lr1e-4-wd1e-4-gradclip1-unbiased-dim256 --lr=1e-4 --activation=swish --save_interval=50 --max_epochs=3000 --plot=True --grad_clip=1 --unbiased=True --wd=1e-4 --discrete_dim=256 --batch_size=1024 --mmd=False

CUDA_VISIBLE_DEVICES=0 python main_randn_synthetic.py --data_name=moons --save_dir=./results_new_mmd/randn/moons-swish-lr1e-4-wd1e-4-gradclip1-unbiased-dim256 --lr=1e-4 --activation=swish --save_interval=50 --max_epochs=3000 --plot=True --grad_clip=1 --unbiased=True --wd=1e-4 --discrete_dim=256 --batch_size=1024 --mmd=False

