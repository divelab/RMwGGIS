# Gradient-Guided Importance Sampling for Learning Binary Energy-Based Models

This is the official implementation of the **RMwGGIS** method proposed in the following paper.

Meng Liu, Haoran Liu, and Shuiwang Ji. "[Gradient-Guided Importance Sampling for Learning Binary Energy-Based Models](https://arxiv.org/abs/2210.05782)".

<p align="center">
<img src="https://github.com/divelab/RMwGGIS/blob/main/assets/RMwGGIS.png" width="600" class="center" alt=""/>
    <br/>
</p>
<p align = "center">
Visualization of learned energy functions on 32-dimensional synthetic discrete datasets.
</p>

There is [an implementation from the community](https://github.com/J-zin/RMwGGIS) as well.

## Requirements
We include key dependencies below.
* PyTorch
* tqdm
* sympy
* distutils

## Run
To run the experiments on synthetic discrete data, please refer to the commands in [`run.sh`](https://github.com/divelab/RMwGGIS/blob/main/RMwGGIS/run.sh).

## Reference
```
@article{liu2022rmwggis,
  title={Gradient-Guided Importance Sampling for Learning Binary Energy-Based Models},
  author={Liu, Meng and Liu, Haoran and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2210.05782},
  year={2022}
}
```
