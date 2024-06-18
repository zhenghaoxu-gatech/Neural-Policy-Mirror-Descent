# Neural Policy Mirror Descent

This repository contains the code for the numerical experiments of the paper [Sample Complexity of Neural Policy Mirror Descent for Policy Optimization on Low-Dimensional Manifolds](https://arxiv.org/abs/2309.13915). It implements the Neural Policy Mirror Descent (NPMD) method for the visualized CartPole environment. 

# Prerequisites

Dependencies are provided in `requirements.txt`. 

# Example

Run `main.py`. Models and results are saved in `model_path`. 

~~~
python main.py --n_iters 200 --resolution "low" --gamma 0.98 --sample_size 4096 --batch_size 256 --epochs 200 --lr 0.001 --model_path "./results/0/"
~~~


# Reference

~~~
@article{xu2023sample,
  title={Sample Complexity of Neural Policy Mirror Descent for Policy Optimization on Low-Dimensional Manifolds},
  author={Xu, Zhenghao and Ji, Xiang and Chen, Minshuo and Wang, Mengdi and Zhao, Tuo},
  journal={arXiv preprint arXiv:2309.13915},
  year={2023}
}
~~~