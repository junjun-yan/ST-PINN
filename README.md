# ST-PINN

An official source code for paper ST-PINN: A Self-Training Physics-Informed Neural Network for Partial Differential Equations, accepted by the IEEE International Joint Conference on Neural Networks, IJCNN 2022. Any communications or issues are welcomed. Please contact shuaicaijunjun@126.com. If you find this repository useful to your research or work, it is really appreciate to star this repository. :heart:

-------------

### Overview

<p align = "justify"> 
    Partial differential equations (PDEs) are an essential computational kernel in physics and engineering. With the development of deep learning, physics-informed neural networks (PINNs), as a mesh-free method, have shown great potential for fast PDE solving. To address the problem of low accuracy and convergence problems of existing PINNs, we propose a selftraining physics-informed neural network, ST-PINN. Specifically,
ST-PINN introduces a pseudo label based self-learning algorithm during training. It employs governing equation as the pseudolabeled evaluation index and selects the highest confidence example from the sample points to attach the pseudo labels. To the best of our knowledge, we are the first to incorporate a self-training mechanism into physics-informed learning. We conduct experiments on five PDEs problems in different fields and scenarios. The results demonstrate that the proposed method allows the network to learn more physical information and benefit convergence. The ST-PINN outperforms existing physicsinformed neural network methods and improves the accuracy by a factor of 1.33x-2.54x.
</p>

<div  align="center">    
    <img src="./pic/ST-PINN.jpg" width=80%/>
</div>

<div  align="center">    
    The architecture of ST-PINN. The blue line displays the training process, while the red line presents the pseudo label generating process.
</div>


### Requirements

1. TensorFlow == 1.15.x
2. Numpy == 1.21.x

Note that the L-BFGS-B optimizer supported by `tf.contrib` is deprecated in TF 2.X, if you want run code in TF 2.X, you can install the TensorFlow Probability (TFP) Package and follow the official tutorial [tensorflow / probability](https://github.com/tensorflow/probability), the L-BFGS-B is packaging in `tfp.optimizer.lbfgs_minimize`


### Dataset

All the PDEs case studies we used in our benchmark are download from [PDEBench Datasets](https://github.com/pdebench/PDEBench), and their files are publicly available on [PDEBench Datasets]([https://sparse.tamu.edu/](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986)).


### Results

<div  align="center">    
    <img src="./pic/performance.jpg" width=80%/>
</div>

<div  align="center">    
    The overall preformance in different formats
</div>
<br>

<div  align="center">    
    <img src="./pic/memory_reducation.jpg" width=70%/>
</div>

<div  align="center">    
    Memory reduction in the CSR&RV format compares to CSR
</div>
<br>

<div  align="center">    
    <img src="./pic/preprocessing.jpg" width=60%/>
</div>

<div  align="center">    
    The pre-processing overload in different formats
</div>



