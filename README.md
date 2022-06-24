### A Convergent and Dimension-Independent Min-Max Optimization Algorithm

This repository contains code for replicating the results in the paper - "A Convergent and Dimension-Independent Min-Max Optimization Algorithm".
The simulations presented in the dataset test our algorithm against baselines in two synthetic settings - low-dimensional test functions and Gaussian mixture datasets - and two real-world dataset - MNIST and CIFAR. 

### File Structure

--The files "Our_Algorithm.m,"   "GDA.m," and "OMD.m" in the folder Low dimensional test functions contain the MATLAB code for the simulations in Figures 1 and 2 of our paper.

--The file "Gaussian_Mixture_code.ipynb" in the folder Gaussian_Mixture_Dataset contains the Python code for the simulations in Figure 3, Table 1 and Appendix E.3 of our paper. This file contains the code for our algorithm, GDA, OMD, and Unrolled GANs, on the four Gaussian mixture dataset.

--The file "CIFAR_Code.ipynb" in the folder CIFAR contains the Python code used for the simulations in Appendix F of our paper. This file contains the code for our algorithm and for GDA, on the CIFAR-10 dataset.

--The file "MNIST_Code.ipynb" in the folder MNIST contains the Python  code used for the simulations in Appendix G. This file contains the code for our algorithm and for GDA, on the MNIST dataset.

--The file "MNIST_decreasing_temperature_Code.ipynb" in the folder MNIST contains the  Python code that was used to generate the results in Appendix H of our paper. This file contains the code for the version of our algorithm with randomized accept/reject step and decreasing temperature schedule.


### References

*A Convergent and Dimension-Independent Min-Max Optimization Algorithm* <br>
Vijay Keswani, Oren Mangoubi, Sushant Sachdeva, Nisheeth K. Vishnoi <br>
ICML 2022




