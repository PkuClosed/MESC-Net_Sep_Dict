# MESC-Net_Sep_Dict

This is a python demo for the paper:<br />
**Chuyang Ye et al., "An Improved Deep Network for Tissue Microstructure
Estimation with Uncertainty Quantification", MedIA 2020.** 

The demo includes both the training and test phase for the deterministic and probabilistic tissue microstructure estimation. Therefore, to run it, both the training and test data (which are images in the NIfTI format) should be prepared. The input diffusion signals should be normalized by the b0 signals.

There are a few dependencies that need to be installed:<br />
**numpy <br />
nibabel <br />
keras <br />
theano <br />
tqdm <br />**

TO DO:
Here is how to run the scripts. For deterministic estimation, run <br />
> time THEANO_FLAGS='device=cuda0,floatX=float32' python MESCNetSepDict.py < list of training normalized diffusion images> < list of training brain mask images > < number of microstructure meaasures to be estimated > < list of training microstructure 1 > ... < list of training microstructure N > < list of test normalized diffusion images > < list of test brain mask images > < input patch size > < output patch size > < upsampling rate > < size of angular dictionary > < size of spatial dictionary > < output directory > 
(Note that the upsampling rate is for future extension and it is set to one in this case.)
For example, <br />
> time THEANO_FLAGS='device=cuda0,floatX=float32' python MESCNetSepDict.py dwis_1.txt masks_1.txt 3 icvfs_1.txt isos_1.txt ods_1.txt dwis_2.txt masks_2.txt 3 1 1 300 300 temp_out

For more questions, please contact me via chuyang.ye@bit.edu.cn or pkuclosed@gmail.com
