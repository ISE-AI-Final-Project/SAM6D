# Instance Segmentation Model (ISM) for SAM-6D 


## Requirements
The code has been tested with
- python 3.9.6
- pytorch 2.0.0
- CUDA 11.3

Create conda environment:

```
conda env create -f environment.yml
conda activate sam6d-ism

# for using SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# for using fastSAM
pip install ultralytics==8.0.135
```


## Data Preparation

Please refer to [[link](https://github.com/JiehongLin/SAM-6D/tree/main/SAM-6D/Data)] for more details.


## Foundation Model Download

Download model weights of [Segmenting Anything](https://github.com/facebookresearch/segment-anything):
```
python download_sam.py
```

Download model weights of [Fast Segmenting Anything](https://github.com/CASIA-IVA-Lab/FastSAM):
```
python download_fastsam.py
```

Download model weights of ViT pre-trained by [DINOv2](https://github.com/facebookresearch/dinov2):
```
python download_dinov2.py
```


## [Senior] Evaluation on LineMod Datasets

To evaluate the model on LineMod datasets, please run the following commands:

##### 1) Run inference script to optain .npz files. Example of inference config file can be seen in `configs/inference/run_inference_linemod_example.yaml`
```
python3 run_inference_linemod.py --config <path/to/inference/config>
```


##### 2) Run evaluation script on .npz files and groud truth images to obtain metrics. Example of inference config file can be seen in `configs/eval/run_eval_linemod_example.yaml`
```
python3 run_eval_linemod.py --config <path/to/eval/config>
```


## Acknowledgements

- [CNOS](https://github.com/nv-nguyen/cnos)
- [SAM](https://github.com/facebookresearch/segment-anything)
- [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
- [DINOv2](https://github.com/facebookresearch/dinov2)

                                                              
