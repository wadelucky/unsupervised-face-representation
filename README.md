# Pre-training strategies and datasets for facial representation learning

This is the PyTorch implementation for [Facial Representation Learning (FRL) paper](http://www.adrianbulat.com/downloads/ECCV2022/face_representation_learning.pdf):
```
@inproceedings{bulat2022pre,
  title={Pre-training strategies and datasets for facial representation learning},
  author={Bulat, Adrian and Cheng, Shiyang and Yang, Jing and Garbett, Andrew and Sanchez, Enrique and Tzimiropoulos, Georgios},
  journal={ECCV},
  year={2022}
}
```

## Model Zoo

They provide bellow some of the models trained in a self-supervised manner. More models to be added later on.
<table>
  <thead>
    <tr style="text-align: right;">
      <th align="center">data</th>
      <th align="center">backbone</th>
      <th align="center">url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">VGG</td>
      <td align="center">ResNet 50</td>
      <td align="center"> <a href="https://www.adrianbulat.com/downloads/ECCV2022/pretrained_models/unsupervised/flr_r50_vgg_face.pth">model</a> </td>
    </tr>
    <tr>
      <td align="center">VGG (1M)</td>
      <td align="center">ResNet 50</td>
      <td align="center"> <a href="https://www.adrianbulat.com/downloads/ECCV2022/pretrained_models/unsupervised/flr_r50_vgg_face_1m.pth">model</a> </td>
    </tr>
    <tr>
      <td align="center">FPR-Flickr</td>
      <td align="center">ResNet 50</td>
      <td align="center"> <a href="https://www.adrianbulat.com/downloads/ECCV2022/pretrained_models/unsupervised/flr_r50_flickr_face.pth">model</a></td>
    </tr>
  </tbody>
</table>

## Installation

To use the code, clone the repo and install the following packages:

```bash
git clone https://github.com/wadelucky/unsupervised-face-representation
```

### Requirements

* Python >= 3.8
* Numpy
* pytorch: [install instructions](https://pytorch.org/get-started/locally/)
* torchvision: ``conda install torchvision -c pytorch``
* apex: [install instructions](https://github.com/NVIDIA/apex#installation)
* OpenCV: ``pip install opencv-python``
* H5Py: ``conda install h5py``
* tensorboard: ``pip install tensorboard``
* pandas

Note, if you are using pytorch > 1.10 and experience issues with apex, please see [#1282](https://github.com/NVIDIA/apex/pulls/1282). Alternatively you can switch to the native pytorch amp.


## Training

~~```bash~~
~~bash scripts/run.sh~~
~~```~~

### In HPC, you may need to modify some hyperparameters.
sbatch face.sbatch

Before running the script make sure to set the appropiate paths. The models released in the paper were trained using 64 K40 GPUs.

## Preparing data

For instructions regarding getting the data, see [DATASET.md](slowfast/datasets/DATASET.md)

## Acknowledgement

We thank the original authors for releasing their code: [SwAV](https://github.com/facebookresearch/swav), [MoCo](https://github.com/facebookresearch/moco), [BYOL](https://github.com/deepmind/deepmind-research/blob/master/byol/byol_experiment.py), and [vissl](https://github.com/facebookresearch/vissl) which we base our code base upon.
