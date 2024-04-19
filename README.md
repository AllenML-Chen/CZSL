# Learning to Infer Unseen Single-/Multi-Attribute-Object Compositions With Graph Networks

## Overview

TensorFlow implementation of [Learning to Infer Unseen Single-/Multi-Attribute-Object Compositions With Graph Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10120982) (T-PAMI 2023).

---

**Introduction**

Compositional Zero-shot Learning (CZSL) has been active in recent years. We thereby construct the Multi-Attribute Dataset (MAD) based on the ImageNet dataset and provide MAD freely to promote related research. MAD is used only for non-commercial research and educational purposes. We hold no liability for any undesirable consequence of using the database. All rights of MAD are reserved. Please follow the process described on the [homepage](http://www.aiar.xjtu.edu.cn/info/1015/2751.htm) and send the signed agreement file to chenhui0622@stu.xjtu.edu.cn for complete use of MAD.

**Details**

MAD includes 158 attributes, 309 objects, 8,030 compositions, and 116,099 images. Each image is annotated with at least one attribute and at most eight salient attributes. The whole dataset is about 12.6 GB. The training set includes 5,630 compositions and 81,371 images. The validation set includes 1k seen and 1k unseen compositions and 27,104 images. The testing set includes 1.4k seen and 1.4k unseen compositions and 42,013 images.

**Organization**

Images and labels are organized by object classes. Each label is organized in a JSON file with the file name corresponding to its image name. That is to say, an image called "img.JPEG" produces a JSON label file called "img.JPEG.json". 

**Evaluation**

We provide the Python code for the implementation of the newly proposed multi-attribute evaluation metrics Hard and Soft in metric.py.


**Citation**

@ARTICLE{10120982,
  author={Chen, Hui and Jiang, Jingjing and Zheng, Nanning},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Learning to Infer Unseen Single-/ Multi-Attribute-Object Compositions With Graph Networks}, 
  year={2023},
  volume={45},
  number={10},
  pages={12022-12037},
  doi={10.1109/TPAMI.2023.3273712}}
