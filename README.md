# CPO: Change-Robust Panorama to Point Cloud Localization
Official PyTorch implementation of **CPO: Change Robust Panorama to Point Cloud Localization (ECCV 2022)** [[Paper]](https://arxiv.org/pdf/2207.05317.pdf)

[<img src="piccolo_overview.jpg" width="500"/>](piccolo_overview.jpg)\
CPO is a fast and robust algorithm for localizing a 2D panorama against a 3D point cloud possibly containing changes.
Instead of focusing on sparse feature points, we make use of the dense color measurements provided from the panorama images.
Specifically, we propose efficient color histogram generation and subsequent robust localization using score maps defined over 2D and 3D.

[<img src="qualitative.png" width="800"/>](qualitative.png)\
In this repository, we provide the implementation of fast histogram generation, which is the key component of CPO that enables candidate pose search and 2D/3D score map generation.
If you have any questions regarding CPO, please leave an issue or contact 82magnolia@snu.ac.kr.
