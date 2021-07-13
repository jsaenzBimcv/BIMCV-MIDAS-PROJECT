# Semantic Segmentation of Vertebra using Convolutional Neural Networks 
--------------------------------------------------------------------------

The approach proposed in this work is based on convolutional neural networks whose output is a mask where each pixel from the input image is classified into one of the possible classes, Background or Vertebrae.

The proposed network architecture is based on the U-Net neural network model [(Ronneberger et al., 2015)](#1).

The Trained Models are provided here.

Prerequisite:

- python 3.6.1
- numpy 1.18.1
- matplotlib 3.1.2
- nibabel 3.0.0
- tensorflow 2.1.0
- keras 2.3.1

### 1. Experiments 

The dataset used in this work was extracted from the <a href="https://bimcv.cipf.es/bimcv-projects/project-midas/">MIDAS corpus</a>.
The used MR images come from scanning sessions corresponding to 100 different patients, each scanning session has a different number of slices.

Sagittal T2-weighted images were used to distinguish the anatomical silhouette of the vertebrae in the lumbar region.
The manual semantic segmentation was carried out by two expert radiologists with a high expertise in skeletal muscle pathology. 

The split into three partitions training, validation and test was done at the level of patient in order to guarantee no 2-D images from the same patient appear in different partitions.

In <a href="https://github.com/jsaenzBimcv/BIMCV-MIDAS-PROJECT/blob/main/Models/Segmentation_Models/Unet2d_Vertebrae/Unet2d_Spine.ipynb">Unet2d_Spine.ipynb</a>, the procedure followed for training the networks is detailed.

The [Table 1](#table1) shows the U-Net model and combination of the configuration parameters that obtained the best results.

<div align="center"> 
<sub> 

| Models Configuration  | mIOU (%)                     |
|------|------------------------------------|
|<a href="https://github.com/jsaenzBimcv/BIMCV-MIDAS-PROJECT/tree/main/Models/Segmentation_Models/Unet2d_Vertebrae/models/Unet2d_Opt-Adam_Lr-0.00033_Epoch-100_Filt-64">Unet2d_Opt-Adam_Lr-0.00033_Epoch-100_Filt-64</a>| 79.31 |
|<a href="https://github.com/jsaenzBimcv/BIMCV-MIDAS-PROJECT/tree/main/Models/Segmentation_Models/Unet2d_Vertebrae/models/Unet2d_Opt-RMSprop_Lr-0.00033_Epoch-100_Filt-32">Unet2d_Opt-RMSprop_Lr-0.00033_Epoch-100_Filt-32</a>| 85.93 |
|<a href="https://github.com/jsaenzBimcv/BIMCV-MIDAS-PROJECT/tree/main/Models/Segmentation_Models/Unet2d_Vertebrae/models/Unet2d_Opt-RMSprop_Lr-0.00033_Epoch-100_Filt-64">Unet2d_Opt-RMSprop_Lr-0.00033_Epoch-100_Filt-64</a>| 84.12 |

</sub>
</div>
<p align="center">
<a id="table1">Table 1:</a> Parameters tested in the U-Net 2D model.
</p>

All models are trained for 100 epochs, In all cases the activation function in the output layer was the softmax and the loss the categorical cross entropy.

You can label your own images by downloading the <a href="https://bimcv.cipf.es/bimcv-projects/project-midas/">MIDAS project</a> model weights.


Intersection over Union (IoU) [(Long et al., 2015)](#2) was used as the metric to compare the performance of the evaluated network architectures.

If you use this code please cite:

J. J. Saenz-Gamboa, M. de la Iglesia-Vayá and J. A. Gómez, "Automatic Semantic Segmentation of Structural Elements related to the Spinal Cord in the Lumbar Region by using Convolutional Neural Networks," 2020 25th International Conference on Pattern Recognition (ICPR), 2021, pp. 5214-5221, <a href="https://doi.org/10.1109/ICPR48806.2021.9412934">doi:10.1109/ICPR48806.2021.9412934.</a>

<a id='references'></a>
## References

<a id="2">[1]</a> Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation. In: International Conference on Medical image computing and computer-assisted intervention. Springer; 2015. p. 234–41.

<a id="1">[2]</a>Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic segmentation. In: Proceedings of the IEEE conference on computer vision and pattern recognition (CVPR). 2015. p. 3431–40.</a>

## Rights and permissions.

 <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>., which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if changes were made. The images or other third party material in this article are included in the article's Creative Commons license, unless indicated otherwise in a credit line to the material. If material is not included in the article's Creative Commons license and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder.

