# 🦖 DINOv2 x Geosciences 🌍
This is the official code repository for our study:

#### *DINOv2 Rocks Geological Image Analysis: Classification, Segmentation, and Interpretability*
- **Authors:** [Florent Brondolo](https://www.linkedin.com/in/flo-brondolo) and [Samuel Beaussant](https://www.linkedin.com/in/samuel-beaussant-a25905197/)
- **Paper:** https://arxiv.org/abs/2407.18100

***
This study investigates the interpretability, classification, and segmentation of CT-scan images of rock samples, with a particular focus on the application of DINOv2 within Geosciences. We compared various segmentation techniques to evaluate their efficacy, efficiency, and adaptability in geological image analysis. The methods assessed include the Otsu thresholding method, clustering techniques (K-means and fuzzy C-means), a supervised machine learning approach (Random Forest), and deep learning methods (UNet and DINOv2). We tested these methods using ten binary sandstone datasets and three multi-class calcite datasets. 

<p align="center">
  <img src="/image_.png" alt="DINOv2" title="DINOv2" width="300"/>
</p>

# 👾 Code
We provide the code in the form of standalone notebooks to facilitate the reproducibility of our results and make them accessible to all (even to GPU-poor people!). The names of all the notebooks are self-explanatory and reproduce a subset of the paper's results.

- The multi-Otsu algorithm has been adapted from [the scikit-image library](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html).
- The K-means algorithm has been adapted from [the OpenCV library](https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html).
- The FCM has been adapted from [this work](https://github.com/jeongHwarr/various_FCM_segmentation/blob/master/FCM.py).
- The UNet has been adapted from [Liang et al. (2022)](https://www.sciencedirect.com/science/article/abs/pii/S0098300422001662).
- The resNet152 model is available [here](https://huggingface.co/microsoft/resnet-152). 

## Data and preprocessing
The raw data used for our experiments are public and freely available:
- Sandstones: [https://www.digitalrocksportal.org/projects/317](https://www.digitalrocksportal.org/projects/317).
- Carbonates: [https://www.digitalrocksportal.org/projects/151](https://www.digitalrocksportal.org/projects/151).

Some notebooks expect data in the form of a NumPy (npy) archive, while others require TIFF (tif) files. In any case, **before running anything**, download the data and store it in a Google Drive folder. Following this you can run the `data_preprocessing.ipy` notebook to transform the raw data into the required formats. 

## Some results
Here, we present a portion of our experimental results, highlighting the performance of seven models: ResNet152, four variations of DINOv2, and two iterations of a UNet. The four versions of DINOv2 include a frozen DINOv2 paired with either a linear or a complex convolutional head, and a LoRA fine-tuned DINOv2 also paired with the same heads. For the UNet models, we utilized the same backbone with two different feature sizes: small (n=32) and large (n=64).

The results clearly demonstrate the superior capability of DINOv2 in interpreting raw rock CT scans. Experimental parameters were consistent across all experiments: 1000 images for training (split between two rock datasets) and 500 images for validation (from a third rock sample). Hyperparameters were set identically across all training sessions.

<p align="center">
  <img src="/iou_models.png" alt="results" title="IoU for various DL models" width="500"/>
</p>

## Model weights
You have the option to either train the models from scratch or perform inference using our pre-trained checkpoints, which can be downloaded from this [link](https://drive.google.com/file/d/1C2UCfMWGNQi2Gv_1wAGp22-SODL1SXQZ/view?usp=sharing). These weights are the product of training with the DINOv2-base backbone (768 features), fine-tuned using LoRA and a simple linear head. The model definition code is available [here](https://github.com/FloFive/DINOv2-X-Geosciences/blob/main/code/DINOv2.ipynb), and the weights were originally used for [PCA evaluation](https://github.com/FloFive/DINOv2-X-Geosciences/blob/main/code/PCA.ipynb).

## Found a bug?
If you spot a bug or have a problem running the code, please open an issue.
If you have any questions or need further assistance, don't hesitate to contact Florent Brondolo ([florent.brondolo@akkodis.com](mailto:florent.brondolo@akkodis.com))
or Samuel Beaussant ([samuel.beaussant@akkodis.com](mailto:samuel.beaussant@akkodis.com)).

# 📚 Citation / Bibtex
If you use our code or find our work helpful, please consider citing it as follows:
```
@article{brondolo2024dinov2,
  title={DINOv2 Rocks Geological Image Analysis: Classification, Segmentation, and Interpretability},
  author={Brondolo, Florent and Beaussant, Samuel},
  journal={arXiv preprint arXiv:2407.18100},
  year={2024}
}
```
