![DINOv2!](/image_.png "DINOv2")
# DINOv2 x Geosciences
This is the official code repository for our study:

#### *DINOv2 Rocks Geological Image Analysis: Classification, Segmentation, and Interpretability*
- **Authors:** Florent Brondolo and Samuel Beaussant
- **Paper:** https://arxiv.org/abs/2407.18100

***
This study investigates the interpretability, classification, and segmentation of CT-scan images of rock samples, with a particular focus on the application of DINOv2 within Geosciences. We compared various segmentation techniques to evaluate their efficacy, efficiency, and adaptability in geological image analysis. The methods assessed include the Otsu thresholding method, clustering techniques (K-means and fuzzy C-means), a supervised machine learning approach (Random Forest), and deep learning methods (UNet and DINOv2). We tested these methods using ten binary sandstone datasets and three multi-class calcite datasets. 

<img src="/image_.png" alt="DINOv2" title="DINOv2" width="300"/>

# Code
We provide the code in the form of standalone notebooks to facilitate the reproducibility of our results and make them accessible even to GPU-poor people ! All the notebooks names are self-explanatory and reproduce a subset of the results of the paper. 

- The multi Otsu algorithm has been adapted from [the scikit-image library](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_multiotsu.html).
- The K-means algorithm has been adapted from [the OpenCV library](https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html).
- The FCM has been adapted from [this work](https://github.com/jeongHwarr/various_FCM_segmentation/blob/master/FCM.py).
- The UNet has been adapted from [Liang et al. (2022)](https://www.sciencedirect.com/science/article/abs/pii/S0098300422001662).

## Data and preprocessing
The raw data used for our experiments are public and freely available at the following links:
- Sandstone dataset : https://www.digitalrocksportal.org/projects/317
- Carbonates : [https://www.digitalrocksportal.org/projects/151](https://www.digitalrocksportal.org/projects/151 "https://www.digitalrocksportal.org/projects/151")

Some notebooks expect to process the data in the form of a numpy (npy) archive, while others expect tif files. In any case, **before running anything**, download the data and store it in a google drive folder. Following this you can run the `data_preprocessing.ipy` notebook to transform the raw data into the required formats. 

## Models weights
You can either train the models from scratch or run inference using our checkpoints. They can be downloaded at the following link: **link to the models weights**.  

## Found a bug ?
If you spot a bug or have a problem running the code, please open an issue.
Please direct other correspondence to Florent Brondolo: [florent.brondolo@akkodis.com](mailto:florent.brondolo@akkodis.com)
or Samuel Beaussant: [samuel.beaussant@akkodis.com](mailto:samuel.beaussant@akkodis.com)

# Bibtex
If you use our code or find our work useful, please consider citing us as
> @article{brondolo2024dinov2,
  title={DINOv2 Rocks Geological Image Analysis: Classification, Segmentation, and Interpretability},
  author={Brondolo, Florent and Beaussant, Samuel},
  journal={arXiv preprint arXiv:2407.18100},
  year={2024}
}

