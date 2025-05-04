# Neutrosophic Image Processing


Author: [Dr. VINOTH D](https://www.linkedin.com/in/dr-vinoth-d-1721a3312/), Department of Mathematics, School of Advanced Sciences, [Vellore Institute of Technology, Vellore](https://vit.ac.in/).
--------
## Introduction
Neutrosophic Image Processing (NIP) is an open-source framework developed in Python to perform image processing using Neutrosophic Sets. This repository provides simple and intuitive tools to manipulate symbolic representations of neutrosophic sets over various universes, as well as mappings between them. The framework is designed to apply neutrosophic sets to image processing tasks, specifically for feature extraction and visualization of membership functions.

## Key Features:

- Neutrosophic Set Membership Functions: Code for defining and calculating the membership, non-membership, and indeterminacy of elements in an image.

- Visualization: Functions to visualize the membership values and functions of neutrosophic sets applied to images.


This repository is inspired by the paper **New Neutrosophic Approach to Image Segmentation** by Guo and Cheng (2009), and explores how neutrosophic sets can be applied to image segmentation, enhancing the results in the presence of uncertainty and ambiguity.

## Repository Structure
- `NeutrosophicSet.py`

This file contains the basic class for implementing neutrosophic sets. It provides the core methods for defining membership, non-membership, and indeterminacy functions.
- `SVNeutrosophicSet.py`

This file implements the Single-Valued Neutrosophic Set (SVNS), an extension of neutrosophic sets that simplifies the membership functions by assuming only one degree of membership.
- `Visualization.py`

Contains functions for visualizing the neutrosophic set values, both for the image and the associated membership functions. This includes plotting membership maps and the overall structure of the neutrosophic sets applied to the image.
- `nip.py`

This script executes the main code for applying neutrosophic sets to sample images. It demonstrates how the framework can be used in practice for image segmentation and other processing tasks.


## Installation
### Prerequisites


- NumPy: For numerical operations.
- Matplotlib: For plotting and visualizing data.
- Seaborn: For enhanced data visualization.

## Usage

### Define the Neutrosophic Set:
Use the NeutrosophicSet.py class to create neutrosophic sets. You can define membership, non-membership, and indeterminacy for each pixel or region in the image.

### Apply to Image:
 Use the nip.py file to load an image and apply the neutrosophic set. The code will process the image, compute the membership functions, and segment it based on the neutrosophic thresholds.

### Visualize Results:
The Visualization.py file contains functions to generate plots of the membership functions, allowing you to visually inspect how the neutrosophic sets behave in relation to the image content.

## Example
```
h = 3
sz = 53
x = cv.imread("./samples/c_man.png", 0)
tru = Neutrosophic_set(x, h, sz).truth_mem()
ind = Neutrosophic_set(x, h, sz).indeter_mem()
fal = Neutrosophic_set(x, h, sz).false_mem()

NS_plots(x, h, sz).tru_viz(folder_path+"tru.png")
NS_plots(x, h, sz).ind_viz(folder_path+"ind.png")
NS_plots(x, h, sz).fal_viz(folder_path+"fal.png")

NS_plots(x, h ,sz).tru_intensity(folder_path+"tru_int.png")
NS_plots(x, h ,sz).ind_intensity(folder_path+"ind_int.png")
NS_plots(x, h ,sz).fal_intensity(folder_path+"fal_int.png")

NS_plots(x, h ,sz).ns_kde(folder_path+"ns_kde.png")
```
The outputs will be saved in the `Results` folder.


## Bibliography

If you use or reference this code in your research or work, please cite the following papers:

- Guo, Y., & Cheng, H. D. (2009). New neutrosophic approach to image segmentation. Pattern Recognition, 42(5), 587-595. DOI: 10.1016/j.patcog.2008.07.004

- Dhatchinamoorthy, V., & Devarasan, E. (2023). An Analysis of Global and Adaptive Thresholding for Biometric Images Based on Neutrosophic Overset and Underset Approaches. Symmetry, 15(5), 1102. DOI: 10.3390/sym15051102

- D, Vinoth and Ezhilmaran Devarasan. A Novel Approach of Residue Neutrosophic Technique for Threshold Based Image Segmentation. Neutrosophic Sets and Systems, 58(1). Link

- Harris, C.R., Millman, K.J., van der Walt, S.J., et al. (2020). Array programming with NumPy. Nature, 585(7825), 357-362. DOI: 10.1038/s41586-020-2649-2

- Hunter, J.D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90-95. DOI: 10.1109/MCSE.2007.55

- Waskom, M., Botvinnik, O., Hobson, P., et al. (2020). Seaborn: statistical data visualization. Journal of Open Source Software, 5(51), 2411. DOI: 10.21105/joss.02411
