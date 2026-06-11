# SimCLR on STL10 dataset from torchvision

## Project Overview

In this project, we explore **self-supervised representation learning** using the SimCLR framework. The main objective is to learn meaningful visual features from images without relying on manual labels, and to evaluate how well these learned representations transfer to downstream tasks.

We use the **STL-10 dataset**, a widely used benchmark for unsupervised and semi-supervised learning. STL-10 consists of a large set of unlabeled images for pretraining, along with a smaller labeled training set and a separate test set for evaluation. The dataset contains natural images across 10 object categories and is specifically designed to test representation learning methods. STL10 contains 500 training images per class, 800 test images per class and 100 000 unlabelled images.

The SimCLR approach is based on contrastive learning, where different augmented views of the same image are pulled closer together in the embedding space, while views from different images are pushed apart. In this project, we implement the full SimCLR pipeline including data augmentation, self-supervised pretraining, and downstream evaluation.

To assess the quality of the learned representations, we evaluate the model using linear classification, k-nearest neighbors (k-NN), and dimensionality reduction techniques such as UMAP for visualization.

## Notebook Preview

This notebook contains interactive visualizations (Plotly) that may not render correctly in GitHub's notebook viewer.

If the notebook preview fails, please:

1. Clone the repository.
2. Install the dependencies.
3. Run the notebook locally using JupyterLab.
