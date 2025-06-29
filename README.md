# 2D to 3D Converter using Deep Learning for Enhanced Dimensional Reconstruction

## Overview

This project is a deep learning-based solution for reconstructing 3D voxel grids from multiple 2D images (front, left, right, back, and top views). This implementation uses a ResNet50 backbone with an attention mechanism to aggregate view features, followed by a 3D convolutional decoder to generate voxel outputs. The project includes synthetic dataset generation, model training with mixed precision and gradient accumulation, evaluation metrics (including Chamfer distance), and visualization tools. It is designed to run on Google Colab with GPU support for efficient training and inference.

## Features

- **Synthetic Dataset Generation**: Creates a dataset of 1000 samples with various shapes (cube, sphere, cylinder, cone, torus, pyramid, prism, ellipsoid) including multi-view images, point clouds, and voxel grids.
- **Enhanced Pix2Vox Model**: Uses ResNet50 for feature extraction, attention-based view aggregation, and a 3D convolutional decoder for voxel reconstruction.
- **Advanced Training**: Implements focal loss, IoU loss, and Dice loss with mixed precision training and gradient accumulation for robust learning.
- **Evaluation**: Provides metrics like voxel accuracy, IoU, and Chamfer distance, along with confusion matrix visualization.
- **Visualization**: Supports interactive 3D voxel visualization using Plotly and fallback Matplotlib visualization.
- **Reconstruction Pipeline**: Allows users to upload images for 3D reconstruction with pre-trained model weights.

## Requirements

To run this project, ensure you have the following dependencies installed:

- Python 3.8+
- PyTorch 2.0+
- torchvision
- NumPy
- Matplotlib
- OpenCV (cv2)
- scikit-image
- scikit-learn
- seaborn
- tqdm
- tensorboard
- Pillow (PIL)
- scipy
- plotly
- google-colab (for Google Colab environment)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Tarun0744/2D_to_3D.git
   cd 2D_to_3D
   ```

2. **Install Dependencies**:
   Install the required Python packages using pip:
   ```bash
   pip install torch torchvision numpy matplotlib opencv-python scikit-image scikit-learn seaborn tqdm tensorboard pillow scipy plotly google-colab
   ```

3. **Set Up Google Colab (Optional)**:
   If running on Google Colab, upload the `2D_to_3D.py` file to your Colab environment. Ensure you enable GPU support in Colab:
   - Go to `Runtime` > `Change runtime type` > Select `GPU`.

## Usage

1. **Prepare the Dataset**:
   - The script automatically generates a synthetic dataset under `/content/shapenet_sample` when you run it.
   - The dataset includes multi-view images, point clouds, and voxel grids for various shapes.

2. **Train the Model**:
   - Run the `2D_to_3D.py` script to train the model:
     ```bash
     python 2D_to_3D.py
     ```
   - The script will:
     - Generate the dataset.
     - Split it into training (80%) and validation (20%) sets.
     - Train the Pix2Vox model for up to 100 epochs with early stopping (patience=15).
     - Save the best model weights as `best_pix2vox.pth`.
     - Plot training/validation loss and metrics (accuracy, IoU).

3. **Evaluate the Model**:
   - After training, the script evaluates the model on the validation set, displaying:
     - A confusion matrix for voxel occupancy.
     - A classification report.
     - Overall voxel accuracy and average Chamfer distance.

4. **Perform Reconstruction**:
   - The script prompts you to upload five images (front, left, right, back, top views) in Google Colab.
   - It processes the images, generates a 3D voxel reconstruction, and visualizes the results.
   - Outputs are saved in the `outputs` folder, including the voxel grid (`voxels.npy`) and input images (`input_view_*.jpg`).

5. **Visualize Results**:
   - Interactive 3D voxel visualizations are generated using Plotly and saved as HTML files.
   - If Plotly fails, a Matplotlib-based visualization is used as a fallback.

## Project Structure

```
2D_to_3D/
├── 2D_to_3D.py        # Main script for dataset generation, training, evaluation, and reconstruction
├── outputs/            # Directory for saving reconstructed voxels and input images
├── shapenet_sample/    # Directory for generated synthetic dataset
└── README.md           # This file
```

## Notes

- **GPU Usage**: The script is optimized for GPU training. Ensure CUDA is available for best performance.
- **Memory Management**: The code includes memory cleanup (`gc.collect()`, `torch.cuda.empty_cache()`) to handle large datasets and models.
- **Pre-trained Model**: If `best_pix2vox.pth` is not found, the reconstruction pipeline will warn you to train the model first.
- **Dataset Customization**: You can modify the `shape_types` list in `MultiViewDataset` to include additional shapes or adjust parameters like `num_points` and `voxel_size`.
- **Colab-Specific**: The file upload functionality relies on Google Colab's `files.upload()`. For non-Colab environments, modify the `select_images` method to accept file paths directly.

## Example

To reconstruct a 3D model from images in Google Colab:
1. Run `2D_to_3D.py`.
2. Wait for the training to complete (or load pre-trained weights).
3. When prompted, upload five images corresponding to the front, left, right, back, and top views.
4. View the visualized input images and the resulting 3D voxel reconstruction.
5. Check the `outputs` folder for saved results.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for bug reports, feature requests, or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the Pix2Vox paper for 3D reconstruction from multi-view images.
- Built with PyTorch and leverages pre-trained ResNet50 weights from torchvision.
- Uses Plotly for interactive 3D visualizations and Matplotlib as a fallback.
