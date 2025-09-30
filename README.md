# RD Model Backbone Shift: ResNet50 to ConvNeXt

## Overview
This repository implements and compares Reverse Distillation (RD) models with different backbone networks - ResNet50 and ConvNeXt - for anomaly detection on the MVTec AD dataset.

## Key Features
- **RD (Reverse Distillation) Implementation**: Teacher-Student framework for anomaly detection
- **Backbone Comparison**: Fair comparison between ResNet50 and ConvNeXt architectures
- **MVTec AD Dataset**: Experiments on industrial anomaly detection benchmark
- **Comprehensive Analysis**: Detailed performance metrics and visualizations

## Project Structure
```
RD4AD-main/
├── resnet.py                    # ResNet backbone implementation
├── de_resnet.py                 # ResNet decoder implementation
├── RD_ConvNeXt_Model.py        # ConvNeXt RD model implementation
├── RD_Trainer.ipynb            # Training notebook for ResNet
├── RD_ConvNeXt_Trainer.ipynb   # Training notebook for ConvNeXt
├── RD_Tester.ipynb             # Fair comparison testing notebook
├── dataset.py                   # Dataset utilities
└── checkpoints/                 # Model checkpoints
```

## Installation
```bash
# Clone the repository
git clone https://github.com/Aronvision/RD-Model-Backbone-Shift.git
cd RD-Model-Backbone-Shift

# Install requirements
pip install torch torchvision numpy scipy scikit-learn matplotlib pandas tqdm
```

## Dataset
This project uses the MVTec AD dataset. Download it from [MVTec AD website](https://www.mvtec.com/company/research/datasets/mvtec-ad) and place it in the `data/` directory.

## Usage

### Training
```python
# Train ResNet50 RD model
# Open RD_Trainer.ipynb and run all cells

# Train ConvNeXt RD model
# Open RD_ConvNeXt_Trainer.ipynb and run all cells
```

### Testing
```python
# Fair comparison between models
# Open RD_Tester.ipynb and run all cells
```

## Key Results

### Performance Comparison (Bottle Dataset)
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| ResNet50 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| ConvNeXt | 0.976 | 0.969 | 1.000 | 0.984 | 0.991 |

### Key Findings
- ResNet50 shows superior performance for anomaly detection tasks
- Local pattern detection capability is crucial for manufacturing defect identification
- Architecture stability and gradient flow significantly impact performance

## Technical Highlights

### ResNet50 RD Model
- BatchNorm + ReLU activation
- Residual connections for gradient flow
- Multi-scale feature extraction
- Channel dimensions: [256, 512, 1024]

### ConvNeXt RD Model
- LayerNorm + GELU activation
- Depthwise separable convolutions
- Hierarchical feature fusion
- Channel dimensions: [96, 192, 384]

## Documentation
- [Portfolio Document](RD4AD-main/RD_Model_Comparison_Portfolio.md) - Comprehensive project analysis
- [Model Guide](RD_Model_ResNet_to_ConvNeXt_Guide.md) - Implementation details and differences

## Citation
If you use this code for research, please cite:
```
@misc{rd-backbone-shift-2024,
  author = {Aronvision},
  title = {RD Model Backbone Shift: ResNet50 to ConvNeXt},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Aronvision/RD-Model-Backbone-Shift}
}
```

## License
This project is licensed under the MIT License.

## Acknowledgments
- MVTec Software for providing the AD dataset
- Original RD paper authors for the methodology
- PyTorch team for the framework
