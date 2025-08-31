# ***LaneNet: Enhanced Curved Lane Detection***

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7.0+-EE4C2C?logo=pytorch)]()
[![CUDA](https://img.shields.io/badge/CUDA-11.0-green?logo=nvidia)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7.0+-blue?logo=opencv)]()
[![Albumentations](https://img.shields.io/badge/Albumentations-0.4.6-orange)]()
[![MMCV](https://img.shields.io/badge/MMCV-1.2.5-yellow)]()
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.2.2-blue)]()
[![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.2.1-F7931E?logo=scikit-learn)]()

LaneNet is a **deep learning-based label assignment module** integrated into the state-of-the-art **Cross Layer Refinement Network (CLRNet)** for lane detection.  
Unlike classical cost functions used for label assignment, LaneNet **learns to match predicted lanes to ground truth dynamically**, leading to **improved performance on challenging curved lane scenarios** while maintaining parity on standard cases.

---

## ***Demo***

> <img width="759" height="174" alt="Screenshot 2025-08-31 at 3 56 34 PM" src="https://github.com/user-attachments/assets/fea3c2ec-ba93-4893-8f34-c57b5891c12e" />
 
> <img width="653" height="555" alt="results" src="https://github.com/user-attachments/assets/cbed602f-ea03-49c6-9450-f4d7ab9375bc" />
  

---

## ***Motivation***

Lane detection is critical for autonomous driving and ADAS systems. Existing anchor-based models like CLRNet rely on handcrafted cost functions to assign predicted lanes to ground truths, which:

- Operate in low-dimensional space  
- Lack flexibility in unseen scenarios  
- Struggle with **curved lanes, occlusions, and lighting variations**  

LaneNet replaces this handcrafted label assignment step with a **learned neural network**, improving the model’s understanding of complex road geometries.

---

## ***Key Contributions***

- **Learned Label Assignment**: LaneNet uses a fully connected neural network to predict match probabilities between predictions and ground truth.
- **Seamless Integration with CLRNet**: LaneNet plugs into the CLRNet architecture without altering its core design.
- **Curve-Focused Training**: Pretrained on a **curve-specific subset** of CULane to specialize in challenging geometries.
- **Performance Gains**:  
  - +2.8% F1 (ResNet34 backbone)  
  - +2.3% F1 (ResNet101 backbone)  
  - +2.96% F1 (DLA34 backbone)  
- **Higher Confidence Thresholds**: Improves detection confidence, enabling stricter filtering for better precision.

---

## ***Architecture***

> <img width="904" height="260" alt="Screenshot 2025-08-31 at 4 01 48 PM" src="https://github.com/user-attachments/assets/75a2c4af-1638-4c3b-9f27-1744580e0e3a" />#

**Pipeline Overview**:
1. **Backbone Feature Extraction**  
   - ResNet or DLA backbone with FPN to extract multi-scale lane features.

2. **Lane Priors & ROIGather**  
   - Anchor-based lane priors guide pooling of lane features.

3. **Lane Prediction Head**  
   - Outputs classification scores, geometric parameters, and auxiliary segmentation maps.

4. **LaneNet Matching Module**  
   - Predicts **match probabilities** for each predicted lane-ground truth pair, replacing handcrafted cost functions.

---

## ***Dataset***

We use the **CULane dataset**, a benchmark for lane detection research:

| Subset           | Description                                | Images |
|------------------|--------------------------------------------|--------|
| Training          | Standard lane images                       | ~88K   |
| Validation        | Standard lane images                       | ~9.7K  |
| Test              | Categorized into 9 scenarios (normal, crowd, dazzle, shadow, curve, etc.) | ~34K   |
| Curve Subset      | Focused dataset of **8,677 curved-lane images** for pretraining LaneNet. | 8,677  |

---

## ***Results***

| Backbone   | Model         | Curve F1 | Δ Improvement |
|------------|--------------|---------|---------------|
| ResNet34   | CLRNet       | 72.77   | -             |
|            | **CLRLaneNet** | **75.57** | **+2.8%**      |
| ResNet101  | CLRNet       | 75.57   | -             |
|            | **CLRLaneNet** | **77.87** | **+2.3%**      |
| DLA34      | CLRNet       | 74.13   | -             |
|            | **CLRLaneNet** | **77.09** | **+2.96%**     |

---

## ***Installation***

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/LaneNet.git
cd LaneNet

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

### ***Training***
```bash
# Pretrain LaneNet on curve subset
python train_lanenet.py --config configs/lanenet_curve.yaml

# Fine-tune CLRNet + LaneNet integration
python train_clrlanenet.py --config configs/clrlanenet.yaml
```

### ***Evaluation***
```bash
python evaluate.py --config configs/clrlanenet.yaml --weights <checkpoint_path>
```

### ***Validation***
```bash
python main.py [configs/path_to_your_config] --[test|validate|demo] --load_from [path_to_clrmatchnet_model] --gpus [gpu_num]
```
For example, run:
```bash
python main.py configs/clrnet/clr_dla34_culane.py --test --load_from=culane_dla34.pth --gpus=1
```
> This code can output the visualization result when testing, add `--view`. We will get the visualization result in `work_dirs/xxx/xxx/xxx/visualization`

### ***Demo***
```bash
python main.py [configs/path_to_your_config] --demo --load_from [path_to_clrmatchnet_model] --gpus [gpu_num] --view
```
---
## ***References***
1.	X. Pan, J. Shi, P. Luo, X. Wang, and X. Tang. Spatial as Deep: ***Spatial CNN for Traffic Scene Understanding.*** In Proceedings of the AAAI Conference on Artificial Intelligence, 2018
2.	T. Zheng et al. ***CLRNet: Cross Layer Refinement Network for Lane Detection.*** In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.


