# LaneNet: Enhanced Curved Lane Detection with Deep Matching Process

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/pytorch-1.9+-red.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-CULane-green.svg)]()

LaneNet is a **deep learning-based label assignment module** integrated into the state-of-the-art **Cross Layer Refinement Network (CLRNet)** for lane detection.  
Unlike classical cost functions used for label assignment, LaneNet **learns to match predicted lanes to ground truth dynamically**, leading to **improved performance on challenging curved lane scenarios** while maintaining parity on standard cases.

---

## Demo

> *(Insert pipeline diagram here)*  
> *(Insert sample detection comparison: CLRNet vs LaneNet)*  

---

## Motivation

Lane detection is critical for autonomous driving and ADAS systems. Existing anchor-based models like CLRNet rely on handcrafted cost functions to assign predicted lanes to ground truths, which:

- Operate in low-dimensional space  
- Lack flexibility in unseen scenarios  
- Struggle with **curved lanes, occlusions, and lighting variations**  

LaneNet replaces this handcrafted label assignment step with a **learned neural network**, improving the model’s understanding of complex road geometries.

---

## Key Contributions

- 📌 **Learned Label Assignment**: LaneNet uses a fully connected neural network to predict match probabilities between predictions and ground truth.
- 🌉 **Seamless Integration with CLRNet**: LaneNet plugs into the CLRNet architecture without altering its core design.
- 🛣️ **Curve-Focused Training**: Pretrained on a **curve-specific subset** of CULane to specialize in challenging geometries.
- 📈 **Performance Gains**:  
  - +2.8% F1 (ResNet34 backbone)  
  - +2.3% F1 (ResNet101 backbone)  
  - +2.96% F1 (DLA34 backbone)  
- ⚡ **Higher Confidence Thresholds**: Improves detection confidence, enabling stricter filtering for better precision.

---

## Architecture

> *(Insert architecture diagram: CLRNet + LaneNet block)*

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

## Dataset

We use the **CULane dataset**, a benchmark for lane detection research:

| Subset           | Description                                | Images |
|------------------|--------------------------------------------|--------|
| Training          | Standard lane images                       | ~88K   |
| Validation        | Standard lane images                       | ~9.7K  |
| Test              | Categorized into 9 scenarios (normal, crowd, dazzle, shadow, curve, etc.) | ~34K   |
| Curve Subset      | Focused dataset of **8,677 curved-lane images** for pretraining LaneNet. | 8,677  |

---

## Results

| Backbone   | Model         | Curve F1 | Δ Improvement |
|------------|--------------|---------|---------------|
| ResNet34   | CLRNet       | 72.77   | -             |
|            | **CLRLaneNet** | **75.57** | **+2.8%**      |
| ResNet101  | CLRNet       | 75.57   | -             |
|            | **CLRLaneNet** | **77.87** | **+2.3%**      |
| DLA34      | CLRNet       | 74.13   | -             |
|            | **CLRLaneNet** | **77.09** | **+2.96%**     |

> *(Insert F1 score comparison bar chart)*

---

## 🛠Implementation Details

- **Framework**: PyTorch  
- **Backbones**: ResNet34, ResNet101, DLA34  
- **Optimizer**: AdamW with cosine learning rate decay  
- **Image Size**: 320 × 800  
- **Thresholds**:  
  - Match threshold: `0.7`  
  - Confidence threshold: `0.43`  

---

## Installation

```bash
git clone https://github.com/<your-username>/LaneNet.git
cd LaneNet

# Install dependencies
pip install -r requirements.txt
