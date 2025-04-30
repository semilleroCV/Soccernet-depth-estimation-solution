# 🥅 SoccerNet Challenge - Monocular Depth Estimation 🏟️

## 📝 Project Description
This repository contains the solution for the SoccerNet Monocular Depth Estimation challenge. The goal is to predict accurate depth maps from RGB images of soccer matches, enabling better 3D spatial understanding of sports scenes.

## 🔍 Key Features
- 🔄 Based on **Depth Anything V2**, state-of-the-art model for depth estimation
- 📊 Evaluation using standard metrics: RMSE, Abs_Rel, SILog, RMSE Log, Sq_Rel
- 🧠 Fine-tuning of pre-trained models for the specific domain of soccer images
- 🎯 Optimized for soccer field scenes with specific color and geometric transformations

## 🧩 Architecture
The solution uses the Depth Anything V2 architecture with different encoders (Small, Base, Large), which combines a pre-trained ViT backbone with a specialized DPT head to estimate precise metric depth.

## 🛠️ Requirements
- Python 3.9
- PyTorch
- CUDA (for accelerated training)

## 📂 Repository Structure
- `data/`: Data loaders and utilities
- `loss/`: Specialized loss functions
- `metrics/`: Evaluation metrics implementation
- `models/`: Model architectures (Depth Anything V2)

## 📊 Results

| Config                  | Abs Rel ×10⁻³ ↓ | RMSE ×10⁻³ ↓ | RMSE Log ×10⁻³ ↓ | Sq Rel ×10⁻⁴ ↓ | SILog ↓ |
|-------------------------|------------------|---------------|-------------------|----------------|----------|
| Best Baseline [Leduc et al., 2024] | 2.429           | 2.343        | 4.002             | 0.121         | 0.400    |
| Ours Half-Res           | 2.781           | 2.516        | 4.350             | 0.125         | 0.435    |
| **Ours Full-Res**       | **1.443**       | **1.590**    | **2.619**         | **0.062**     | **0.261** |


## 🚀 Usage
```bash
# Training
bash train.sh

# Evaluation
python evaluate_depth.py --pred_dir [predictions_directory] --gt_dir [ground_truth_directory]

# Inference
python challenge.py --encoder vitl --checkpoint_path [checkpoint_path] --input_dir [input_directory] --output_dir [output_directory]
```
Training options can be modified in the train.sh file.
