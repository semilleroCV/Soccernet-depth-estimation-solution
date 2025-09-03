# 🥇 CVPR 2025 Winner - SoccerNet Monocular Depth Estimation Challenge

Team Hand-on Computer Vision

Members: [Fabian Perez](https://github.com/factral/), [Henry Mantilla](https://github.com/HenryMantilla), [Jorge García](https://github.com/jorge1b3), [Cristian Rey](https://github.com/CristianR8), Jose Sarmiento and [Hoover Rueda-Chacón](https://www.hfarueda.com/)

## 🏟️ Project Description

This repository contains the **winning solution** to the SoccerNet Monocular Depth Estimation Challenge at **CVPR 2025**. Our method predicts accurate depth maps from RGB images of soccer matches, enabling improved 3D spatial understanding in sports analytics.

## 📈 Challenge Results

🏆 **1st Place** at the SoccerNet Monocular Depth Estimation Challenge (CVPR 2025)

| Rank | Team                     | RMSE   | AbsRel | RMSElog | SqRel  | SIlog |
|------|--------------------------|--------|--------|---------|--------|-------|
| 1️⃣   | Hands-On Computer Vision | 0.00242 | 0.00164 | 0.00432 | 2e-05 | 0.43  |
| 2    | HUST-iPad                | 0.00258 | 0.00179 | 0.00468 | 3e-05 | 0.47  |
| 3    | bupt miclab              | 0.00268 | 0.00186 | 0.00484 | 3e-05 | 0.48  |
| 4    | jacekm                   | 0.00275 | 0.00207 | 0.00500 | 3e-05 | 0.50  |
| 5    | hvrl                     | 0.00282 | 0.00228 | 0.00502 | 3e-05 | 0.50  |

📺 [Presentation Video](https://youtu.be/5W_7nTs2FDo)  
📊 [Official Leaderboard](https://www.codabench.org/competitions/6864/#/results-tab)
📰 [Paper](https://arxiv.org/pdf/2508.19182v1)

---

## 🔍 Key Features

- 🔄 Based on **Depth Anything V2**, a state-of-the-art model for depth estimation.
- 📊 Evaluated using standard metrics: RMSE, AbsRel, SILog, RMSElog, SqRel.
- 🧠 Fine-tuned for soccer-specific imagery and temporal consistency.
- 🖼️ Preserves sharp edges and fine details via specialized losses.
- 📏 Full-resolution training for enhanced accuracy.

## 🧩 Architecture
The solution uses the Depth Anything V2 Large architecture, which combines a pre-trained ViT backbone with a specialized DPT head to estimate precise metric depth.

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


---


feel free to use it and modify it as you wish, if you find this code useful please give a star ⭐ to the repository.
