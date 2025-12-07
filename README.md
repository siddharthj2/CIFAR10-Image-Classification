CIFAR-10 Image Classification using CNN and ResNet-18

This project implements and compares two deep learning models for image classification on the CIFAR-10 dataset:

A custom Convolutional Neural Network (CNN) trained from scratch

A transfer learning approach using a pre-trained ResNet-18 model

The objective is to analyze how architecture design and training strategy influence model performance under resource constraints.


📊 Dataset

The CIFAR-10 dataset consists of 60,000 32x32 RGB images across 10 classes:

airplane, automobile, bird, cat

deer, dog, frog, horse

ship, truck

Split:

50,000 training images

10,000 testing images

Download:
🔗 https://www.cs.toronto.edu/~kriz/cifar.html

🏗️ Model Architectures
1. Baseline CNN (Trained from Scratch)

3 convolutional blocks (32, 64, 128 channels)

ReLU activation + MaxPool

Fully connected layers: 512 → 10

Softmax classification

Design Goals:

Lightweight

Efficient training

Strong baseline performance

2. ResNet-18 (Transfer Learning)

Pre-trained on ImageNet

All convolutional layers frozen

Final fully connected layer replaced

Only classification head trained

Goal:
Evaluate transfer learning performance with minimal fine-tuning.

⚙️ Training Configuration

Epochs: 5

Optimizer: Adam (lr = 1e-3)

Loss: Cross-Entropy

Batch size: 64

Hardware: CPU/GPU

Metrics tracked per epoch:

Loss

Accuracy

📈 Results
Baseline CNN Performance
Metric	Score
Accuracy	0.7580
Precision	0.7635
Recall	0.7580
F1-Score	0.7592

Observations:

Rapid convergence

Strong generalization

Some confusion among visually similar classes

ResNet-18 Performance
Metric	Score
Accuracy	0.4028
Precision	0.4131
Recall	0.4028
F1-Score	0.3963

Observations:

Minimal improvement with training

Predictions close to random for many classes

Poor adaptation of pre-trained features

🥊 Quantitative Comparison
Model	Accuracy	Precision	Recall	F1-Score
Baseline CNN	0.7580	0.7635	0.7580	0.7592
ResNet-18	0.4028	0.4131	0.4028	0.3963

Winner:
🚀 Baseline CNN

🔍 Key Insights

Transfer learning does not always guarantee better performance

Freezing the ResNet feature extractor severely limits adaptation

Input resolution mismatch (32×32 vs 224×224) reduces feature quality

Under limited training time, simpler models may outperform larger ones

🧪 Evaluation Artifacts

This project includes:

Training curves

Confusion matrices

Saved metrics as JSON

For reproducibility, see:
results/

💡 How to Run

Clone the repo:

git clone https://github.com/<your-username>/CIFAR10-Image-Classification.git
cd CIFAR10-Image-Classification


Open the notebook:

Via Google Colab

Or Jupyter Notebook

Run all cells to train and evaluate models
(GPU recommended for speed)

🔧 Requirements

If needed, install dependencies:

torch
torchvision
numpy
pandas
matplotlib
seaborn
scikit-learn

📄 Report

A full academic report is included in report.pdf, summarizing:

Dataset

Methodology

Architectures

Results

Discussion

Conclusion

References

🚀 Future Work

Unfreeze ResNet layers

Resize inputs to 224×224

Train for 20–50 epochs

Introduce learning rate scheduling

Explore WRN, DenseNet, ViT

👨‍🎓 Author

Name: Siddharth jindal
Course: Deep Learning 


📝 License

This project is open-source under the MIT License.

⭐ Acknowledgements

CIFAR-10 dataset

🎉 Summary

This repository demonstrates that simpler custom architectures can outperform poorly adapted transfer-learning models under limited training conditions and constrained computational resources.
