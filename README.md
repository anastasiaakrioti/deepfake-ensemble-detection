# A Unified Ensemble Framework for Detecting Manipulated and AI-Generated Facial Content


This repository serves as the landing page for the implementation code of the research paper: **"A Unified Ensemble Framework for Detecting Manipulated and AI-Generated Facial Content"**.

The project proposes a robust deepfake detection framework leveraging an ensemble of four state-of-the-art CNN architectures, optimized via an adaptive weighted soft-voting strategy.

## 🚀 Implementation & Code

To ensure reproducibility and provide access to the exact environment and datasets used, the full source code (including preprocessing, training, and evaluation) is hosted on **Kaggle**.

<a href="https://www.kaggle.com/code/anastasiaakr/ensemble-framework">
  <img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle" height="40"/>
</a>

👉 **[Click here to view the Full Implementation on Kaggle](https://www.kaggle.com/code/anastasiaakr/ensemble-framework)**

---

## Abstract

Deepfake video has become a serious and technological threat, as it is used to spread fake news, misinformation and manipulate public opinion. Advanced techniques for synthesizing and editing multimedia content make the detection of such manipulated videos particularly challenging, creating the need for reliable detection solutions. This paper proposes a Deepfake detection method based on Ensemble Learning, aiming to improve the efficiency of classifying images as real or fake. The implementation leverages an ensemble of four pre-trained convolutional neural networks — EfficientNet-B0, ResNet50, MobileNetV3, and DenseNet121— combined with an adaptive weighted soft voting strategy. The analysis is conducted on three distinct datasets, each with unique characteristics, as well as on a constructed Unified dataset, with all three datasets combined. The classifier is developed using the Python programming language and modern deep learning libraries; its performance is evaluated using metrics such as AUC-ROC, Accuracy, and Sensitivity. The experimental results demonstrate that the proposed ensemble outperforms individual baselines, confirming the potential of a multi-architecture strategy as an effective tool in addressing the deepfake detection problem. 
The contribution of this study lies not only in its technical approach to detecting deepfake content but also in emphasizing the severity of the issue, highlighting the need for advanced mechanisms to protect against the spread of fake audiovisual material

## Methodology

### 1. Backbone Architectures
We utilize four distinct CNN architectures to capture diverse features:
* **ResNet50:** Robust baseline with residual connections.
* **EfficientNet-B0:** optimized for balance between depth, width, and resolution.
* **MobileNetV3-Large:** Lightweight architecture with Squeeze-and-Excitation blocks.
* **DenseNet121:** Focuses on feature reuse through dense connectivity.

### 2. Ensemble Strategy
Instead of simple averaging, we employ a **Grid Search-optimized Weighted Soft Voting** strategy. Weights are assigned based on the validation performance of each backbone to minimize generalization error.

## Datasets

The model was evaluated on:
1.  **DFDC (DeepFake Detection Challenge):** Diverse generation methods (GANs, etc.).
2.  **DeepFake MNIST+:** Focuses on facial animation/expressions.
3.  **EXP:** High-quality manually edited images (Photoshop) by experts.
4.  **Unified Dataset:** A combination of the above to test "in-the-wild" robustness.

## Requirements

The code is designed to run in a Kaggle Notebook environment (GPU accelerator recommended). Main dependencies include:
* `torch` & `torchvision`
* `numpy` & `pandas`
* `scikit-learn`
* `matplotlib` & `seaborn`
* `Pillow`

## Reference
If you use this code or findings, please refer to the original implementation link:
[https://www.kaggle.com/code/anastasiaakr/ensemble-framework](https://www.kaggle.com/code/anastasiaakr/ensemble-framework)
