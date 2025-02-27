# Infant Audio Classification: Cry, Scream, and Normal Utterances

## Project Overview
This project aims to develop a robust ensemble-based audio classification system that distinguishes between infant cries, screams, and normal utterances. The approach integrates two powerful models, **YAMNet** and **Wav2Vec2**, to leverage both pre-trained knowledge and fine-tuned accuracy for this specific classification task.

The system processes raw audio input, extracts relevant features, and classifies the sounds into three categories:
- **Crying**
- **Screaming**
- **Normal Utterances**

An ensemble technique is used to combine the predictions from YAMNet and Wav2Vec2, ensuring improved accuracy and robustness.

---
## Features
- Multi-dataset integration for diverse and balanced training.
- Preprocessing techniques including noise reduction and pitch normalization.
- Fine-tuned **YAMNet** and **Wav2Vec2** models.
- **Ensemble learning** to improve classification accuracy.
- **Google Colab integration** for ease of execution.
- Performance evaluation through accuracy, precision, recall, and confusion matrices.

---
## Theory and Model Explanation

### **1. YAMNet**
YAMNet is a deep neural network trained on **AudioSet** to recognize over 500 different sound classes. It uses **Mel spectrograms** as input features and applies a **MobileNet** architecture.

**Modifications in this project:**
- Fine-tuned the model on our specific dataset.
- Adjusted output layers to classify three specific categories.

### **2. Wav2Vec2**
Wav2Vec2 is a self-supervised **speech representation learning** model by Facebook AI. Unlike YAMNet, it learns **contextualized representations** directly from raw audio signals, making it robust to noise and variations.

**Modifications in this project:**
- Fine-tuned on labeled infant audio data.
- Adjusted classification head for cry, scream, and normal utterance detection.

### **3. Ensemble Model**
We implemented an ensemble approach by combining outputs from both models using:
- **Averaging Probabilities**: Taking the mean of both models' predictions.
- **Majority Voting**: Selecting the most frequent predicted class.

This helps in reducing errors and improving overall accuracy.

---
## Dataset
The project utilizes multiple datasets:
- **Infant Cry Audio Corpus from KAGGLE**
- **Human Screaming Detection Dataset from KAGGLE**
- **Children speech Audioset 4**

All datasets were preprocessed to ensure:
- **Consistent sample rates**
- **Uniform bit-depth normalization**
- **Proper segmentation and labeling**

---
## Setup Instructions
### **1. Google Colab Setup**
This project is implemented in **Google Colab** for ease of execution.

**Steps to Run on Colab:**
1. Open Google Colab: [Colab Link](https://colab.research.google.com/)
2. Upload the dataset to **Google Drive**.
3. Mount Google Drive in Colab:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
4. Clone the GitHub repository:
    ```bash
    !git clone https://github.com/Arjun-08/Developing-an-Ensemble-Model-for-Detecting-Infant-Cries-Screams-and-Normal-Utterances.git
    cd Developing-an-Ensemble-Model-for-Detecting-Infant-Cries-Screams-and-Normal-Utterances
    ```

    


5. Install required dependencies:
    ```python
    !pip install -r requirements.txt
    ```
6. Run the training and inference scripts (detailed below).

---
## Training, Testing, and Validation

### **Dataset Split:**
- **70% Training**
- **15% Validation**
- **15% Testing**

### **Evaluation Metrics:**
- **Accuracy**
- **Precision, Recall, F1-score**
- **Confusion Matrices**
- **ROC Curves**

---
## Inference Instructions
Once the model is trained, you can perform inference on new audio files.

### **Run Inference on a Test Audio File:**
```python
from model import run_inference
prediction = run_inference('/content/drive/MyDrive/frontera/extracted_data/Screaming/---1_cCGK4M_out.wav')
print("Predicted Label:", prediction)
```

### **Example Prediction Output:**
```bash
Predicted Label: [3] (crying)
```

---
## Results
### **YAMNet Model Performance:**
```
Epoch 1/10
accuracy: 0.6756 - loss: 0.9839 - val_accuracy: 0.7826 - val_loss: 0.9807
...
Epoch 10/10
accuracy: 0.8676 - loss: 0.4984 - val_accuracy: 0.7826 - val_loss: 0.7915
```

### **Wav2Vec2 Model Performance:**
```
Epoch 1   Validation Loss: 0.815121
Epoch 2   Validation Loss: 0.833583
Epoch 3   Validation Loss: 0.837320
```

### **Ensemble Model Performance:**
```
Train Loss: 0.6878751118977865
Test Accuracy: 0.7681
Test Precision: 0.5900
Test Recall: 0.7681
Test F1 Score: 0.6674
```

---
## References
- [YAMNet Model](https://tfhub.dev/google/yamnet/1)
- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477)
- [AudioSet Dataset](https://research.google.com/audioset/)


---
## Contact
For questions or collaborations, reach out via [nvarjunmani07@gmail.com](mailto:nvarjunmani07@gmail.com).

---
### **Acknowledgments**
Special thanks to the Team FRONTERA HEALTH and dataset providers that made this research possible!

