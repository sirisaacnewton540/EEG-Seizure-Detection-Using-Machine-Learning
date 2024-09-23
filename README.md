# EEG Seizure Detection Using Machine Learning

## Overview

This project aims to apply machine learning to detect epileptic seizures from electroencephalogram (EEG) data. Epilepsy affects approximately 50 million people worldwide, with many cases being undiagnosed or diagnosed late due to the complexity of seizure detection. EEG is the primary diagnostic tool for epilepsy; however, manual analysis of EEG data is time-consuming and prone to errors due to the subtle patterns that differentiate seizure events from normal brain activity. 

This project utilizes EEG data to build machine learning models that classify seizure and non-seizure events. By applying feature extraction techniques and state-of-the-art models like CNN-LSTM, we aim to create an automated, accurate, and real-time seizure detection system that could be integrated into clinical practice.

## Results
![1](https://github.com/user-attachments/assets/eee979a2-54d8-4e52-b349-9e4f3490ad76)

![2](https://github.com/user-attachments/assets/2e01712f-0433-4fb7-ac2c-45a221c43b02)


## Project Structure

```
├── data/                    # Contains the EEG dataset (included in the repository)
├── notebooks/                # Jupyter notebooks with detailed analysis and experiments
├── models/                   # Trained models and performance metrics
├── results/                  # Confusion matrix, accuracy reports, and other results
├── src/                      # Source code for feature extraction, modeling, and evaluation
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

## Dataset

The dataset used in this project is an openly available EEG dataset, which contains recordings of brain activity from 5 different classes:
1. **Class 1:** Seizure activity.
2. **Class 2:** Non-seizure activity (EEG recording from the tumor region).
3. **Class 3:** Non-seizure activity (EEG recording from a healthy brain region).
4. **Class 4:** Eyes closed (non-seizure).
5. **Class 5:** Eyes open (non-seizure).

The dataset consists of 178 features, which are EEG signal values at different time points, for 4097 data points per recording. Our goal is to perform binary classification, where Class 1 (seizure) is classified against all other classes (non-seizure). The dataset has been preprocessed to simplify access, converting the raw data into CSV format for ease of analysis.

## Research Objectives

- **Primary Objective:** Develop a robust machine learning model that can classify EEG signals as seizure or non-seizure events, achieving high sensitivity and specificity.
- **Secondary Objective:** Investigate the effectiveness of combining feature-based machine learning models (e.g., Random Forest, SVM) with deep learning techniques (e.g., CNN-LSTM) for temporal and spatial pattern recognition.
- **Exploratory Objective:** Compare the performance of traditional machine learning models versus deep learning architectures in detecting and classifying epileptic seizures.

## Background Knowledge

### Understanding EEG Signals

Electroencephalogram (EEG) measures electrical activity in the brain and is widely used in neurology to detect abnormalities, such as epileptic seizures. Seizures cause sudden, excessive electrical discharges in the brain, detectable through characteristic patterns in EEG recordings. However, distinguishing between these abnormal patterns and normal brain activity is challenging due to the noisy and complex nature of EEG signals.

### Seizure Detection in Clinical Practice

In clinical practice, neurologists manually analyze EEG data to detect seizures, which is highly subjective and error-prone. Automated detection systems can reduce the burden on neurologists by providing quick, objective, and reliable diagnoses. Machine learning, particularly models that can capture both temporal and spatial features, is an ideal approach for this task.

## Methodology

### Data Preprocessing

1. **Handling Missing Data:** 
   - Missing values were imputed using the mean value of the corresponding feature columns.
   
2. **Normalization:**
   - The EEG signal values were normalized to a common scale to ensure that variations in magnitude between different subjects do not affect model training.
   
3. **Reshaping Data for Deep Learning Models:**
   - For deep learning models like CNN-LSTM, the data was reshaped to a 3D format (samples, time_steps, features) to account for the temporal nature of the EEG signals.

### Feature Extraction

EEG signals contain a rich set of features that span across both the time domain and the frequency domain. We extracted various features to enhance model performance:

- **Time Domain Features:** Mean, variance, skewness, kurtosis, and zero-crossing rate.
- **Frequency Domain Features:** Using Fourier Transform and Welch’s Method, we extracted power spectral densities (PSD) in different frequency bands (alpha, beta, gamma, delta, theta).

These features were used for training traditional machine learning models like Random Forest and SVM.

### Modeling Approaches

#### 1. **Random Forest Classifier:**
   - Random Forest was used as a baseline model due to its robustness in handling high-dimensional data.
   - Key metrics: **Accuracy**, **F1-Score**, **Precision**, **Recall**.

#### 2. **Support Vector Machine (SVM):**
   - SVM was used to classify EEG signals based on extracted features. SVM is effective at drawing decision boundaries between different classes in high-dimensional space.

#### 3. **CNN-LSTM:**
   - **Convolutional Neural Network (CNN)** captures spatial patterns in the EEG signals, useful for identifying local features like spikes or waves associated with seizure activity.
   - **Long Short-Term Memory (LSTM)** is used to capture temporal dependencies across time-series data, making it ideal for EEG signals which are inherently sequential.

### Model Training and Evaluation

- The dataset was split into training and test sets using an 80-20 ratio.
- **Random Forest** and **SVM** were trained on the extracted features from the time and frequency domains.
- **CNN-LSTM** was trained directly on the raw EEG data after reshaping the input into the 3D format.
- Cross-validation was used to ensure generalization and prevent overfitting.

## Results

### Random Forest and SVM Performance:

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Random Forest         | 91.4%    | 90.2%     | 89.7%  | 89.9%    |
| SVM                  | 88.6%    | 87.8%     | 87.2%  | 87.5%    |

- **Random Forest** performed better than SVM in terms of accuracy, likely due to its ensemble nature and ability to handle feature interactions well.
- **SVM** also showed good performance, but struggled slightly in capturing complex non-linear patterns in the data.

### CNN-LSTM Performance:

| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| CNN-LSTM (Deep Model) | 93.7%    | 93.1%     | 92.5%  | 92.8%    |

The CNN-LSTM model outperformed traditional models in all metrics, demonstrating its ability to capture both local spatial patterns (via CNN) and long-term temporal dependencies (via LSTM). This improvement is especially notable in the **Recall** and **F1-Score**, indicating that the model is effective at detecting true seizure events while minimizing false negatives.

### Confusion Matrix for CNN-LSTM:

![2](https://github.com/user-attachments/assets/dc044f2f-6f73-4bba-b7df-3bfc236f5482)


- The CNN-LSTM model had very few false positives and false negatives, indicating high sensitivity and specificity.
- The confusion matrix shows that the model correctly classified most seizure and non-seizure events, providing a reliable basis for real-time seizure detection.

## Results Interpretation

The results of the CNN-LSTM model suggest that combining **spatial pattern recognition** (CNN) with **temporal sequence learning** (LSTM) is highly effective for analyzing EEG data. By capturing both the local features (e.g., spikes in EEG signals) and the temporal evolution of the signals, the model is able to detect seizures more accurately than traditional methods.

- **High Recall (92.5%)**: Indicates the model’s ability to detect actual seizure events with minimal false negatives.
- **High Precision (93.1%)**: Ensures that false positives (non-seizure events incorrectly classified as seizures) are minimized, making it suitable for clinical applications where over-diagnosis could be problematic.

These findings suggest that deep learning methods, particularly CNN-LSTM, can significantly improve automated seizure detection and provide a foundation for further real-time clinical implementations.

## Conclusion

This project demonstrates the potential of using machine learning and deep learning techniques to accurately detect epileptic seizures from EEG data. The CNN-LSTM model outperformed traditional machine learning models by effectively capturing both spatial and temporal dependencies in the EEG signals.

This work could be extended for **real-time seizure detection**, where automated alerts could be sent to healthcare professionals when seizures are detected, providing faster interventions and improving patient outcomes.

## Future Work

1. **Real-time Seizure Monitoring:**
   - Developing a system that continuously monitors EEG signals and sends real-time alerts during seizure events.
   
2. **Model Improvement:**
   - Exploring advanced deep learning techniques such as **transformers** for further improving temporal pattern recognition.
   
3. **Larger Dataset:**
   - Testing the model on larger, more diverse datasets to assess its generalization to different demographics and EEG devices.
   
4. **Explainability of the Model:**
   - Implementing methods like **Grad-CAM** to visualize which regions of the EEG signal contribute most to the model’s predictions, making it more interpretable for clinical use.

## Dependencies

- Python 3.7+
- TensorFlow 2.0+
- Scikit-learn
- NumPy
- Matplotlib
- Pandas

## Contact

For any inquiries or contributions, feel free to reach out:

- **Pushpendra Singh**
- [Email](mailto:spushpendra540@gmail.com)
- [GitHub](https://github.com/sirisaacnewton540)
