# Title: Improved Pneumonia Classification Using Deep Learning with Class Balancing and Explainability Enhancements
Author: Hyun Woo Kim
 Code Repository: https://github.com/hyunwookim129/pneumonia_detection_GradCAM
 
Introduction
This report details the development, training, and evaluation of an enhanced deep learning model for pneumonia classification using chest X-ray images. Compared to previous iterations, this version incorporates class balancing, additional regularization techniques, and Grad-CAM for interpretability. The dataset remains imbalanced, necessitating corrective strategies such as class weighting and adjusted decision thresholds.

Dataset Description
1. Training Set:
Total images: 4187
Class distribution:
NORMAL: 1080
PNEUMONIA: 3107
Class imbalance was mitigated using class weighting to prevent bias towards the majority class.

3. Validation Set:
Total images: 1045
Class distribution:
NORMAL: 269
PNEUMONIA: 776
Ensured a meaningful validation set for better model generalization.

4. Test Set:
Total images: 624
Class distribution:
NORMAL: 234
PNEUMONIA: 390

Data Augmentation
To improve generalization and combat overfitting, the training data underwent the following augmentations:
Rescaling: Normalize pixel values to [0, 1].
Rotation: Up to 20 degrees.
Shifting: Horizontal and vertical shifts by up to 20%.
Shearing: Shear intensity of 20%.
Zooming: Random zoom by up to 20%.
Horizontal Flipping: Introduced variations in image orientation.

Model Architecture
The CNN-based model follows an improved design with additional regularization and explainability features:
Three convolutional layers (Conv2D)
Filter sizes: 32, 64, 128
L2 regularization (0.001) applied to convolutional layers
Batch Normalization and MaxPooling to stabilize training
Global Average Pooling Layer
Reduces overfitting by replacing Flatten layers with averaging operations
Fully Connected Layers:
Dense(128, activation='relu') with L2 regularization
Dense(64, activation='relu') with L2 regularization
Dropout (0.5) applied for additional regularization
Output Layer:
Dense(1, activation='sigmoid') for binary classification

Training Process
Loss Function: Binary Crossentropy
Optimizer: Adam with mixed precision training
Learning Rate Schedule: ReduceLROnPlateau with an initial learning rate of 0.0001, decaying by a factor of 0.5 when validation loss stops improving (min_lr = 1e-5)
Batch Size: 32
Class Weights:
NORMAL: 1.938
PNEUMONIA: 0.674
Results
1. Training Performance
Final Training Accuracy: 90.98%
Final Validation Accuracy: 92.63%
Final Training Loss: 0.4559
Final Validation Loss: 0.4312

2. Evaluation Metrics

Overall Accuracy: 85%
AUC-ROC: 0.93
AUC-PR: 0.95

3. Detailed Metric Analysis
Normal Class:
Precision: 77% (77% of predicted Normal cases are correct).
Recall: 85% (85% of actual Normal cases are identified).
Pneumonia Class:
Precision: 90% (90% of predicted Pneumonia cases are correct).
Recall: 85% (85% of actual Pneumonia cases are identified).
ROC Curve
AUC: 0.93, indicating strong separation between classes.

Precision-Recall Curve
AUC-PR: 0.95, confirming strong predictive performance.

Calibration Curve
The model is slightly underconfident, with predicted probabilities generally lower than the true likelihood of pneumonia.

Explainability: Grad-CAM Visualization
a. Appearance
The heatmap highlights certain vertical regions along the midline of the chest and scattered areas near the lung fields. Some “hotspots” (green/yellow) suggest that the model is focusing on (or finding relevant) features in these areas.
b. Interpretation
If the model was trained to detect pneumonia, the highlighted regions might be areas the network associates with infiltrates or indicators of pathology. That said, some activation is also near the mediastinum (center of the chest), which can indicate the model is using non-pulmonary cues or picking up borderline edges.

Potential Issues

Midline Dominance
The bright vertical streak in the center could suggest the model is partly focusing on the heart, spine, or mediastinal shadow instead of strictly the lung fields. This sometimes happens if the dataset contains confounding features or if certain data augmentations have shifted the model’s focus.

Scattered Peripheral Activations
The orange/green patches at the image edges could mean the model is responding to the boundary or external aspects (e.g., ribs or image corners). This is more common when the training dataset had consistent features like text markers, or if the classifier inadvertently learned extraneous signals correlated with pneumonia labels.

Unclear Lung Coverage
While some color is in the lung fields, it’s not fully clear whether the most suspicious lung zones (e.g., lower lobes) are being highlighted. If the primary pneumonia signs are subtle, the Grad-CAM might miss them or highlight them only weakly.
Current Visualization: The heatmap indicates the model is partially focusing on the lungs but also significantly on the mediastinal region. This may be due to either legitimate cues in the center of the chest or spurious correlations.

Interpretation Caveat: Grad-CAM only shows what the model believes is relevant for its classification. It does not confirm that these anatomical areas are truly indicative of pneumonia. For medical validation, domain expertise and clinical context remain essential.
Strengths of the Model

Improved Pneumonia Detection
High AUC-ROC and AUC-PR values indicate strong classification ability.
Balanced Classification Performance
Adjusting the decision threshold from 0.5 to 0.3 increased recall and improved overall classification.

Enhanced Explainability
Grad-CAM heatmaps offer insights into model focus areas.
Robust to Class Imbalance
Applied class weighting and decision threshold adjustments to reduce bias.
Limitations

Validation Fluctuations
Validation accuracy varied across epochs, indicating potential instability.

Threshold Sensitivity
The model performed best with a threshold of 0.3, deviating from the standard 0.5.

Calibration Issues
The calibration curve suggests minor underconfidence in probability estimation.
Recommendations for Improvement

Further Class Imbalance Handling
Consider oversampling the normal class to provide more balanced feature learning.
Enhanced Regularization Techniques

Implement dropout or stronger L2 penalties in fully connected layers.
Use of External Datasets
Expanding the dataset with additional sources may improve generalization.

Refining Explainability Approaches
Explore alternative explainability tools like Grad_CAM++, LIME or SHAP.

Future Directions
Multiclass Pneumonia Classification
Expand the model to detect bacterial vs. viral pneumonia subtypes.
Incorporation of Clinical Metadata
Utilize patient history and symptoms to refine predictions.

Conclusion
This study presents an enhanced pneumonia classification model incorporating class balancing, regularization, and explainability. With an AUC-ROC of 0.93 and AUC-PR of 0.95, the model demonstrates strong discriminatory power. However, further calibration and stability improvements are necessary for real-world medical applications.

