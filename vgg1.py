import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Flatten, concatenate
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import time
import pickle

# Loading tabular dataset
df = pd.read_csv('heart.csv')

# Extracting features and target variable
X = df.drop('HeartDiseaseorAttack', axis=1)
y = df['HeartDiseaseorAttack']

# Convert target variable to categorical
y = to_categorical(y, num_classes=2)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Loading the pre-trained VGG16 model without the top (fully connected) layers
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in vgg_model.layers:
    layer.trainable = False

# Define three input layers for your tabular data
input_shape_1 = (21,)
input_shape_2 = (21,)
input_shape_3 = (21,)
input_1 = Input(shape=input_shape_1)
input_2 = Input(shape=input_shape_2)
input_3 = Input(shape=input_shape_3)

# Define three dense layers for each input
output_1 = Dense(50176, activation='relu')(input_1)
output_2 = Dense(50176, activation='relu')(input_2)
output_3 = Dense(50176, activation='relu')(input_3)

# Concatenate the outputs
merged = concatenate([output_1, output_2, output_3])
merged = tf.reshape(merged, [-1, 224, 224, 3])

# Pass through the VGG16 base
x = vgg_model(merged)
x = Flatten()(x)

# Output layer for binary classification
output_layer = Dense(2, activation='softmax')(x)

# Create a new model with the modified input shape
Vggmodel = Model(inputs=[input_1, input_2, input_3], outputs=output_layer)

# Compile the model with binary crossentropy loss and Adam optimizer
Vggmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
print(Vggmodel.summary())

start_time = time.time()
# Train the model
history = Vggmodel.fit([X_train_scaled] * 3, y_train, epochs=10, batch_size=32, validation_data=([X_test_scaled] * 3, y_test))

# Save the model to a pickle file
with open('vgg_model.pkl', 'wb') as file:
    pickle.dump(Vggmodel, file)

end_time_training = time.time()

# Time taken for training
training_time = end_time_training - start_time
print("Training Time:", training_time, "seconds")

# Start time for prediction
start_time_prediction = time.time()

# Predict on test set
y_pred = Vggmodel.predict([X_test_scaled] * 3)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)


# End time for prediction
end_time_prediction = time.time()

# Time taken for prediction
prediction_time = end_time_prediction - start_time_prediction
print("Prediction Time:", prediction_time, "seconds")

# Calculate accuracy
accuracy = accuracy_score(y_test_classes, y_pred_classes)
print("Accuracy:", accuracy)


# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test_classes, y_pred[:, 1])
roc_auc = auc(fpr, tpr)
print("ROC AUC:", roc_auc)

# Plot ROC curve and save it
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')  # Save ROC curve plot
plt.show()


# Calculate precision-recall curve
precision, recall, _ = precision_recall_curve(y_test_classes, y_pred[:, 1])

# Plot precision-recall curve and save it
plt.figure()
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.savefig('precision_recall_curve.png')  # Save precision-recall curve plot
plt.show()

# Calculate AUPR
aupr = auc(recall, precision)
print("AUPR:", aupr)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
tn, fp, fn, tp = conf_matrix.ravel()

# Calculate sensitivity and specificity
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

# Plot confusion matrix and save it
plt.figure()
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.xticks([0, 1], ['Predicted 0', 'Predicted 1'])
plt.yticks([0, 1], ['True 0', 'True 1'])
plt.tight_layout()

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', color='white' if conf_matrix[i, j] > conf_matrix.max() / 2 else 'black')

plt.savefig('confusion_matrix.png')  # Save confusion matrix plot
plt.show()
