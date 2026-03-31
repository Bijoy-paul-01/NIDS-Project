import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack', 'difficulty'
]

dataset = pd.read_csv("Project/KDDTrain+.txt", names=columns)


for col in reversed(dataset.columns):
    if dataset[col].dtype == 'object':
        attack_col = col
        break
dataset.rename(columns={attack_col: 'attack'}, inplace=True)
print("Detected attack column:", 'attack')
print("Dataset shape:", dataset.shape)
print("Columns:", dataset.columns)
print("First 5 rows:\n", dataset.head())
print("\nAttack type distribution:\n", dataset['attack'].value_counts())


plt.figure(figsize=(10,6))
sns.countplot(y='attack', data=dataset, order=dataset['attack'].value_counts().index)
plt.title("Attack Distribution")
plt.show()


categorical_features = dataset.select_dtypes(include=['object']).columns
categorical_features = categorical_features.drop('attack')  
print("Categorical features:", categorical_features)


le_dict = {}  
for col in categorical_features:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])
    le_dict[col] = le  


label_encoder = LabelEncoder()
dataset['attack'] = label_encoder.fit_transform(dataset['attack'])
dataset.fillna(0, inplace=True)


numerical_features = dataset.select_dtypes(include=['int64','float64']).columns
numerical_features = numerical_features.drop('attack')  
scaler = StandardScaler()
dataset[numerical_features] = scaler.fit_transform(dataset[numerical_features])


X = dataset.drop('attack', axis=1)
y = dataset['attack']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if X_train.shape[0] > 0 and X_test.shape[0] > 0:
    print("✅ Model is ready to train! Training can start now.")


plt.figure(figsize=(12,10))
sns.heatmap(dataset.corr(), cmap='coolwarm', center=0)
plt.title("Feature Correlation")
plt.show()


dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
print("✅ Decision Tree model has been trained.")
y_pred_dt = dt_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))


rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
print("✅ Random Forest model has been trained.")
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))


svm_model = SVC(random_state=42, probability=True)
svm_model.fit(X_train, y_train)
print("✅ SVM model has been trained.")
y_pred_svm = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))


models = {'Decision Tree': y_pred_dt, 'Random Forest': y_pred_rf, 'SVM': y_pred_svm}
for name, pred in models.items():
    print(f"\n{name} Classification Report:\n", classification_report(y_test, pred))
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{name} Confusion Matrix")
    plt.show()


param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2,5,10]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
print("Tuned Random Forest Accuracy:", accuracy_score(y_test, y_pred_best_rf))


nn_model = Sequential()
nn_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
nn_model.add(Dropout(0.3))
nn_model.add(Dense(32, activation='relu'))
nn_model.add(Dense(len(np.unique(y)), activation='softmax'))

nn_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = nn_model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.2)
print("✅ Neural Network model has been trained.")


nn_eval = nn_model.evaluate(X_test, y_test)
print("Neural Network Accuracy:", nn_eval[1])
print("\n✅ All ML and DL models have been trained successfully!")


plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('NN Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('NN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


def detect_attack(sample):
    sample_df = pd.DataFrame([sample], columns=X.columns)

    
    for col in categorical_features:
        if col in sample_df.columns:
            le_col = le_dict[col]  
            val = sample_df[col][0]
            if val in le_col.classes_:
                sample_df[col] = le_col.transform([val])
            else:
                sample_df[col] = 0  

    
    sample_df[numerical_features] = scaler.transform(sample_df[numerical_features])

    
    pred = best_rf.predict(sample_df)[0]
    confidence = max(best_rf.predict_proba(sample_df)[0])
    traffic_status = "Malicious" if pred != 0 else "Normal"
    attack_name = label_encoder.inverse_transform([pred])[0]

    return {
    "Traffic Status": traffic_status,
    "Attack Type": attack_name,
    "Confidence": confidence
}


sample_input = X_test.iloc[0].to_dict()
detection = detect_attack(sample_input)
print("\nAttack Detection Output:\n", detection)

import joblib

joblib.dump(best_rf, "best_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_dict, "label_encoders.pkl")
joblib.dump(label_encoder, "attack_encoder.pkl")
print("\n✅ All models, scalers, and encoders have been saved successfully! You are ready to deploy the NIDS.")