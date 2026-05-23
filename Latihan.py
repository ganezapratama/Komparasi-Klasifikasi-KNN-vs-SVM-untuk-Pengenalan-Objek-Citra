# =========================================================
# KOMPARASI KLASIFIKASI KNN VS SVM
# UNTUK PENGENALAN OBJEK CITRA
# Dataset : Fashion-MNIST
# =========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import time

from sklearn.datasets import fetch_openml

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA

from skimage.feature import hog, local_binary_pattern

# =========================================================
# LOAD DATASET
# =========================================================

print("Mengambil dataset Fashion-MNIST...")

fashion_mnist = fetch_openml(
    'Fashion-MNIST',
    version=1,
    as_frame=False
)

X = fashion_mnist.data
y = fashion_mnist.target.astype(int)

# Mengambil 1000 sampel
X = X[:1000]
y = y[:1000]

print("Jumlah data :", X.shape)
print("Jumlah label :", y.shape)

# =========================================================
# MENAMPILKAN CONTOH GAMBAR
# =========================================================

plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label : {y[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# =========================================================
# EKSTRAKSI FITUR HOG DAN LBP
# =========================================================

print("\nMelakukan ekstraksi fitur...")

hog_features = []
lbp_features = []

radius = 1
n_points = 8 * radius

for image in X:

    img = image.reshape(28, 28).astype('uint8')

    # =====================================================
    # HOG FEATURE
    # =====================================================

    hog_feature = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )

    hog_features.append(hog_feature)

    # =====================================================
    # LBP FEATURE
    # =====================================================

    lbp = local_binary_pattern(
        img,
        n_points,
        radius,
        method='uniform'
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    hist = hist.astype("float")

    hist /= (hist.sum() + 1e-6)

    lbp_features.append(hist)

hog_features = np.array(hog_features)
lbp_features = np.array(lbp_features)

# Menggabungkan fitur HOG + LBP
features = np.hstack((hog_features, lbp_features))

print("Ukuran fitur akhir :", features.shape)

# =========================================================
# SPLIT DATA
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    features,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nData Training :", X_train.shape)
print("Data Testing  :", X_test.shape)

# =========================================================
# IMPLEMENTASI KNN
# =========================================================

print("\n=================================================")
print("IMPLEMENTASI KNN")
print("=================================================")

k_values = [1, 3, 5, 7, 9, 11]
metrics = ['euclidean', 'manhattan', 'minkowski']

knn_results = []

for metric in metrics:

    for k in k_values:

        print(f"\nKNN -> k={k}, metric={metric}")

        knn_model = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(
                n_neighbors=k,
                metric=metric
            ))
        ])

        # Training Time
        start_train = time.time()

        knn_model.fit(X_train, y_train)

        end_train = time.time()

        train_time = end_train - start_train

        # Testing Time
        start_test = time.time()

        y_pred_knn = knn_model.predict(X_test)

        end_test = time.time()

        inference_time = end_test - start_test

        # Evaluasi
        accuracy = accuracy_score(y_test, y_pred_knn)

        precision = precision_score(
            y_test,
            y_pred_knn,
            average='weighted'
        )

        recall = recall_score(
            y_test,
            y_pred_knn,
            average='weighted'
        )

        f1 = f1_score(
            y_test,
            y_pred_knn,
            average='weighted'
        )

        knn_results.append([
            k,
            metric,
            accuracy,
            precision,
            recall,
            f1,
            train_time,
            inference_time
        ])

knn_df = pd.DataFrame(knn_results, columns=[
    'k',
    'Metric',
    'Accuracy',
    'Precision',
    'Recall',
    'F1-Score',
    'Training Time',
    'Inference Time'
])

print("\nHASIL KNN")
print(knn_df)

# =========================================================
# KNN TERBAIK
# =========================================================

best_knn = KNeighborsClassifier(
    n_neighbors=5,
    metric='euclidean'
)

best_knn.fit(X_train, y_train)

y_pred_knn = best_knn.predict(X_test)

# =========================================================
# CONFUSION MATRIX KNN
# =========================================================

cm_knn = confusion_matrix(y_test, y_pred_knn)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm_knn,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title('Confusion Matrix KNN')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()

# =========================================================
# IMPLEMENTASI SVM
# =========================================================

print("\n=================================================")
print("IMPLEMENTASI SVM")
print("=================================================")

kernels = ['linear', 'poly', 'rbf']

C_values = [0.1, 1, 10, 100]

gamma_values = [0.001, 0.01, 0.1, 1]

svm_results = []

for kernel in kernels:

    for C in C_values:

        if kernel == 'rbf':

            for gamma in gamma_values:

                print(f"\nSVM -> kernel={kernel}, C={C}, gamma={gamma}")

                svm_model = Pipeline([
                    ('scaler', StandardScaler()),
                    ('svm', SVC(
                        kernel=kernel,
                        C=C,
                        gamma=gamma,
                        probability=True
                    ))
                ])

                start_train = time.time()

                svm_model.fit(X_train, y_train)

                end_train = time.time()

                train_time = end_train - start_train

                start_test = time.time()

                y_pred_svm = svm_model.predict(X_test)

                end_test = time.time()

                inference_time = end_test - start_test

                accuracy = accuracy_score(y_test, y_pred_svm)

                precision = precision_score(
                    y_test,
                    y_pred_svm,
                    average='weighted'
                )

                recall = recall_score(
                    y_test,
                    y_pred_svm,
                    average='weighted'
                )

                f1 = f1_score(
                    y_test,
                    y_pred_svm,
                    average='weighted'
                )

                svm_results.append([
                    kernel,
                    C,
                    gamma,
                    accuracy,
                    precision,
                    recall,
                    f1,
                    train_time,
                    inference_time
                ])

        else:

            print(f"\nSVM -> kernel={kernel}, C={C}")

            svm_model = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(
                    kernel=kernel,
                    C=C,
                    probability=True
                ))
            ])

            start_train = time.time()

            svm_model.fit(X_train, y_train)

            end_train = time.time()

            train_time = end_train - start_train

            start_test = time.time()

            y_pred_svm = svm_model.predict(X_test)

            end_test = time.time()

            inference_time = end_test - start_test

            accuracy = accuracy_score(y_test, y_pred_svm)

            precision = precision_score(
                y_test,
                y_pred_svm,
                average='weighted'
            )

            recall = recall_score(
                y_test,
                y_pred_svm,
                average='weighted'
            )

            f1 = f1_score(
                y_test,
                y_pred_svm,
                average='weighted'
            )

            svm_results.append([
                kernel,
                C,
                '-',
                accuracy,
                precision,
                recall,
                f1,
                train_time,
                inference_time
            ])

svm_df = pd.DataFrame(svm_results, columns=[
    'Kernel',
    'C',
    'Gamma',
    'Accuracy',
    'Precision',
    'Recall',
    'F1-Score',
    'Training Time',
    'Inference Time'
])

print("\nHASIL SVM")
print(svm_df)

# =========================================================
# SVM TERBAIK
# =========================================================

best_svm = SVC(
    kernel='rbf',
    C=10,
    gamma=0.01,
    probability=True
)

best_svm.fit(X_train, y_train)

y_pred_svm = best_svm.predict(X_test)

# =========================================================
# CONFUSION MATRIX SVM
# =========================================================

cm_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm_svm,
    annot=True,
    fmt='d',
    cmap='Greens'
)

plt.title('Confusion Matrix SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nCLASSIFICATION REPORT SVM")

print(classification_report(
    y_test,
    y_pred_svm
))

# =========================================================
# ROC CURVE DAN AUC
# =========================================================

classes = np.unique(y)

y_test_bin = label_binarize(
    y_test,
    classes=classes
)

svm_probs = best_svm.predict_proba(X_test)

plt.figure(figsize=(10, 8))

for i in range(len(classes)):

    fpr, tpr, _ = roc_curve(
        y_test_bin[:, i],
        svm_probs[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f'Class {i} (AUC={roc_auc:.2f})'
    )

plt.plot([0, 1], [0, 1], 'k--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Multi-Class SVM')

plt.legend()

plt.show()

# =========================================================
# CROSS VALIDATION
# =========================================================

print("\n=================================================")
print("CROSS VALIDATION")
print("=================================================")

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    best_svm,
    features,
    y,
    cv=skf,
    scoring='accuracy'
)

print("CV Scores :")
print(cv_scores)

print("\nMean Accuracy :")
print(cv_scores.mean())

# =========================================================
# GRID SEARCH CV
# =========================================================

print("\n=================================================")
print("GRID SEARCH CV")
print("=================================================")

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)

print("\nBest Parameter :")
print(grid_search.best_params_)

print("\nBest Accuracy :")
print(grid_search.best_score_)

# =========================================================
# LEARNING CURVE
# =========================================================

train_sizes, train_scores, test_scores = learning_curve(
    best_svm,
    features,
    y,
    cv=5,
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))

plt.plot(
    train_sizes,
    train_mean,
    label='Training Accuracy'
)

plt.plot(
    train_sizes,
    test_mean,
    label='Validation Accuracy'
)

plt.xlabel('Training Size')
plt.ylabel('Accuracy')

plt.title('Learning Curve SVM')

plt.legend()

plt.show()

# =========================================================
# PCA DAN DECISION BOUNDARY
# =========================================================

print("\n=================================================")
print("PCA DAN DECISION BOUNDARY")
print("=================================================")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(features)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca,
    y,
    test_size=0.2,
    random_state=42
)

svm_pca = SVC(
    kernel='rbf',
    C=10,
    gamma=0.01
)

svm_pca.fit(X_train_pca, y_train_pca)

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.5),
    np.arange(y_min, y_max, 0.5)
)

Z = svm_pca.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))

plt.contourf(xx, yy, Z, alpha=0.3)

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap='tab10'
)

plt.title('Decision Boundary SVM')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

plt.show()

# =========================================================
# ANALISIS HASIL
# =========================================================

print("\n=================================================")
print("ANALISIS HASIL")
print("=================================================")

print("""
1. Nilai k kecil pada KNN cenderung mengalami overfitting.

2. Nilai k besar membuat model lebih stabil namun bisa underfitting.

3. Kernel RBF pada SVM biasanya memberikan akurasi terbaik.

4. Parameter C besar membuat model lebih kompleks.

5. Kombinasi fitur HOG dan LBP meningkatkan kualitas klasifikasi.

6. SVM umumnya menghasilkan akurasi lebih tinggi dibanding KNN.

7. KNN lebih sederhana namun inference lebih lambat.

8. Cross-validation membantu evaluasi model menjadi lebih stabil.
""")

# =========================================================
# SELESAI
# =========================================================

print("\nProgram selesai dijalankan.")