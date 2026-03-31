# ============================================================
# SCIKIT-LEARN MAIN CONCEPTS
# ============================================================
# Scikit-learn is Python's primary machine learning library.
# It provides a consistent API for: preprocessing, model training,
# evaluation, and pipeline construction.

import numpy as np
import pandas as pd
from sklearn import datasets

# ============================================================
# 1. DATASETS
# ============================================================
# Scikit-learn ships with toy datasets for quick experimentation.

from sklearn.datasets import (
    load_iris, load_digits, load_boston,
    make_classification, make_regression, make_blobs,
)

# Built-in datasets
iris = load_iris()
X, y = iris.data, iris.target     # X: features, y: labels
print("--- Dataset ---")
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("X shape:", X.shape)         # (150, 4)

# Synthetic datasets (useful for demos)
X_cls, y_cls = make_classification(n_samples=1000, n_features=10,
                                    n_informative=5, random_state=42)
X_reg, y_reg = make_regression(n_samples=500, n_features=8,
                                noise=10, random_state=42)
X_blob, y_blob = make_blobs(n_samples=300, centers=4, random_state=42)


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42,
    stratify=y,         # preserve class distribution
)
print("\n--- Train/Test Split ---")
print("Train:", X_train.shape, "Test:", X_test.shape)


# ============================================================
# 3. PREPROCESSING
# ============================================================
# Always fit on training data only, then transform train and test.

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    PolynomialFeatures, Binarizer,
)

print("\n--- Preprocessing ---")

# --- Scaling ---
scaler = StandardScaler()           # mean=0, std=1
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform
X_test_scaled  = scaler.transform(X_test)        # transform only (no refit!)

# MinMaxScaler: rescale to [0, 1]
mm = MinMaxScaler()
X_mm = mm.fit_transform(X_train)

# RobustScaler: uses median and IQR — less sensitive to outliers
rb = RobustScaler()
X_rb = rb.fit_transform(X_train)

# --- Encoding categorical variables ---
le = LabelEncoder()
y_encoded = le.fit_transform(["cat", "dog", "cat", "bird"])
print("LabelEncoder:", y_encoded)           # [1 2 1 0]

ohe = OneHotEncoder(sparse_output=False)
ohe.fit_transform([["cat"], ["dog"], ["bird"]])  # 2D input required

# --- Polynomial features ---
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_train[:5, :2])     # expands features


# ============================================================
# 4. THE ESTIMATOR API
# ============================================================
# Every model in scikit-learn follows the same interface:
#   estimator.fit(X_train, y_train)    -- learn from data
#   estimator.predict(X_test)          -- make predictions
#   estimator.score(X_test, y_test)    -- evaluate (default metric)
#   estimator.get_params()             -- inspect hyperparameters
#   estimator.set_params(**params)     -- set hyperparameters


# ============================================================
# 5. CLASSIFICATION
# ============================================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

print("\n--- Classification ---")

# Logistic Regression
lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X_train_scaled, y_train)
print("LogisticRegression accuracy:", lr.score(X_test_scaled, y_test))

# Decision Tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
print("DecisionTree accuracy:", dt.score(X_test, y_test))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("RandomForest accuracy:", rf.score(X_test, y_test))

# Support Vector Machine
svm = SVC(kernel="rbf", C=1.0, random_state=42)
svm.fit(X_train_scaled, y_train)
print("SVM accuracy:", svm.score(X_test_scaled, y_test))

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
print("KNN accuracy:", knn.score(X_test_scaled, y_test))


# ============================================================
# 6. REGRESSION
# ============================================================

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

print("\n--- Regression ---")

lin = LinearRegression()
lin.fit(X_r_train, y_r_train)
print("LinearRegression R²:", lin.score(X_r_test, y_r_test))
print("Coefficients:", lin.coef_)
print("Intercept:", lin.intercept_)

ridge = Ridge(alpha=1.0)           # L2 regularization
ridge.fit(X_r_train, y_r_train)
print("Ridge R²:", ridge.score(X_r_test, y_r_test))

lasso = Lasso(alpha=0.1)           # L1 regularization (sparse coefficients)
lasso.fit(X_r_train, y_r_train)
print("Lasso R²:", lasso.score(X_r_test, y_r_test))


# ============================================================
# 7. CLUSTERING (UNSUPERVISED)
# ============================================================

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

print("\n--- Clustering ---")

# K-Means
km = KMeans(n_clusters=4, random_state=42, n_init="auto")
km.fit(X_blob)
print("KMeans labels:", np.unique(km.labels_))
print("Inertia:", km.inertia_)          # sum of squared distances to centers
print("Centers shape:", km.cluster_centers_.shape)

# DBSCAN — density-based, finds clusters of arbitrary shape
db = DBSCAN(eps=0.5, min_samples=5)
db.fit(X_blob)
print("DBSCAN labels:", np.unique(db.labels_))  # -1 = noise points

# Gaussian Mixture Model
gm = GaussianMixture(n_components=4, random_state=42)
gm.fit(X_blob)
print("GMM predictions:", gm.predict(X_blob[:5]))


# ============================================================
# 8. DIMENSIONALITY REDUCTION
# ============================================================

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

print("\n--- Dimensionality Reduction ---")

# PCA — linear projection to maximize variance
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("PCA shape:", X_pca.shape)                       # (150, 2)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", pca.explained_variance_ratio_.sum())

# Choose n_components by explained variance threshold
pca_95 = PCA(n_components=0.95)   # keep enough components for 95% variance
pca_95.fit(X)
print("Components for 95% variance:", pca_95.n_components_)

# t-SNE — non-linear, good for visualization (not for ML pipelines)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)
print("t-SNE shape:", X_tsne.shape)


# ============================================================
# 9. MODEL EVALUATION — CLASSIFICATION METRICS
# ============================================================

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve,
)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)   # class probabilities

print("\n--- Classification Metrics ---")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="macro"))
print("Recall   :", recall_score(y_test, y_pred, average="macro"))
print("F1 Score :", f1_score(y_test, y_pred, average="macro"))
print("\nClassification Report:\n", classification_report(y_test, y_pred,
      target_names=iris.target_names))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ROC-AUC (for binary or multi-class with one-vs-rest)
auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
print("ROC-AUC:", auc)


# ============================================================
# 10. MODEL EVALUATION — REGRESSION METRICS
# ============================================================

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    root_mean_squared_error, r2_score,
)

y_r_pred = lin.predict(X_r_test)

print("\n--- Regression Metrics ---")
print("MAE :", mean_absolute_error(y_r_test, y_r_pred))
print("MSE :", mean_squared_error(y_r_test, y_r_pred))
print("RMSE:", root_mean_squared_error(y_r_test, y_r_pred))
print("R²  :", r2_score(y_r_test, y_r_pred))


# ============================================================
# 11. CROSS-VALIDATION
# ============================================================

from sklearn.model_selection import (
    cross_val_score, cross_validate,
    KFold, StratifiedKFold, LeaveOneOut,
)

print("\n--- Cross-Validation ---")

# Basic k-fold CV
scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy")
print("CV scores:", scores)
print("Mean:", scores.mean(), "Std:", scores.std())

# Stratified k-fold (preserves class balance in each fold)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_skf = cross_val_score(rf, X, y, cv=skf, scoring="f1_macro")
print("Stratified CV F1:", scores_skf.mean())

# cross_validate returns train scores, fit times, etc.
results = cross_validate(rf, X, y, cv=5,
                         scoring=["accuracy", "f1_macro"],
                         return_train_score=True)
print("Test accuracy:", results["test_accuracy"].mean())
print("Train accuracy:", results["train_accuracy"].mean())


# ============================================================
# 12. HYPERPARAMETER TUNING
# ============================================================

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

print("\n--- Hyperparameter Tuning ---")

# Grid Search — exhaustive search over a parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth":    [None, 3, 5],
    "min_samples_split": [2, 5],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42),
                    param_grid, cv=3, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
print("Best CV score:", grid.best_score_)
print("Test score:", grid.score(X_test, y_test))

# Randomized Search — samples a fixed number of parameter combinations
from scipy.stats import randint
param_dist = {
    "n_estimators": randint(50, 300),
    "max_depth":    [None, 3, 5, 10],
}
rand = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                          param_dist, n_iter=10, cv=3,
                          scoring="accuracy", random_state=42)
rand.fit(X_train, y_train)
print("RandomSearch best params:", rand.best_params_)


# ============================================================
# 13. PIPELINES
# ============================================================
# Chains preprocessing + model into one object.
# Prevents data leakage: fit() on training data only.

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer

print("\n--- Pipeline ---")

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),   # fill missing values
    ("scaler",  StandardScaler()),                  # normalize
    ("clf",     LogisticRegression(max_iter=200)),  # model
])

pipe.fit(X_train, y_train)
print("Pipeline accuracy:", pipe.score(X_test, y_test))

# Shorthand (auto-names steps)
pipe2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
pipe2.fit(X_train, y_train)

# Pipelines work with GridSearchCV — use __ to target nested params
param_grid_pipe = {
    "logisticregression__C": [0.1, 1.0, 10.0],
}
grid_pipe = GridSearchCV(pipe2, param_grid_pipe, cv=5)
grid_pipe.fit(X_train, y_train)
print("Pipeline GridSearch best C:", grid_pipe.best_params_)


# ============================================================
# 14. COLUMN TRANSFORMER
# ============================================================
# Apply different preprocessing to different columns.

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline

df_mixed = pd.DataFrame({
    "age":    [25, 30, None, 40],
    "salary": [50000, 60000, 70000, None],
    "dept":   ["HR", "Eng", "Eng", "HR"],
    "hired":  [1, 0, 1, 0],
})
X_mixed = df_mixed.drop("hired", axis=1)
y_mixed = df_mixed["hired"]

num_features = ["age", "salary"]
cat_features = ["dept"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler",  StandardScaler()),
])
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features),
])

full_pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression()),
])
print("\n--- ColumnTransformer ---")
print(full_pipe.fit(X_mixed, y_mixed))


# ============================================================
# 15. FEATURE SELECTION
# ============================================================

from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2,
    RFE, SelectFromModel, VarianceThreshold,
)

print("\n--- Feature Selection ---")

# Filter method: select K best features by statistical test
selector = SelectKBest(score_func=f_classif, k=2)
X_best = selector.fit_transform(X, y)
print("SelectKBest shape:", X_best.shape)
print("Selected features:", selector.get_support())

# Wrapper method: Recursive Feature Elimination
rfe = RFE(estimator=LogisticRegression(max_iter=200), n_features_to_select=2)
rfe.fit(X_train_scaled, y_train)
print("RFE selected:", rfe.support_)
print("RFE ranking:", rfe.ranking_)

# Embedded method: SelectFromModel (uses feature_importances_ or coef_)
sfm = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
sfm.fit(X_train, y_train)
X_sfm = sfm.transform(X_test)
print("SelectFromModel shape:", X_sfm.shape)

# Remove low-variance features
vt = VarianceThreshold(threshold=0.1)
X_vt = vt.fit_transform(X)
print("After VarianceThreshold:", X_vt.shape)


# ============================================================
# 16. SAVING & LOADING MODELS
# ============================================================
import joblib

print("\n--- Saving & Loading ---")

joblib.dump(rf, "random_forest.joblib")     # save
loaded_rf = joblib.load("random_forest.joblib")  # load
print("Loaded model accuracy:", loaded_rf.score(X_test, y_test))

# Clean up demo file
import os
os.remove("random_forest.joblib")


# ============================================================
# QUICK REFERENCE CHEAT SHEET
# ============================================================
#
# DATA
#   load_iris() / make_classification()      Built-in & synthetic datasets
#   train_test_split(X, y, test_size=0.2)    Split data
#
# PREPROCESSING
#   StandardScaler / MinMaxScaler            Feature scaling
#   LabelEncoder / OneHotEncoder             Encode categories
#   SimpleImputer(strategy="mean")           Fill missing values
#   PolynomialFeatures(degree=2)             Generate interactions
#   fit_transform(X_train)                   Fit on train only
#   transform(X_test)                        Apply to test (no refit)
#
# MODELS — CLASSIFICATION
#   LogisticRegression()                     Linear, probabilistic
#   DecisionTreeClassifier()                 Tree-based, interpretable
#   RandomForestClassifier()                 Ensemble of trees
#   GradientBoostingClassifier()             Boosted ensemble
#   SVC(kernel="rbf")                        Support Vector Machine
#   KNeighborsClassifier(n_neighbors=5)      Instance-based
#
# MODELS — REGRESSION
#   LinearRegression()                       Ordinary least squares
#   Ridge(alpha=) / Lasso(alpha=)            L2 / L1 regularization
#   RandomForestRegressor()                  Ensemble regression
#
# CLUSTERING
#   KMeans(n_clusters=k)                     Centroid-based
#   DBSCAN(eps=, min_samples=)               Density-based
#   GaussianMixture(n_components=k)          Probabilistic
#
# DIMENSIONALITY REDUCTION
#   PCA(n_components=)                       Linear projection
#   TSNE(n_components=2)                     Non-linear visualization
#
# EVALUATION — CLASSIFICATION
#   accuracy_score / f1_score / roc_auc_score
#   classification_report / confusion_matrix
#
# EVALUATION — REGRESSION
#   mean_absolute_error / mean_squared_error
#   root_mean_squared_error / r2_score
#
# CROSS-VALIDATION
#   cross_val_score(model, X, y, cv=5)       Quick CV
#   StratifiedKFold(n_splits=5)              Balanced folds
#   cross_validate(..., return_train_score)  Full results
#
# TUNING
#   GridSearchCV(model, param_grid, cv=)     Exhaustive search
#   RandomizedSearchCV(model, param_dist)    Sampled search
#
# PIPELINES
#   Pipeline([("step", transformer), ...])   Chain steps
#   make_pipeline(scaler, model)             Shorthand
#   ColumnTransformer([...])                 Per-column transforms
#
# FEATURE SELECTION
#   SelectKBest(f_classif, k=)               Filter by stats test
#   RFE(estimator, n_features_to_select=)    Recursive elimination
#   SelectFromModel(estimator)               Embedded importance
#
# PERSISTENCE
#   joblib.dump(model, "file.joblib")        Save
#   joblib.load("file.joblib")               Load
