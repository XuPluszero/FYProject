import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
random_state = 42
np.random.seed(random_state)

# Radiomics features were extracted using PyRadiomics and saved as CSV files.
# Define file paths for features (edema and tumor for each modality) and the labels file
feature_files = {
    'adc': {'edema': 'features_edema_all_adc.csv', 'tumor': 'features_tumor_all_adc.csv'},
    't1c': {'edema': 'features_edema_all_t1c.csv', 'tumor': 'features_tumor_all_t1c.csv'},
    't2':  {'edema': 'features_edema_all_t2.csv',  'tumor': 'features_tumor_all_t2.csv'}
}
label_file = 'info_tumor.csv'

# Load the labels (IDH mutation status)
labels_df = pd.read_csv(label_file)
# Ensure ID column is of type int for merging
labels_df['ID'] = labels_df['ID'].astype(int)
# Keep only relevant columns from label file
labels_df = labels_df[['ID', 'IDHmutation']]

# Dictionaries to store processed feature data and labels for each modality
X_data = {}
y_data = {}

# 1 & 2. Merge features for each modality and align with labels
for mod, paths in feature_files.items():
    # Load radiomic features for edema and tumor regions of this modality
    df_edema = pd.read_csv(paths['edema'])
    df_tumor = pd.read_csv(paths['tumor'])
    # Drop irrelevant columns (columns B to N: "Mask" and "general_info_*")
    drop_cols_edema = [c for c in df_edema.columns if c not in ['Image'] and (c == 'Mask' or c.startswith('general_info'))]
    drop_cols_tumor = [c for c in df_tumor.columns if c not in ['Image'] and (c == 'Mask' or c.startswith('general_info'))]
    df_edema = df_edema.drop(columns=drop_cols_edema)
    df_tumor = df_tumor.drop(columns=drop_cols_tumor)
    # Merge edema and tumor features on the 'Image' (patient ID) column
    merged = pd.merge(df_edema, df_tumor, on='Image', suffixes=('_edema', '_tumor'))
    # Create numeric patient ID from 'Image' (assuming format "patient_XXX")
    merged['ID'] = merged['Image'].apply(lambda s: int(s.split('_')[1]))
    # Merge with labels on ID to get the IDH mutation status for each patient
    merged = pd.merge(merged, labels_df, on='ID')
    # Sort by patient ID to ensure consistent order and reset index
    merged = merged.sort_values('ID').reset_index(drop=True)
    # Separate features (X) and target label (y)
    y = merged['IDHmutation'].values  # binary labels (0 = wild-type, 1 = mutant)
    X = merged.drop(columns=['Image', 'ID', 'IDHmutation'])
    # 3. Preprocess: handle missing values by filling with the median of each feature
    X = X.fillna(X.median())
    # Standardize features to zero mean and unit variance
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    # Store the preprocessed features and labels for this modality
    X_data[mod] = X_scaled
    y_data[mod] = y

# 4. Feature selection using Lasso (L1-regularized linear model) with cross-validation to choose the best alpha
selected_features = {}  # to store selected feature names for each modality
for mod, X in X_data.items():
    y = y_data[mod]
    # Initialize Lasso with cross-validation (5-fold) to select regularization strength
    lasso = LassoCV(cv=5, random_state=random_state)
    lasso.fit(X, y)
    # Determine which features have non-zero coefficients (selected features)
    coef_mask = lasso.coef_ != 0
    selected_feat_names = X.columns[coef_mask]
    # If no features selected (all coefficients zero), fall back to selecting all features
    if len(selected_feat_names) == 0:
        selected_feat_names = X.columns
    selected_features[mod] = selected_feat_names
    # Reduce feature set to selected features
    X_data[mod] = X_data[mod][selected_feat_names]
    print(f"{mod.upper()}: Selected {len(selected_feat_names)} features out of {X.shape[1]} after Lasso feature selection.")

# 5. Define multiple classifiers for evaluation
classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=1000, solver='liblinear', random_state=random_state),
    "RandomForest"     : RandomForestClassifier(n_estimators=100, random_state=random_state),
    "SVM"              : SVC(kernel='rbf', probability=True, random_state=random_state),
    "XGBoost"          : XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state),
    "LightGBM"         : LGBMClassifier(random_state=random_state, verbosity=-1)
}

# Stratified 5-fold cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

# 6. Evaluate classifiers for each modality and plot ROC curves
best_models = {}      # store best classifier (name) for each modality
best_probs = {}       # store out-of-fold probability predictions of best model for each modality
for mod in ['adc', 't1c', 't2']:
    X = X_data[mod]
    y = y_data[mod]
    print(f"\nEvaluating classifiers for modality: {mod.upper()}")
    plt.figure()  # new figure for this modality's ROC curves
    best_auc = 0.0
    best_clf_name = None
    best_prob = None
    # Evaluate each classifier using cross_val_predict (to get probability predictions for ROC/AUC)
    for name, clf in classifiers.items():
        # Obtain out-of-fold predicted probabilities for the positive class (IDH mutant)
        y_prob = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
        # Calculate AUC and accuracy for these predictions
        auc_val = roc_auc_score(y, y_prob)
        acc_val = accuracy_score(y, (y_prob >= 0.5).astype(int))
        print(f"  {name}: AUC = {auc_val:.3f}, Accuracy = {acc_val:.3f}")
        # Plot the ROC curve for this classifier
        fpr, tpr, _ = roc_curve(y, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.2f})")
        # Track the best classifier based on AUC
        if auc_val > best_auc:
            best_auc = auc_val
            best_clf_name = name
            best_prob = y_prob
    # Finalize ROC plot for this modality
    plt.title(f"{mod.upper()} Modality - ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()
    # Store the best model's name and its probability predictions for this modality
    print(f"  -> Best classifier for {mod.upper()}: {best_clf_name} (AUC = {best_auc:.3f})")
    best_models[mod] = best_clf_name
    best_probs[mod] = best_prob

# 7. Ensemble predictions from all three modalities
# Ensure that the label arrays are identical for all modalities (they should be, after sorting by ID)
assert np.array_equal(y_data['adc'], y_data['t1c']) and np.array_equal(y_data['adc'], y_data['t2'])
y = y_data['adc']  # this is the ground truth labels for all patients

# (a) Averaging probabilities from each modality's best model
avg_prob = (best_probs['adc'] + best_probs['t1c'] + best_probs['t2']) / 3.0
avg_pred = (avg_prob >= 0.5).astype(int)

# (b) Stacking ensemble: use a Logistic Regression as meta-classifier on top of modality predictions
stack_input = np.column_stack((best_probs['adc'], best_probs['t1c'], best_probs['t2']))
meta_clf = LogisticRegression(random_state=random_state)
meta_clf.fit(stack_input, y)  # train meta-classifier on out-of-fold predictions of base models
stack_prob = meta_clf.predict_proba(stack_input)[:, 1]
stack_pred = (stack_prob >= 0.5).astype(int)

# (c) Voting ensemble: majority vote on binary predictions from each modality
pred_adc = (best_probs['adc'] >= 0.5).astype(int)
pred_t1c = (best_probs['t1c'] >= 0.5).astype(int)
pred_t2 = (best_probs['t2'] >= 0.5).astype(int)
vote_sum = pred_adc + pred_t1c + pred_t2
vote_pred = (vote_sum >= 2).astype(int)  # 2 or more votes for mutant => predict mutant
# For ROC analysis of voting, use the average vote as a score (0, 1/3, 2/3, or 1)
vote_score = vote_sum / 3.0

# Calculate performance metrics for ensemble methods
auc_avg = roc_auc_score(y, avg_prob)
acc_avg = accuracy_score(y, avg_pred)
auc_stack = roc_auc_score(y, stack_prob)
acc_stack = accuracy_score(y, stack_pred)
auc_vote = roc_auc_score(y, vote_score)
acc_vote = accuracy_score(y, vote_pred)

print("\nEnsemble Strategies Performance:")
print(f"  Averaging: AUC = {auc_avg:.3f}, Accuracy = {acc_avg:.3f}")
print(f"  Stacking (meta-classifier): AUC = {auc_stack:.3f}, Accuracy = {acc_stack:.3f}")
print(f"  Voting (majority rule): AUC = {auc_vote:.3f}, Accuracy = {acc_vote:.3f}")

# 8. Compare ensemble strategies vs single-modality models by plotting all ROC curves together
plt.figure()
# Plot ROC for each modality's best model
for mod in ['adc', 't1c', 't2']:
    fpr, tpr, _ = roc_curve(y, best_probs[mod])
    auc_val = roc_auc_score(y, best_probs[mod])
    plt.plot(fpr, tpr, label=f"{mod.upper()} best (AUC={auc_val:.2f})")
# Plot ROC for each ensemble method
fpr_avg, tpr_avg, _ = roc_curve(y, avg_prob)
fpr_stack, tpr_stack, _ = roc_curve(y, stack_prob)
fpr_vote, tpr_vote, _ = roc_curve(y, vote_score)
plt.plot(fpr_avg, tpr_avg, label=f"Ensemble Avg (AUC={auc_avg:.2f})", linestyle='--', linewidth=2)
plt.plot(fpr_stack, tpr_stack, label=f"Ensemble Stacking (AUC={auc_stack:.2f})", linestyle='--', linewidth=2)
plt.plot(fpr_vote, tpr_vote, label=f"Ensemble Voting (AUC={auc_vote:.2f})", linestyle='--', linewidth=2)
plt.title("ROC Curves - Single Modalities vs Ensemble Strategies")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()
