import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

print("--- Starting Full Expanded Visualization & Machine Learning Pipeline ---")

# Step 1: Read raw data to capture structural missing value tracking maps
df_raw = pd.read_csv('Credit_Card.csv', sep=';')

# CORRECTED: Assigned to Figure 6 to prevent overlapping filenames
print("Generating Figure 6: Missing Values Heatmap...")
plt.figure(figsize=(10, 5))
sns.heatmap(df_raw.isnull(), cbar=False, cmap='viridis')
plt.title('Figure 6: Missing Values Heatmap')
plt.tight_layout()
plt.savefig('Figure_6_Missing_Values_Heatmap.png', dpi=300)
plt.close()

# Step 2: Read clean data cache for core model processing
df = pd.read_csv('Cleaned_Credit_Card.csv')

print("Generating Figure 1: Distribution of LIMIT_BAL...")
plt.figure(figsize=(8, 4))
sns.histplot(df['LIMIT_BAL'], kde=True, bins=30, color='skyblue')
plt.title('Figure 1: Distribution of LIMIT_BAL')
plt.tight_layout()
plt.savefig('Figure_1_Distribution_of_LIMIT_BAL.png', dpi=300)
plt.close()

print("Generating Figure 2: Distribution of AGE...")
plt.figure(figsize=(8, 4))
sns.histplot(df['AGE'], kde=True, bins=20, color='salmon')
plt.title('Figure 2: Distribution of AGE')
plt.tight_layout()
plt.savefig('Figure_2_Distribution_of_AGE.png', dpi=300)
plt.close()

print("Generating Figure 3: Boxplot of LIMIT_BAL...")
plt.figure(figsize=(8, 3))
sns.boxplot(x=df['LIMIT_BAL'], color='lightgreen')
plt.title('Figure 3: Boxplot of LIMIT_BAL')
plt.tight_layout()
plt.savefig('Figure_3_Boxplot_of_LIMIT_BAL.png', dpi=300)
plt.close()

print("Generating Figure 4: Distribution of SEX...")
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='SEX', palette='pastel')
plt.title('Figure 4: Distribution of SEX (1=Male, 2=Female)')
plt.tight_layout()
plt.savefig('Figure_4_Distribution_of_SEX.png', dpi=300)
plt.close()

print("Generating Figure 5: Distribution of EDUCATION...")
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x='EDUCATION', palette='muted')
plt.title('Figure 5: Distribution of EDUCATION')
plt.tight_layout()
plt.savefig('Figure_5_Distribution_of_EDUCATION.png', dpi=300)
plt.close()

# Prepare feature boundaries and split datasets
X = df.drop(columns=['ID', 'default.payment.next.month', 'risk_leak'])
y = df['default.payment.next.month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True, class_weight='balanced', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

matrix_scores = []
plt.figure(figsize=(10, 8))

print("\nTraining Classifiers and generating ROC Curves...")
for name, algorithm in models.items():
    algorithm.fit(X_train_scaled, y_train)
    predictions = algorithm.predict(X_test_scaled)
    
    matrix_scores.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions),
        'Recall': recall_score(y_test, predictions),
        'F1 Score': f1_score(y_test, predictions)
    })
    
    proba = algorithm.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

# CORRECTED: Kept as Figure 7 to match sequential evaluation flows
print("Generating Figure 7: ROC Curve Plot Comparison...")
plt.plot([0, 1], [0, 1], 'k--', label='Base Rate (Chance)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Figure 7: ROC Curve Plot Comparison')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('Figure_7_ROC_Curve_Comparison.png', dpi=300)
plt.close()

results_df = pd.DataFrame(matrix_scores)
print("\n=== PERFORMANCE GRID ===")
print(results_df.to_string(index=False))

print("\nGenerating Figure 8: Model Performance Comparison Bar Chart...")
df_melted = pd.melt(results_df, id_vars='Model', var_name='Metric', value_name='Score')
plt.figure(figsize=(12, 6))
sns.barplot(data=df_melted, x='Model', y='Score', hue='Metric', palette='Set2')
plt.title('Figure 8: Model Performance Comparison Dashboard')
plt.ylabel('Score Value')
plt.xticks(rotation=15)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('Figure_8_Performance_Comparison.png', dpi=300)
plt.close()

print("\nRunning Grid Search Optimization for Random Forest...")
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search = GridSearchCV(estimator=RandomForestClassifier(class_weight='balanced', random_state=42), 
                           param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_scaled_df, y_train)
best_rf = grid_search.best_estimator_

print("\nCalculating SHAP interpretability values...")
shap_sample = X_test_scaled_df.sample(200, random_state=42)
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(shap_sample)

if isinstance(shap_values, list):
    shap_vals_matrix = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_vals_matrix = shap_values[:, :, 1]
else:
    shap_vals_matrix = shap_values

print("Generating Figure 9: SHAP Beeswarm Plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals_matrix, shap_sample, show=False)
plt.title('Figure 9: SHAP Feature Importance Summary (Beeswarm)', fontsize=12, pad=20)
plt.tight_layout()
plt.savefig('Figure_9_SHAP_Summary_Beeswarm.png', dpi=300)
plt.close()

print("Generating Figure 10: SHAP Bar Plot...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_vals_matrix, shap_sample, plot_type="bar", show=False)
plt.title('Figure 10: SHAP Feature Importance (Average Magnitude Bar)', fontsize=12, pad=20)
plt.tight_layout()
plt.savefig('Figure_10_SHAP_Summary_Bar.png', dpi=300)
plt.close()

print("\n🎉 ALL ASSETS GENERATED SUCCESSFULLY IN YOUR PROJECT REPOSITORY DIRECTORY!")