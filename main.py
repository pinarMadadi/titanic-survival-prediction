# Titanic Survival Prediction: A Comprehensive ML Workflow
# IMPORT LIBRARIES
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings("ignore")
import os
os.makedirs("results", exist_ok=True)

# COLOR PALETTE
COLOR_DIED, COLOR_SURVIVED = "#E53E3E", "#319795"
PALETTE_MAIN = [COLOR_DIED, COLOR_SURVIVED]
COLOR_FEMALE, COLOR_MALE = "#fbb4ae", "#b3cde3"
PALETTE_SEX = [COLOR_FEMALE, COLOR_MALE]
COLOR_BASELINE, COLOR_HIGHLIGHT = "#7D8795", "#EB7B30"
COLOR_PRIMARY = "#549F9E"
sns.set_theme(style="whitegrid")


# DATA LOADING
# Load the Titanic training and test datasets from GitHub.

train_url = "https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/train.csv"
test_url = "https://raw.githubusercontent.com/agconti/kaggle-titanic/master/data/test.csv"

train, test = pd.read_csv(train_url), pd.read_csv(test_url)
print(f"Training data shape: {train.shape}")
print(f"Test data shape: {test.shape}")
print(f"Training data columns: {list(train.columns)}")
print(f"Test data columns: {list(test.columns)}")

# EXPLORATORY DATA ANALYSIS (EDA)
# Compare survival rates between male and female passengers.

sex_survival_rate = train.groupby("Sex")["Survived"].mean()
print(f"Survival rate by sex:\n{sex_survival_rate}")
plt.bar(sex_survival_rate.index,sex_survival_rate.values,color=PALETTE_SEX)
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Sex")
plt.show()

# Compare the age distribution of survivors and non-survivors.

plt.hist(train[train["Survived"] == 0]["Age"].dropna(),
         bins=44,color=COLOR_DIED,alpha=0.8,label="Died")

plt.hist(train[train["Survived"] == 1]["Age"].dropna(),
         bins=44,color=COLOR_SURVIVED,alpha=0.8,label="Survived")

plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution")
plt.legend()
plt.show()

# Explore survival patterns by passenger class,
# sex, age, fare, and missing values.

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.barplot(data=train,x="Pclass",y="Survived",hue="Sex",palette=PALETTE_SEX,ax=axes[0, 0])
axes[0, 0].set_title("1. Survival Rate by Pclass & Sex",fontsize=12,fontweight="bold")
axes[0, 0].set_ylabel("Survival Rate")

sns.kdeplot( data=train, x="Age", hue="Survived", common_norm=False, 
            fill=True, palette=[COLOR_DIED, COLOR_SURVIVED], alpha=0.4,ax=axes[0, 1])
axes[0, 1].set_title("2. Age Distribution by Survival Status", fontsize=12,fontweight="bold")

sns.boxplot(data=train[train["Fare"] < 150],x="Pclass",y="Fare",hue="Survived",
            palette=[COLOR_DIED, COLOR_SURVIVED],ax=axes[1, 0])
axes[1, 0].set_title("3. Fare Distribution (Fare < 150) by Pclass",fontsize=12,fontweight="bold")

sns.heatmap(train.isnull(),cbar=False,cmap="magma",yticklabels=False,ax=axes[1, 1])
axes[1, 1].set_title("4. Missing Data Map (Bright lines = Missing)",fontsize=12,fontweight="bold")
plt.tight_layout()
plt.show()

# CHILDREN VS ALL PASSENGERS

children_survival_rate = round(train[train["Age"] < 10]["Survived"].mean(),2)
print(f"Children survival rate: {children_survival_rate}" )

all_passenger_survival_rate = round(train["Survived"].mean(),2)
print(f"All passengers survival rate: {all_passenger_survival_rate}" )

survival_rate = {"children": children_survival_rate , "all": all_passenger_survival_rate}
plt.bar(survival_rate.keys(),survival_rate.values(),color=[COLOR_HIGHLIGHT, COLOR_BASELINE])
plt.xlabel("Category")
plt.ylabel("Survival Rate")
plt.title("Children Survival Rate vs All Passenger Survival Rate")
plt.show()

# FEATURE ENGINEERING
# Extract passenger titles from names
# and group rare titles together.
for df in [train, test]:
    df["title"] = (df["Name"].str.extract(r",\s*([^.]+)\.")[0].str.strip())
    df["title"] = (df["title"].replace(["Mlle", "Ms"], "Miss").replace("Mme", "Mrs"))
    df["title"] = df["title"].where(
    df["title"].isin(["Mr", "Miss", "Mrs", "Master"] ),"Rare")
print(f"Title value counts:\n{train['title'].value_counts()}")
print(f"Survival rate by title:\n{train.groupby('title')['Survived'].mean().round(2)}")

# SURVIVAL BY TITLE AND PCLASS

survival_by_title_pclass = (train.groupby(["Pclass", "title"])["Survived"].mean().round(2).unstack())
print(f"Survival rate by Pclass and Title:\n{survival_by_title_pclass}" )
plt.figure(figsize=(8, 7))
sns.heatmap(survival_by_title_pclass,annot=True,cmap="nipy_spectral",fmt=".2f")
plt.title("Survival Rate by Pclass and Title")
plt.xlabel("Title")
plt.ylabel("Pclass")
plt.show()

survival_by_title_pclass.plot( kind="bar",figsize=(8, 4),colormap="Set2")
plt.xlabel("Pclass")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Pclass")
plt.xticks([0, 1, 2], ["1st", "2nd", "3rd"],rotation=0)
plt.ylim(0, 1)
plt.legend(title="Title")
plt.show()

# DATA PREPROCESSING

print(f"Missing values in training data:\n{train.isnull().sum()[train.isnull().sum() > 0]}")
print(f"Missing values in test data:\n{test.isnull().sum()[test.isnull().sum() > 0]}")

# Check median age for each title.
print(f"Median age by title:\n{train.groupby('title')['Age'].median()}")

# PREPROCESS FUNCTION
def preprocess(df):
    df = df.copy()
    df["family_size"] = df["SibSp"] + df["Parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    df["has_cabin"] = df["Cabin"].notnull().astype(int)
    df["family_category"] = pd.cut(df["family_size"],bins=[0, 1, 4, 7, 20],
        labels=["Alone", "Small", "Medium", "Large"]).astype(str)
    return df

class GroupSizeTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.ticket_counts_ = X["Ticket"].value_counts()
        return self

    def transform(self, X):
        X = X.copy()

        X["group_size"] = (
            X["Ticket"]
            .map(self.ticket_counts_)
            .fillna(1)
        )

        X["fare_per_person"] = X["Fare"] / X["group_size"]

        return X


# Custom transformer for leakage-safe imputation.

# Imputation statistics are learned separately within each training fold
# when used inside cross-validation.

class TitanicImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.age_medians_ = X.groupby("title")["Age"].median()
        self.fare_medians_ = X.groupby("Pclass")["Fare"].median()
        self.embarked_mode_ = X["Embarked"].mode()[0]
        self.fare_pp_medians_ = X.groupby("Pclass")["fare_per_person"].median()
        return self
    def transform(self, X):
        X = X.copy()
        X["Age"] = X["Age"].fillna(X["title"].map(self.age_medians_))
        X["Fare"] = X["Fare"].fillna(X["Pclass"].map(self.fare_medians_))
        X["Embarked"] = X["Embarked"].fillna(self.embarked_mode_)
        X["fare_per_person"] = X["fare_per_person"].fillna(X["Pclass"].map(self.fare_pp_medians_))
        return X


# Apply preprocessing.
train, test = preprocess(train), preprocess(test)
print(f"Embarked values in training data:\n{train['Embarked'].value_counts()}")

# FAMILY SIZE ANALYSIS
family_survival_rate = (train.groupby("family_size")["Survived"].mean().round(2))
print(f"Family survival rate:\n{family_survival_rate}")
print(f"Missing values in training data:\n {train[['Age', 'Embarked', 'has_cabin', 'Fare']].isnull().sum()}")
print(f"Missing values in test data:\n {test[['Age', 'Embarked', 'has_cabin', 'Fare']].isnull().sum()}")
plt.bar(family_survival_rate.index.astype(str), family_survival_rate.values,color=COLOR_PRIMARY)
plt.xlabel("Family Size")
plt.ylabel("Survival Rate")
plt.title("Survival Rate by Family Size")
plt.show()

# FEATURE SELECTION AND TRAIN-VALIDATION SPLIT
passenger_ids_test = test["PassengerId"]
drop_cols = ["PassengerId", "Name" , "Cabin"]
X = train.drop(drop_cols + ["Survived"], axis=1)
y = train["Survived"]
X_test = test.drop(columns=drop_cols)
num_cols = ["Age", "Fare", "is_alone", "fare_per_person" , "has_cabin", "family_size" , "group_size"]
cat_cols = ["Sex", "Embarked", "title", "Pclass", "family_category"]
preprocessor = ColumnTransformer([
    ("num", "passthrough", num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols)
])

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# FEATURE SELECTION:
# FORWARD AND BACKWARD ABLATION STUDY

# Evaluate a specific feature set using cross-validation.
# The preprocessing and SVM are fitted independently within each CV fold.

def Evaluate_feature_set(num_features, cat_features, X, y, cv=5):
    selected_features = num_features + cat_features

    current_preprocessor = ColumnTransformer([
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"),
         cat_features)
    ])

    current_svm = Pipeline([
        ("group_size", GroupSizeTransformer()),
        ("impute" , TitanicImputer()),
        ("preprocessor", current_preprocessor),
        ("scale", StandardScaler(with_mean=True)),
        ("svc", SVC(C=1, gamma="scale", kernel="rbf",
                    max_iter=10000, random_state=42))
    ])

    score = cross_val_score(
        current_svm, X, y,
        cv=cv, scoring="accuracy"
    )

    return score.mean(), score.std()

# FORWARD ABLATION STUDY

baseline_num = ["Age", "Fare"]
baseline_cat = ["Sex", "Embarked", "Pclass"]

baseline_mean, baseline_std = Evaluate_feature_set(
    baseline_num, baseline_cat, X_train, y_train
)

title_num = baseline_num.copy()
title_cat = baseline_cat + ["title"]
title_mean, title_std = Evaluate_feature_set(
    title_num, title_cat, X_train, y_train
)

family_size_num = title_num + ["family_size"]
family_size_cat = title_cat.copy()
family_size_mean, family_size_std = Evaluate_feature_set(
    family_size_num, family_size_cat, X_train, y_train
)

is_alone_num = family_size_num + ["is_alone"]
is_alone_cat = family_size_cat.copy()
is_alone_mean, is_alone_std = Evaluate_feature_set(
    is_alone_num, is_alone_cat, X_train, y_train
)

family_category_num = is_alone_num.copy()
family_category_cat = is_alone_cat + ["family_category"]
family_category_mean, family_category_std = Evaluate_feature_set(
    family_category_num, family_category_cat, X_train, y_train
)

full_num = family_category_num + ["has_cabin"]
full_cat = family_category_cat.copy()
full_mean, full_std = Evaluate_feature_set(
    full_num, full_cat, X_train, y_train
)

for name, mean, std in [
    ("Baseline", baseline_mean, baseline_std),
    ("Title", title_mean, title_std),
    ("Family size", family_size_mean, family_size_std),
    ("Is alone", is_alone_mean, is_alone_std),
    ("Family category", family_category_mean, family_category_std),
    ("Has cabin", full_mean, full_std)
]:
    print(f"{name} accuracy: {mean:.4f} (+/- {std:.4f})")


ablation_results = pd.DataFrame({
    "feature_set": ["baseline", "+ title", "+ family_size", "+ is_alone",
                    "+ family_category", "+ has_cabin"],
    "mean CV accuracy": [
        baseline_mean, title_mean, family_size_mean,
        is_alone_mean, family_category_mean, full_mean
    ],
    "std CV accuracy": [
        baseline_std, title_std, family_size_std,
        is_alone_std, family_category_std, full_std
    ]
}).set_index("feature_set").round(4)

print(ablation_results)

# BACKWARD ABLATION STUDY
backward_results = {}

for feature in ["Age", "Fare"]:
    num = [f for f in baseline_num if f != feature]
    mean, std = Evaluate_feature_set(num, baseline_cat, X_train, y_train)
    backward_results[f"- {feature.lower()}"] = (mean, std)

for feature in ["Sex", "Embarked", "Pclass"]:
    cat = [f for f in baseline_cat if f != feature]
    mean, std = Evaluate_feature_set(baseline_num, cat, X_train, y_train)
    backward_results[f"- {feature.lower()}"] = (mean, std)

for name, (mean, std) in backward_results.items():
    print(f"Remove {name[2:]} accuracy: {mean:.4f} (+/- {std:.4f})")


backward_ablation_results = pd.DataFrame({
    "feature_set": ["baseline"] + list(backward_results.keys()),
    "mean CV accuracy": [baseline_mean] +
                        [v[0] for v in backward_results.values()],
    "std CV accuracy": [baseline_std] +
                       [v[1] for v in backward_results.values()]
}).set_index("feature_set").round(4)

print(backward_ablation_results)

# MODEL PIPELINES

logistic_pipeline = Pipeline([
    ("group_size", GroupSizeTransformer()),
    ("impute" , TitanicImputer()),
    ("preprocessor", preprocessor),
    ("scale", StandardScaler(with_mean=True)),
    ("logisticregression", LogisticRegression())
])

rf_pipeline = Pipeline([
    ("group_size", GroupSizeTransformer()),
    ("impute" , TitanicImputer()),
    ("preprocessor", preprocessor),
    ("randomforestclassifier", RandomForestClassifier(random_state=42))
])

tree_pipeline = Pipeline([
    ("group_size", GroupSizeTransformer()),
    ("impute" , TitanicImputer()),
    ("preprocessor", preprocessor),
    ("decisiontreeclassifier", DecisionTreeClassifier(random_state=42))
])

knn_pipeline = Pipeline([
    ("group_size", GroupSizeTransformer()),
    ("impute" , TitanicImputer()),
    ("preprocessor", preprocessor),
    ("scale", StandardScaler(with_mean=True)),
    ("knn", KNeighborsClassifier())
])

svc_pipeline = Pipeline([
    ("group_size", GroupSizeTransformer()),
    ("impute" , TitanicImputer()),
    ("preprocessor", preprocessor),
    ("scale", StandardScaler(with_mean=True)),
    ("svc", SVC(probability=True , max_iter=-1, random_state=42))
])

xgb_pipeline = Pipeline([
    ("group_size", GroupSizeTransformer()),
    ("impute" , TitanicImputer()),
    ("preprocessor", preprocessor),
    ("xgbclassifier", XGBClassifier(
        random_state=42, eval_metric="logloss"
    ))
])

# HYPERPARAMETER TUNING FUNCTION

def tune_model(pipeline, param_grid, X, y, cv=5):
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv,
        scoring="accuracy", n_jobs=-1
    )

    grid_search.fit(X, y)
    print(f"Best params for {pipeline.steps[-1][0]}:")
    print(grid_search.best_params_)

    return grid_search.best_estimator_, grid_search

param_grid_svm = [
    {
        "svc__kernel": ["linear"],
        "svc__C": [0.1, 1, 10]
    },
    {
        "svc__kernel": ["rbf"],
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale", "auto", 0.01, 0.1, 1]
    }
]

param_grid_xgb = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.01, 0.1, 0.2],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.8, 1.0]
}

param_grid_knn = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 13],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"]
}

param_grid_rf = {
    "randomforestclassifier__n_estimators": [100, 200, 300],
    "randomforestclassifier__max_depth": [None, 5, 10],
    "randomforestclassifier__min_samples_split": [2, 5, 10],
    "randomforestclassifier__min_samples_leaf": [1, 2, 4],
    "randomforestclassifier__max_features": ["sqrt", "log2"]
}

param_grid_tree = {
    "decisiontreeclassifier__max_depth": [3, 4, 5, 6, 7],
    "decisiontreeclassifier__min_samples_split": [2, 5, 10],
    "decisiontreeclassifier__min_samples_leaf": [1, 2, 4]
}

param_grid_logistic = {
    "logisticregression__C": [0.01, 0.1, 1, 10, 100],
    "logisticregression__max_iter": [500 , 1000],
    "logisticregression__solver": ["liblinear"],
    "logisticregression__penalty": ["l1", "l2"]
}

# TUNE ALL MODELS

models_and_grids = {
    "Logistic Regression": (logistic_pipeline, param_grid_logistic),
    "Decision Tree": (tree_pipeline, param_grid_tree),
    "Random Forest": (rf_pipeline, param_grid_rf),
    "KNN": (knn_pipeline, param_grid_knn),
    "SVM": (svc_pipeline, param_grid_svm),
    "XGBoost": (xgb_pipeline, param_grid_xgb)
}

# Dictionary for storing the best model
# found for each algorithm.
tuned_models = {}
# Dictionary for storing complete GridSearchCV objects.
grid_searches = {}
# Run GridSearchCV for every model.
for name, (pipeline, param_grid) in models_and_grids.items():
    print("\n" + "=" * 60)
    print(f"TUNING: {name}")
    print("=" * 60)
    best_model, grid_search = tune_model(pipeline,param_grid,X_train,y_train)
    tuned_models[name] = best_model
    grid_searches[name] = grid_search

# COMPARE TUNED MODELS

# Compare the tuned models using their mean cross-validation accuracy.
# The held-out validation set is not used for model selection.

model_comparison = pd.DataFrame({
    "model": list(grid_searches.keys()),
    "CV accuracy": [grid_searches[name].best_score_
                    for name in grid_searches]
}).sort_values("CV accuracy", ascending=False).reset_index(drop=True)

model_comparison["CV accuracy"] = model_comparison["CV accuracy"].round(4)

print("\n" + "=" * 60)
print("TUNED MODEL COMPARISON")
print("=" * 60)
print(model_comparison)

# MODEL COMPARISON PLOT

plt.figure(figsize=(9 , 6))
sns.barplot(data=model_comparison,x="CV accuracy",y="model",palette="Blues_r")
plt.title("Tuned Model Comparison (CV Accuracy)",fontsize=12,fontweight="bold")
plt.xlim(0.75, 0.88)
plt.tight_layout()
plt.savefig("results/model_comparison.png",dpi=300,bbox_inches="tight")
plt.show()

# SVM GRID SEARCH RESULTS

def extract_svc_results(grid_search):
    results = pd.DataFrame(grid_search.cv_results_)
    results = results[results["param_svc__kernel"].isin(["linear", "rbf"])]
    columns = [
        "param_svc__kernel", "param_svc__C",
        "param_svc__gamma", "mean_test_score", "std_test_score"
    ]
    return (
        results[columns]
        .sort_values("mean_test_score", ascending=False)
        .rename(columns={
            "param_svc__kernel": "kernel",
            "param_svc__C": "C",
            "param_svc__gamma": "gamma",
            "mean_test_score": "accuracy",
            "std_test_score": "Std"
        })
        .reset_index(drop=True)
    )
svc_results_table = extract_svc_results(grid_searches["SVM"])
print("\nSVM Grid Search Results:")
print(svc_results_table)

# SELECT THE BEST TUNED MODEL

selected_model_name = model_comparison.iloc[0]["model"]
selected_model = tuned_models[selected_model_name]

print(f"\nBest tuned model: {selected_model_name}")
print("\nBest hyperparameters:")
print(grid_searches[selected_model_name].best_params_)

# FINAL VALIDATION EVALUATION

# Evaluate the selected model once on the held-out validation set.
# This validation set was not used during hyperparameter tuning.'

y_val_pred = selected_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"\nValidation accuracy of {selected_model_name}: {val_accuracy:.4f}")

print(f"\nClassification report for {selected_model_name} "
      f"on validation set:")

print(classification_report(
    y_val, y_val_pred,
    target_names=["Died", "Survived"]
))


# CONFUSION MATRIX

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_val, y_val_pred,display_labels=["Died(0)", "Survived(1)"],cmap="Blues", ax=ax)
plt.title(f"Confusion Matrix: {selected_model_name}",fontsize=12, fontweight="bold")
plt.grid(False)
plt.savefig("results/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# ROC CURVE

fig, ax = plt.subplots(figsize=(9, 6))
RocCurveDisplay.from_estimator(
    selected_model, X_val, y_val,name=selected_model_name, ax=ax,curve_kwargs={"alpha": 0.85})
plt.plot([0, 1], [0, 1], "k--",label="Chance level (AUC = 0.5)")
plt.title(f"ROC Curve: {selected_model_name}",fontsize=12, fontweight="bold")
plt.xlabel("False Positive Rate (FPR)", fontsize=10)
plt.ylabel("True Positive Rate (TPR / Recall)", fontsize=10)
plt.legend(loc="lower right", fontsize=9, frameon=True)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("results/roc_curve.png", dpi=300, bbox_inches="tight")
plt.show()

from sklearn.metrics import roc_auc_score
y_val_proba = selected_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_proba)
print(f"Validation ROC-AUC: {val_auc:.4f}")

# FEATURE IMPORTANCE

if selected_model_name in ["Random Forest", "XGBoost", "Decision Tree"]:

    model_step = selected_model.steps[-1][0]
    importances = selected_model[model_step].feature_importances_

    fitted_preprocessor = selected_model.named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out()

    clean_feature_importance = [
        name.split("__")[-1] for name in feature_names
    ]

    feature_name_df = pd.DataFrame({
        "Feature": clean_feature_importance,
        "Importance": importances
    }).sort_values("Importance", ascending=False)

    print(f"\nFeature Importance for {selected_model_name}:")
    print(feature_name_df.head(10))

    sns.barplot(
        x=feature_name_df["Importance"],
        y=feature_name_df["Feature"],
        hue=feature_name_df["Feature"],
        palette="Blues_r",
        legend=False
    )

    plt.title(f"Feature Importance in {selected_model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Feature Importance in {selected_model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.savefig("results/feature_importance.png", dpi=300, bbox_inches="tight")
    plt.show()

else:
    print(f"\n{selected_model_name} does not support direct feature_importances_" )

# ENSEMBLE MODEL

# Build an ensemble combining:
# SVM + Random Forest + Logistic Regression.
# VotingClassifier will fit the supplied estimators
# internally.

ensemble_model = VotingClassifier(
    estimators=[
        ("svm", tuned_models["SVM"]),
        ("random_forest", tuned_models["Random Forest"]),
        ("logistic", tuned_models["Logistic Regression"])
    ],
    voting="soft"
    )

ensemble_model.fit(X_train, y_train)
ensemble_predicted = ensemble_model.predict(X_val)

ensemble_accuracy = accuracy_score(y_val, ensemble_predicted)
print(f"Ensemble Validation Accuracy: {ensemble_accuracy:.4f}")

# SHAP ANALYSIS FOR TREE-BASED MODELS

def SHAP(s_model, model_name):
    # 1. Separate preprocessing from the final classifier
    preprocessor_pipeline = Pipeline(s_model.steps[:-1])
    classifier_model = s_model.steps[-1][1]

    # 2. Transform validation data using the fitted preprocessing pipeline
    X_val_transformed = preprocessor_pipeline.transform(X_val)

    # 3. Extract clean feature names after preprocessing
    fitted_column_trans = s_model.named_steps["preprocessor"]
    raw_feature_names = fitted_column_trans.get_feature_names_out()

    clean_feature_names = [
        col.split("__")[-1] for col in raw_feature_names
    ]

    # Convert processed data to DataFrame for readable SHAP plots
    X_val_processed_df = pd.DataFrame(
        X_val_transformed,
        columns=clean_feature_names
    )

    # 4. Select the appropriate SHAP explainer
    if model_name in ["Random Forest", "Decision Tree", "XGBoost"]:
        explainer = shap.TreeExplainer(classifier_model)
        shap_values = explainer(X_val_processed_df)

    elif model_name == "Logistic Regression":
        explainer = shap.LinearExplainer(
            classifier_model,
            X_val_processed_df
        )
        shap_values = explainer(X_val_processed_df)

    elif model_name in ["SVM", "KNN"]:
        background_data = shap.sample(
            X_val_processed_df,
            50,
            random_state=42
        )

        explainer = shap.KernelExplainer(
            classifier_model.predict_proba,
            background_data
        )

        shap_values = explainer(
            X_val_processed_df,
            silent=True
        )

    else:
        print(f"SHAP analysis is not available for {model_name}.")
        return

    # 5. Select SHAP values for the Survived class
    if len(shap_values.shape) == 3:
        shap_values_survived = shap_values[:, :, 1]
    else:
        shap_values_survived = shap_values

    # 6. SHAP Summary Beeswarm Plot
    plt.figure(figsize=(10, 8))

    shap.summary_plot(
        shap_values_survived,
        X_val_processed_df,
        show=False
    )

    plt.title(
        f"SHAP Summary Plot ({model_name})",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig("results/shap_summary.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 7. SHAP Feature Importance Bar Plot
    plt.figure(figsize=(10, 6))

    shap.summary_plot(
        shap_values_survived,
        X_val_processed_df,
        plot_type="bar",
        show=False
    )

    plt.title(
        f"SHAP Feature Importance Bar Plot ({model_name})",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.show()


# Run SHAP only for the selected final model
SHAP(selected_model, selected_model_name)