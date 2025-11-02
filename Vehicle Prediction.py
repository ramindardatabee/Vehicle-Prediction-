import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import warnings
# Suppress warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("//Users//mona//Desktop//BCIT//Advanced Data Analytics//COMP-4254-NET - Adv Topics Data Analytics//Vehicle Price (1).csv")
df = df.dropna(subset=['price'])  # Drop rows with missing target
X = df.drop(columns=['price'])
y = df['price']

# Encode categorical variables
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Feature selection
selector = SelectKBest(score_func=chi2, k=3)
selector.fit(X_imputed, y)
top_features = X_imputed.columns[selector.get_support()]
print("\nTop 3 selected features based on Chi-Square test:", list(top_features))

# Final feature set
X_final = X_imputed[top_features]
X_np = X_final.values
y_np = y.values

# Define scalers
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

# Apply K-Fold Cross-Validation for each scaler
for scaler_name, scaler in scalers.items():
    print(f"\n========== Evaluating with {scaler_name} ==========")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_list, precision_list, recall_list, f1_list = [], [], [], []

    fold = 1
    for train_idx, test_idx in kf.split(X_np):
        X_train, X_test = X_np[train_idx], X_np[test_idx]
        y_train, y_test = y_np[train_idx], y_np[test_idx]

        # Scale the features using the current scaler
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = LogisticRegression(fit_intercept=True, solver='liblinear', class_weight='balanced')
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        # Store metrics
        accuracy_list.append(acc)
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

        print(f"\nFold {fold}")
        print("Accuracy :", acc)
        print("Precision:", prec)
        print("Recall   :", rec)
        print("F1 Score :", f1)
       # print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        fold += 1

    # Final summary for current scaler
    print(f"\n--- Summary for {scaler_name} ---")
    print(f"Average Accuracy : {np.mean(accuracy_list):.4f} (±{np.std(accuracy_list):.4f})")
    print(f"Average Precision: {np.mean(precision_list):.4f} (±{np.std(precision_list):.4f})")
    print(f"Average Recall   : {np.mean(recall_list):.4f} (±{np.std(recall_list):.4f})")
    print(f"Average F1 Score : {np.mean(f1_list):.4f} (±{np.std(f1_list):.4f})")


# --- REGRESSION SECTION ---

# Define regression features
regression_features = ['name', 'description', 'make', 'model', 'year', 'engine', 'cylinders',
                       'fuel', 'mileage', 'transmission', 'trim', 'body', 'doors',
                       'exterior_color', 'interior_color', 'drivetrain']

X_reg = df[regression_features].fillna('missing')
y_reg = df['price'].values


#Encode categorical columns
for col in X_reg.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X_reg[col] = le.fit_transform(X_reg[col].astype(str))

X_reg = X_reg.values

# Define base regression models
def get_reg_models():
    return [
        ElasticNet(),
        SVR(gamma='scale'),
        DecisionTreeRegressor(),
        AdaBoostRegressor(),
        RandomForestRegressor(n_estimators=10),
        ExtraTreesRegressor(n_estimators=10)
    ]

# Regression evaluation
def evaluate_reg_model(y_true, preds, model):
    mse = mean_squared_error(y_true, preds)
    rmse = round(np.sqrt(mse), 3)
    print(f"RMSE: {rmse} - {model.__class__.__name__}")

# Train base and stacked regression models
def train_reg_models(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    models = get_reg_models()
    dfPred_val = pd.DataFrame()

    for i, model in enumerate(models):
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        dfPred_val[str(i)] = preds

    stacked_model = LinearRegression()
    stacked_model.fit(dfPred_val, y_val)

    print("\n** Evaluate Base Regression Models **")
    dfPred_test = pd.DataFrame()
    for i, model in enumerate(models):
        preds = model.predict(X_test)
        dfPred_test[str(i)] = preds
        evaluate_reg_model(y_test, preds, model)

    stacked_preds = stacked_model.predict(dfPred_test)
    print("\n** Evaluate Stacked Regression Model **")
    evaluate_reg_model(y_test, stacked_preds, stacked_model)

# --- CLASSIFICATION SECTION ---

# Prepare classification data (binning price)
df_clf = df.copy()
df_clf['price_bin'] = pd.qcut(df_clf['price'], q=3, labels=[0, 1, 2])
y_clf = df_clf['price_bin']

# Drop columns and impute missing values
X_clf = df_clf.drop(columns=['price', 'name', 'price_bin'])

for col in X_clf.select_dtypes(include=['number']).columns:
    X_clf[col] = X_clf[col].fillna(X_clf[col].median())

for col in X_clf.select_dtypes(include=['object']).columns:
    X_clf[col] = X_clf[col].astype(str).fillna('missing')

for col in X_clf.columns:
    if X_clf[col].dtype == 'object':
        le = LabelEncoder()
        X_clf[col] = le.fit_transform(X_clf[col].astype(str))

# Define base classification models
def get_clf_models():
    return [
        LogisticRegression(max_iter=1000),
        DecisionTreeClassifier(),
        AdaBoostClassifier(),
        RandomForestClassifier(n_estimators=10)
    ]

# Classification evaluation
def evaluate_clf_model(y_true, preds, model):
    precision = round(precision_score(y_true, preds, average='macro'), 2)
    recall = round(recall_score(y_true, preds, average='macro'), 2)
    f1 = round(f1_score(y_true, preds, average='macro'), 2)
    accuracy = round(accuracy_score(y_true, preds), 2)
    print(f"Precision: {precision} Recall: {recall} F1: {f1} Accuracy: {accuracy} - {model.__class__.__name__}")

# Train base and stacked classification models
def train_clf_models(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.70, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    models = get_clf_models()
    dfPred_val = pd.DataFrame()

    for i, model in enumerate(models):
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        dfPred_val[str(i)] = preds

    stacked_model = LogisticRegression(max_iter=1000)
    stacked_model.fit(dfPred_val, y_val)

    print("\n** Evaluate Base Classification Models **")
    dfPred_test = pd.DataFrame()
    for i, model in enumerate(models):
        preds = model.predict(X_test)
        dfPred_test[str(i)] = preds
        evaluate_clf_model(y_test, preds, model)

    stacked_preds = stacked_model.predict(dfPred_test)
    print("\n** Evaluate Stacked Classification Model **")
    evaluate_clf_model(y_test, stacked_preds, stacked_model)

# Run both tasks
train_reg_models(X_reg, y_reg)
train_clf_models(X_clf, y_clf)


# ---------------------- New Evaluation Code Starts Here ----------------------

import matplotlib.pyplot as plt

# Show features used in regression
print("\nSelected Features for Regression:")
print(regression_features)


# Cross-validation evaluation for regression
def evaluate_regression_models_with_cv(X, y):
    print("\n=== Cross-Validation RMSE Summary ===")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = get_reg_models()
    model_names = [model.__class__.__name__ for model in models]

    results = {name: [] for name in model_names}
    predictions = {}
    actuals = {}

    for model, name in zip(models, model_names):
        rmse_list = []
        all_preds = []
        all_actual = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, preds))
            rmse_list.append(rmse)

            all_preds.extend(preds)
            all_actual.extend(y_test)

        results[name] = rmse_list
        predictions[name] = np.array(all_preds)
        actuals[name] = np.array(all_actual)

    # Print RMSE table
    print("{:<25} {:>10} {:>10}".format("Model", "Avg RMSE", "Std Dev"))
    for name in model_names:
        avg_rmse = np.mean(results[name])
        std_rmse = np.std(results[name])
        print("{:<25} {:>10.2f} {:>10.2f}".format(name, avg_rmse, std_rmse))

    # Identify best model
    best_model = min(model_names, key=lambda name: np.mean(results[name]))
    print(f"\nBest Model Based on RMSE: {best_model}")

    # Plot actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(actuals[best_model], predictions[best_model], alpha=0.5, color='teal')
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Actual vs Predicted Prices ({best_model})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

evaluate_regression_models_with_cv(X_reg, y_reg)


######################################

