# ============================================================
# Ridge Logistic Regression for Toxic Comment Classification
# ============================================================
# This script builds a Ridge (L2-regularized) logistic regression
# model from scratch and trains it on the Jigsaw toxic comment
# dataset. It predicts 6 toxicity labels: toxic, severe_toxic,
# obscene, threat, insult, identity_hate.
#
# WHAT IS RIDGE (L2) REGULARIZATION?
# Without regularization, the model might assign crazy-high
# weights to certain features, overfitting to the training data.
# Ridge adds a penalty proportional to the SQUARE of each weight.
# This pushes weights toward zero (keeping them small and stable)
# but never fully zeroes them out — every feature stays in the mix.
#
# HOW IS THIS DIFFERENT FROM THE LASSO (L1) VERSION?
# Lasso uses "soft thresholding" which can push weights all the
# way to zero, effectively deleting features. Ridge is gentler —
# it shrinks weights but keeps them all. That's the ONLY
# difference. Everything else (sigmoid, gradient descent, etc.)
# is the same.
# ============================================================

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os


# ============================================================
# PART 1: Define the Ridge Logistic Regression class
# ============================================================
# This is the actual model. Think of it as a blueprint for a
# machine that learns to classify comments as toxic or not.
# We build it as a class so we can create one instance per
# toxicity label (6 total).
# ============================================================

class RidgeLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression with L2 (Ridge) regularization,
    built from scratch using gradient descent.
    """

    def __init__(self, alpha=0.01, learning_rate=0.1, max_iter=1000,
                 tol=1e-4, fit_intercept=True):
        """
        Set up the model's settings (called "hyperparameters").
        These control HOW the model learns, not WHAT it learns.

        Parameters:
        -----------
        alpha : float (default=0.01)
            Regularization strength. Higher alpha = stricter penalty
            on large weights. Think of it as "how tight the leash is."

        learning_rate : float (default=0.1)
            How big of a step we take each iteration when adjusting
            weights. Too big = we overshoot the best answer. Too
            small = we take forever to get there.

        max_iter : int (default=1000)
            Maximum number of times we loop through the training
            process. The model might stop early if it converges
            (stops improving meaningfully).

        tol : float (default=1e-4)
            Tolerance for convergence. If the weights change by
            less than this amount between iterations, we consider
            the model "done learning."

        fit_intercept : bool (default=True)
            Whether to include a bias/intercept term. Almost always
            True. The intercept lets the model shift its baseline
            prediction up or down regardless of feature values.
        """
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def _sigmoid(self, z):
        """
        The sigmoid function: converts any number into a probability
        between 0 and 1. This is how we turn raw scores into
        predictions like "72% chance this comment is toxic."

        How it works:
        - sigmoid(0) = 0.5 (50/50 chance, totally uncertain)
        - sigmoid(large positive number) is close to 1.0 (very likely toxic)
        - sigmoid(large negative number) is close to 0.0 (very likely NOT toxic)

        The np.clip part just prevents math errors when numbers
        get extremely large or small (caps them at -500 to 500).
        """
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        """
        Train the model: find the best weight for each feature.

        This is where the actual learning happens. The model:
        1. Starts with all weights at zero (no opinion yet)
        2. Makes predictions with current weights
        3. Checks how wrong the predictions are (the "residual")
        4. Calculates which direction to adjust each weight (the "gradient")
        5. Adds the Ridge penalty to the gradient (pushes weights toward zero)
        6. Takes a step in the improved direction
        7. Repeats until predictions stop improving

        Parameters:
        -----------
        X : numpy array, shape (number_of_comments, number_of_features)
            The feature values for each comment (slang count, word count, etc.)
        y : numpy array, shape (number_of_comments,)
            The true labels for each comment (0 = not toxic, 1 = toxic)
        """
        # Make sure y is a numpy array (not a pandas Series)
        y = np.asarray(y)

        # n = how many comments we have
        # p = how many features each comment has
        n, p = X.shape

        # Store the unique class labels (should be [0, 1])
        self.classes_ = np.unique(y)

        # Initialize all feature weights to zero
        # The underscore at the end is a scikit-learn convention meaning
        # "this attribute only exists after the model has been trained"
        self.coef_ = np.zeros(p)

        # Initialize the intercept (bias term) to zero
        self.intercept_ = 0.0

        # ---- THE TRAINING LOOP ----
        # Each pass through this loop is one "iteration" of learning
        for iteration in range(self.max_iter):

            # STEP A: Make predictions with current weights
            # X @ self.coef_ is the dot product: multiply each feature
            # by its weight and add them all up. Then add the intercept.
            # Finally, push through sigmoid to get a probability (0 to 1).
            raw_score = X @ self.coef_ + self.intercept_
            p_hat = self._sigmoid(raw_score)

            # STEP B: Calculate the residual (how wrong we are)
            # If p_hat=0.9 and true label=1, residual = -0.1 (close, small error)
            # If p_hat=0.9 and true label=0, residual = 0.9 (way off, big error)
            residual = p_hat - y

            # STEP C: Calculate the gradient for each feature weight
            # The gradient tells us: "if I increase this weight a tiny bit,
            # does the error go up or down, and by how much?"
            # (X.T @ residual) / n gives the average gradient across all comments
            grad_coef = (X.T @ residual) / n

            # STEP D: ADD THE RIDGE (L2) PENALTY TO THE GRADIENT
            # ===================================================
            # THIS IS THE KEY DIFFERENCE FROM THE LASSO VERSION.
            # ===================================================
            # Lasso (L1) uses soft thresholding, which can push weights
            # all the way to zero (deleting features entirely).
            #
            # Ridge (L2) adds (alpha * current_weight) to the gradient.
            # This means:
            #   - If a weight is large and positive, the penalty pushes it down
            #   - If a weight is large and negative, the penalty pushes it up
            #   - If a weight is near zero, the penalty barely touches it
            # The result: weights get SHRUNK toward zero but never reach zero.
            # Every feature stays in the model, just with a smaller influence.
            grad_coef = grad_coef + self.alpha * self.coef_

            # STEP E: Update the weights
            # Move in the OPPOSITE direction of the gradient
            # (opposite because gradient points toward MORE error,
            # and we want LESS error)
            coef_new = self.coef_ - self.learning_rate * grad_coef

            # STEP F: Update the intercept separately
            # Note: we do NOT apply the Ridge penalty to the intercept.
            # The intercept just shifts the baseline and shouldn't be penalized.
            if self.fit_intercept:
                self.intercept_ -= self.learning_rate * np.mean(residual)

            # STEP G: Check convergence — did the weights barely change?
            # We look at the single biggest weight change across all features.
            delta = np.max(np.abs(coef_new - self.coef_))

            # Save the new weights for the next iteration
            self.coef_ = coef_new

            # If the biggest change is smaller than our tolerance,
            # the model has converged (settled into a solution) — stop early
            if delta < self.tol:
                print(f"  Model converged at iteration {iteration}")
                break

        # Return the model itself (a scikit-learn convention that allows
        # chaining like model.fit(X, y).predict(X))
        return self

    def predict_proba(self, X):
        """
        Output probability scores for each comment.

        Returns a 2-column array:
        - Column 0: probability of NOT being toxic
        - Column 1: probability of BEING toxic

        Example output for one comment: [0.3, 0.7] means
        "30% chance it's clean, 70% chance it's toxic"
        """
        # Calculate probability of being toxic
        p = self._sigmoid(X @ self.coef_ + self.intercept_)

        # Stack [prob_not_toxic, prob_toxic] side by side
        return np.stack([1 - p, p], axis=1)

    def predict(self, X, threshold=0.5):
        """
        Output final yes/no predictions (0 or 1).

        If the predicted probability >= 0.5 (the threshold), we say
        "yes, this comment is toxic" (predict 1).
        Otherwise, we say "no, it's not toxic" (predict 0).
        """
        # Get the raw probability, compare to threshold, convert to 0 or 1
        return (self._sigmoid(X @ self.coef_ + self.intercept_) >= threshold).astype(int)

    def score(self, X, y):
        """
        Calculate accuracy: what fraction of predictions are correct.

        Example: if we predict 950 out of 1000 comments correctly,
        accuracy = 0.95 (95%)
        """
        return np.mean(self.predict(X) == y)


# ============================================================
# PART 2: Load and prepare the data
# ============================================================

print("Loading data...")

# Read the CSV file with all the features the team built
data = pd.read_csv("01_data/01_processed/train_set_with_features.csv")

print(f"Dataset shape: {data.shape[0]} comments, {data.shape[1]} columns")

# These are the 6 toxicity labels we want to predict
target_columns = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

# These columns should NOT be used as features:
# - "id" is just an identifier, not a useful signal
# - "comment_text" is raw text (our numeric features already capture its info)
# - The target columns themselves can't be used as inputs (that would be cheating!)
columns_to_exclude = ["id", "comment_text"] + target_columns

# Everything that's left is a feature column
feature_columns = [col for col in data.columns if col not in columns_to_exclude]

print(f"Number of features: {len(feature_columns)}")
print(f"Features: {feature_columns}")

# Separate the data into:
# X = the feature values (what the model sees as input)
# y = the target labels (what the model is trying to predict)
# .values converts from a pandas DataFrame to a numpy array
X = data[feature_columns].values
y = data[target_columns].values


# ============================================================
# PART 3: Split into training and test sets
# ============================================================

print("\nSplitting data into train (80%) and test (20%)...")

# train_test_split randomly divides the data into two groups:
# - Training set (80%): the model learns from this
# - Test set (20%): we check how well the model performs on data it has NEVER seen
#
# random_state=42 is a fixed seed so we get the same split every time.
# This makes our results reproducible (anyone can re-run and get the same numbers).
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"Training set: {X_train.shape[0]} comments")
print(f"Test set: {X_test.shape[0]} comments")


# ============================================================
# PART 4: Scale the features (standardization)
# ============================================================

print("\nScaling features...")

# StandardScaler transforms each feature so that:
# - Its average (mean) becomes 0
# - Its spread (standard deviation) becomes 1
#
# WHY DO WE NEED THIS?
# Features have wildly different ranges. For example:
# - word_count might go from 0 to 500
# - uppercase_ratio goes from 0 to 1
# Without scaling, the model would pay way more attention to
# word_count just because its numbers are bigger — not because
# it's actually more important. Scaling puts every feature on
# the same playing field so the model can judge them fairly.

scaler = StandardScaler()

# fit_transform on training data: learn the mean and std, then scale
X_train_scaled = scaler.fit_transform(X_train)

# transform on test data: use the SAME mean and std from training
# (we do NOT re-learn these from test data — that would be "data leakage,"
# which means the model sneaks a peek at test data it shouldn't see yet)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# PART 5: Train one Ridge Logistic Regression per toxicity label
# ============================================================
# We train 6 separate models — one for each toxicity type.
# Each model answers one specific yes/no question:
#   Model 1: "Is this comment toxic?"
#   Model 2: "Is this comment severely toxic?"
#   Model 3: "Is this comment obscene?"
#   Model 4: "Is this comment a threat?"
#   Model 5: "Is this comment an insult?"
#   Model 6: "Does this comment contain identity hate?"
# ============================================================

# Dictionary to store our 6 trained models
trained_models = {}

# Dictionaries to store prediction probabilities
# (Davvi1 needs these for stacking — he'll use your model's "confidence
# scores" as inputs to HIS model)
all_train_predictions = {}
all_test_predictions = {}

print("\n" + "=" * 60)
print("TRAINING MODELS")
print("=" * 60)

# Loop through each of the 6 toxicity labels
for i, label in enumerate(target_columns):
    print(f"\n--- Training model for: {label} ---")

    # Create a fresh Ridge Logistic Regression model with our chosen settings
    model = RidgeLogisticRegression(
        alpha=0.01,           # regularization strength (how strict the penalty is)
        learning_rate=0.1,    # step size for gradient descent
        max_iter=1000,        # maximum training iterations
        tol=1e-4              # convergence tolerance
    )

    # Train the model on the training data for this specific label
    # y_train[:, i] grabs column i from the targets array
    # (i=0 means "toxic", i=1 means "severe_toxic", i=2 means "obscene", etc.)
    model.fit(X_train_scaled, y_train[:, i])

    # Store the trained model in our dictionary
    trained_models[label] = model

    # Generate probability predictions on BOTH train and test sets
    # predict_proba returns [prob_NOT_toxic, prob_IS_toxic]
    # We grab column 1 ([:, 1]) because we want the "is toxic" probability
    train_proba = model.predict_proba(X_train_scaled)[:, 1]
    test_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Store these probabilities — Davvi1 will use them as inputs for stacking
    all_train_predictions[label] = train_proba
    all_test_predictions[label] = test_proba

    # Generate hard predictions (0 or 1) on the test set for evaluation
    test_preds = model.predict(X_test_scaled)

    # Calculate performance metrics
    # Accuracy: what fraction of predictions are correct
    accuracy = accuracy_score(y_test[:, i], test_preds)

    # F1 Score: a balanced measure that considers both:
    #   - Precision (of comments we flagged as toxic, how many actually were?)
    #   - Recall (of all actually toxic comments, how many did we catch?)
    # F1 is the harmonic mean of precision and recall.
    # zero_division=0 prevents an error if there are no positive predictions.
    f1 = f1_score(y_test[:, i], test_preds, zero_division=0)

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")


# ============================================================
# PART 6: Save outputs for Davvi1's stacking/ensemble
# ============================================================
# Davvi1's job is to take the predictions from YOUR model,
# Aarushi's tree model, and Klaas's SVM model, and combine
# them into one final "stacked" model. For that, he needs
# your model's predicted probabilities saved as CSV files.
# ============================================================

print("\n" + "=" * 60)
print("SAVING OUTPUTS")
print("=" * 60)

# Create an output folder inside the models directory
output_dir = "00_analysis/02_models/ridge_outputs"
os.makedirs(output_dir, exist_ok=True)

# Save the predicted probabilities as CSV files
# Each file has 6 columns (one per toxicity label) and one row per comment
train_preds_df = pd.DataFrame(all_train_predictions)
test_preds_df = pd.DataFrame(all_test_predictions)

train_preds_df.to_csv(f"{output_dir}/ridge_train_predictions.csv", index=False)
test_preds_df.to_csv(f"{output_dir}/ridge_test_predictions.csv", index=False)
print(f"Saved prediction CSVs to {output_dir}/")

# Save the scaler and trained models as .pkl (pickle) files
# This lets anyone reload and reuse them later without retraining
joblib.dump(scaler, f"{output_dir}/ridge_scaler.pkl")
joblib.dump(trained_models, f"{output_dir}/ridge_models.pkl")
print("Saved scaler and trained models as .pkl files")

print("\n" + "=" * 60)
print("DONE! Ridge Logistic Regression complete.")
print("=" * 60)
