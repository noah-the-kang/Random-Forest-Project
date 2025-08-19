import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, gammaln  # sigmoid function and log gamma
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

class BinomialDecisionTree:
    """A decision tree that splits based on binomial log-likelihood"""

    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def binomial_log_likelihood(self, y, n, p):
        """Calculate complete binomial log-likelihood including binomial coefficient"""
        # Avoid log(0) by clipping p
        p = np.clip(p, 1e-15, 1-1e-15)

        # Complete binomial log-likelihood: log(n choose y) + y*log(p) + (n-y)*log(1-p)
        # log(n choose y) = log(n!) - log(y!) - log((n-y)!)
        # Using gammaln: log(n!) = gammaln(n+1)
        log_binomial_coeff = gammaln(n + 1) - gammaln(y + 1) - gammaln(n - y + 1)
        log_likelihood_terms = y * np.log(p) + (n - y) * np.log(1 - p)

        return np.sum(log_binomial_coeff + log_likelihood_terms)

    def optimal_p(self, y, n):
        """Find optimal probability that maximizes binomial log-likelihood"""
        if len(y) == 0:
            return 0.5
        # MLE for binomial is sum(y)/sum(n)
        return np.sum(y) / np.sum(n) if np.sum(n) > 0 else 0.5

    def evaluate_split(self, X, y, n, feature_idx, threshold):
        """Evaluate a potential split based on binomial log-likelihood gain"""
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        # Check minimum samples constraint
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return -np.inf

        # Calculate log-likelihood for parent node
        p_parent = self.optimal_p(y, n)
        ll_parent = self.binomial_log_likelihood(y, n, np.full_like(y, p_parent, dtype=float))

        # Calculate log-likelihood for left and right children
        y_left, n_left = y[left_mask], n[left_mask]
        y_right, n_right = y[right_mask], n[right_mask]

        p_left = self.optimal_p(y_left, n_left)
        p_right = self.optimal_p(y_right, n_right)

        ll_left = self.binomial_log_likelihood(y_left, n_left, np.full_like(y_left, p_left, dtype=float))
        ll_right = self.binomial_log_likelihood(y_right, n_right, np.full_like(y_right, p_right, dtype=float))

        # Return log-likelihood gain
        return ll_left + ll_right - ll_parent

    def find_best_split(self, X, y, n):
        """Find the best split across all features and thresholds"""
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]
        n_samples = X.shape[0]

        for feature_idx in range(n_features):
            # Get unique values for this feature
            unique_values = np.unique(X[:, feature_idx])

            # Try splits at midpoints between unique values
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                gain = self.evaluate_split(X, y, n, feature_idx, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def build_tree(self, X, y, n, depth=0):
        """Recursively build the decision tree"""
        # Base cases
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1):
            return {'is_leaf': True, 'prediction': self.optimal_p(y, n)}

        # Find best split
        feature, threshold, gain = self.find_best_split(X, y, n)

        # If no good split found, make leaf
        if feature is None or gain <= 0:
            return {'is_leaf': True, 'prediction': self.optimal_p(y, n)}

        # Create split
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # Recursively build subtrees
        left_tree = self.build_tree(X[left_mask], y[left_mask], n[left_mask], depth + 1)
        right_tree = self.build_tree(X[right_mask], y[right_mask], n[right_mask], depth + 1)

        return {
            'is_leaf': False, #'leaf' is the same as 'terminal node' (no more split is needed)
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }

    def fit(self, X, y, n):
        """Fit the decision tree"""
        self.tree = self.build_tree(X, y, n)
        return self

    def predict_single(self, x, tree):
        """Predict a single sample"""
        if tree['is_leaf']:
            return tree['prediction']

        if x[tree['feature']] <= tree['threshold']:
            return self.predict_single(x, tree['left'])
        else:
            return self.predict_single(x, tree['right'])

    def predict(self, X):
        """Predict probabilities for samples"""
        if self.tree is None:
            raise ValueError("Tree not fitted. Call fit() first.")

        predictions = []
        for x in X:
            pred = self.predict_single(x, self.tree)
            predictions.append(pred)

        return np.array(predictions)


class BinomialRandomForest:
    """Random Forest with binomial log-likelihood splitting"""

    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split #smallest persons in a node
        self.min_samples_leaf = min_samples_leaf   #how many leaves
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []

    def _get_n_features(self, n_total_features):
        """Determine number of features to consider at each split"""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_total_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_total_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_total_features)
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_total_features)
        else:
            return n_total_features

    def fit(self, X, y, n):
        """Fit the random forest"""
        np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        n_features_per_tree = self._get_n_features(n_features)

        self.trees = []
        self.feature_indices = []

        for i in range(self.n_estimators):
            # Bootstrap sampling
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
            n_bootstrap = n[bootstrap_indices]

            # Random feature selection
            feature_indices = np.random.choice(n_features, n_features_per_tree, replace=False)
            X_subset = X_bootstrap[:, feature_indices]

            # Fit tree
            tree = BinomialDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(X_subset, y_bootstrap, n_bootstrap)

            self.trees.append(tree)
            self.feature_indices.append(feature_indices)

        return self

    def predict(self, X):
        """Predict probabilities using ensemble average"""
        if not self.trees:
            raise ValueError("Forest not fitted. Call fit() first.")

        predictions = np.zeros((X.shape[0], len(self.trees)))

        for i, (tree, feature_indices) in enumerate(zip(self.trees, self.feature_indices)):
            X_subset = X[:, feature_indices]
            predictions[:, i] = tree.predict(X_subset)

        # Return average prediction across all trees
        return np.mean(predictions, axis=1)


# Example usage with your data
def main():
    # Your data
    nvec = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900])
    yvec = np.array([10, 20, 30, 80, 100, 120, 210, 240, 270])
    pvec = yvec / nvec
    x1 = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    x2 = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])

    # Combine features
    X = np.column_stack([nvec, yvec, x1, x2])

    print("Data Summary:")
    print(f"Features (X): nvec, yvec, x1, x2")
    print(f"Target (pvec): {pvec}")
    print(f"Sample size: {len(pvec)}")

    # Fit the binomial random forest
    rf = BinomialRandomForest(n_estimators=50, max_depth=5, random_state=42)
    rf.fit(X, yvec, nvec)

    # Make predictions
    predictions = rf.predict(X)

    # Compare predictions with actual probabilities
    print("\nPredictions vs Actual:")
    print("Actual   Predicted   Difference")
    print("-" * 35)
    for i in range(len(pvec)):
        diff = abs(pvec[i] - predictions[i])
        print(f"{pvec[i]:.4f}   {predictions[i]:.4f}      {diff:.4f}")

    # Calculate some metrics
    mse = np.mean((pvec - predictions) ** 2)
    mae = np.mean(np.abs(pvec - predictions))

    print(f"\nModel Performance:")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")

    # Also compare with standard Random Forest for reference
    print("\n" + "="*50)
    print("Comparison with Standard Random Forest:")

    rf_standard = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    rf_standard.fit(X, pvec)
    pred_standard = rf_standard.predict(X)

    mse_standard = np.mean((pvec - pred_standard) ** 2)
    mae_standard = np.mean(np.abs(pvec - pred_standard))

    print(f"Standard RF - MSE: {mse_standard:.6f}, MAE: {mae_standard:.6f}")
    print(f"Binomial RF - MSE: {mse:.6f}, MAE: {mae:.6f}")

if __name__ == "__main__":
    main()


