# Binomial Random Forest
This repository contains a custom implementation of a decision tree and a random forest specifically designed for binomial data. Unlike standard regression trees that minimize mean squared error, this model uses a binomial log-likelihood as the criterion for finding optimal splits. This approach is ideal for datasets where each observation consists of a number of trials (n) and a number of successes (y), and the goal is to predict the underlying success probability (p).

## 📚 Key Features
- `BinomialDecisionTree`: A decision tree class that builds its structure by recursively finding the split that maximizes the gain in binomial log-likelihood.
- `BinomialRandomForest`: An ensemble method that combines multiple BinomialDecisionTree instances. It uses bootstrap aggregation and random feature subsets to reduce overfitting and improve prediction accuracy.
- `binomial_log_likelihood`: The core function that calculates the log-likelihood for a set of binomial observations. It is used to evaluate the quality of a potential split. scipy.special.gammaln is used for numerical stability when computing the binomial coefficient.
- `optimal_p`: A utility function that computes the Maximum Likelihood Estimate (MLE) of the success probability p for a given node. For binomial data, this is simply the total number of successes divided by the total number of trials.

## 💻 Dependencies
The following libraries are required to run the code:
- `numpy`
- `scipy`
- `pandas`
- `scikit-learn`
- `matplotlib`

You can install them using pip:
```
pip install numpy scipy pandas scikit-learn matplotlib
```

## 🚀 How to Use
The `main()` function provides a complete example of how to use the BinomialRandomForest class.
1. **Define your data:** Create arrays for your features (`X`), number of successes (`y`), and number of trials (`n`).
2. Instantiate the model:

```
from your_module import BinomialRandomForest

rf = BinomialRandomForest(n_estimators=50, max_depth=5, random_state=42)
```

3. Fit the model: Call the fit method with your data.
```
rf.fit(X, y, n)
```

4. Make predictions: Use the predict method to get probability predictions for new data points.

```
predictions = rf.predict(X_new)
```

The example in `main()` also includes a comparison with scikit-learn's standard `RandomForestRegressor`, demonstrating the difference in performance.


## 🎯 Use Case
This model is particularly useful for problems where the outcome is a ratio or a count of successes within a fixed number of trials. A classic example is predicting the conversion rate of an online ad, where n is the number of ad views and y is the number of clicks. Standard regression models may not be the best fit for this kind of data, while this custom implementation is tailored to the underlying statistical properties.
