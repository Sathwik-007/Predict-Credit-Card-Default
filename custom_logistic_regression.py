import numpy as np

class CustomLogisticRegression:
    """
    Logistic Regression from scratch with L2 Regularization and Class Weights.
    """
    
    def __init__(self, epochs=1000, lr=0.01, lambda_param=0.1, class_weights=None):
        self.epochs = epochs
        self.lr = lr
        self.lambda_param = lambda_param
        self.class_weights = class_weights
        self.w = None
        self.b = None
        self.loss_history = []

    def _sigmoid(self, y):

        # clipping to a range of -500 to 500 to prevent overflow errors on exponent.
        y = np.clip(y, -500, 500)
        return 1 / (1 + np.exp(-y))

    def _binary_cross_entropy(self, y_true, y_pred, sample_weights):
        
        # add epsilon to prevent log(0) error
        epsilon = 1e-15

        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        weighted_loss = loss * sample_weights

        # calculate Regularisation loss
        # formula: Actual formula before differentiation as described in previous cell
        L2 = (self.lambda_param / (2 * len(y_true))) * np.sum(np.square(self.w))

        # add and return 
        return np.mean(weighted_loss) + L2

    def fit(self, X, y):

        # make sure X and y are np arrays
        X = np.array(X)
        y = np.array(y)
        
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # compute sample_wieghts
        sample_weights = np.ones(n_samples)
        if self.class_weights:
            for class_label, weight in self.class_weights.items():
                sample_weights[y == class_label] = weight

        for _ in range(self.epochs):

            # y = mx + c
            y_pred = np.dot(X, self.w) + self.b

            # pass throught sigmoid to get probabilities between 0 to 1.
            y_probs = self._sigmoid(y_pred)

            # calcualte loss
            loss = self._binary_cross_entropy(y, y_probs, sample_weights)
            self.loss_history.append(loss)

            # compute error
            err = (y_probs - y) * sample_weights

            # update gradients
            # updating weights matrices with L2 penalty
            # Differentiation of dL2/dw
            dw = (1 / n_samples) * np.dot(X.T, err) + (self.lambda_param / n_samples) * self.w
            db = (1 / n_samples) * np.sum(err)
            
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def _predict_proba(self, X):
        # for given example perform dot product with trained weights and bias
        y_pred = np.dot(X, self.w) + self.b
        # squeeze the probs through sigmoid function
        y_probs = self._sigmoid(y_pred)

        # We need class of 0 and 1
        return np.vstack((1 - y_probs, y_probs)).T

    def _predict(self, X, threshold=0.5):
        # compute probabilities
        probabilities = self._predict_proba(X)[:, 1]
        # return if greater than agreed threshold
        return (probabilities >= threshold)
    