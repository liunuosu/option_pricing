import numpy as np
from sklearn.linear_model import LinearRegression

class AHBS:

    def __init__(self):
        self.models = []

    def fit(self, moneyness, maturity, iv_data):
        """
        Fit AHBS model over time.

        Parameters:
        - moneyness: shape (T, N)
        - maturity: shape (T, N)
        - iv_data: shape (T, N)
        """
        self.models = []
        T, N = iv_data.shape

        for t in range(T):
            x1 = moneyness[t]
            x2 = maturity[t]
            y = iv_data[t]

            # Mask invalid entries
            mask = ~np.isnan(y)
            x1 = x1[mask]
            x2 = x2[mask]
            y = y[mask]

            X = np.stack([
                np.ones_like(x1),
                x1,
                x1**2,
                x2,
                x2**2,
                x1 * x2
            ], axis=1)

            model = LinearRegression()
            model.fit(X, y)
            self.models.append(model)

    def predict(self, moneyness, maturity):
        """
        Predict implied volatility using fitted AHBS models.

        Parameters:
        - moneyness: shape (T, N)
        - maturity: shape (T, N)

        Returns:
        - predicted_iv: shape (T, N)
        """
        T, N = moneyness.shape
        preds = np.full((T, N), np.nan)

        for t in range(T):
            model = self.models[t]

            x1 = moneyness[t]
            x2 = maturity[t]
            X = np.stack([
                np.ones_like(x1),
                x1,
                x1**2,
                x2,
                x2**2,
                x1 * x2
            ], axis=1)
            preds[t] = model.predict(X)

        return preds
