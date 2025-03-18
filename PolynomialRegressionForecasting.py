import numpy as np


class MultiLinearRegression:
    def __init__(self, x, y, w, b, alpha, _lambda):
        self.w = w
        self.b = b
        self.x, self.scaler = self.z_score_scaling(x)
        self.y = y
        self.m = x.shape[0]
        self.n = x.shape[1]
        self.alpha = alpha
        self._lambda = _lambda
        self.last_dw_dj_sum = 1E+10
        self.last_db_dj_sum = 1E+10
        self.num = 0

    def z_score_scaling(self, x):
        x_std = np.std(x, axis=0)
        x_mean = np.mean(x, axis=0)
        if not x_std.any():
            return np.array([np.zeros(x.shape[1])]), (x_mean, x_std)

        return (x - x_mean) / x_std, (x_mean, x_std)

    def f_wb(self, i):
        return np.dot(self.w, self.x[i]) + self.b

    def compute_cost(self):
        cost = 0
        regularized_cost = 0

        for i in range(self.m):
            cost += (self.f_wb(i) - self.y[i]) ** 2
        for i in range(self.w):
            regularized_cost += i**2

        return cost / (2 * self.m) + (regularized_cost * self._lambda) / (2 * self.m)

    def compute_gradient(self):
        f_wb = np.dot(self.x, self.w) + self.b
        loss = f_wb - self.y.reshape(-1, 1)  # Reshape self.y to be a column vector
        regularized_term = (self._lambda / self.m) * self.w
        dw_dj = np.dot(self.x.T, loss) / self.m + regularized_term.reshape(-1, 1)  # Transpose self.x to align dimensions
        db_dj = np.mean(loss)
        return dw_dj, db_dj

    def gradient_descent(self):
        self.num += 1
        dw_dj, db_dj = self.compute_gradient()
        # if self.num == 30:
        #     self.alpha = self.alpha * 10**(1/3)
        #     self.num = 0
        #     print('OOOONNNN')
        #
        # if np.linalg.norm(dw_dj) > np.linalg.norm(self.last_dw_dj_sum):
        #     self.alpha = self.alpha / 10**(1/3)
        #     print('OOOOOFFFF')
        #
        # self.last_dw_dj_sum = np.linalg.norm(self.last_dw_dj_sum)

        # print(self.w)
        # print(dw_dj)
        # print(self.alpha)
        # print('----')
        self.w = self.w - self.alpha*dw_dj
        self.b = self.b - self.alpha*db_dj

    def predict(self, x):
        if not np.any(self.scaler[1]):
            scaled_x = np.zeros(self.n)
            return np.dot(scaled_x, self.w) + self.b
        scaled_x = (x - self.scaler[0]) / self.scaler[1]
        return np.dot(scaled_x, self.w) + self.b