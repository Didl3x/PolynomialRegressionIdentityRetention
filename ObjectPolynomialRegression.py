import numpy as np
import PolynomialRegressionForecasting
import random

class DetObject:
    def __init__(self, center_point, id):
        self.center_point = center_point
        self.last_center_point = np.array([0, 0])
        self.v = np.array([0, 0])
        self.id = id
        self.updated = False
        self.x = []
        self.y_x = []
        self.y_y = []
        self.current_index = 0
        self.frames_gone = 0

        self.w_x = np.array([[0, 0, 0, 0, 0, 0]]).T
        self.b_x = 0
        self.w_y = np.array([[0, 0, 0, 0, 0, 0]]).T
        self.b_y = 0
        self.alpha_x = 2E-1
        self.alpha_y = 2E-1
        self._lambda = 1E-2
        self.bgr = (random.randint(1, 254), random.randint(1, 254), random.randint(1, 254))

        self.predicted = []

    def update(self, x):
        y_x = self.center_point[0]
        y_y = self.center_point[1]
        if len(self.x) == 10:
            self.x[self.current_index] = [x**(i+1) for i in range(6)]
            self.y_x[self.current_index] = y_x
            self.y_y[self.current_index] = y_y
        else:
            self.x.append([x**(i+1) for i in range(6)])
            self.y_x.append(y_x)
            self.y_y.append(y_y)
        obj_x = PolynomialRegressionForecasting.MultiLinearRegression(np.array(self.x), np.array(self.y_x), self.w_x, self.b_x, self.alpha_x, self._lambda)
        obj_y = PolynomialRegressionForecasting.MultiLinearRegression(np.array(self.x), np.array(self.y_y), self.w_y, self.b_y, self.alpha_y, self._lambda)
        # print(self.x, self.y_x, self.y_y)
        for i in range(15000):
            obj_x.gradient_descent()
            obj_y.gradient_descent()

        self.w_x = obj_x.w
        self.b_x = obj_x.b

        self.w_y = obj_y.w
        self.b_y = obj_y.b

        self.alpha_x = obj_x.alpha
        self.alpha_y = obj_y.alpha

        next_x = x + 1
        self.v[0] = obj_x.predict(np.array([next_x**(u+1) for u in range(6)]))
        self.v[1] = obj_y.predict(np.array([next_x**(u+1) for u in range(6)]))
        self.current_index += 1
        if self.current_index == 10:
            self.current_index = 0

        self.predicted = np.zeros((15, 2))
        count = 0
        for i in range(x, x + 15):
            self.predicted[count] = np.array([int(obj_x.predict(np.array([i**(u+1) for u in range(6)]))), int(obj_y.predict(np.array([i**(u+1) for u in range(6)])))]).reshape((2,))
            count += 1
        # print(self.predicted)
        print(self.y_x, self.y_y)

    def kill_invisible(self):
        if not self.updated:
            del self