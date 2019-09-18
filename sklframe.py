from sklearn.datasets import load_boston
from sklearn.pipeline import Pipeline

X, y = load_boston(return_X_y = True)
pipe = Pipeline([ ("scale", StandardScaler()),
                  ("model", LinearRegression())\n
               ])
