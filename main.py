import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns

iris = datasets.load_iris()

x = iris['data']            # sepal length, sepal width, petal length, petal width
y = iris['target']          # 0 -> sentosa, 1 -> versicolor, 2 -> virginica

x = StandardScaler().fit_transform(x)   # scaling the data to 0 mean and unit variance for comparison amongst features

covariance_of_data = np.cov(x, rowvar=False)    # symmetrical and positive definite hence the covariance matrix gives
#                                               # orthogonal eigen vectors and real eigen values

eigenvalues, eigenvectors = np.linalg.eig(covariance_of_data)   # eigenvalues[:5]
#                                                               # array([2.93808505,0.9201649,0.14774182,0.02085386])
#                                                           eigenvectors[:5]
#                                                           array([[0.52106591,-0.37741762,-0.71956635,0.26128628],
#                                                           [-0.26934744, -0.92329566,  0.24438178, -0.12350962],
#                                                           [ 0.5804131 , -0.02449161,  0.14212637, -0.80144925],
#                                                           [ 0.56485654, -0.06694199,  0.63427274,  0.52359713]])

# we can see from eigen values that most of the contribution is from the first 2 eigen values, hence the first 2 eigen
# vectors

component_one = x.dot(eigenvectors.T[0])
component_two = x.dot(eigenvectors.T[1])

final_df = pd.DataFrame(component_one, columns=['Component 1'])
final_df['Component 2'] = component_two

final_df['Y'] = y

plt.figure(figsize=(20, 10))
sns.scatterplot(x="Component 1", y="Component 2", data=final_df, hue="Y", s=200)
