import pandas as pd
df = pd.read_csv(filepath_or_buffer = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None)
df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
print(df.tail())

# split data table into data X and class labels y
X = df.ix[:,0:4].values
y = df.ix[:,4].values

#heatplot using matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2],c=X[:,3], cmap = plt.hot())
plt.show()
#scale features if necessary, not necessary here
import numpy as np

mean_vec = np.mean(X, axis=0)

X=X-mean_vec
#calculating cov_mat or cor_mat
#cov_mat = (X - mean_vec).T.dot((X - mean_vec)) / (X.shape[0]-1)

# can use corelation insead
cor_mat = np.corrcoef(X.T)#

# eigen decomposition

eig_vals, eig_vecs = np.linalg.eig(cor_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


#eig_vals, eig_vecs = np.linalg.eig(cov_mat)
#print('Eigenvectors \n%s' %eig_vecs)
#print('\nEigenvalues \n%s' %eig_vals)



# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

#picking most informative features , EXPLAINED VARIANCE: HOW MUCH INFO COULD BE ATTRIBUTED TO EACH OF THE PRINCIPAL COMPONENTS
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in (eig_vals)]
cum_var_exp = np.cumsum(var_exp)

print(cum_var_exp)

# PROJECTION MATRIX
proj_mat = eig_vecs[ : , 0:2]
print(proj_mat.shape)

#PROJECT ALONG NEW AXES
Y = X.dot(proj_mat)

print(Y.shape)
plt.scatter(Y[:,0],Y[:,1])
plt.show()
