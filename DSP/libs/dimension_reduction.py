import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets

from sklearn.decomposition import (PCA, IncrementalPCA,
                                   KernelPCA, TruncatedSVD,
                                   FastICA, MiniBatchDictionaryLearning,
                                   SparsePCA)

from sklearn.manifold import (MDS, Isomap,
                              TSNE, LocallyLinearEmbedding)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.random_projection import (GaussianRandomProjection,
                                       SparseRandomProjection)

from sklearn.svm import LinearSVC

from sklearn.neighbors import (KNeighborsClassifier,
                               NeighborhoodComponentsAnalysis)
                               
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class dimension_reduction(object):

    def __init__(self, features, n_ch):
        super(dimension_reduction, self).__init__()
        self.data = data
        # self.n_ch = n_ch
        # self.n_features = n_ch * sum(
        #     [f.dim_per_channel for f in self.data])
        # self.output = np.zeros(self.n_ch * self.n_features)

    def process_fit(self, data):
        return np.hstack([f.fit(data) for f in self.data])

    def process_transform(self, data):
        return np.hstack([f.transform(data) for f in self.data])

    def __repr__(self):
        return "%s.%s(%s)" % (
            self.__class__.__module__,
            self.__class__.__name__,
            str([str(f) for f in self.data])
        )

class data(object):

    def __repr__(self):
        return "%s.%s()" % (
            self.__class__.__module__,
            self.__class__.__name__
        )

#------------------------------------------------------------------------
# n_neighbors = 3
# random_state = 0

# # Load Digits dataset
# digits = datasets.load_digits()
# X, y = digits.data, digits.target

# # Split into train/test
# X_train, X_test, y_train, y_test = \
#     train_test_split(X, y, test_size=0.5, stratify=y,
#                      random_state=random_state)

# dim = len(X[0])
# n_classes = len(np.unique(y))
#------------------------------------------------------------------------

class PCA():

    def __init__(self, n_components, random_state):
        self.pca = PCA(n_components=n_components,
                random_state=random_state))
        
    def fit(self, x, y):
        
        return self.pca.fit(x, y)

    def transform(self, x)

        return self.pca.transform(x)


# Reduce dimension to 2 with Incremental PCA
inc_pca = make_pipeline(StandardScaler(),
                        IncrementalPCA(n_components=2))

# Reduce dimension to 2 with Kernel PCA
kpca = make_pipeline(StandardScaler(),
                     KernelPCA(kernel="rbf",
                               n_components=2,
                               gamma=None,
                               fit_inverse_transform=True,
                               random_state=random_state,
                               n_jobs=1))
                     
# Reduce dimension to 2 with Sparse PCA
sparsepca = make_pipeline(StandardScaler(),
                          SparsePCA(n_components=2,
                                    alpha=0.0001,
                                    random_state=random_state,
                                    n_jobs=-1))
                          
# Reduce dimension to 2 with Singular Value Decomposition [SVD]
SVD = make_pipeline(StandardScaler(),
                    TruncatedSVD(n_components=2,
                                 algorithm='randomized',
                                 random_state=2019,
                                 n_iter=5))

# Reduce dimension to 2 with Gaussian Random Projection [GRP]
GRP = make_pipeline(StandardScaler(),
                    GaussianRandomProjection(n_components=2,
                                             eps = 0.5,
                                             random_state=random_state))

# Reduce dimension to 2 with LinearDiscriminantAnalysis
lda = make_pipeline(StandardScaler(),
                    LinearDiscriminantAnalysis(n_components=2))

# Reduce dimension to 2 with NeighborhoodComponentAnalysis
nca = make_pipeline(StandardScaler(),
                    NeighborhoodComponentsAnalysis(n_components=2,
                                                   random_state=random_state))

# Reduce dimension to 2 with Sparse Random Projection [SRP]
SRP = make_pipeline(StandardScaler(),
                    SparseRandomProjection(n_components=2,
                                           density = 'auto',
                                           eps = 0.5,
                                           random_state=random_state,
                                           dense_output = False))

# Reduce dimension to 2 with MultiDimensional Scaling [MDS]                   
mds = make_pipeline(StandardScaler(),
                    MDS(n_components=2,
                        n_init=12,
                        max_iter=1200,
                        metric=True,
                        n_jobs=4,
                        random_state=random_state))

# Reduce dimension to 2 with Isomap                  
isomap = make_pipeline(StandardScaler(),
                       Isomap(n_components=2,
                              n_jobs = 4,
                              n_neighbors = 5))

# Reduce dimension to 2 with MiniBatch Dictionary Learning
miniBatchDictLearning = make_pipeline(StandardScaler(),
                                      MiniBatchDictionaryLearning(n_components=2,
                                                                  batch_size = 200,
                                                                  alpha = 1,
                                                                  n_iter = 25,
                                                                  random_state=random_state))

# Reduce dimension to 2 with Independent Composent Analysis [ICA]
FastICA = make_pipeline(StandardScaler(),
                        FastICA(n_components=2,
                                algorithm = 'parallel',
                                whiten = True,
                                max_iter = 100,
                                random_state=random_state))

# Reduce dimension to 2 with T-distributed Stochastic Neighbor Embedding [T-SNE]
tsne = make_pipeline(StandardScaler(),
                     TSNE(n_components=2,
                          learning_rate=300,
                          perplexity = 30,
                          early_exaggeration = 12,
                          init = 'random',
                          random_state=random_state))

# Reduce dimension to 2 with Locally Linear Embedding [LLE]
lle = make_pipeline(StandardScaler(),
                    LocallyLinearEmbedding(n_components=2,
                                           n_neighbors = 10,
                                           method = 'modified',
                                           n_jobs = 4,
                                           random_state=random_state))

# Reduce dimension to 2 with L1-based feature selection
lsvc = make_pipeline(StandardScaler(),
                     LinearSVC(C=0.01,
                               penalty="l1",
                               dual=False))

#------------------------------------------------------------------------
# Use a nearest neighbor classifier to evaluate the methods
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
#------------------------------------------------------------------------

# Make a list of the methods to be compared
dim_reduction_methods = [('PCA', pca),
                         ('LDA', lda),
                         ('NCA', nca),
                         ('INC PCA', inc_pca),
                         ('KPCA', kpca),
                         ##('Sparced PCA', sparsepca),
                         ('SVD', SVD),
                         ('GRP', GRP),
                         ('SRP', SRP),
                         #('MDS', mds),
                         ('IsoMap', isomap),
                         ('MBD', miniBatchDictLearning),
                         ('ICA', FastICA),
                         #('TSNE', tsne),
                         ('LLE', lle),]

plt.figure(figsize=(24, 36))
for i, (name, model) in enumerate(dim_reduction_methods):
    plt.subplot(3, 4, i + 1, aspect=1)

    # Fit the method's model
    model.fit(X_train, y_train)

    # Fit a nearest neighbor classifier on the embedded training set
    knn.fit(model.transform(X_train), y_train)

    # Compute the nearest neighbor accuracy on the embedded test set
    acc_knn = knn.score(model.transform(X_test), y_test)

    # Embed the data set in 2 dimensions using the fitted model
    X_embedded = model.transform(X)
    df = pd.DataFrame(np.concatenate((X_embedded, np.reshape(y, (-1, 1))), axis=1))

    # Plot the projected points and show the evaluation score
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, s=30, cmap='Set1')
    plt.title("{}, KNN (k={})\nTest accuracy = {:.2f}".format(name,
                                                              n_neighbors,
                                                              acc_knn))
    
    for i, number in enumerate(y_test):
        plt.annotate(number,
                     df.loc[df[2]==number,[0,1]].mean(),
                     horizontalalignment='center',
                     verticalalignment='center',
                     weight='bold',
                     size='20')
plt.show()