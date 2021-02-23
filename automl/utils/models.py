from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

models_list = ['CatBoostClassifier', "Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
               "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
               "Naive Bayes", "QDA"]

MODELS_STR_TO_OBJECT = {'CatBoostClassifier': CatBoostClassifier(verbose=False),
                        'Nearest Neighbors': KNeighborsClassifier(3),
                        'Linear SVM': SVC(kernel="linear", C=0.025),
                        'RBF SVM': SVC(gamma=2, C=1),
                        'Gaussian Process': GaussianProcessClassifier(1.0 * RBF(1.0)),
                        'Decision Tree': DecisionTreeClassifier(max_depth=5),
                        'Random Forest': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                        'Neural Net': MLPClassifier(alpha=1, max_iter=1000),
                        'AdaBoost': AdaBoostClassifier(),
                        'Naive Bayes': GaussianNB(),
                        'QDA': QuadraticDiscriminantAnalysis()}
