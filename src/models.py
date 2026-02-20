from sklearn.svm._classes import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB

def get_models():
    clf1 = XGBClassifier()
    clf2 = KNeighborsClassifier()
    clf3 = MLPClassifier()
    clf4 = SVC()
    clf5 = GaussianNB()
    
    voting_clf = VotingClassifier(
        estimators=[
            ('xgb', clf1), 
            ('knn', clf2), 
            ('mlp', clf3),
            ('svm', clf4),
            ('gnb',clf5)
        ],
        voting='soft',
        weights=[5, 2, 1, 3, 4]
    )

    models = {
        'XGBClassifier': clf1,
        'KNeighborsClassifier': clf2,
        'MLPClassifier': clf3,
        'SVM': clf4,
        'GaussianNB': clf5,
        'WeightedVoting': voting_clf,
    }
    return models
