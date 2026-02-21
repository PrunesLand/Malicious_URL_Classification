from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB

def get_models():
    clf1 = XGBClassifier(n_jobs=-1)
    clf2 = KNeighborsClassifier(n_jobs=-1)
    clf3 = MLPClassifier(max_iter=400, verbose=True)
    clf4 = SGDClassifier(loss='log_loss', n_jobs=-1, max_iter=1000)
    clf5 = GaussianNB()
    
    voting_clf = VotingClassifier(
        estimators=[
            ('xgb', clf1), 
            ('knn', clf2), 
            ('mlp', clf3),
            ('lsvm', clf4),
            ('gnb',clf5)
        ],
        voting='soft',
        weights=[5, 2, 1, 3, 4],
        n_jobs=-1,
    )

    models = {
        'XGBClassifier': clf1,
        'KNeighborsClassifier': clf2,
        'MLPClassifier': clf3,
        'LinearSVM': clf4,
        'GaussianNB': clf5,
        'WeightedVoting': voting_clf,
    }
    return models
