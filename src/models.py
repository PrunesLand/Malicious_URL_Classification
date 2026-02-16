from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

def get_models():
    clf1 = XGBClassifier()
    clf2 = KNeighborsClassifier()
    clf3 = MLPClassifier()
    
    voting_clf = VotingClassifier(
        estimators=[
            ('xgb', clf1), 
            ('knn', clf2), 
            ('mlp', clf3)
        ],
        voting='soft',
        weights=[3, 2, 1]
    )

    models = {
        'XGBClassifier': clf1,
        'KNeighborsClassifier': clf2,
        'MLPClassifier': clf3,
        'WeightedVoting': voting_clf,
    }
    return models
