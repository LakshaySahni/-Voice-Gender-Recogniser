import numpy as np
from datetime import datetime
from copy import deepcopy

from pandas import read_csv, to_numeric
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def cout(message):
    message = str(datetime.isoformat(datetime.now())) + ": " + message
    print(message)


class Prediction:

    def __init__(self):
        cout("Reading data")
        data = read_csv('voice.csv')
        data_x = data.iloc[:, :data.shape[1] - 1]
        data_x = data_x[['meanfun', 'IQR']]
        data_y = data.iloc[:, data.shape[1] - 1:data.shape[1]]
        data_x = data_x.apply(lambda x: np.log(x + 1))
        scaler = MinMaxScaler(feature_range=(0, 1))

        data_x = scaler.fit_transform(data_x)

        train_x, test_x, train_y, test_y = train_test_split(
            data_x, data_y, test_size=0.33, random_state=42)

        for i in range(len(train_y)):
            if train_y.iloc[i].get_value('label') == 'female':
                train_y.iloc[i].set_value('label', int(0))
            else:
                train_y.iloc[i].set_value('label', int(1))
        for i in range(len(test_y)):
            if test_y.iloc[i].get_value('label') == 'female':
                test_y.iloc[i].set_value('label', int(0))
            else:
                test_y.iloc[i].set_value('label', int(1))

        train_x = np.array(train_x)
        train_y = np.array(train_y, dtype='int')
        test_x = np.array(test_x)
        test_y = np.array(test_y, dtype='int')
        train_y = train_y.flatten(order='C')
        cout("Training SGD Classifier")
        clf = SGDClassifier()
        clf.fit(train_x, train_y)
        predictions = clf.predict(test_x)
        cout("Testing accuracy:" + str(accuracy_score(test_y, predictions)))
        cout("Training Decision Tree")
        classifier = DecisionTreeClassifier(
            random_state=42, min_samples_split=50)
        classifier = classifier.fit(train_x, train_y)
        predictions = classifier.predict(test_x)
        cout("Testing accuracy:" + str(classifier.score(test_x, test_y)))
        cout("Training AdaBoostClassifier")
        ada_clf = AdaBoostClassifier()
        ada_clf.fit(train_x, train_y)
        predictions = ada_clf.predict(test_x)
        cout("Testing accuracy:" + str(accuracy_score(test_y, predictions)))
        cout("Training Random Forest Classifier")
        rf_clf = RandomForestClassifier()
        rf_clf.fit(train_x, train_y)
        predictions = rf_clf.predict(test_x)
        cout("Testing accuracy:" + str(accuracy_score(test_y, predictions)))
        cout("Training Gaussian Naive Bayes")
        gnb_clf = GaussianNB()
        gnb_clf.fit(train_x, train_y)
        predictions = gnb_clf.predict(test_x)
        cout("Testing accuracy:" + str(accuracy_score(test_y, predictions)))
        cout("Training Support Vector Classifier")
        sv_clf = SVC()
        sv_clf.fit(train_x, train_y)
        predictions = sv_clf.predict(test_x)
        cout("Testing accuracy:" + str(accuracy_score(test_y, predictions)))

        self.sgd_classifier = deepcopy(clf)
        del clf
        self.dt_classifier = deepcopy(classifier)
        del classifier
        self.ada_clf = deepcopy(ada_clf)
        del ada_clf
        self.rf_clf = deepcopy(rf_clf)
        del rf_clf
        self.gnb_clf = deepcopy(gnb_clf)
        del gnb_clf
        self.sv_clf = deepcopy(sv_clf)
        del sv_clf

    def predict(self, audio_vector):
        audio_vector = audio_vector / 1000
        needed_audio_vector = np.array(
            [audio_vector[5], audio_vector[12]]).astype('float')
        results = []
        results.append(self.sgd_classifier.predict(needed_audio_vector)[0])
        results.append(self.dt_classifier.predict(needed_audio_vector)[0])
        results.append(self.ada_clf.predict(needed_audio_vector)[0])
        results.append(self.rf_clf.predict(needed_audio_vector)[0])
        results.append(self.gnb_clf.predict(needed_audio_vector)[0])
        results.append(self.sv_clf.predict(needed_audio_vector)[0])
        print results
        was_it = raw_input("Were 3 or more predictions correct? (y/n)")
        from collections import Counter
        majority = Counter(results).most_common(n=1)[0][0]
        if majority == 0:
            majority = ',"female"'
            opp = ',"male"'
        else:
            majority = ',"male"'
            opp = ',"female"'
        if was_it == 'y':
            fd = open('voice.csv', 'a')
            fd.write('\n' + ','.join(map(str, audio_vector)) + majority)
            fd.close()
        else:
            fd = open('voice.csv', 'a')
            fd.write('\n' + ','.join(map(str, audio_vector)) + opp)
            fd.close()
