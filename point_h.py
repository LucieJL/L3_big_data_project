from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score



''' Encoding function (needs a list) '''
def labelEncoder(df, list_att):
    le = preprocessing.LabelEncoder()

    for att in list_att:
        le.fit(df[att].unique())
        df[att] = le.transform(df[att])


def printMetrics(y_test,y_test_pred):
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_test_pred))
    print("Exactitude :", accuracy_score(y_test, y_test_pred))
    print("Exactitude équilibrée :", balanced_accuracy_score(y_test, y_test_pred))
    print("Précision :", precision_score(y_test, y_test_pred, average='macro'))
    print("Rappel :", recall_score(y_test, y_test_pred, average='macro'))
    print("F1-score :", f1_score(y_test, y_test_pred, average='macro'))


def printFinalMetrics(accuracies, balanced_accuracies):
    print('min accuracy : ' + str(min(accuracies)))
    print('max accuracy : ' + str(max(accuracies)))
    print('mean accuracy : ' + str(sum(accuracies) / len(accuracies)))
    print()
    print('min balanced accuracy : ' + str(min(balanced_accuracies)))
    print('max balanced accuracy : ' + str(max(balanced_accuracies)))
    print('mean balanced accuracy : ' + str(sum(balanced_accuracies) / len(balanced_accuracies)))


def predictAttributeNaiveBayes(df, X, y, verbose=False):

    dfSelected = df[X + [y]]

    dfTest = dfSelected[dfSelected[y].isna()]
    dfTrain = dfSelected[~dfSelected[y].isna()]

    nb_classifiers = []
    accuracies = []
    balanced_accuracies = []

    if verbose:
        print("CategoricalNB")
        print("Starting predicting", y, "attribute...")

    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(dfTrain):
        if verbose:
            print("TRAIN:", train_index, "TEST:", test_index)

        nb = CategoricalNB()
        nb_classifiers.append(nb)

        X_train = dfTrain.iloc[train_index].loc[:, dfTrain.columns != y]
        y_train = dfTrain.iloc[train_index][y]
        X_test = dfTrain.iloc[test_index].loc[:, dfTrain.columns != y]
        y_test = dfTrain.iloc[test_index][y]

        nb.fit(X_train, y_train)
        y_test_pred = nb.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_test_pred))
        balanced_accuracies.append(balanced_accuracy_score(y_test, y_test_pred))

        if verbose:
            printMetrics(y_test, y_test_pred)

    if verbose:
        printFinalMetrics(accuracies, balanced_accuracies)

    df.loc[dfTest.index, y] = nb_classifiers[accuracies.index(max(accuracies))].predict(dfTest.loc[:, dfTrain.columns != y])


def predictAttributeRandomForest(df, X, y, verbose=False):

    dfSelected = df[X + [y]]

    dfTest = dfSelected[dfSelected[y].isna()]
    dfTrain = dfSelected[~dfSelected[y].isna()]

    rf_classifiers = []
    accuracies = []
    balanced_accuracies = []

    if verbose:
        print("RandomForestClassifier")
        print("Starting predicting", y, "attribute...")

    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(dfTrain):
        if verbose:
            print("TRAIN:", train_index, "TEST:", test_index)

        rf = RandomForestClassifier()
        rf_classifiers.append(rf)

        X_train = dfTrain.iloc[train_index].loc[:, dfTrain.columns != y]
        y_train = dfTrain.iloc[train_index][y]
        X_test = dfTrain.iloc[test_index].loc[:, dfTrain.columns != y]
        y_test = dfTrain.iloc[test_index][y]

        rf.fit(X_train, y_train)
        y_test_pred = rf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_test_pred))
        balanced_accuracies.append(balanced_accuracy_score(y_test, y_test_pred))

        if verbose:
            printMetrics(y_test, y_test_pred)

    if verbose:
        printFinalMetrics(accuracies, balanced_accuracies)

    df.loc[dfTest.index, y] = rf_classifiers[accuracies.index(max(accuracies))].predict(dfTest.loc[:, dfTrain.columns != y])