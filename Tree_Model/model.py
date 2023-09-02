import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from numpy import *

from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

import random

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text


from sklearn.neural_network import MLPRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import roc_curve, auc


def read_data(run_num, prob):
    normalise = False

    df = pd.read_csv('abalone.data', header=None)
    df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight',
                  'Shell_weight', 'Rings']
    df['Sex'] = df['Sex'].replace(['M', 'F', 'I'], [0, 1, 2])
    df["age_bins"] = pd.cut(x=df["Rings"], bins=[0, 7, 10, 15, np.inf], labels=[1, 2, 3, 4])
    # convert_dict = {'age_bins': int64}
    # df = df.astype(convert_dict)
    data_inputx = df.iloc[:, 0:8]
    data_inputy = df.iloc[:, 9]  # this is target - so that last col is selected from data
    # data_in = genfromtxt("abalone.data", delimiter=",")
    # data_inputx = data_in[:, 1:9]  # all features 0, 1, 2, 3, 4, 5, 6, 7
    # data_inputy = data_in[:, 0]  # this is target - so that last col is selected from data

    if normalise == True:
        transformer = Normalizer().fit(data_inputx)  # fit does nothing.
        data_inputx = transformer.transform(data_inputx)

    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.40, random_state=run_num)

    return x_train, x_test, y_train, y_test


def scipy_models(x_train, x_test, y_train, y_test, type_model, run_num, problem):
    print(run_num, ' is our exp run')

    tree_depth = 2

    if problem == 'classifification':
        if type_model == 1:  # https://scikit-learn.org/stable/modules/tree.html  (see how tree can be visualised)
            #model = DecisionTreeClassifier(random_state=0, max_depth=tree_depth)
            model = DecisionTreeClassifier(random_state=0)
            # model.fit(x_train, y_train)
        elif type_model == 2:
            model = RandomForestClassifier(n_estimators=300)

        elif type_model == 3:
            model = ExtraTreesClassifier(n_estimators=100, max_depth=tree_depth, random_state=run_num)


    # Train the model using the training sets

    model.fit(x_train, y_train)

    if type_model == 1:
        r = export_text(model)
        print(r)

    # Make predictions using the testing set
    y_pred_test = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    if problem == 'classifification':
        perf_test = accuracy_score(y_pred_test, y_test)
        perf_train = accuracy_score(y_pred_train, y_train)
        cm = confusion_matrix(y_pred_test, y_test)
        # print(cm, 'is confusion matrix')
        # auc = roc_auc_score(y_pred, y_test, average=None)

    return perf_test  # ,perf_train


def main():
    max_expruns = 10

    forest_all = np.zeros(max_expruns)
    tree_all = np.zeros(max_expruns)
    extratree_all = np.zeros(max_expruns)

    hidden = 8

    prob = 'classifification'  # classification  or regression
    # prob = 'regression' #  classification  or regression

    # classifcation accurary is reported for classification and RMSE for regression

    print(prob, ' is our problem')

    for run_num in range(0, max_expruns):
        x_train, x_test, y_train, y_test = read_data(run_num, prob)

        acc_tree = scipy_models(x_train, x_test, y_train, y_test, 1, run_num, prob)  # Decision Tree
        acc_forest = scipy_models(x_train, x_test, y_train, y_test, 2, run_num,
                                  prob)  # Random Forests
        acc_extratree = scipy_models(x_train, x_test, y_train, y_test, 3, run_num,
                                     prob)  # Extra Trees

        tree_all[run_num] = acc_tree
        forest_all[run_num] = acc_forest
        extratree_all[run_num] = acc_extratree


    print(tree_all, ' tree_all')
    print(np.mean(tree_all), ' tree _all')
    print(np.std(tree_all), ' tree _all')

    print(forest_all, hidden, ' forest_all')
    print(np.mean(forest_all), ' forest _all')
    print(np.std(forest_all), ' forest _all')
    #
    # print(extratree_all, ' extra tree_all')
    # print(np.mean(extratree_all), ' extra tree _all')
    # print(np.std(extratree_all), ' extra tree_all')

def jupyter_notebook():
    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt

    from sklearn import datasets
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import Normalizer
    from sklearn import tree

    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier

    import random

    from sklearn.datasets import load_iris
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import export_text

    from sklearn.neural_network import MLPRegressor

    from sklearn.tree import DecisionTreeRegressor

    from sklearn.ensemble import RandomForestClassifier

    from sklearn.ensemble import ExtraTreesClassifier

    from sklearn.ensemble import RandomForestRegressor

    from sklearn.ensemble import ExtraTreesRegressor

    from sklearn.metrics import roc_curve, auc

    normalise = False

    df = pd.read_csv('abalone.data', header=None)
    df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight',
                  'Shell_weight', 'Rings']

    df['Sex'] = df['Sex'].replace(['M', 'F', 'I'], [0, 1, 2])
    # df["Age_bins"]=pd.cut(x=df["Rings"], bins=[0,7,10,15,np.inf], labels=[1,2,3,4])
    df["Age_bins"] = pd.cut(x=df["Rings"], bins=[0, 7, 10, 15, np.inf], labels=[0, 1, 2, 3])
    # df["Rings"]=pd.cut(x=df["Rings"], bins=[0,7,10,15,np.inf], labels=[0,1,2,3]
    convert_dict = {'Age_bins': int64}
    df = df.astype(convert_dict)
    data_inputx = df.iloc[:, 0:8]
    data_inputy = df.iloc[:, 9]  # this is target - so that last col is selected from data
    x_train, x_test, y_train, y_test = train_test_split(data_inputx, data_inputy, test_size=0.40, random_state=0)

    max_depth = 2
    model = DecisionTreeClassifier(random_state=0)
    model.fit(x_train, y_train)
    #
    fn = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight',
          'Shell_weight']
    # cn=['Child','Teen','Adult','Old']
    cn = ['1', '2', '3', '4']
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=600)
    # tree.plot_tree(model, feature_names=fn, class_names=cn, filled=True)
    # fig.savefig('DT_tree.png')
    #
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score

    pred = model.predict(x_test)
    pred_prob = model.predict_proba(x_test)

    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh = {}

    n_class = 4

    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:, i], pos_label=i)

    # plotting
    plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label='Class 1 vs Rest')
    plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Class 2 vs Rest')
    plt.plot(fpr[2], tpr[2], linestyle='--', color='blue', label='Class 3 vs Rest')
    plt.plot(fpr[3], tpr[3], linestyle='--', color='red', label='Class 4 vs Rest')
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('DT Multiclass ROC', dpi=300)
    for i in range(n_class):
        print(metrics.auc(fpr[i], tpr[i]))

    model2 = DecisionTreeClassifier(random_state=0)
    path = model2.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    plt.savefig('Post-pruning 1')

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(x_train, y_train)
        clfs.append(clf)
    print(
        "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
            clfs[-1].tree_.node_count, ccp_alphas[-1]
        )
    )

    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    plt.savefig('Post-pruning 2')

    train_scores = [clf.score(x_train, y_train) for clf in clfs]
    test_scores = [clf.score(x_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    ax.legend()
    plt.savefig('Post-pruning 3')

    alpha_dict = {}
    for i in range(len(ccp_alphas)):
        alpha_dict[ccp_alphas[i]] = abs(train_scores[i] - test_scores[i])
    minval = min(alpha_dict.values())
    res = [k for k, v in alpha_dict.items() if v == minval]

    model3 = DecisionTreeClassifier(random_state=0, ccp_alpha=0.010134845949880164)
    model3.fit(x_train, y_train)
    pred = model3.predict(x_test)
    accuracy_score(y_test, pred)

    fig, axes = plt.subplots(nrows=1, ncols=1, dpi=300)
    tree.plot_tree(model3, feature_names=fn, class_names=cn, filled=True)
    axes.set_title('Post-pruning Decision Tree')
    fig.savefig('Post-pruning_DT_tree.png')

    pred = model3.predict(x_test)
    pred_prob = model3.predict_proba(x_test)

    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh = {}

    n_class = 4

    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:, i], pos_label=i)

    # plotting
    plt.clf()
    plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label='Class 1 vs Rest')
    plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Class 2 vs Rest')
    plt.plot(fpr[2], tpr[2], linestyle='--', color='blue', label='Class 3 vs Rest')
    plt.plot(fpr[3], tpr[3], linestyle='--', color='red', label='Class 4 vs Rest')
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('pruning DT Multiclass ROC', dpi=300)
    for i in range(n_class):
        print(metrics.auc(fpr[i], tpr[i]))
    #----------------------------------take a long running time----------------------------------
    # max_expruns = 10
    # rf_estimators = np.zeros(max_expruns)
    # forest_all = np.zeros(max_expruns)
    # n_trees = [1, 20, 40, 60, 80, 100, 200, 300, 400, 500]
    # for num in range(0, max_expruns):
    #     n_estimators = n_trees[num]
    #     for run_num in range(0, max_expruns):
    #         rf = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    #         rf.fit(x_train, y_train)
    #         y_pred_test = rf.predict(x_test)
    #         y_pred_train = rf.predict(x_train)
    #         perf_test = accuracy_score(y_pred_test, y_test)
    #         perf_train = accuracy_score(y_pred_train, y_train)
    #         # cm = confusion_matrix(y_pred_test, y_test)
    #         # print(cm, 'is confusion matrix')
    #         forest_all[run_num] = accuracy_score(y_pred_test, y_test)
    #
    #     rf_estimators[num] = np.mean(forest_all)
    # print(rf_estimators)

    # n_trees = [1, 10, 20, 30 ,40, 50, 100, 200, 300, 400]
    # fig, ax = plt.subplots()
    # ax.set_xlabel("number of trees")
    # ax.set_ylabel("accuracy")
    # ax.set_title("Accuracy vs Number of Trees in the Ensembles Increases")
    # # ax.plot(n_trees, rf_estimators, marker="o", label="train", drawstyle="steps-post")
    # ax.plot(n_trees, rf_estimators, marker="o", drawstyle="steps-post")
    # fig.savefig('random forest with different num of trees.png')
    # plt.show()

    # fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
    best_rf = RandomForestClassifier(n_estimators=300, random_state=0)
    best_rf.fit(x_train, y_train)
    # tree.plot_tree(best_rf.estimators_[0], feature_names = fn,class_names=cn,filled = True)

    pred = best_rf.predict(x_test)
    pred_prob = best_rf.predict_proba(x_test)

    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh = {}

    n_class = 4

    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:, i], pos_label=i)

    # plotting
    plt.clf()
    plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label='Class 1 vs Rest')
    plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Class 2 vs Rest')
    plt.plot(fpr[2], tpr[2], linestyle='--', color='blue', label='Class 3 vs Rest')
    plt.plot(fpr[3], tpr[3], linestyle='--', color='red', label='Class 4 vs Rest')
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('Multiclass ROC of Random Forest with 300 trees in the ensembles', dpi=300)
    for i in range(n_class):
        print(metrics.auc(fpr[i], tpr[i]))

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import GradientBoostingRegressor

    import xgboost as xgb

    gd_boost = GradientBoostingClassifier(n_estimators=100, random_state=0, learning_rate=0.1)
    gd_boost.fit(x_train, y_train)
    # Make predictions using the testing set
    y_pred_test = gd_boost.predict(x_test)
    y_pred_train = gd_boost.predict(x_train)
    perf_test = accuracy_score(y_pred_test, y_test)

    pred = gd_boost.predict(x_test)
    pred_prob = gd_boost.predict_proba(x_test)

    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh = {}

    n_class = 4

    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:, i], pos_label=i)

    # plotting
    plt.clf()
    plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label='Class 1 vs Rest')
    plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Class 2 vs Rest')
    plt.plot(fpr[2], tpr[2], linestyle='--', color='blue', label='Class 3 vs Rest')
    plt.plot(fpr[3], tpr[3], linestyle='--', color='red', label='Class 4 vs Rest')
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('Multiclass ROC of Gradient Boosting', dpi=300)
    for i in range(n_class):
        print(metrics.auc(fpr[i], tpr[i]))

    xg_boost = xgb.XGBClassifier(learning_rate=0.1, n_estimators=100, random_state=0)
    xg_boost.fit(x_train, y_train)
    # Make predictions using the testing set
    y_pred_test = xg_boost.predict(x_test)
    y_pred_train = xg_boost.predict(x_train)
    perf_test = accuracy_score(y_pred_test, y_test)

    pred = xg_boost.predict(x_test)
    pred_prob = xg_boost.predict_proba(x_test)

    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh = {}

    n_class = 4

    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_prob[:, i], pos_label=i)

    # plotting
    plt.clf()
    plt.plot(fpr[0], tpr[0], linestyle='--', color='orange', label='Class 1 vs Rest')
    plt.plot(fpr[1], tpr[1], linestyle='--', color='green', label='Class 2 vs Rest')
    plt.plot(fpr[2], tpr[2], linestyle='--', color='blue', label='Class 3 vs Rest')
    plt.plot(fpr[3], tpr[3], linestyle='--', color='red', label='Class 4 vs Rest')
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('Multiclass ROC of XGBoost', dpi=300)
    for i in range(n_class):
        print(metrics.auc(fpr[i], tpr[i]))

    # auc score for each class in every model
    auc_all = {}
    auc_all['DT'] = [0.7854770685873681,
                     0.5972015177214224,
                     0.6069037698544341,
                     0.5757192870782941]
    auc_all['Pruning'] = [0.9049366266421686,
                          0.6630992974816154,
                          0.7282886705325036,
                          0.7081349458943349]
    auc_all['RF'] = [0.942766005721175,
                     0.7462335741576529,
                     0.7829829089316755,
                     0.8648854232972629]
    auc_all['GDboost'] = [0.9389311279078563,
                          0.7445447716081626,
                          0.7881198497896789,
                          0.873583704646722]
    auc_all['XGboost'] = [0.9400744731491821,
                          0.7386757792554597,
                          0.7763116864160508,
                          0.8641183959261618]
    acc_all = {}
    acc_all['DT'] = 0.5336923997606223
    acc_all['Pruning'] = 0.5942549371633752
    acc_all['RF'] = 0.6146020347097546
    acc_all['GDboost'] = 0.6163973668461998
    acc_all['XGboost'] = 0.6122082585278277

    plt.clf()
    x_class = ['Class 1', 'Class 2', 'Class 3', 'Class 4']
    # plotting
    plt.plot(x_class, auc_all['DT'], label='Decision Tree')
    plt.plot(x_class, auc_all['Pruning'], label='Prunning')
    plt.plot(x_class, auc_all['RF'], label='Random Forest')
    plt.plot(x_class, auc_all['GDboost'], label='GDboost')
    plt.plot(x_class, auc_all['XGboost'], label='XGboost')
    plt.title('AUC of Each Class for Each Model')
    plt.xlabel('Class')
    plt.ylabel('AUC')
    plt.legend(loc='best')
    plt.savefig('overall auc')

    plt.clf()
    x_class = ['Decision Tree', 'Prunning', 'Random Forest', 'GDboost', 'XGboost']
    y_class = [acc_all['DT'], acc_all['Pruning'], acc_all['RF'], acc_all['GDboost'], acc_all['XGboost']]
    # plotting
    plt.plot(x_class, y_class)
    plt.title('Accuracy score vs Models')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.savefig('overall accuracy')

    from sklearn.model_selection import learning_curve
    import numpy as np

    X = df.iloc[:, 0:8]
    y = df.iloc[:, 9]
    importances = best_rf.feature_importances_
    std = np.std([best_rf.feature_importances_ for tree in best_rf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the impurity-based feature importances of the forest
    plt.clf()
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.savefig('importance_rf.png')

    from sklearn.metrics import mean_squared_error

    data_dmatrix = xgb.DMatrix(data=X, label=y)
    preds = xg_boost.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE: %f" % (rmse))
    params = {"objective": "reg:linear", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
              'max_depth': 5, 'alpha': 10}
    cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                        num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
    cv_results.head()
    print((cv_results["test-rmse-mean"]).tail(1))
    xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

    plt.clf()
    xgb.plot_importance(xg_reg)
    plt.rcParams['figure.figsize'] = [8, 6]
    plt.savefig('importance_xg.png')




if __name__ == '__main__':
    main()
    jupyter_notebook()