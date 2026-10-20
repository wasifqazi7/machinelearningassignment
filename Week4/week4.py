# id:9--9-9-0
# id:9--18-9-0 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve, auc


path = "/Users/wasirq/Workspace/University/Machine Learning/Assignment4/week4.csv"
with open(path) as f:
    lines = f.readlines()
header_idx = [(i,line.strip()) for i, line in enumerate(lines) if line.startswith("#")]
# The first header at 0 and the otehr at 1339
df1 = pd.read_csv(path, skiprows=1, nrows=1338, header=None)   
df2 = pd.read_csv(path, skiprows=1340, header=None)    

def question_1_a():
    print('The # appears at index (index,id): ',header_idx)
    print('\n')
    print('The shape of the two datasets')
    print(df1.shape, df2.shape)
    print('\n')
    print('The shape of the datasets - 1')
    print(df1.head())
    print('\n')
    print('The shape of the datasets - 2')
    print(df2.head())

    # Dataset 1
    X1 = df1.iloc[:, :2].values
    y1 = df1.iloc[:, 2].values

    # plt.scatter(X1[y1 == 1, 0], X1[y1 == 1, 1], label="+1")
    # plt.scatter(X1[y1 == -1, 0], X1[y1 == -1, 1], label="-1")

    # plt.xlabel("x1")
    # plt.ylabel("x2")
    # plt.title("Dataset-1")
    # plt.legend()
    # plt.show()

    # # Dataset 2
    # X2 = df2.iloc[:, :2].values
    # y2 = df2.iloc[:, 2].values
    # plt.scatter(X2[y2 == 1, 0], X2[y2 == 1, 1], label="+1")
    # plt.scatter(X2[y2 == -1, 0], X2[y2 == -1, 1], label="-1")
    # plt.xlabel("x1")
    # plt.ylabel("x2")
    # plt.title("Dataset-2")
    # plt.legend()
    # plt.show()
    

    degree = [1, 2, 3, 4, 5, 6]
    cvals = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for q in degree:
        X_poly = PolynomialFeatures(degree=q, include_bias=False).fit_transform(X1)
        mean_f1, std_f1 = [], []
        for C in cvals:
            model = LogisticRegression(C=C, max_iter=1000)
            scores = cross_val_score(model, X_poly, y1, cv=5, scoring="f1")
            mean_f1.append(scores.mean())
            std_f1.append(scores.std())
        plt.errorbar(cvals, mean_f1, yerr=std_f1, capsize=3, marker="o", label=f"q={q}")
    plt.xscale("log")
    plt.grid(True, alpha=0.3)
    plt.xlabel("C", fontsize=14)
    plt.ylabel("F1 score (5-fold CV)", fontsize=14)
    plt.xticks(cvals, [str(c) for c in cvals], rotation=45)
    plt.yticks(fontsize=12)
    plt.minorticks_off()  
    plt.title("Dataset 1: choosing q and C", fontsize=14)
    plt.legend()
    plt.show()
    
    # Prediction using q = 2 and c = 1.
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X1poly = poly.fit_transform(X1)
    model = LogisticRegression(C=1, max_iter=1000)
    model.fit(X1poly, y1)
    y1_pred = model.predict(X1poly)
    plt.scatter(X1[y1 == 1, 0], X1[y1 == 1, 1], s=40, alpha=0.4, label="true +1")
    plt.scatter(X1[y1 == -1, 0], X1[y1 == -1, 1], s=40, alpha=0.4, label="true -1")
    # predicted labels
    plt.scatter(X1[y1_pred == 1, 0], X1[y1_pred == 1, 1], marker="+", c="k", s=20, label="predicted +1")
    plt.scatter(X1[y1_pred == -1, 0], X1[y1_pred == -1, 1], marker=".", c="r", s=8, label="predicted -1")
    plt.xlabel("x1", fontsize=14); plt.ylabel("x2", fontsize=14)
    plt.title("Dataset 1: Logistic Regression (q=2, C=1)", fontsize=14)
    plt.legend(fontsize=10); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


def question_1_b():
    # Dataset 1
    X1 = df1.iloc[:, :2].values
    y1 = df1.iloc[:, 2].values
    meanf1, stdf1 = [], []
    kvals = list(range(1, 52, 2)) # 1-52 with skip = 2 
    
    for k in kvals:
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, X1, y1, cv=5, scoring="f1")
        meanf1.append(scores.mean())
        stdf1.append(scores.std())
    
    meanf1 = np.array(meanf1)
    stdf1 = np.array(stdf1)

    plt.figure(figsize=(8, 5))
    plt.plot(kvals, meanf1, marker="o", label="mean F1")
    plt.fill_between(kvals, meanf1 - stdf1, meanf1 + stdf1, alpha=0.2, label="± 1 std")
    plt.ylim(0.85, 1.0)
    plt.xlabel("k", fontsize=14)
    plt.ylabel("F1 score 5 fold CV", fontsize=14)
    plt.title("Dataset 1 - choosing k for kNN", fontsize=14)
    plt.legend(fontsize=11); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
    
    
    # train on k = 17
    def knn_final():
        knn = KNeighborsClassifier(n_neighbors=17).fit(X1, y1)
        y1_pred_knn = knn.predict(X1)
        
        plt.scatter(X1[y1 == 1, 0], X1[y1 == 1, 1], s=40, alpha=0.4, label="true +1")
        plt.scatter(X1[y1 == -1, 0], X1[y1 == -1, 1], s=40, alpha=0.4, label="true -1")
        plt.scatter(X1[y1_pred_knn == 1, 0], X1[y1_pred_knn == 1, 1], marker="+", c="k", s=20, label="predicted +1")
        plt.scatter(X1[y1_pred_knn == -1, 0], X1[y1_pred_knn == -1, 1], marker=".", c="r", s=8, label="predicted -1")
        plt.xlabel("x1", fontsize=14); plt.ylabel("x2", fontsize=14)
        plt.title("Dataset 1 - Knn Classifier with k = 17)", fontsize=14)
        plt.legend(fontsize=10); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()
    knn_final()
    
def question_1_c():
        # Dataset 1
    X1 = df1.iloc[:, :2].values
    y1 = df1.iloc[:, 2].values
    X_train, X_test, y_train, y_test = train_test_split(
    X1, y1, test_size=0.2, stratify=y1, random_state=0)
    # logistic regression (q=2, C=1)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    lr = LogisticRegression(C=1, max_iter=1000)
    lr.fit(poly.fit_transform(X_train), y_train)
    y_pred_lr = lr.predict(poly.transform(X_test))

    knn = KNeighborsClassifier(n_neighbors=17).fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    # baseline
    most_freq = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    random_clf = DummyClassifier(strategy="uniform", random_state=0).fit(X_train, y_train)
    y_pred_mf = most_freq.predict(X_test)
    y_pred_rand = random_clf.predict(X_test)

    for name, y_pred in [("Logistic Regression", y_pred_lr), ("kNN", y_pred_knn),
                            ("Most frequent", y_pred_mf), ("Random", y_pred_rand)]:
        print(name)
        print(confusion_matrix(y_test, y_pred))
        print()
        
    return knn, lr, poly, X_test, y_test, y_pred_mf, y_pred_rand
            

def question_1_d(knn, lr, poly, X_test, y_test, y_pred_mf, y_pred_rand):
    score_lr = lr.predict_proba(poly.transform(X_test))[:, 1]
    score_knn = knn.predict_proba(X_test)[:, 1]
    fpr_lr,tpr_lr, _ = roc_curve(y_test, score_lr)
    fpr_knn,tpr_knn, _ = roc_curve(y_test, score_knn)
    fpr_mf,tpr_mf, _ = roc_curve(y_test, y_pred_mf)
    fpr_rand,tpr_rand, _ = roc_curve(y_test, y_pred_rand)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_lr,tpr_lr,label=f"Logistic Regression(AUC={auc(fpr_lr, tpr_lr):.3f})")
    plt.plot(fpr_knn,tpr_knn,label=f"kNN(AUC={auc(fpr_knn, tpr_knn):.3f})")
    plt.plot([0, 1],[0, 1],"k--",label="Random classifier")

    plt.scatter(fpr_mf[1],tpr_mf[1],c="green", marker="s",label="Most frequent")
    plt.scatter(fpr_rand[1],tpr_rand[1],c="red",marker="^",label="Random baseline")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Dataset 1 - ROC curves")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    #question_1_a()
    #question_1_b()
    knn, lr, poly, Xtest, ytest, y_pred_mf, y_pred_rand = question_1_c()
    question_1_d(knn, lr, poly, Xtest, ytest, y_pred_mf, y_pred_rand)