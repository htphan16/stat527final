import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import pylab as pl

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.decomposition import PCA
from pca import pca
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("divorce.csv",sep=';')

X = df.drop(['Class'], axis=1)
y = df['Class']

# dtc_scores = []
# cvs_scores = []


#-----------------------------
# Decision Tree
for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    dtc = DecisionTreeClassifier()
    dtc_fit = dtc.fit(X_train, y_train)
    dtc_score = dtc_fit.score(X_test, y_test)
    # cvs = cross_val_score(dtc_fit, X_test, y_test, cv = 5)
    # cvs_scores.append(np.mean(cvs))
    dtc_scores.append(dtc_score)

print(np.mean(dtc_scores))
print(np.mean(cvs_scores))

# DTC Accuracy Scores
# print(np.mean([0.9679, 0.9694, 0.9688, 0.9741, 0.97, 0.975, 0.9709,0.9738, 0.9679, 0.9726]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
dtc = DecisionTreeClassifier(random_state=1)
dtc_fit = dtc.fit(X_train, y_train)
dtc_score = dtc_fit.score(X_test, y_test)

text_representation = tree.export_text(dtc)
print(text_representation)

with open("decistion_tree.log", "w") as fout:
    fout.write(text_representation)

fig = plt.figure(figsize=(30,30))
_ = tree.plot_tree(dtc,filled=True)
fig.savefig("decision_tree_6.png")
plt.clf()

# y_pred = dtc.predict(X_test)

# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
# dta = accuracy_score(y_pred,y_test)

# print('Decision tree accuracy: {:.2f}%'.format(dta*100))

# scoreListDT = []
# for i in range(2,50):
#     DTclassifier = DecisionTreeClassifier(max_leaf_nodes=i)
#     DTclassifier.fit(X_train, y_train)
#     scoreListDT.append(DTclassifier.score(X_test, y_test))
    
# plt.plot(range(2,50), scoreListDT)
# plt.xticks(np.arange(2,50,2))
# plt.xlabel("Leaf")
# plt.ylabel("Score")
# plt.savefig('dtc_plot.png')
# # plt.show()
# DTAccMax = max(scoreListDT)
# print("DT Acc Max: {:.2f}%".format(DTAccMax*100))



#-----------------------------
cvs_scores_rf = []
rfc_scores = []

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
rfc=RandomForestClassifier(n_estimators=200)
rfc_fit = rfc.fit(X_train, y_train)
estimator = rfc.estimators_[5]
fig = plt.figure(figsize=(15, 10))
_ = tree.plot_tree(rfc.estimators_[0],
          class_names=True, 
          filled=True, impurity=True, 
          rounded=True)
fig.savefig("random_tree_5.png")
plt.clf()

for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    rfc=RandomForestClassifier(n_estimators=10)
    rfc_fit = rfc.fit(X_train, y_train)
    rfc_score = rfc_fit.score(X_test, y_test)
    # cvs_rf = cross_val_score(rfc_fit, X_test, y_test, cv = 5)
    # cvs_scores_rf.append(np.mean(cvs_rf))
    rfc_scores.append(rfc_score)

print(np.mean(rfc_scores))


# RTC Accuracy Scores
# print(np.mean([0.9765, 0.9797, 0.9753, 0.9774, 0.9821, 0.9729, 0.9726, 0.9774, 0.9732, 0.9782]))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# rfc=RandomForestClassifier(n_estimators=1000)
# rfc_fit = rfc.fit(X_train, y_train)
# rfc_score = rfc_fit.score(X_test, y_test)

# y_pred_rfc = rfc.predict(X_test)
# print(classification_report(y_test, y_pred_rfc))
# print(confusion_matrix(y_test, y_pred_rfc))

# rfc_score = rfc_fit.score(X_test, y_test)
# print('Random forest accuracy: {:.2f}%'.format(rfc_score*100))

# importances = rfc.feature_importances_
# std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
# feature_names = [f"feature {i}" for i in range(X.shape[1])]
# forest_importances = pd.Series(importances, index=feature_names)

# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=std, ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()
# plt.show()

plt.figure(figsize=(5,5))
sorted_idx = rfc.feature_importances_.argsort()
plt.barh(df.columns[sorted_idx][0:10], rfc.feature_importances_[sorted_idx][0:10])
plt.xlabel("Random Forest Feature Importance")
plt.savefig("rf_10variables_6.png")
plt.clf()
# scoreListRF = []
# for i in range(2,55):
#     RFclassifier = RandomForestClassifier(n_estimators = 200, random_state = 1, max_leaf_nodes=i)
#     RFclassifier.fit(X_train, y_train)
#     scoreListRF.append(RFclassifier.score(X_test, y_test))

# plt.plot(range(2,55), scoreListRF)
# plt.xticks(np.arange(2,55,5))
# plt.xlabel("RF Value")
# plt.ylabel("Score")
# plt.savefig('randForest.png')
# # plt.show()
# RFAccMax = max(scoreListRF)
# print("RF Acc Max: {:.2f}%".format(RFAccMax*100))

# plt.clf()

# print(X_train.iloc[:,0])
# print(X_train.iloc[:,1])
# plt.scatter(X_train.iloc[:,0], X_train.iloc[:,1])

# X_train, X_test = X_train.values, X_test.values

# LRclassifier = LogisticRegression(solver='liblinear', max_iter=5000)
# LRclassifier.fit(X_train, y_train)

# y_pred = LRclassifier.predict(X_test)

# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
# LRAcc = accuracy_score(y_pred,y_test)

# print('Logistic regression accuracy: {:.2f}%'.format(LRAcc*100))

#-----------------------------
# Logistic Regression
cvs_scores_lrc = []
lrc_scores = []

for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    LRclassifier = LogisticRegression(solver="lbfgs")
    lrc_fit = LRclassifier.fit(X_train, y_train)
    lrc_score = lrc_fit.score(X_test, y_test)
    cvs_lrc = cross_val_score(lrc_fit, X_test, y_test, cv = 5)
    cvs_scores_lrc.append(np.mean(cvs_lrc))
    lrc_scores.append(lrc_score)

print(np.mean(lrc_scores))
print(np.mean(cvs_scores_lrc))


# Logistic using solver liblinear accuracy scores
# print(np.mean([0.9730, 0.9759, 0.9715, 0.9747, 0.9771, 0.9747, 0.9735, 0.9779, 0.9753, 0.9698]))

# Logistic using solver lbfgs accuracy scores
# print(np.mean([0.9762, 0.9765, 0.9818, 0.9788, 0.9821, 0.9774, 0.9832, 0.9824, 0.9803, 0.9759]))

# KNC Accuracy scores

# print(np.mean([0.9762, 0.9771, 0.9765, 0.9776, 0.9794, 0.9779, 0.9762, 0.9791, 0.9809, 0.9726]))

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# linear_svc=SVC(kernel='linear')
# linear_svc_fit = linear_svc.fit(X_train, y_train)


#-----------------------------
# Linear SVC

linear_svc_scores = []
cvs_scores_linear_svc = []

for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    linear_svc=SVC(kernel='linear')
    linear_svc_fit = linear_svc.fit(X_train, y_train)
    linear_svc_score = linear_svc_fit.score(X_test, y_test)
    cvs_linear_svc = cross_val_score(linear_svc_fit, X_test, y_test, cv = 5)
    cvs_scores_linear_svc.append(np.mean(cvs_linear_svc))
    linear_svc_scores.append(linear_svc_score)

print(np.mean(linear_svc_scores))

# Linear SVC accuracy scores
# print(np.mean([0.9835, 0.9832, 0.9824, 0.9797, 0.9794, 0.9821, 0.985, 0.9803, 0.9829, 0.9838]))

# y_pred_linear_svc = linear_svc.predict(X_test)
# print(classification_report(y_test, y_pred_linear_svc))
# print(confusion_matrix(y_test, y_pred_linear_svc))

# linear_svc_score = linear_svc_fit.score(X_test, y_test)
# print('Linear SVC accuracy: {:.2f}%'.format(linear_svc_score*100))

# support_vector_indices = linear_svc.support_
# print(support_vector_indices)

# support_vectors_per_class = linear_svc.n_support_
# print(support_vectors_per_class)

# support_vectors = linear_svc.support_vectors_
# print(support_vectors[:,0])
# print(support_vectors[:,1])

# print(support_vectors)

# print(X_train.values[:,0])
# print(X_train.values[:,1])

# ax.scatter(X_train[:,0], X_train[:,1], c=y_train)

# Z = linear_svc.decision_function(xy).reshape(XX.shape)
# ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
#            linestyles=['--', '-', '--'])

# ax.scatter(linear_svc.support_vectors_[:, 0], linear_svc.support_vectors_[:, 1], s=100,
#            linewidth=1, facecolors='none', edgecolors='k')
# plt.show()
# plt.plot(support_vectors[:,0], support_vectors[:,1])

# plt.title('Linearly separable data with support vectors')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.show()


# Linear SVC with PCA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
pca = PCA(n_components=2).fit(X_train)
pca_2d = pca.transform(X_train)

# print(pca_2d)
# pca = PCA(n_components=2).fit(X)
# pca_2d = pca.transform(X)
print(pca.explained_variance_ratio_)
# print(pca.components_)

# Initialize
# model = pca()
# # Fit transform
# out = model.fit_transform(X_train)
# print(out['topfeat'])
# model.plot()

# my_list = pca.components_[0]
# max_item = max(my_list)
# print([index for index, item in enumerate(my_list) if item == max_item])

# my_list2 = pca.components_[1]
# max_item2 = max(my_list2)
# print([index2 for index2, item2 in enumerate(my_list2) if item2 == max_item2])

svmClassifier_2d = SVC(kernel='linear').fit(pca_2d, y_train)
# pl.xlim(-10, 10)
# pl.ylim(-10, 10)
for i in range(0, pca_2d.shape[0]):
    if y_train.values[i] == 0:
        c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r', marker='+')
    elif y_train.values[i] == 1:
        c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g', marker='o')
pl.legend([c1, c2], ['Not Divorced', 'Divorced'],loc='upper right') 

# pca = PCA()
# x_new = pca.fit_transform(X)

# def myplot(score,coeff,labels=None):
#     xs = score[:,0]
#     ys = score[:,1]
#     n = coeff.shape[0]
#     scalex = 1.0/(xs.max() - xs.min())
#     scaley = 1.0/(ys.max() - ys.min())
#     plt.scatter(xs * scalex,ys * scaley, c = y_train)
#     for i in range(0,3):
#         plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
#         if labels is None:
#             plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
#         else:
#             plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
# plt.xlim(-1,1)
# plt.ylim(-1,1)
# plt.xlabel("PC{}".format(1))
# plt.ylabel("PC{}".format(2))
# plt.grid()

# # #Call the function. Use only the 2 PCs.
# myplot(pca_2d[:,0:2],np.transpose(pca.components_[0:2, :]))
# plt.show()

# # Accuracy
# for i in range(0,100):
#     pca_2d_test = pca.transform(X_test)
#     y_pred_linear_svc = svmClassifier_2d.predict(pca_2d_test)
#     linear_svc_score = svmClassifier_2d.score(pca_2d_test, y_test)
#     linear_svc_scores.append(linear_svc_score)
# print(classification_report(y_test, y_pred_linear_svc))
# print(confusion_matrix(y_test, y_pred_linear_svc))
# print(np.mean(linear_svc_scores))
# linear_svc_score = svmClassifier_2d.score(pca_2d_test, y_test)
# print('Linear SVC accuracy: {:.2f}%'.format(linear_svc_score*100))

# cvs_pca_svm = cross_val_score(svmClassifier_2d, pca_2d_test, y_test, cv = 5)
# print(cvs_pca_svm)

# # Plot SVC
x_min, x_max = pca_2d[:, 0].min() - 1, pca_2d[:,0].max() + 1
y_min, y_max = pca_2d[:, 1].min() - 1, pca_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .01),   np.arange(y_min, y_max, .01))
Z = svmClassifier_2d.predict(np.c_[xx.ravel(),  yy.ravel()])
Z = Z.reshape(xx.shape)

pl.contour(xx, yy, Z, colors='k', alpha=0.5)
pl.title('Support Vector Classifier')
# pl.axis('off')
pl.savefig("svc_p2.png")

plt.clf()

#-----------------------------
# K-neighbors classifier

knc_scores = []

for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    knc = KNeighborsClassifier(n_neighbors=5)
    knc_fit = knc.fit(X_train, y_train)
    knc_score = knc_fit.score(X_test, y_test)
    knc_scores.append(knc_score)

print(np.mean(knc_scores))

# print(np.mean([0.9788, 0.9768, 0.9774, 0.9759, 0.9774, 0.9788, 0.9771, 0.9762, 0.9803, 0.9774]))
# y_pred_knc = knc.predict(X_test)
# print(classification_report(y_test, y_pred_knc))
# print(confusion_matrix(y_test, y_pred_knc))

# knc_score = knc_fit.score(X_test, y_test)
# print('K-neighbors classifier accuracy: {:.2f}%'.format(knc_score*100))

# plt.clf()

# scoreListknn = []
# for i in range(1,30):
#     KNclassifier = KNeighborsClassifier(n_neighbors = i)
#     KNclassifier.fit(X_train, y_train)
#     scoreListknn.append(KNclassifier.score(X_test, y_test))
    
# plt.plot(range(1,30), scoreListknn)
# plt.xticks(np.arange(1,30,1))
# plt.xlabel("K value")
# plt.ylabel("Score")
# plt.savefig('knc_plot.png')
# plt.show()
# KNAccMax = max(scoreListknn)
# print("KNN Acc Max: {:.2f}%".format(KNAccMax*100))
