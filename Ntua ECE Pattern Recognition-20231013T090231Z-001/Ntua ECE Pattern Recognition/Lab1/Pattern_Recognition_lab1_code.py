# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 21:31:27 2021

@author: ktsin
"""

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import where
from numpy import meshgrid
from numpy import arange
from numpy import hstack
from sklearn.datasets import make_blobs
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from matplotlib import cm
from matplotlib.lines import Line2D
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import  VotingClassifier,BaggingClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier

"step1"
Y_train1=[]
X_train=[]
Y_test1=[]
X_test=[]
x1=[]
x2=[]

#read train txt
with open(r'D:\Ktsin_Files\Downloads\pr_lab1_data\train1.txt') as f:
    
    pk = f.readlines()
    
    
    for i in pk:
        w=[]
        wi= i.split()
        for i in wi:
            w.append(float(i))
        Y_train1.append(w[0])
        for i in range(1,len(w)):
            x1.append(w[i])
X_train=np.reshape(x1,(len(x1)//256,256))
Y_train =np.asarray(Y_train1)

    


    

#Read test.txt
with open(r'D:\Ktsin_Files\Downloads\pr_lab1_data\test.txt') as f:
    
    pk = f.readlines()
    for i in pk:
        w=[]
        wi= i.split()
        for i in wi:
            w.append(float(i))
        Y_test1.append(w[0])
        for i in range(1,len(w)):
            x2.append(w[i])
X_test=np.reshape(x2,(len(x2)//256,256))
Y_test=np.asarray(Y_test1)


"step 2"
X_train2 =[]   
X_train3=[]  
for j in range(len(X_train)):        
 X_train2.append( np.reshape(X_train[j],(16,16)))
 X_train3.append( X_train[j])

plt.imshow( X_train2[130],cmap='gray')
plt.show()



"step3"
plt.figure()

l = [None] * 10
for i in range(0,10):
 l[i] = X_train2[Y_train1.index(i)]
 #plt.imshow(l[i])
for i in range(0,10): 
     plt.subplot(5, 5, i+1)
     plt.imshow(l[i])
plt.show()  




"step 4 and step 5"
values=[]
for i in range(len(Y_train)):
    if Y_train[i]==0: 
        values.append(X_train[i,153])
mean=sum(values)/len(values)
print(mean)


var=np.var(values)
print(var)




"step 6 and step 7 and 8 "
k=[]
k2 =[]
# X_train[Y_train.index(i)]
Y_train = np.array(Y_train)
ap = np.where(Y_train==0.0)
k3=[]
for i in ap[0]:
    k3.append(X_train[i])
    k2.append(X_train2[i][9][9])
    k.append(X_train2[i])

mt =  np.mean(k2)

print(mt)
dia = np.var(k2)
print(dia)
k3 = np.array(k3)

k4=[]
k5=[]


for i in range(256):
    k4 = k3[:,i]
    
    k5.append(k4)

kpl=[0.0]*16
for i in k:
    kpl = i+kpl
    
  
mtola =kpl/len(k) 

diaola=[]
#digit 0 with the mean values
plt.imshow(mtola)
plt.show() 
 

for i in k5:
      diaola.append(np.var(i))
 
diaola = np.reshape(diaola,(16,16))
#digit 0 with the var values
plt.imshow(diaola)
plt.show()  
 


"step 9"
def diamesh(number):
    
    k3=[]
    k4=[]
    k5=[]
    k=[]
    diaola=[]
    ap = np.where(Y_train==number)
    
    for i in ap[0]:
      k3.append(X_train[i])
      k.append(X_train2[i])
    kpl=[0.0]*16
    k3 = np.array(k3)
    for i in k:
      kpl = i+kpl  
    #mtola =kpl/len(k)
    mtola=[]
    for i in range(256):
       k4 = k3[:,i]
       k5.append(k4)  
    for i in k5:
      mtola.append(np.mean(i))
      diaola.append(np.var(i))
    diaola = np.reshape(diaola,(16,16))  
    mtola = np.reshape(mtola,(16,16))  
    return mtola,diaola 


mt=[]
dias=[]    
plt.figure()
for ij in range(10):
    
    mt1,dias1 = diamesh(ij*1.0)
    mt.append(mt1)
    dias.append(dias1)
for ij in range(10): 
# print(ij)
 plt.subplot(5, 5, ij+1)
 plt.imshow(mt[ij])
plt.show()



"step 10"

k44 =[]

mt2 = np.reshape(mt,(10,256))

for i in mt2:
  k44.append(np.linalg.norm(X_test[100]-i))

             
kloth = np.argmin(k44)*1.0 
print(kloth)



"step 11"


prov=[]
for j in X_test:
    k42=[]
    for i in mt2:
       k42.append(np.linalg.norm(j-i))
    prov.append(np.argmin(k42)*1.0)

Y_test1 = np.array(Y_test)
prov1 = np.array(prov)
pososto = np.where(Y_test1==prov1,1,0)
print(pososto) 
sum1 = sum(pososto)/len(pososto) 
print(sum1) 


"step12"
class EuclideanClassifier(BaseEstimator, ClassifierMixin):  
    """Classify samples based on the Euclidean distance from the mean feature value"""

    def __init__(self, classes=None):
        # Χ_mean: numpy.ndarray of shape (n_classes, n_features)
        self.X_mean_ = None
        self.classes = None
        self.predictions=None


    def fit(self, X, y):
        """
        This should fit classifier. All the "work" should be done here.
        
        Calculates self.X_mean_ based on the mean 
        feature values in X for each class.
        
        self.X_mean_ becomes a numpy.ndarray of shape 
        (n_classes, n_features)
        
        fit always returns self.
        """
        ## X: (n_training examples)x(n_features)
        ## y is an 1d vector: 1x(number of training examples)
        # Check that X and y have correct shape
        (X,y) = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes = unique_labels(y)
        
        # Number of features
        (_, n_features) = X.shape
        # Initialization
        self.X_mean_ = np.zeros(( (len(self.classes)), n_features ))
        
        # Compute X_mean_
        for label in self.classes:
            label = int(label)
            labelX = X[y == label]
            labelX = np.mean(labelX, axis=0)
            self.X_mean_[label] = np.add(self.X_mean_[label], labelX) 
        
        return self
        
        


    def predict(self, X):
        """
        Make predictions for X based on the
        euclidean distance from self.X_mean_
        """
        ## X: (n_tests)x(n_features)
        
        # Input validation
        X = check_array(X)
        
        # Initialize y_predict
        (n_tests, _) = X.shape 
        y_predict = np.zeros((n_tests, 1))
        
        for test in range(len(X)):
            eucl_classify = []
            for train in self.X_mean_:
                # Euclidean distance
                
                eucl_classify.append(norm(train - X[test]) )
            # Predict the min euclidean distance for each test
            y_predict[test] = np.argmin(eucl_classify)
                
        return y_predict
    
    
    def score(self, X, y_truth):
        """
        Return accuracy score on the predictions
        for X based on ground truth y_truth
        """
        y_predict = self.predict(X)
        return ( np.sum([y_predict == y_truth]) ) / ( len(y_predict) )
    

Euclidis = EuclideanClassifier()
_ = Euclidis.fit(X_train, Y_train) 
 
score= Euclidis.score(X_test,Y_test)




"step 13"


"""
   Αρχικά συνενώνω τους πίνκαες train και test για να έχω 
   καλύτερη αξιοποίηση των δεδομένων
   
"""
X=np.concatenate((X_train,X_test),axis=0)
Y=np.concatenate((Y_train,Y_test),axis=0)


scores = cross_val_score(EuclideanClassifier(), X, Y,cv=KFold(n_splits=5, random_state=1,shuffle=True),scoring="accuracy")
print('scores on each of the 5 splits: ',scores)
print("E_CAverage Score: " + str(np.mean(scores)))
      


"13th_b step"
from sklearn.decomposition import PCA
pca= PCA(n_components=2)
X_train_PCA= pca.fit_transform(X_train)


#X0=np.mean(X_train[:,:153],axis=1)
#X1=np.mean(X_train[:,153:256],axis=1)
#Xfinal=np.vstack((X0,X1)).T
X0=X_train_PCA[:,0]
X1=X_train_PCA[:,1]
Xfinal=X_train_PCA


fig, ax = plt.subplots()
n_classes = 10
colors = ['red', 'yellow', 'blue', 'green', 'black', 'purple', 'orange', 'silver', 'grey', 'brown']
plot_step = 0.05
min1,max1=min(X0)-1,max(X0)+1
min2,max2=min(X1)-1,max(X1)+1
x1grid = arange(min1, max1, plot_step)
x2grid = arange(min2, max2,plot_step)
xx, yy = meshgrid(x1grid, x2grid)
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
grid = hstack((r1,r2))
model = EuclideanClassifier()
_=model.fit(Xfinal,Y_train)
predict=model.predict(grid)
zz = predict.reshape(xx.shape)
cmap = cm.get_cmap("rainbow",10)
out = ax.contourf(xx, yy, zz+0.5, alpha=0.8,  levels = np.append(np.unique(Y_train),max(Y_train)+1),colors=colors)

for i in range(10):
        ax.scatter(
            X0[Y_train==i], X1[Y_train==i],
            c=(colors[i]), label=i,
            s=80, alpha=0.8, edgecolors='k'
        )

    
ax.set_ylabel("feature 1")
ax.set_xlabel("feature 0")
ax.set_title("decision areas")
ax.legend()
plt.show()









"step 13th c"



def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
   

    
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores,fit_times,_ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        scoring = "accuracy",
        train_sizes=train_sizes,
        return_times=True
        
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    plt.legend(loc="best")
    
    

    return plt


title = "Learning Curves (Naive Bayes)"
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

estimator = EuclideanClassifier()
plot_learning_curve(estimator, title, X, Y,  ylim=(0.6, 1.01), cv=cv, n_jobs=-1)
plt.show()




"step 14"
def calculate_priors(X, y):
    labels=[]
    num_in_labels=[]
    labels, num_in_labels = np.unique(Y_train, return_counts=True)
    a_priori=num_in_labels/sum(num_in_labels)
    return a_priori



print("apriori",calculate_priors(X_train,Y_train))



"step 15"
class CustomNBClassifier(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""
    
    def __init__(self, use_unit_variance=False):
        self.use_unit_variance = use_unit_variance
        self.classes=None
        self.X_mean=None
        self.X_var=None
        self.a_priori = None
        


    def fit(self, X, Y):
        ## X: (n_training examples)x(n_features)
        ## y is an 1d vector: 1x(number of training examples)
        
       # Check that X and y have correct shape
        (X,Y) = check_X_y(X, Y)
        # Store the classes seen during fit
        self.classes = unique_labels(Y)
        
        # Number of features
        (_, n_features) = X_train.shape
        # Initialization
        self.X_mean = np.zeros(( (len(self.classes)), n_features ))
        self.X_var=np.zeros(( (len(self.classes)), n_features ))
        # Compute X_mean_ and X_var
        for label in self.classes:
            label = int(label)
            labelX = X[Y == label]
            labelM = np.mean(labelX, axis=0)
            labelV=np.var(labelX,axis=0)
            self.X_mean[label] = np.add(self.X_mean[label], labelM) 
            self.X_var[label]=np.add(self.X_var[label],labelV)
        
        
        
        
       #Compute a_priori possibilities
        labels=[]
        num_in_labels=[]
        labels, num_in_labels = np.unique(Y, return_counts=True)
        self.a_priori=num_in_labels/sum(num_in_labels)
        return self
        
       


    def predict(self, X):
        
        
        # Initialize y_predict
        (n_tests, _) = X.shape 
        y_predict = np.zeros((n_tests, 1))
        
        #predict
        for test in range(len(X)):
            bayes_classify=[]
          
            
            for train1,train2,train3 in zip(range(len(self.X_mean)),range(len(self.X_var)),range(len(self.a_priori))):
           
                divide=np.true_divide(((X[test]-self.X_mean[train1])**2),(2*self.X_var[train2] + 10**(-5)))
                exp=np.exp(-divide)
                pithanofaneia=np.true_divide(1,np.sqrt(2*np.pi*self.X_var[train2])+ 10**(-5))* exp
                
                bayes=self.a_priori[train3]*np.prod(pithanofaneia)
                
                bayes_classify.append(bayes)
            
              
            y_predict[test]=np.argmax(bayes_classify)
        
        
        
        return y_predict

    def score(self, X, y_truth):
        y_predict = self.predict(X)
        score=np.sum(np.where(y_predict.T == y_truth,1,0))
        return score/len(y_predict)
        



clf = CustomNBClassifier()
clf.fit(X_train, Y_train)
y_predict=clf.predict(X_test)


"ypologizw to score"
# 5-Fold Cross Validation
print("Current Score: " + str(clf.score(X_test, Y_test)))
scores1 = cross_val_score(CustomNBClassifier(), X, Y, 
                         cv=KFold(n_splits=5, random_state=1,shuffle=True), 
                         scoring="accuracy")
print('CustomNBClassifier scores on each of the 5 splits: ',scores1)
print("CustomNBClassifier CV Average Score: " + str(np.mean(scores1)))



"15c sygkrish ylopoihsewn"

# 5-Fold Cross Validation
scores2 = cross_val_score(GaussianNB(), X, Y, 
                         cv=KFold(n_splits=5, random_state=1,shuffle=True), 
                         scoring="accuracy")
print('sklearn NBGaussian scores on each of the 5 splits: ',scores2)
print("sklearn NBGaussian CV Average Score: " + str(np.mean(scores2)))



"step 16"


class CustomNBClassifiervar(BaseEstimator, ClassifierMixin):
    """Custom implementation Naive Bayes classifier"""
    
    def __init__(self, use_unit_variance=False):
        self.use_unit_variance = use_unit_variance
        #self.classes=None
        self.X_mean=None
        #self.X_var=None
        #self.a_priori =None
        


    def fit(self, X, Y):
        ## X: (n_training examples)x(n_features)
        ## y is an 1d vector: 1x(number of training examples)
        
       # Check that X and y have correct shape
        (X,Y) = check_X_y(X, Y)
        # Store the classes seen during fit
        self.classes = unique_labels(Y)
        
        # Number of features
        (_, n_features) = X_train.shape
        # Initialization
        self.X_mean = np.zeros(( (len(self.classes)), n_features ))
        self.X_var=np.ones(( (len(self.classes)), n_features ))
        # Compute X_mean
        for label in self.classes:
            label = int(label)
            labelX = X[Y == label]
            labelM = np.mean(labelX, axis=0)
            self.X_mean[label] = np.add(self.X_mean[label], labelM) 
            
        
        
        
        
       #Compute a_priori possibilities
        labels=[]
        num_in_labels=[]
        labels, num_in_labels = np.unique(Y, return_counts=True)
        self.a_priori=num_in_labels/sum(num_in_labels)
        return self
        
       


    def predict(self, X):
        #print("length",len(X))
        #X = check_array(X)
        
        # Initialize y_predict
        (n_tests, _) = X.shape 
        y_predict = np.zeros((n_tests, 1))
        
        #predict
        for test in range(len(X)):
            bayes_classify=[]
          
            
            for train1,train2,train3 in zip(range(len(self.X_mean)),range(len(self.X_var)),range(len(self.a_priori))):
            #for train1,train2 in zip(self.X_mean,self.X_var):
                divide=np.true_divide(((X[test]-self.X_mean[train1])**2),(2*self.X_var[train2] + 10**(-5)))
                exp=np.exp(-divide)
                pithanofaneia=np.true_divide(1,np.sqrt(2*np.pi*self.X_var[train2])+ 10**(-5))* exp
                #print("product",np.prod(pithanofaneia))
                bayes=self.a_priori[train3]*np.prod(pithanofaneia)
                
                bayes_classify.append(bayes)
            
              
            y_predict[test]=np.argmax(bayes_classify)
        
        
        
        return y_predict

    def score(self, X, y_truth):
        y_predict = self.predict(X)
        score=np.sum(np.where(y_predict.T == y_truth,1,0))
        return score/len(y_predict)
    
    
    
clf_var = CustomNBClassifiervar()
clf_var.fit(X_train, Y_train)
y_predict = clf_var.predict(X_test)
"ypologizw to score"
# 5-Fold Cross Validation
print("Current Score: " + str(clf_var.score(X_test, Y_test)))
scores1 = cross_val_score(CustomNBClassifiervar(), X, Y, 
                         cv=KFold(n_splits=5, random_state=1,shuffle=True), 
                         scoring="accuracy")
print('CustomNBClassifiervar scores on each of the 5 splits: ',scores1)
print("CustomNBClassifiervar CV Average Score: " + str(np.mean(scores1)))    




"step 17"

clfs=np.array([KNeighborsClassifier(n_neighbors=1),KNeighborsClassifier(n_neighbors=3),KNeighborsClassifier(n_neighbors=5),KNeighborsClassifier(n_neighbors=15)])



for clf in clfs:
    scores= cross_val_score(clf, X, Y, cv=KFold(n_splits=5,random_state=1,shuffle=True), scoring='accuracy')
    print("mean_scores",scores.mean(),"scores_std",scores.std())
    
    
"search of the best parameters for the SVC classifier/ tuning of the parameters"
"create the parameter grid"
parameters = {'kernel':('poly','linear','rbf'), 'C':[1, 10,100],"gamma": ['scale',0.1, 1, 10]}

grid=GridSearchCV(estimator=SVC(), param_grid={'kernel':[ 'poly','linear','rbf'], 'C':[1, 10,100] , 'gamma':[0.01,0.001]})
grid.fit(X_train,Y_train)
print("best_score",grid.best_score_,"best parameters",grid.best_params_)
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid.cv_results_['params']):
    print( mean, std * 2, params)


"step 18_a"

clfs = np.array(
        [KNeighborsClassifier(n_neighbors=1),KNeighborsClassifier(n_neighbors=5),SVC(kernel='poly',C=1,gamma='scale',probability=True),SVC(kernel='rbf',gamma='scale',probability=True),
        SVC(kernel='linear',probability=True) ])

clfs[0].fit(X_train, Y_train)
y_pre0 = clfs[0].predict(X_test)
score0 = clfs[0].score(X_test, Y_test)
print("clf0 accuracy score: " + str(score0))

clfs[1].fit(X_train, Y_train)
y_pre1 = clfs[1].predict(X_test)
score1 = clfs[1].score(X_test, Y_test)
print("clf1 accuracy score: " + str(score1))

clfs[2].fit(X_train, Y_train)
y_pre2 = clfs[2].predict(X_test)
score2 = clfs[2].score(X_test, Y_test)
print("clf2 accuracy score: " + str(score2))

clfs[3].fit(X_train, Y_train)
y_pre3 = clfs[3].predict(X_test)
score3 = clfs[3].score(X_test, Y_test)
print("clf3 accuracy score: " + str(score3))

clfs[4].fit(X_train, Y_train)
y_pre4 = clfs[4].predict(X_test)
score4 = clfs[4].score(X_test, Y_test)
print("clf4 accuracy score: " + str(score4))

for i in range(5):
    plot_confusion_matrix(clfs[i], X_test, Y_test) 
    plt.show()
'Hard Voting'
eclf1 = VotingClassifier(estimators=[('1-NN', clfs[0]), ('5-NN', clfs[1]), ('Svc-poly', clfs[2]),('SVC-rbf',clfs[3]),('SVC-linear',clfs[4])], voting='hard')
eclf1=eclf1.fit(X_train, Y_train)
y_hard=eclf1.predict(X_test)
print("Hard VotingClassifier score: " , eclf1.score(X_test, Y_test))

'Soft Voting only classifiers that calculate probs'
eclf2 = VotingClassifier(estimators=[('1-NN', clfs[0]),('SVC-rbf',clfs[3]),('SVC-linear',clfs[4])], voting='soft')
eclf2=eclf2.fit(X_train, Y_train)
y_soft=eclf2.predict(X_test)
print("Soft VotingClassifier score: " , eclf2.score(X_test, Y_test))






"step 18_b"

models=np.array([KNeighborsClassifier(n_neighbors=5),SVC(kernel='poly',gamma='scale',C=1,probability=True),DecisionTreeClassifier()])
for i in range(3):
    clf_bag = BaggingClassifier(base_estimator=models[i],n_estimators=10, random_state=0).fit(X_train, Y_train)
    predict_bag=clf_bag.predict(X_test)
    print("BaggingClassifier score :",i, clf_bag.score(X_test,Y_test))

Tree=DecisionTreeClassifier()
Tree=Tree.fit(X_train,Y_train)
predictct=Tree.predict(X_test)
print("Random Forest accuracy score",Tree.score(X_test,Y_test))

