import numpy as np
import pandas as pd
from numpy.linalg import inv
 #For reproducibility
np.random.seed(123)

# Read the data
data = pd.read_csv('Dataset/train_all.csv', index_col='id')
#test = pd.read_csv('Dataset/test_all.csv', index_col='id')
# Make the class column contain 0 or 1 only
data['class'] = data['class'] - 1
# Get the number of features
test , train = np.split(data.sample(frac=1),[int(.2*len(data))],)

train_class_1 = train[train['class']==0]
train_class_2 = train[train['class']==1]
predictors_1 = train_class_1.drop(['class'], axis=1)
predictors_2 = train_class_2.drop(['class'], axis=1)
test_x=  test.drop(['class'], axis=1)
test_y = test.drop(['B','G','R'], axis=1)
x_test = np.matrix(test_x)
y_test = np.matrix(test_y)

cov_1 = predictors_1.cov()
cov_2 =  predictors_2.cov()

sigma_1 = np.matrix(cov_1)
sigma_2 = np.matrix(cov_2)
sigma_1_inv = inv(sigma_1)
sigma_2_inv = inv(sigma_2)
dif_sigma = sigma_2_inv - sigma_1_inv
tr_1 =  np.matrix(predictors_1)
tr_2 = np.matrix(predictors_2)
meu_1= [tr_1[0].mean(),tr_1[1].mean(),tr_1[2].mean()]
meu_2 = [tr_2[0].mean(),tr_2[1].mean(),tr_2[2].mean()]
meu_1_t = np.transpose(meu_1)
meu_2_t = np.transpose(meu_2)

s_M_1 =  np.dot(sigma_1_inv,meu_1)
s_M_2 =  np.dot(sigma_2_inv,meu_2)

dif_eq= s_M_2 - s_M_1
S_M_1 = np.dot(np.dot(meu_1_t,sigma_1_inv),meu_1)
S_M_2 = np.dot(np.dot(meu_2_t,sigma_2_inv),meu_2)
third_term = S_M_2 - S_M_1
ratio_of_sigma = np.divide(np.linalg.det(sigma_1), np.linalg.det(sigma_2))
ln = np.log(ratio_of_sigma)

size_1 = predictors_1.__sizeof__()
size_2 = predictors_2.__sizeof__()
size = size_1+ size_2
ratio_of_class1 = size_1 /  size
ratio_of_class2 = size_2 /  size
th= ratio_of_class2 / ratio_of_class1
predict = []

secend_term = th + ln

for i  in range(len(x_test)):
                       x= x_test[0]
                       x_T = np.transpose(x)
                       quadritic = np.dot(np.dot(x,dif_sigma),x_T)
                       liner_model = np.dot(np.dot(2,x),dif_eq.reshape(3,1))
                       first_term =(quadritic - liner_model) + third_term

                       if ( first_term < secend_term) :
                             predict.append(1)
                       else:
                             predict.append( 0)



