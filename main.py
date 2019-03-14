import data_preprocessing as dp
import matplotlib.pyplot as plt
import feature_scaling as fs
import numpy as np
import neural_network as nn
import neural_network_helper as nnh
import pandas as pd
from keras import models
from keras import layers
from keras import regularizers

df=dp.processFile('train.csv')
dfTest=dp.processFile('test.csv')
df=dp.processData(df)
dfTest=dp.processData(dfTest)

initial_dataset=df.values
testDataset=dfTest.values
##Randonly shuffles dataset before separation into training and cros validation to prevent any overfitting based on sorting
np.random.shuffle(initial_dataset)
##
dataset_length=np.size(initial_dataset,0)
#Splitting dataset 80/20 into training and cross validation respectively
training_set=initial_dataset[0:round(dataset_length*0.8),:]
cross_valid_set=initial_dataset[(round(dataset_length*0.8))+1:,:]
##
##Removing the ID and expected values from the training and CV and assigning to to y
training_set=initial_dataset
print(training_set.shape)
print(cross_valid_set.shape)
y_train=training_set[:,1:2]
y_val=cross_valid_set[:,1:2]
id_train=training_set[:,0:1]
id_val=cross_valid_set[:,0:1]
id_test=testDataset[:,0:1]
training_set=np.delete(training_set, np.s_[0:2], axis=1)
cross_valid_set=np.delete(cross_valid_set, np.s_[0:2], axis=1)
testDataset=np.delete(testDataset,np.s_[0:1],axis=1)

##
##Normaling the training and CV datasets
#np.random.rand(891,8)
X_train=fs.normalize(training_set.astype(float))
X_val=fs.normalize(cross_valid_set.astype(float))
X_test=fs.normalize(testDataset.astype(float))
##

nnConfig={
"hiddenLayers" :3,
"nodes":400,
"layerSizes": [np.size(X_train,1),100,50,25,np.size(y_train,1)]

}



#model = models.Sequential()
#model.add(layers.Dense(512, activation='relu', input_shape=(8, )))
#model.add(layers.Dense(128, activation='relu'))
#model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(1, activation='sigmoid'))

#model.compile(optimizer='Adam',
              #loss='binary_crossentropy',
              #metrics=['accuracy'])
#model.fit(X_train, y_train, epochs=150, batch_size=16)

#Y_pred = model.predict(X_val)
#score=model.evaluate(X_val,y_val,verbose=0)
#nnh.kerasEval(Y_pred,y_val)
#print(score)

neuralNatwork = nn.Neural_Network(X_train,X_val,y_train,y_val,id_train,id_val,nnConfig)
neuralNatwork.initializeWeights()

neuralNatwork.trainNetwork()
print(np.mean((neuralNatwork.initialNNParams==neuralNatwork.trainedParams).astype(int)))

nnh.predict(neuralNatwork.Xtrain,neuralNatwork.yTrain,neuralNatwork.trainedParamsShaped)
nnh.predict(neuralNatwork.Xval,neuralNatwork.yVal,neuralNatwork.trainedParamsShaped)
testPred=nnh.testPred(X_test,neuralNatwork.trainedParamsShaped)
print(id_test.shape)
testRes=pd.DataFrame(data=np.column_stack((id_test,testPred)),columns=['PassengerId','Survived'])
testRes=testRes.astype(int)


#testRes=testRes.drop(testRes.columns[0],axis=1)
print(testRes)
testRes.to_csv("submissions.csv",index=False)
#X=fs.normalize(df.values)
