import numpy as np
import matplotlib.pyplot as plt
import math
import neural_network_helper as nnh
from scipy.optimize import minimize

class Neural_Network:
    def __init__(self,Xtrain,Xval,yTrain,yVal,idTrain,idVal,netConfig):
        self.Xtrain=Xtrain
        self.Xval=Xval
        self.yTrain=yTrain
        self.yVal=yVal
        self.idTrain=idTrain
        self.idVal=idVal
        self.hiddenLayers=netConfig["hiddenLayers"]
        self.noOfHiddenLayerNodes=netConfig["nodes"]
        self.layerSizes=netConfig["layerSizes"]
        self.noOfLayers=self.hiddenLayers+2
        self.inputLayerSize=np.size(Xtrain,1)
        self.outputLayerSize=np.size(yTrain,1)
        self.lamba=2
        self.params=[]

    def initializeWeights(self):
        i=0
        while i <self.noOfLayers-1:
            self.params.append(nnh.randomInitializeWeights(self.layerSizes[i],self.layerSizes[i+1]))
            if i==0:
                self.initialNNParams=self.params[i].flatten()
            else:
                self.initialNNParams=np.hstack([self.initialNNParams,self.params[i].flatten()])
            i+=1

    def trainNetwork(self):
        self.train()


    def train(self):
        inArgs=(self.Xtrain,self.yTrain,self.lamba,self.layerSizes)
        res=minimize(nnh.gCostFunction,self.initialNNParams,method='CG',args=inArgs,options={"maxiter":100,"disp":True},jac=nnh.gGradient,)
        print(res.message)
        print(self.initialNNParams)
        self.trainedParams=res.x
        self.trainedParamsShaped=nnh.reshapeParams(self.trainedParams,self.layerSizes)
        
