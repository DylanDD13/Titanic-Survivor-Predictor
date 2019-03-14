import numpy as np
import math


def randomInitializeWeights(inLayer,outLayer):
    epsilonInit=(math.sqrt(6))/(math.sqrt(inLayer+outLayer))
    weights=np.random.rand(outLayer,1+inLayer)*2*epsilonInit-epsilonInit
    return weights

def sigmoid(z):
    return 1.0 / (1.0+np.exp(-z))
def reshapeParams(params,layers):
    i=0
    step=0
    reshaped=[]
    while i <len(layers)-1:
        if i == len(layers)-2:
            reshaped.append(np.reshape(params[step:None],(layers[i+1],layers[i]+1)))
        else:
            reshaped.append(np.reshape(params[step:(layers[i+1]*(layers[i]+1))+step],(layers[i+1],layers[i]+1)))
            step=step+(layers[i+1]*(layers[i]+1))
        i+=1
    return reshaped

def forwardProp(Theta,*args):
    X,y,l,layerSizes=args[0],args[1],args[2],args[3]
    thetas=reshapeParams(Theta,layerSizes)
    m=np.size(X,0)
    propVals=[]
    size=m
    prevInput=X
    deltas=[]
    regSum=0
    for index,theta in enumerate(thetas):
        size=np.size(prevInput,0)
        currInputs=np.hstack((np.ones((size,1)),prevInput))
        currZ=np.dot(currInputs,theta.T)
        currA=sigmoid(currZ)
        propVals.append({
            "input":currInputs,
            "Z":currZ,
            "A":currA
        })
        prevInput=currA
        thetaSq=np.square(theta[:,1:None])
        regSum+=np.sum(thetaSq.T)

    
    sumOfCost=-y.T.dot(np.log(prevInput))-((1-y).T.dot(np.log(1-prevInput)))
    J=np.multiply((1/m),np.sum(sumOfCost))
    regValue=np.multiply(l/(2*m),regSum)
    J=J+regValue
    prevGrad=np.zeros((1,1))
    thetasCopy=thetas.copy()
    for x in range(0,len(propVals)):
        currCorrection=propVals.pop()
        a=currCorrection["A"]
        inp=currCorrection["input"]
        if x==0:
            currGrad=np.subtract(a,y)
        else:
            theta=thetasCopy.pop()
            zSigGrad=np.multiply(a,np.subtract(1,a))
            currGrad=np.multiply(np.dot(prevGrad,theta[:,1:None]),zSigGrad)
        delta=np.dot(currGrad.T,inp)
        deltas.append(delta)
        prevGrad=currGrad
    deltas.reverse()
    retTheta=np.zeros((1,1))
    for index,grad in enumerate(deltas):
        unRegGrad=np.multiply(1/m,grad)
        currTheta=thetas[index]
        regVal=np.multiply(l/m,currTheta[:,1:None])
        size=np.size(regVal,0)
        regValBias=np.hstack((np.zeros((size,1)),regVal))
        regGrad=np.add(unRegGrad,regValBias)
        if index==0:
            retTheta=regGrad.flatten()
        else:
            retTheta=np.hstack([retTheta,regGrad.flatten()])
    return J,retTheta

def gCostFunction(initialNNParams,*args):
    cost,grad =forwardProp(initialNNParams,*args)
    return cost
def gGradient(initialNNParams,*args):
    cost,grad =forwardProp(initialNNParams,*args)
    return grad


def predict(X,y,thetas):
    prevInput=X
    for index,theta in enumerate(thetas):
         size=np.size(prevInput,0)
         currInputs=np.hstack((np.ones((size,1)),prevInput))
         currZ=np.dot(currInputs,theta.T)
         currA=sigmoid(currZ)
         prevInput=currA
    outPut=prevInput
    p=(outPut>0.5).astype(int)
    p=((p==(y.astype(int))).astype(int))
    accuracy=np.mean(p)
    print(accuracy)

def testPred(X,thetas):
    prevInput=X
    for index,theta in enumerate(thetas):
         size=np.size(prevInput,0)
         currInputs=np.hstack((np.ones((size,1)),prevInput))
         currZ=np.dot(currInputs,theta.T)
         currA=sigmoid(currZ)
         prevInput=currA
    outPut=prevInput
    p=(outPut>0.5).astype(int)
    return p

def kerasEval(pred,y):
    predSig=sigmoid(pred)
    print(print(np.c_[predSig, y]))
    p=(predSig>0.5).astype(int)
    #print(p)
    print(np.sum(p))
    print(np.size(p))
    p=((p==(y.astype(int))).astype(int))
    accuracy=np.mean(p)
    print(accuracy)

def seededWeights(inLayer,outLayer):
    W=np.zeros((outLayer,inLayer+1))
    print(W.shape)
    #W=np.reshape(np.sin(np.array(0:np.size(W)),W.shape)
    W=np.sin(np.arange(np.size(W))).reshape(W.shape)
    print(W.shape)
    return W

def checkNNGrad(l):
    hidden_layer_size=5
    input_layer_size=3
    num_labels=3
    m=5
    Theta1 = seededWeights(input_layer_size,hidden_layer_size)
    Theta2 = seededWeights(hidden_layer_size,num_labels)
    X=seededWeights(input_layer_size-1,m)
    y=np.add(1,np.remainder(np.arange(0,m).reshape((1,m)),num_labels).T)
    nnpParams=np.hstack([Theta1.flatten(),Theta2.flatten()])
    args=(X,y,l,input_layer_size,hidden_layer_size)
    costFunc= lambda x: forwardProp(x,args)
    j,params=costFunc(nnpParams)
    numGrad=checkNumericalGrad(costFunc,nnpParams)
    print(np.c_[params, numGrad])
    diff1=np.linalg.norm(np.subtract(numGrad,params))
    diff2=np.linalg.norm(np.add(numGrad,params))
    diff=diff1/diff2
    print(diff> 1e-4)
def checkNumericalGrad(j,theta):
    numGrad=np.zeros(theta.shape)
    perturb=np.zeros(theta.shape)
    e=1e-4
    for i in range(0,np.size(theta)):
        perturb[i]=e
        loss1=j(np.subtract(theta,perturb))[0]
        loss2=j(np.add(theta,perturb))[0]
        numGrad[i] = (loss2 - loss1) / (2*e)
        perturb[i] = 0
    return numGrad
