from parameters import *;
from UtilityFunctions import *;
from DataItem import *;
from pybrain.structure import *;
from pybrain.datasets import *;
import numpy;
import random;
import heapq;
import copy;
from Score1 import *;
from QueueItem import *;
from pybrain.tools.shortcuts import buildNetwork;
from DataItem import *;
from pybrain.structure import *;
from pybrain.datasets import *;
from pybrain.supervised.trainers import *;
from copy import *;
from pybrain.utilities import percentError;


#from FeedForward import *;



#read input file to get a data list
dataList=readInput("./Dataset/Patterns_with_type.txt");   

#get the list of regular and individual items 
(dataListR,dataListI)=divide(dataList);



#creating the training pool
random.shuffle(dataList);
#sLen=beamSize;
#if len(dataList)<sLen:
	#sLen=len(dataList);
sLen=40
trainingpool=dataList[:sLen];



#the priority queue
pq=[];

#initial neural network with dummy queue item
nodeCount= len(dataList[0].data);
nn=buildNetwork(5,3,5,hiddenclass=SigmoidLayer,outclass=SigmoidLayer,bias=True);

dt=DataItem([0,0,0,1,1],[1,1],1,3,'I');
#print nn.activate(dt.data)

print nn.params	
print 'new ones'

#nn=trainNN(nn,dt);

nnTemp=deepcopyNN(nn)

trainer=BackpropTrainer(nnTemp,batchlearning=True);
output=nn.activate(dt.data); #Getting the output
print output;
#print output;
output=output[:dt.startIndex].tolist()+dt.target+output[dt.endIndex:].tolist();
print nnTemp.params;
print output;

ds = SupervisedDataSet(5, 5);
ds.addSample(dt.data,output);

trainer.setData(ds);

trainer.trainEpochs(1);
#nn=trainer.module;

print nnTemp.params;

#print nn.activate(dt.data)
#print nn.params



