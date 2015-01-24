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
nn=createFeedForwardNetwork(nodeCount,nodeCount,nodeCount);

dataItem=QueueItem([],nn,0.0);
pq.append(dataItem);

#main loop
for i in range(trainSize):
	print i, ' ', len(pq),' ',pq[len(pq)-1].score;
	#print pq[len(pq)-1].dataList;
	
	
	
	tempPQ=[];
	#work for individual items in the pq
	for qi in pq:
		dl=qi.dataList;
		qnn=qi.nn;
		
		#debugging statement
		#print dl;
		
		for ti in trainingpool: 
			#creating copies of the training list and neural network
			newdl=deepcopy(qi.dataList);
			newNn=deepcopyNN(qi.nn);#might be redundant
			
			newdl.append(ti);
			newNn=trainNN(newNn,ti);
			
			accuracyR=testList(newNn,dataListR);
			accuracyI=testList(newNn,dataListI);
			
			scoreValue=score(accuracyR,accuracyI);
			newQI=QueueItem(newdl,newNn,scoreValue);
			
			#pushpop if max length reached
			if len(tempPQ)>=beamSize:
				heapq.heappushpop(tempPQ,newQI);
			#otherwise push only
			else:
				heapq.heappush(tempPQ,newQI);
			pq=tempPQ;
				
		#for tpqi in tempPQ:
		#	if len(pq)>=beamSize:
		#		heapq.heappushpop(pq,tpqi);
		#	else:
		#		heapq.heappush(pq,tpqi);

print pq[len(pq)-1].score;

#print dataItem.target;
#print dataItem.endIndex;

#output=nn.activate(dataItem.data); #Getting the output
#print output;

#dataItem=dataList[0];
#nn=trainNN(nn,dataItem);
#print dataItem.target[4];
#print dataItem.dataType;



#print testItem(nn,dataItem);

#print nn.activate(dataItem.data);
	
#print ds;
#print data[0];

