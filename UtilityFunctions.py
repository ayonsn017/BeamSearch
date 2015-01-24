from DataItem import *;
from pybrain.structure import *;
from pybrain.datasets import *;
from pybrain.supervised.trainers import *;
from copy import *;
from pybrain.tools.shortcuts import buildNetwork;

#function to read input file
def readInput(fileName):
	f=open(fileName,'r');
	a=f.readline();
	#print len(a);
	attributes=a.split();
	attributes=attributes[1:];#Getting the list of attributes
	#print len(attributes); #total attribute length will be half of this


	f.readline();

	dataList=[];
	target=[];
	for line in f:
		temp=line.split();
	
		if(len(temp) ==0):
			continue; 
	
		temp2=temp[58:];
		#finding the start index of the target
		#print temp2;
	

	
		startIndex=temp2.index('0');
		tempStartIndex=len(temp2);
		if '1' in temp[58:]:
			tempStartIndex=temp2.index('1');
	

		if tempStartIndex<startIndex:
			startIndex=tempStartIndex;
		#print temp2[::-1];	
	
		endIndex=len(temp2)-temp2[::-1].index('0');
		tempEndIndex=0;
		if '1' in temp[58:]:
			tempEndIndex=len(temp2)-temp2[::-1].index('1');
	
		#finding the end index of the target
		if tempEndIndex>endIndex:
			endIndex=tempEndIndex
	
		tempData=DataItem(temp[6:57],temp2[startIndex:endIndex],startIndex,endIndex,temp[0]);
		dataList.append(tempData);
		
	return dataList;
	
#divide dataItems into two lists
def divide(dataList):
	dataListR=[];
	dataListI=[];
	for dataItem in dataList:
		if dataItem.dataType=='R':
			dataListR.append(dataItem);
		else:
			dataListI.append(dataItem);
			
	return (dataListR,dataListI);
	
#Function to create a feedforward neural network	
def createFeedForwardNetwork(inputNodeCount, hiddenNodeCount, outputNodeCount):
	nn=buildNetwork(inputNodeCount, hiddenNodeCount, outputNodeCount,hiddenclass=SigmoidLayer,outclass=SigmoidLayer,bias=True);
	return nn;
	
def deepcopyNN(nnOriginal):
	
	nn=buildNetwork(nnOriginal['in'].dim,nnOriginal['hidden0'].dim,nnOriginal['out'].dim,hiddenclass=SigmoidLayer,outclass=SigmoidLayer,bias=True);
	nn._setParameters(nnOriginal.params);
	
	return nn;


#trains the neural network for only one epoch. does not update all nodes from hidden to output layer
def trainNN(nn,dataItem):
	
	nodeCount=len(dataItem.data);
	output=nn.activate(dataItem.data); #Getting the output
	#print output;
	output=output[:dataItem.startIndex].tolist()+dataItem.target+output[dataItem.endIndex:].tolist();


	#print output;

	ds = SupervisedDataSet(nodeCount, nodeCount);
	ds.addSample(dataItem.data,output);
	
	nnTemp=deepcopyNN(nn);
	#nnTemp=nn.copy();
	#nnTemp.sortModules();
	
	trainer = BackpropTrainer(nnTemp, ds);

	trainer.train();#train for only one epoch

	#print nn.activate(dataItem.data);
	return nnTemp;
	
#test a single data item on the neural network
def testItem(nn,dataItem):
	output=nn.activate(dataItem.data); #Getting the output
	output=output[dataItem.startIndex:dataItem.endIndex];
	#print output;
	#print dataItem.target;
	
	for i in range(len(output)):
		if output[i]>0.5:
			val=1;
		else:
			val=0;
		#print val,' ',dataItem.target[i]
		
		if val!=dataItem.target[i]:
			return 0;
	return 1;

#test a list and return accuracy
def testList(nn,dataList):
	count=0;
	
	for dataItem in dataList:
		count+=testItem(nn,dataItem);
	return float(count)/len(dataList);
