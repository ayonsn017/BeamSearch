class DataItem:
    dataType='';
    data=[];	#the attribute values
    target=[];	#the actual value of the missing attributes
    startIndex=0;#the strart index of the missing attributes
    endIndex=0;#the end index+1 of the missing attributes
    dataType='';#type of the data item. possible values S and I
    def __init__(self, data, target,startIndex,endIndex,dataType):
		temp1=[];
		for c in data:
			temp1.append(int(c));
		self.data=temp1;
		
		temp2=[];	
		for c in target:
			temp2.append(int(c));
		self.target=temp2;            
		self.startIndex=startIndex;
		self.endIndex=endIndex;
		self.dataType=dataType;
