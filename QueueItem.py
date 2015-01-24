

#Object that will be put in queue
class QueueItem:
	def __init__(self,dataList,nn,score):
		self.dataList=dataList;
		self.nn=nn;
		self.score=score;
	def __cmp__(self, other):
		return cmp(self.score,other.score);
	

	
	
		





