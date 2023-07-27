import time

class SpeedTracker:
    def __init__(self):
        self.averageSpeed=0
        self.countSeen=0
        self.countUnseen=0
        self.minFrameSeen=15
        self.maxFrameNotSeen=5
        self.currentObjectId=None
        self.speedList=[]
        self.startPosition=[]
        self.lastPosition=[]
        self.latestTimeStamp=None
        self.firstTimeStamp=None
    def __getCenter(self,object):
        x1 = object[0]
        y1 = object[1]
        x2 = object[2]
        y2 = object[3]
        centerX = (x1+x2)//2
        centerY = (y1+y2)//2
        # return(centerX,centerY)
        return(x1,y1)
    def __selectNewObject(self,outputs, timeNow):
        #get the latest object, with highest id
        outputs.sort(key=lambda x: x[4], reverse=True) 
        newObject = outputs[0] 
        self.currentObjectId=newObject[4]
        # print(f'current object id={self.currentObjectId}')
        #reset everything
        self.countSeen=0 
        self.countUnseen=0
        newX, newY = self.__getCenter(newObject)
        self.firstPosition=(newX,newY)
        self.latestTimeStamp=timeNow
        self.firstTimeStamp=timeNow
    def __calcLatestSpeed(self):
        timeDelta = self.latestTimeStamp - self.firstTimeStamp
        # print(f'latest time stamp={self.latestTimeStamp}')
        # print(f'first time stamp={self.firstTimeStamp}')
        # print(f'delta={timeDelta}')
        distanceDelta = abs(self.firstPosition[0] - self.lastPosition[0]) #assuming only a difference on x axis for now
        # print(f'distance delta={distanceDelta}')
        if timeDelta!=0:
            newSpeed = distanceDelta/timeDelta
            # print(f'newSpeed={newSpeed}')
            return (newSpeed)
        else: 
            return 0
    def __updateAverageSpeed(self, newSpeed):
        self.speedList.append(newSpeed)
        if len(self.speedList)>30:
            rm = self.speedList.pop(0)
        if len(self.speedList)>0:
            self.averageSpeed = sum(self.speedList)/len(self.speedList)
    def updateSpeed(self,outputs, timeNow):
        if len(outputs)==0:
            return self.averageSpeed
        currentObject = next((output for output in outputs if output[4] == self.currentObjectId), None)    
        if currentObject:
            self.countSeen+=1
            self.countUnseen=0
            newX, newY = self.__getCenter(currentObject)
            self.lastPosition=(newX, newY)
            # print(f'object id={currentObject[4]}   new X={newX}')
            self.latestTimeStamp=timeNow
            # print(f' object id = {currentObject[4]} - seen={self.countSeen}')
        else:
            self.countUnseen+=1
            # print(f' object id = {self.currentObjectId} - unseen={self.countUnseen}')
            if self.countUnseen>=self.maxFrameNotSeen:
                # print('too many unseen')
                if self.countSeen >= self.minFrameSeen:
                    newSpeed = self.__calcLatestSpeed()
                    self.__updateAverageSpeed(newSpeed)
                self.__selectNewObject(outputs, timeNow)
        return (self.averageSpeed) 
    def getSpeed(self):
        return self.averageSpeed
        
##todo: make sure we don't break at the start or if outputs is empty