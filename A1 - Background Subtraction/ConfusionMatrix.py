import cv2
import os
import numpy as np

cm = np.zeros((2,2))

pathmy = "/Users/dharmishasharma/Documents/SEM 2/COL-780 CV/assign1/HallAndMonitor/output3D4binclean/"
pathgt = "/Users/dharmishasharma/Documents/SEM 2/COL-780 CV/assign1/HallAndMonitor/groundtruth/"
filenamesgt=[]
for file in os.listdir(pathgt):
    filenamesgt.append(file)
filenamesgt.sort()
#print(filenames)
myf=[]
gtf=[]
for f in filenamesgt:
    frame = cv2.imread(os.path.join(pathgt, f),0)
    gtf.append(frame)

filenames = []
for i in range(0,len(gtf)):
    filenames.append("frame"+str(i)+".png")

for f in filenames:
    frame = cv2.imread(os.path.join(pathmy, f),0)
    myf.append(frame)

x,y = myf[0].shape
n = len(myf)
for i in range(n):
    for h in range(x):
        for w in range(y):
            if(myf[i][h,w]==255 and gtf[i][h,w]==255):
                cm[1][1]+=1
            elif(myf[i][h,w]==0 and gtf[i][h,w]==0):
                cm[0][0]+=1
            elif(myf[i][h,w]==255 and gtf[i][h,w]==0):
                cm[0][1]+=1
            elif(myf[i][h,w]==0 and gtf[i][h,w]==255):
                cm[1][0]+=1

#print(cm[1][1]/(cm[1][1]+cm[0][1]))

print("Accuracy: ",((cm[0][0]+cm[1][1])/(x*y*n))*100)

print(cm)



            


