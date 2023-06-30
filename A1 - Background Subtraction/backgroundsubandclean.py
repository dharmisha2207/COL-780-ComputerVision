import cv2
import os
import numpy as np
from math import exp

def read_frames(path,d):
    
    filenames=[]
    for file in os.listdir(path):
        filenames.append(file)
    filenames.sort()
    #print(filenames)
    frames=[]
    for f in filenames:
        frame = cv2.imread(os.path.join(path, f),d)
        frames.append(frame)
    return frames

def gaussian(x,d,b=255/64):
    if d.get(x) is not None:
        return d.get(x)
    else:
        x_plot = np.linspace(1.5, 253.5,64)
        xy = x_plot - x
        d[x] = np.exp(-xy**2/(2*(b/2)**2))/((b/2)*np.sqrt(2*np.pi))
        return d.get(x)

def background_sub(path,d,op):
    #pre-calculate gaussian
    for i in range(0,256):
        gaussian(i,d)
    
    #Read frames
    frames = read_frames(path,1)
    x,y, z= frames[0].shape
    
    #Initialize models
    modelr=np.zeros((x,y,64))
    modelb=np.zeros((x,y,64))
    modelg=np.zeros((x,y,64))
    
    #Make base model
    for h in range(0,x):
        for w in range(0,y):
            modelr[h][w]+=gaussian(frames[0][h,w][0],d)
            modelb[h][w]+=gaussian(frames[0][h,w][1],d)
            modelg[h][w]+=gaussian(frames[0][h,w][2],d)
    
    n = len(frames)
    
    #Initialize gradient
    gradr = np.ones((x,y))
    gradb = np.ones((x,y))
    gradg = np.ones((x,y))
    
    #initialize learning rate
    c = 1
    gt = 300*2/(1+exp(-1*(c-100)/20))
    
    b = (255*3)/64
    
    #Update model for each frame
    for i in range(0,n):
        for h in range(0,x):
            for w in range(0,y):
                ir = frames[i][h,w][0]
                ib = frames[i][h,w][1]
                ig = frames[i][h,w][2]
                
                #find min dist
                distr = []
                distb = []
                distg = []
                for j in range(64):
                    if(modelr[h][w][j]>1/64):
                        distr.append(abs(((j*4+(j*4+3))/2)-ir))
                    if(modelb[h][w][j]>1/64):
                        distb.append(abs(((j*4+(j*4+3))/2)-ib))
                    if(modelg[h][w][j]>1/64):
                        distg.append(abs(((j*4+(j*4+3))/2)-ig))
                distr = min(distr)
                distb = min(distb)
                distg = min(distg)
                
                #Foreground detection
                if((distr/(1+gradr[h][w]))+(distb/(1+gradb[h][w]))+distg/(1+gradg[h][w]))>b:
                    gradr[h][w]=(gt-1)*(gradr[h][w]/gt)+(0.3*distr/gt)
                    gradb[h][w]=(gt-1)*(gradb[h][w]/gt)+(0.3*distb/gt)
                    gradg[h][w]=(gt-1)*(gradg[h][w]/gt)+(0.3*distg/gt)
                    frames[i][h,w] = 255
                else:
                    gradr[h][w]=(gt-1)*(gradr[h][w]/gt)+(distr/gt)
                    gradb[h][w]=(gt-1)*(gradb[h][w]/gt)+(distb/gt)
                    gradg[h][w]=(gt-1)*(gradg[h][w]/gt)+(distg/gt)
                    frames[i][h,w] = 0
            
            modelr[h][w]=(0.95*modelr[h][w])+(0.05*gaussian(ir,d)/gt)
            modelr[h][w]=modelr[h][w]/np.sum(modelr[h][w])
            modelb[h][w]= (0.95*modelb[h][w])+(0.05*gaussian(ib,d)/gt)
            modelb[h][w]=modelb[h][w]/np.sum(modelb[h][w])
            modelg[h][w]= (0.95*modelg[h][w])+(0.05*gaussian(ig,d)/gt)
            modelg[h][w]=modelr[h][w]/np.sum(modelg[h][w])
        c+=1
        gt = 300*2/(1+exp(-1*(c-100)/20))
        status = cv2.imwrite(r""+op+'/frame0000%d.png'%i,frames[i])
        print(status,i)
 
def read_frames_clean(path):
    filenames = []
    for file in os.listdir(path):
        filenames.append(file)
    n = len(filenames)
    filenames=[]
    for i in range(n-1):
        filenames.append("frame0000"+str(i)+".png")
    #print(filenames)
    frames=[]
    for f in filenames:
        frame = cv2.imread(os.path.join(path, f),0)
        frames.append(frame)
    return frames
def clean(path,op):
    frames = read_frames_clean(path)
    x,y = frames[0].shape
    for i in range(0,len(frames)): 
        oi = np.zeros((x,y))
        for h in range(x):
            for w in range(y):
                oi[h][w]=frames[i][h,w]
        ii = np.zeros((x,y))
        for h in range(x):
            for w in range(y):
                if(h==0 and w ==0):
                    ii[h][w] = oi[h][w]
                elif(h==0):
                    ii[h][w] = oi[h][w]+ii[h][w-1]
                elif(w==0):
                    ii[h][w] = oi[h][w]+ii[h-1][w]
                else:
                    ii[h][w] = oi[h][w]+ii[h][w-1]+ii[h-1][w]-ii[h-1][w-1]
        h1 = 3
        w1 = 3
        for h in range(0,x,3):
            for w in range(0,y,3):
                if(h+h1)>=x or (w+w1)>=y:
                    continue
                if(ii[h+h1][w+w1]-ii[h+h1][w]-ii[h][w+w1]+ii[h][w])<(255*(h1+1)*(w1+1))*0.25:
                    for a in range(h,h+h1+1):
                        for b in range(w,w+w1+1):
                            frames[i][a,b]=0
                elif(ii[h+h1][w+w1]-ii[h+h1][w]-ii[h][w+w1]+ii[h][w])>=(255*(h1+1)*(w1+1))*0.75:
                    for a in range(h,h+h1+1):
                        for b in range(w,w+w1+1):
                            frames[i][a,b]=255
        status = cv2.imwrite(r''+op+'/frame%d.png'%i,frames[i])
        print(status,i)


def bgsubandclean(path):
    d = {}
    #call background_sub where first param is source data path, d is a dictionary, second param 
    #is output data path
    background_sub(path,d,'res1')
    # call clean where first param is source data path and second param is output data path
    clean('res1','result1')

#call bgsubandclean where param contains source data path. It calls the background_sub module and clean module
bgsubandclean("cropped")           

            
    
            
