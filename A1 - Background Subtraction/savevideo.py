import cv2
import os


filenames = []
for file in os.listdir('result1'):
    filenames.append(file)
n = len(filenames)
filenames=[]
for i in range(n-1):
    filenames.append("frame"+str(i)+".png")
#print(filenames)
frames=[]
for fr in range(1,len(filenames)):
    f = filenames[fr]
    img = cv2.imread(os.path.join('result1', f))
    frames.append(img)
    height, width, layers = img.shape
    size = (width,height)
    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 20, size)
 
for i in range(len(frames)):
    out.write(frames[i])
out.release()
print("done")
