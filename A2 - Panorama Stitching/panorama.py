import numpy as np
import cv2 as cv
import math
import math
import random
import os
def grad(img,kernel):
    h,w = img.shape
    Ix = np.zeros((h,w))
    Iy = np.zeros((h,w))

    for i in range(h):
        for j in range(w):
            if(j==0):
                x = -(kernel)*img[i][j+1]
            elif(j==w-1):
                x = kernel*img[i][j-1]
            else:
                x = kernel*img[i][j-1]+(-(kernel))*img[i][j+1]
            Ix[i][j]=x
    
    for i in range(h):
        for j in range(w):
            if(i==0):
                x = -(kernel)*img[i+1][j]
            elif(i==h-1):
                x = kernel*img[i-1][j]
            else:
                x = kernel*img[i-1][j]+(-(kernel))*img[i+1][j]
            Iy[i][j]=x

    return Ix,Iy

def corner_detection(gray, h, w):
    Ix, Iy = grad(gray,0.5)
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy
    window = 3
    n = int(window / 2)
    R = np.zeros((h, w))
    for i in range(n, h - n):
        for j in range(n, w - n):
            a = 0
            b = 0
            c = 0
            sx = i - n
            ex = i + window - n
            sy = j - n
            ey = j + window - n
            for x in range(sx, ex):
                for y in range(sy, ey):
                    a += Ixx[x][y]
                    c += Iyy[x][y]
                    b += Ixy[x][y]
            b = 2 * b
            l1 = (a + c + math.sqrt(b ** 2 + (a - c) ** 2)) / 2
            l2 = (a + c - math.sqrt(b ** 2 + (a - c) ** 2)) / 2
            r = (l1 * l2) - 0.06 * ((l1 + l2) ** 2)
            # print(r)
            if r>100000:
                R[i][j] = r
    return R

def non_maximal_supression(R,h,w):
    corners = []
    window = 30
    n = int(window/2)
    for i in range(n,h-n):
        for j in range(n,w-n):
            if R[i][j]>100000:
                maxi = 0
                sx = i-n
                ex = i+window-n
                sy = j-n
                ey = j+window-n
                maxi = R[sx:ex, sy:ey].max()
                # print(maxi)
                if(maxi != R[i][j]):
                    R[i][j]=0
                elif(R[i][j]>100000):
                    # print("here")
                    corners.append((i,j))
    return corners

def harris_corner_detection(filename):
    img = cv.imread(filename,0)
    resimg = cv.imread(filename)
    gray = np.float32(img)
    gray = cv.GaussianBlur(gray, (3, 3),0)
    h,w = img.shape
    R = corner_detection(gray,h,w)
    # print(R)
    corners = non_maximal_supression(R,h,w)
    for (x,y) in corners:
       resimg[x][y]=(0,255,0)
    # save the resulting image
    cv.imwrite('corners.png',resimg)
    print(len(corners),corners)
    return corners

def matchssd(frame1, frame2, corners1, corners2, window = 30):
    img1 = cv.imread(frame1,0)
    img2 = cv.imread(frame2,0)
    offset = window//2
    matches = []
    matchdict = {}
    keys = []
    for i in corners1:
        test = []
        x = i[0]
        y = i[1]
        sx = x-offset
        ex = x+offset
        sy = y-offset
        ey = y+offset
        for h in range(sx,ex+1):
            for w in range(sy, ey+1):
                if (h,w) in corners2: 
                    test.append([h,w])
        #print("test",test,x,y)
        res = []
        if(len(test)!=0):
            frame1i = []
            for h in range(sx,ex+1):
                    for w in range(sy, ey+1):
                       frame1i.append(img1[h,w])
            f1i = np.array(frame1i)
            mini = 1000000
            for j in test:
                x2 = j[0]
                y2 = j[1]
                sx2 = x2-offset
                ex2 = x2+offset
                sy2 = y2-offset
                ey2 = y2+offset
                ssd = 0
                frame2i = []
                for h in range(sx2,ex2+1):
                    for w in range(sy2, ey2+1):
                       frame2i.append(img2[h,w]) 
                f2i = np.array(frame2i)   
                s = np.sum((f1i[:]-f2i[:])**2)
                ssd = s/(window**2)
                # print("ssd",ssd)
                if(mini>ssd and ssd>25):   
                    res = j
                mini = min(mini,ssd)
        # print(ssd)
        if(len(res)!=0):
            matchdict[ssd] = ([i,res])
            keys.append(ssd)
    keys.sort()
    print(keys)
    # print(matchdict)
    for i in range(0,3):
        matches.append(matchdict[keys[i]])
        print(keys[i])
    print(matches)
    print("length",len(matches))
    img11 = cv.imread(frame1)
    img22 = cv.imread(frame2)
    h,w,z = img11.shape
    width = 2*w
    img = np.zeros((h,width,3))
    img[:,:w,:] = img11[:,:,:]
    img[:,w:,:] = img22[:,:,:]
    print("lines")
    # for j in  range(20):
    #     i = random.randint(0,len(match)-1)
        
    #     # print(match[i][0][1], match[i][0][0], match[i][1][1]+w, match[i][1][0])
    #     cv.line(img,(match[i][0][1], match[i][0][0]), (match[i][1][1]+w, match[i][1][0]),(0, 255, 0),1)
    for i in matches:
        cv.line(img,(i[0][1], i[0][0]), (i[1][1]+w, i[1][0]),(0, 255, 0),1)
    cv.imwrite('matching.png',img)
    return matches

def affine(match):
    test = match 
    # print("affine test:", test)
    x11 = test[0][0][0]
    y11 = test[0][0][1]
    x12 = test[0][1][0]
    y12 = test[0][1][1]
    x21 = test[1][0][0]
    y21 = test[1][0][1]
    x22 = test[1][1][0]
    y22 = test[1][1][1]
    x31 = test[2][0][0]
    y31 = test[2][0][1]
    x32 = test[2][1][0]
    y32 = test[2][1][1]
    # h_arr = np.array(([x11,y11,1,0,0,0],[0,0,0,x11,y11,1],[x21,y21,1,0,0,0],[0,0,0,x21,y21,1],[x31,y31,1,0,0,0],[0,0,0,x31,y31,1]))
    h_arr = np.array((
        [x12,y12,1,0,0,0],
        [0,0,0,x12,y12,1],
        [x22,y22,1,0,0,0],
        [0,0,0,x22,y22,1],
        [x32,y32,1,0,0,0],
        [0,0,0,x32,y32,1]))
    if np.linalg.det(h_arr) == 0.0:
        return []
    v_arr = np.array(([x11],[y11],[x21],[y21],[x31],[y31]))
    a = np.linalg.solve(h_arr,v_arr)
    # print(a)
    affine_mat = np.array(([a[0][0], a[1][0], a[2][0]], [a[3][0],a[4][0],a[5][0]],[0,0,1]))
    # print("Affine:",affine_mat)
    return affine_mat

def stitch(img,frame2,affine_mat,op):
    f1c = img
    f2c = cv.imread(frame2)
    h,w,z = f2c.shape
    h1,w1,z = f1c.shape
    c1 = np.array(([0,0,1]))
    c2 = np.array(([0,w,1]))
    c3 = np.array(([h,0,1]))
    c4 = np.array(([h,w,1]))
    ca1 = np.matmul(affine_mat,c1).astype(int)
    ca2 = np.matmul(affine_mat,c2).astype(int)
    ca3 = np.matmul(affine_mat,c3).astype(int)
    ca4 = np.matmul(affine_mat,c4).astype(int)
    print(ca1, ca2, ca3, ca4)
    winh = max(ca1[0], ca2[0], ca3[0], ca4[0],h,h1)
    winw = max(ca1[1], ca2[1], ca3[1], ca4[1],w,w1)
    print(winh,winw)
    new_img = np.zeros((winh,winw,3))
    # affine_inv = np.linalg.inv(affine_mat)
    minx = min(ca1[0], ca2[0], ca3[0], ca4[0])
    miny = min(ca1[1], ca2[1], ca3[1], ca4[1])
    offsetw = winw-w
    offseth = winh-h
    new_img[:h1,:w1] = f1c[:h1,:w1]
    affine_inv = np.linalg.inv(affine_mat)
    for i in range(winh):
        for j in range(w1,winw):
            x = np.array(([i,j,1]))
            t = np.matmul(affine_inv,x).astype(int)
            x = t[0]
            y = t[1]
            mini = 0
            if(t[0]>=0 and t[0]<h and t[1]>=0 and t[1]<w):
                new_img[i][j]= f2c[x][y]
            else:
                new_img[i][j] = [0,0,0]
    print("done stitching")

    return new_img
    
def getcorners(files):
    corners = []
    for i in files:
        corners.append(harris_corner_detection(i))
        print("Corners found !")
    return corners

def getmatches(files):
    corners = getcorners(files)
    matches = []
    for i in range(0,len(files)-1):
        matches.append(matchssd(files[i], files[i+1], corners[i], corners[i+1]))
        print(i," matching done!")
    return matches

def getaffines(files):
    matches = getmatches(files)
    affines = []
    for i in range(0,len(matches)):
        # print(i)
        a = affine(matches[i])
        # while(len(a)==0):
        #     a = affine(matches[i])
        # if(len(a)!=0):
        affines.append(a)
  
    return(affines)

def stitchall(files,op,affines):
    affine_mat = affines[0]
    img = cv.imread(files[0])
    img = stitch(img,files[1], affine_mat,op)
    cv.imwrite('stitch1.jpg',img)
    print("1 stitching done!")
    for i in range(2,len(files)):
        affine_mat = np.matmul(affine_mat,affines[i-1])
        img = stitch(img,files[i], affine_mat,op)
        cv.imwrite('stitch'+str(i)+'.jpg',img)
        print(i," stitching done!")
    cv.imwrite(op,img)

def getfiles(folder_path):
    filenames = []
    for i in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, i)):
            filenames.append(folder_path+i)
    filenames.sort()
    return filenames

def panorama(path,result):
    filenames = getfiles(path)
    print(filenames)
    filenames = ['source1/third.png','source1/second.png','source1/original.png']
    # detecting direction
    affine = getaffines(filenames[:2])
    # print(affine)
    c1 = np.array(([0,0,1]))
    ca1 = np.matmul(affine,c1).astype(int)
    # print(ca1)
    if ca1[0][1]<0:
        filenames.sort(reverse=True)
        print(filenames)
    affines = getaffines(filenames)
    stitchall(filenames,result,affines)

panorama('source1/','res.jpg')