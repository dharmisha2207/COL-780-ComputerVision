import cv2 as cv
import numpy as np

def get_vanishing_points(corners):
    vp = []
    for i in corners:
        l1 = [(np.cross(i[0],i[1])),np.cross(i[2],i[3])]
        l2 = [(np.cross(i[0],i[2])),np.cross(i[1],i[3])]
        # l1 = [(np.cross(i[0],i[1])),np.cross(i[2],i[3])]
        # l2 = [(np.cross(i[0],i[3])),np.cross(i[1],i[2])]
        # print("lines", v1,v2)
        vp1 = np.cross(l1[0],l1[1])
        vp2 = np.cross(l2[0],l2[1])
        vp.append(vp1/vp1[2])
        vp.append(vp2/vp2[2])
    # print(vp)
    return vp

def calib(corners):
    vp = get_vanishing_points(corners)
    A = []
    C = []

    for i in range(0,len(vp)-1,2):
        a1 = vp[i][0]
        b1 = vp[i][1]
        c1 = vp[i][2]
        a2 = vp[i+1][0]
        b2 = vp[i+1][1]
        c2 = vp[i+1][2]
        A.append([a1*a2, (c1*a2+a1*c2),b1*b2,(c1*b2+b1*c2),c1*c2])
    # print("A")
    # for i in A:
    #     print(i)
    U,D,V = np.linalg.svd(A)

    # print(V)

    w11,w13,w22,w23,w33 = V[-1,:]

    x = [   [w11],
            [w13],
            [w22],
            [w23],
            [w33]]

    # res = np.matmul(A,x)
    # print("res", res)
    finalW = [  [w11,0,w13],
                [0, w22,w23],
                [w13, w23, w33]]

    return finalW/finalW[2][2]
    # return finalW

corners = [[(788,448,1),(3447,1114,1),(977,1748,1),(2967,2667,1)],
            [(600,872,1),(3760,738,1),(832,2520,1),(3191,2060,1)],
            [(378,839,1),(3474,511,1),(581,2164,1),(3435,2612,1)],
            [(617,611,1),(3140,838,1),(703,2598,1),(3051,2035,1)],
            [(671,682,1),(3241,809,1),(427,2125,1),(3505,2121,1)],
            [(450,1092,1),(3657,1183,1),(722,2258,1),(3206,2234,1)]]

def get_K(corners):
    w = calib(corners)
    # print("W")
    # for i in w:
    #     print(i)
    # w = np.linalg.inv(w)
    k = np.linalg.cholesky(w)
    k = np.linalg.inv(k)
    k = np.transpose(k)
    k/=k[2][2]
    print("Camera Calibration done!")
    return k

def pyramid(ip,P,op,xmin,ymin,xmax,ymax,x,y,z,h):
    X1 = [[xmin],[ymin],[z],[1]]
    X2 = [[xmin],[ymax],[z],[1]]
    X3 = [[xmax],[ymin],[z],[1]]
    X4 = [[xmax],[ymax],[z],[1]]
    X5 = [[(xmin+xmax//2)],[(ymin+ymax//2)],[z-h],[1]]

    x1 = np.matmul(P,X1)
    x2 = np.matmul(P,X2)
    x3 = np.matmul(P,X3)
    x4 = np.matmul(P,X4)
    x5 = np.matmul(P,X5)

    x1=x1/x1[2]
    x2=x2/x2[2]
    x3=x3/x3[2]
    x4=x4/x4[2]
    x5=x5/x5[2]

    p1 = (int(x1[0]*x)),int(x1[1]*y)
    p2 = (int(x2[0]*x)),int(x2[1]*y)
    p3 = (int(x3[0]*x)),int(x3[1]*y)
    p4 = (int(x4[0]*x)),int(x4[1]*y)
    p5 = (int(x5[0]*x)),int(x5[1]*y)
    # print(p1)
    img = cv.imread(ip)
    img = cv.circle(img, p1, 10, (255, 0, 0), -1)
    img = cv.circle(img, p2, 10, (255, 0, 0), -1)
    img = cv.circle(img, p3, 10, (255, 0, 0), -1)
    img = cv.circle(img, p4, 10, (255, 0, 0), -1)
    img = cv.circle(img, p5, 10, (255, 0, 0), -1)
    pts1 = np.array([[p1[0],p1[1]],[p3[0],p3[1]],[p5[0],p5[1]]])
    pts2 = np.array([[p1[0],p1[1]],[p2[0],p2[1]],[p5[0],p5[1]]])
    pts3 = np.array([[p4[0],p4[1]],[p3[0],p3[1]],[p5[0],p5[1]]])
    pts4 = np.array([[p2[0],p2[1]],[p4[0],p4[1]],[p5[0],p5[1]]])
    pts5 = np.array([[p1[0],p1[1]],[p2[0],p2[1]],[p3[0],p3[1]],[p4[0],p4[1]]])
    img = cv.fillPoly(img,pts=[pts1],color = (255,153,153))
    img = cv.fillPoly(img,pts=[pts2],color = (255,153,153))
    img = cv.fillPoly(img,pts=[pts3],color = (255,153,153))
    img = cv.fillPoly(img,pts=[pts4],color = (255,153,153))
    img = cv.fillPoly(img,pts=[pts5],color = (255,153,153))
    img = cv.line(img,p1,p3,(255,0,0),20)
    img = cv.line(img,p1,p2,(255,0,0),20)
    img = cv.line(img,p1,p5,(255,0,0),20)
    img = cv.line(img,p2,p5,(255,0,0),20)
    img = cv.line(img,p3,p5,(255,0,0),20)
    img = cv.line(img,p4,p5,(255,0,0),2)
    img = cv.line(img,p2,p4,(255,0,0),2)
    img = cv.line(img,p3,p4,(255,0,0),2)
    # return img
    cv.imwrite(op,img)
    print("Pyramid Imposed!!")
    # return img

def cube(ip,P,op,xmin,ymin,xmax,ymax,x,y,z):
    X1 = [[xmin],[ymin],[z],[1]]
    X2 = [[xmax],[ymin],[z],[1]]
    X3 = [[xmin],[ymin],[z-(ymax-ymin)],[1]]
    X4 = [[xmax],[ymin],[z-(ymax-ymin)],[1]]
    X5 = [[xmin],[ymax],[z],[1]]
    X6 = [[xmax],[ymax],[z],[1]]
    X7 = [[xmin],[ymax],[z-(ymax-ymin)],[1]]
    X8 = [[xmax],[ymax],[z-(ymax-ymin)],[1]]

    x1 = np.matmul(P,X1)
    x2 = np.matmul(P,X2)
    x3 = np.matmul(P,X3)
    x4 = np.matmul(P,X4)
    x5 = np.matmul(P,X5)
    x6 = np.matmul(P,X6)
    x7 = np.matmul(P,X7)
    x8 = np.matmul(P,X8)

    x1=x1/x1[2]
    x2=x2/x2[2]
    x3=x3/x3[2]
    x4=x4/x4[2]
    x5=x5/x5[2]
    x6=x6/x6[2]
    x7=x7/x7[2]
    x8=x8/x8[2]

    p1 = (int(x1[0]*x)),int(x1[1]*y)
    p2 = (int(x2[0]*x)),int(x2[1]*y)
    p3 = (int(x3[0]*x)),int(x3[1]*y)
    p4 = (int(x4[0]*x)),int(x4[1]*y)
    p5 = (int(x5[0]*x)),int(x5[1]*y)
    p6 = (int(x6[0]*x)),int(x6[1]*y)
    p7 = (int(x7[0]*x)),int(x7[1]*y)
    p8 = (int(x8[0]*x)),int(x8[1]*y)
    print(p1)
    img = cv.imread(ip)
    img = cv.circle(img, p1, 10, (255, 0, 0), -1)
    img = cv.circle(img, p2, 10, (255, 0, 0), -1)
    img = cv.circle(img, p3, 10, (255, 0, 0), -1)
    img = cv.circle(img, p4, 10, (255, 0, 0), -1)
    img = cv.circle(img, p5, 10, (255, 0, 0), -1)
    img = cv.circle(img, p6, 10, (255, 0, 0), -1)
    img = cv.circle(img, p7, 10, (255, 0, 0), -1)
    img = cv.circle(img, p8, 10, (255, 0, 0), -1)
    img = cv.fillPoly(img,pts=[np.array([[p1[0],p1[1]],[p2[0],p2[1]],[p6[0],p6[1]],[p5[0],p5[1]]])],color = (255,153,153))
    img = cv.fillPoly(img,pts=[np.array([[p1[0],p1[1]],[p2[0],p2[1]],[p4[0],p4[1]],[p3[0],p3[1]]])],color = (255,153,153))
    img = cv.fillPoly(img,pts=[np.array([[p1[0],p1[1]],[p3[0],p3[1]],[p7[0],p7[1]],[p5[0],p5[1]]])],color = (255,153,153))
    img = cv.fillPoly(img,pts=[np.array([[p7[0],p7[1]],[p8[0],p8[1]],[p6[0],p6[1]],[p5[0],p5[1]]])],color = (255,153,153))
    img = cv.fillPoly(img,pts=[np.array([[p4[0],p4[1]],[p8[0],p8[1]],[p3[0],p3[1]],[p7[0],p7[1]]])],color = (255,153,153))
    img = cv.fillPoly(img,pts=[np.array([[p4[0],p4[1]],[p2[0],p2[1]],[p6[0],p6[1]],[p8[0],p8[1]]])],color = (255,153,153))
    img = cv.line(img,p1,p2,(255,0,0),15)
    img = cv.line(img,p2,p6,(255,0,0),1)
    img = cv.line(img,p6,p5,(255,0,0),1)
    img = cv.line(img,p1,p5,(255,0,0),15)
    img = cv.line(img,p1,p3,(255,0,0),15)
    img = cv.line(img,p5,p7,(255,0,0),15)
    img = cv.line(img,p3,p7,(255,0,0),15)
    img = cv.line(img,p3,p4,(255,0,0),15)
    img = cv.line(img,p4,p8,(255,0,0),15)
    img = cv.line(img,p8,p7,(255,0,0),15)
    img = cv.line(img,p6,p8,(255,0,0),1)
    img = cv.line(img,p2,p4,(255,0,0),15)
    cv.imwrite(op,img)
    print("Cube Imposed!!")
    # return img
    

def cubeandpyr(ip,P,op,xcmin,ycmin,xcmax,ycmax, xpmin, ypmin, xpmax, ypmax, x,y,z,h):
    cube(ip,P,op,xcmin,ycmin,xcmax,ycmax,x,y,z)
    pyramid(op,P, op, xpmin, ypmin, xpmax, ypmax, x, y, z, h)

def calculate_P(f): 
    if f =='z9.jpg':
        xt = np.array([   [1697/1000,2441/1000,1],
            [2407/1000,1384/1000,1],
            [3446/1000,1395/1000,1],
            [2753/1000,1735/1000,1],
            [3107/1000,1043/1000,1]])
    
    elif f =='z9skew.jpg':
        xt = np.array([   [1206/1000,2742/1000,1],
                [1992/1000,1772/1000,1],
                [2939/1000,1814/1000,1],
                [2302/1000,2088/1000,1],
                [2605/1000,1526/1000,1]]) 

    elif f =='chess8.jpg':
        xt = np.array([   [978/1000,2665/1000,1],
            [1387/1000,2251/1000,1],
            [1376/1000,1024/1000,1],
            [1781/1000,617/1000,1],
            [3403/1000,1003/1000,1],
            [2987/1000,213/1000,1]])

    #chess8skew
    # xt = np.array([   [707/1000,2000/1000,1],
    #     [1184/1000,2246/1000,1],
    #     [1330/1000,1555/1000,1],
    #     [1661/1000,1321/1000,1],
    #     [2875/1000,1535/1000,1],
    #     [2764/1000,1114/1000,1]])

    x = np.transpose(xt)
    if f =='z9.jpg' or f =='z9skew.jpg':
        Xt = np.array([ [0,2,9,1],
                    [3,4,9,1],
                    [3,7,9,1],
                    [2,5,9,1],
                    [4,6,9,1]])
    elif f =='chess8.jpg':
        Xt = np.array([ [0,0,8,1],
                        [1,1,8,1],
                        [4,1,8,1],
                        [5,2,8,1],
                        [4,6,8,1],
                        [6,5,8,1]])

    X = np.transpose(Xt)
    K = get_K(corners)
    Kinv = np.linalg.inv(K)
    rhs = np.matmul(Kinv,x)
    # print("rhs:",rhs.shape)
    Xt = np.transpose(X)
    A = []

    for i in range(0,5):
        A.append([Xt[i][0], Xt[i][1], Xt[i][2],0,0,0,0,0,0,1,0,0])
        A.append([0,0,0,Xt[i][0], Xt[i][1], Xt[i][2],0,0,0,0,1,0])
        A.append([0,0,0,0,0,0,Xt[i][0], Xt[i][1], Xt[i][2],0,0,1])

    A = np.array(A)

    b = []
    for i in range(0,5):
        b.append([rhs[0][i]])
        b.append([rhs[1][i]])
        b.append([rhs[2][i]])

    b = np.array(b)


    Rt = np.linalg.lstsq(A, b)

    Rt = Rt[0]

    finalRt = np.array([ [Rt[0][0], Rt[1][0], Rt[2][0], Rt[9][0]],
            [Rt[3][0], Rt[4][0], Rt[5][0], Rt[10][0]],
            [Rt[6][0], Rt[7][0], Rt[8][0], Rt[11][0]]])

    P = np.matmul(K,finalRt)
    P = P/P[2][3]
    print("Projection matrix calculated!")
    return P

def AR():
    files = ['z9.jpg','z9skew.jpg','chess8.jpg']
    for i in files:
        P = calculate_P(i)
        z = 0
        if i=='z9.jpg' or i == 'z9skew.jpg':
            z = 9
        else:
            z = 8
        print(z)
        cube(i, P, 'AR/cube'+i, 2,2,4,4,1000, 1000, z)
        pyramid(i, P, 'AR/pyramid'+i,2,2,4,5,1000,1000,z,3)
        cubeandpyr(i, P, 'AR/cubeandpyramid'+i, 2, 2, 3, 3, 2,4,3,5, 1000, 1000, z, 1)

AR()

