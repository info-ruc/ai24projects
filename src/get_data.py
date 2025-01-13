from paddleocr import PaddleOCR
import os
import pandas as pd
import numpy as np
import math
import random
import re
import API
import cv2

def sharpen_image(image_path,s_dic):
    # 读取图片
    image = cv2.imread(image_path)
    sharpening_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    # 应用锐化滤波器
    sharpened_image = cv2.filter2D(image, -1, sharpening_kernel)
    # 保存锐化后的图片
    print(s_dic+'sharpened_'+image_path)
    cv2.imwrite(s_dic+'sharpened_'+image_path, sharpened_image)

#统计数字数量
def count_digits(s):
    return sum(1 for char in s if char.isnumeric())/len(s)
#只保留数字
def extract_digits(s):
    return re.findall(r'\d+', s)

# 获取距离
def distance(p1,p2):
    ans=[]
    for i in range(4):
        for j in range(4):
            an=math.sqrt((p1[i][0]-p2[j][0])**2+(p1[i][1]-p2[j][1])**2)
            ans.append(an)
    return np.array(ans).min()/500
def get_left_distance(p1,p2):
    an=math.sqrt((p1[1][0]-p2[0][0])**2+16*(p1[1][1]-p2[0][1])**2)
    return an/500

#利用ocr获得数据
def get_picture(image_path):
    # 初始化 OCR 模型
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')  # 支持中文
    # 文字检测和识别
    results = ocr.ocr(image_path, cls=True)
    return results

#获取属性
def half_auto_get_ans(por,results):
    length=len(por)
    count=0;answers=np.zeros(length);ans=[]
    for line in results[0]:
        count+=1
        for i in range(length):
            index=line[1][0].find(por[i][0])
            if index>-1:
                answers[i]=count
    maxs=np.full(length, -100.0);indexs=np.zeros(length);count=0
    for line in results[0]:
        count+=1
        for i in range(length):
            if answers[i]==0:
                continue
            elif por[i][1]=="i":
                d=distance(line[0],results[0][int(answers[i]-1)][0])
                cd=count_digits(line[1][0])
                if cd-d>maxs[i]:
                    indexs[i]=count
                    maxs[i]=cd-d
            elif por[i][1]=="c":
                d=-get_left_distance(results[0][int(answers[i]-1)][0],line[0])
                if d>maxs[i]:
                    indexs[i]=count
                    maxs[i]=d
    for i in range(length):
        if answers[i]==0:
            ans.append("#Error#")
        elif por[i][1]=="i":
            annn=extract_digits(results[0][int(indexs[i]-1)][1][0])
            if(len(annn)):
                ans.append(annn[0])
            else:
                ans.append(results[0][int(indexs[i]-1)][1][0])
        elif por[i][1]=="c":
            ans.append(results[0][int(indexs[i]-1)][1][0])
    return ans

#获取对应文件的数据
def half_auto_get_data(path,por,type,aimpath):
    col=[]
    for i in por :
        col.append(i[0])
    data = pd.DataFrame(columns=col)
    if type=="f":
        print(path)
        sharpen_image(path,"./sharpened/")
        data.loc[len(data)]=half_auto_get_ans(por,get_picture('./sharpened/sharpened_'+path))
    elif type=="d":
        fatherLists = os.listdir(path)
        for c_path in fatherLists:
            child_path=path+c_path
            sharpen_image(child_path,"./sharpened/")
            data.loc[len(data)]=half_auto_get_ans(por,get_picture('./sharpened/sharpened_'+child_path))
    data.to_csv(aimpath)
    return data

