import os
from PIL import ImageFont, Image, ImageDraw
import numpy as np
import re
from matplotlib import pyplot as plt
from math import pi
import random
import cv2
from random import *
import hgtk
import string


def get_rad(theta, phi, gamma):
    return (deg_to_rad(theta),
            deg_to_rad(phi),
            deg_to_rad(gamma))

def get_deg(rtheta, rphi, rgamma):
    return (rad_to_deg(rtheta),
            rad_to_deg(rphi),
            rad_to_deg(rgamma))

def deg_to_rad(deg):
    return deg * pi / 180.0

def rad_to_deg(rad):
    return deg * 180.0 / pi




class ImageTransformer(object):
    """ Perspective transformation class for image
        with shape (height, width, #channels) """

    def __init__(self, image):
        self.image = image
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

    """ Wrapper of Rotating a Image """

    def rotate_along_axis(self, theta=0, phi=0, gamma=0, dx=0, dy=0, dz=0):
        # Get radius of rotation along 3 axes
        rtheta, rphi, rgamma = get_rad(theta, phi, gamma)

        # Get ideal focal length on z axis
        # NOTE: Change this section to other axis if needed
        d = np.sqrt(self.height ** 2 + self.width ** 2)
        self.focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz = self.focal

        # Get projection matrix
        mat = self.get_M(rtheta, rphi, rgamma, dx, dy, dz)
        rotated_img = cv2.warpPerspective(self.image.copy(), mat, (self.width, self.height))
        A = np.resize(rotated_img, (rotated_img.shape[0], rotated_img.shape[1], 1))
        return (A)

    """ Get Perspective Projection Matrix """

    def get_M(self, theta, phi, gamma, dx, dy, dz):
        w = self.width
        h = self.height
        f = self.focal

        # Projection 2D -> 3D matrix
        A1 = np.array([[1, 0, -w / 2],
                       [0, 1, -h / 2],
                       [0, 0, 1],
                       [0, 0, 1]])

        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([[1, 0, 0, 0],
                       [0, np.cos(theta), -np.sin(theta), 0],
                       [0, np.sin(theta), np.cos(theta), 0],
                       [0, 0, 0, 1]])

        RY = np.array([[np.cos(phi), 0, -np.sin(phi), 0],
                       [0, 1, 0, 0],
                       [np.sin(phi), 0, np.cos(phi), 0],
                       [0, 0, 0, 1]])

        RZ = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],
                       [np.sin(gamma), np.cos(gamma), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([[1, 0, 0, dx],
                      [0, 1, 0, dy],
                      [0, 0, 1, dz],
                      [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([[f, 0, w / 2, 0],
                       [0, f, h / 2, 0],
                       [0, 0, 1, 0]])

        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1)))

def one_hot (label,text_length,dictionary):
    label = np.array(label)
    n_data = label.shape[0]
    label_length = len(dictionary)
    y_batch = np.zeros((n_data,text_length,label_length))
    for i in range(len(y_batch)):
        y_batch[i] = single_text_encoder(label[i],text_length,dictionary)
    return(np.array(y_batch))

def single_text_encoder (text,text_length,dictionary):
    label_length = len(dictionary)
    onehot = np.zeros((text_length,label_length))
    char = single_decompose(text,text_length)
    
    for i in range(text_length):
        c = char[i]
        if c not in dictionary:
            continue
        
        index = dictionary[c]
        onehot[i,index] = 1
        
    return(onehot)


def rotate(img, rotation_range):
    if np.max(img)>2:
        img = img/255

    img = 1-img
    rn1 = np.random.randint(-rotation_range, rotation_range)
    rn2 = np.random.randint(-rotation_range, rotation_range)
    rn3 = np.random.randint(-rotation_range, rotation_range)
    it = ImageTransformer(img)
    rotated_img = it.rotate_along_axis(theta= rn1 , phi=rn2, gamma=rn3, dx=0, dy=0, dz=0)
    rotated_img = np.resize(rotated_img,(rotated_img.shape[0],rotated_img.shape[1],1))
    return(1-rotated_img)

def shift(img, shift_range):
    if np.max(img)>2:
        img = img/255

    img = 1-img
    rn1 = np.random.randint(-shift_range, shift_range)
    rn2 = np.random.randint(-shift_range, shift_range)
    it = ImageTransformer(img)
    rotated_img = it.rotate_along_axis(theta= 0 , phi= 0, gamma= 0, dx=rn1, dy=rn2, dz=0)
    return(1-rotated_img)

def shrink(img, rate):
    if np.max(img)>2:
        img = img/255

    if np.random.random() > 0.5:
        rn = rate*np.random.random()
        w = img.shape[1]
        h = img.shape[0]
        new_img = np.zeros((h,w,1))
        for i in range(h):
            background_color = img[0,0]
            resized = int(w - (rn*i*w/h))
            A = cv2.resize(img[i:i+1,:], (resized, 1), interpolation=cv2.INTER_CUBIC)
#             A = np.resize(A,(A.shape[0]))
            B = np.array([background_color]*w)
            B = np.resize(B,(B.shape[0]))
            B[0:resized]= A
            B = np.roll(B,int((w-resized)/2))
            B = np.resize(B,(B.shape[0],1))
            new_img[i] = B
            new_img = np.resize(new_img,(new_img.shape[0],new_img.shape[1],1))
        return(new_img)
    
    else:
        rn = rate*np.random.random()
        w = img.shape[1]
        h = img.shape[0]
        new_img = np.zeros((h,w,1))
        for i in range(h):
            background_color = img[0,0]
            resized = int((1-rn)*w+rn*i*w/h)
            A = cv2.resize(img[i:i+1,:], (resized, 1), interpolation=cv2.INTER_CUBIC)
#             A = np.resize(A,(A.shape[0]))
            B = np.array([background_color]*w)
            B = np.resize(B,(B.shape[0]))
            B[0:resized]= A
            B = np.roll(B,int((w-resized)/2))
            B = np.resize(B,(B.shape[0],1))
            
            new_img[i] = B
            new_img = np.resize(new_img,(new_img.shape[0],new_img.shape[1],1))
        return(new_img)
    
def tilt (img,tilt_range):
    if np.max(img)>2:
        img = img/255

    width = img.shape[1]
    height = img.shape[0]
    img = np.array(img)
    tilt_scale_width = np.random.randint(-tilt_range,tilt_range+1)
    l = tilt_scale_width
    int_tilt = int(tilt_scale_width)

    for i in range(height):
        img[i,:] = np.roll(img[i,:],int_tilt-int(l/2))
        tilt_scale_width = tilt_scale_width - tilt_scale_width/height
        int_tilt = int(tilt_scale_width)
        
    img = np.resize(img,(img.shape[0],img.shape[1],1))
    return(img)

def g_blur(img,sigma):
    if np.max(img)>2:
        img = img/255

    rn = sigma*np.random.random()
    img = cv2.GaussianBlur(img,(5,5),rn)
    img = np.resize(img,(img.shape[0],img.shape[1],1))
    return(img)

def reverse(img):
    if np.max(img)>2:
        img = img/255
    
    img = np.resize(img,(img.shape[0],img.shape[1],1))
    return(1-img)

def random_blank(n):
    space = np.random.randint(n+1)
    blank = " " * space
    return(blank)


def random_special(n):
    case = [':', '-', ';', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '{', '}', '[', ']', '?', '"', "'", '~', '>', '<', '/',
           ' ', '=', '+', '_']
    n_case = len(case)
    n_iter = np.random.randint(n)
    text = ""
    for i in range(n_iter):
        idx = np.random.randint(n_case)
        text = text + case[idx]
    return(text)



def text_img_generator_clean(width,height,kor_source,eng_source,font_path):

    
    font_list = os.listdir(font_path)
    n_font = len(font_list)
    font_rand_idx = np.random.randint(n_font)
    language_rn =  np.random.random() 
    text_length_rn = np.random.random()
    number_add_rn = np.random.random()
    
    
    #한글텍스트생성
    if language_rn < 0.4:
        R0 = np.random.randint(len(kor_source))
        text_img = kor_source[R0][0]
        text_label = kor_source[R0][0]
        
        if text_length_rn > 0.02:
            R1 = np.random.randint(len(kor_source))
            T1 = kor_source[R1][0]
            text_img = text_img + random_blank(4) + T1
            text_label = text_label + T1
                
        if text_length_rn > 0.05:
            R2 = np.random.randint(len(kor_source))
            T2 = kor_source[R2][0]
            text_img = text_img + random_blank(4) + T2
            text_label = text_label + T2
                
        if text_length_rn > 0.20:
            R3 = np.random.randint(len(kor_source))
            T3 = kor_source[R3][0]
            text_img = text_img + random_blank(2) + T3
            text_label = text_label + T3

        if text_length_rn > 0.40:
            R4 = np.random.randint(len(kor_source))
            T4 = kor_source[R4][0]
            text_img = text_img + random_blank(1) + T4
            text_label = text_label + T4
            
        text_length = len(text_img)

    #혼합텍스트생성
    elif language_rn > 0.4 and language_rn <= 0.6:
        R0 = np.random.randint(len(kor_source))
        R1 = np.random.randint(len(eng_source))
        kor_text = kor_source[R0]
        eng_text = eng_source[R1]
        
      

        if language_rn >= 0.55:
            text_img = kor_text + eng_text
            text_label = text_img
            
        elif language_rn > 0.5 and language_rn <= 0.55:
            text_img = kor_text[0] + eng_text + kor_text[1]
            text_label = text_img
            
        elif language_rn > 0.45 and language_rn <= 0.5:
            text_img = eng_text[0:2] + kor_text + eng_text[2:4]
            text_label = text_img
            
        else:
            text_img = eng_text + kor_text
            text_label = text_img
            
    #영문텍스트생성
    elif language_rn > 0.6 and language_rn <= 0.8:
        R0 = np.random.randint(len(eng_source))
        L0 = np.random.randint(1,3)
        text_img = eng_source[R0][0:L0]
        text_label = eng_source[R0][0:L0]
        if text_length_rn > 0.02 :
            R1 = np.random.randint(len(eng_source))
            L1 = np.random.randint(1,3)
            T1 = eng_source[R1][0:L1]
            text_img = text_img + random_blank(4) + T1
            text_label = text_label + T1
                
        if text_length_rn > 0.05 :
            R2 = np.random.randint(len(eng_source))
            L2 = np.random.randint(1,3)
            T2 = eng_source[R2][0:L2]
            
            text_img = text_img + random_blank(3) + T2
            text_label = text_label + T2
           
        if text_length_rn > 0.20:
            R3 = np.random.randint(len(eng_source))
            L3 = np.random.randint(1,3)
            T3 = eng_source[R3][0:L3]
            text_img = text_img + random_blank(2) + T3
            text_label = text_label + T3
            
        if text_length_rn > 0.40:
            R4= np.random.randint(len(eng_source))
            L4= np.random.randint(1,3)
            T4= eng_source[R1][0:L4]
            text_img = text_img + random_blank(1) + T4
            text_label = text_label + T4
        text_length = len(text_img)
        
    #숫자텍스트생성
    else:
        length = np.random.randint(3,11)
        text_img = str(np.random.randint(0,10))
        text_label = text_img
        for a in range(length):
            A = str(np.random.randint(0,10))
            text_img = text_img + A
            text_label = text_label + A
            if np.random.random() < 0.2:
                text_img = text_img + random_blank(2)
        
        
    #숫자치환    
    if number_add_rn < 0.2:
        length = np.random.randint(0,3)
        num = str(np.random.randint(0,10))
        for a in range(length):
            num = num + str(np.random.randint(0,10))
            
        start = np.random.randint(0,max((len(text_img)-length),1))
        text_img = list(text_label[:-1])
        text_img[start:start + length] = num
        text_img = "".join(text_img)
        text_label = text_img

        
    font_type = font_list[font_rand_idx]
    img = Image.new('RGB', (width, height), color = (255,255,255))
    d = ImageDraw.Draw(img)
    
    #이미지생성
    margin_start = 5

    for char in text_img:
        font_size = 18
        font = ImageFont.truetype(font_path+font_type, font_size)
        top_margin = 6
        d.text((margin_start,top_margin), char, fill=(0,0,0), font=font)
        text_shape = font.getsize(char)
        if " " in char:
            margin_start = margin_start + text_shape[0]
            continue
            
        margin_start = margin_start + text_shape[0]

    
    img = np.array(img.convert('L'))/255
    
        
    img = np.reshape(img,(height,width))
    return(img, text_label)


def text_img_generator_mild(width,height,kor_source,eng_source,font_path):

    
    font_list = os.listdir(font_path)
    n_font = len(font_list)
    font_rand_idx = np.random.randint(n_font)
    rand_text_color = np.random.randint(0,64)
    rand_bg_color = np.random.randint(192,256)
    rand_line_color = np.random.randint(64,128)
    width_point1, width_point2 = np.random.randint(0,width,2)
    height_point1, height_point2 = np.random.randint(0,height,2)
    language_rn =  np.random.random() 
    text_length_rn = np.random.random()
    number_add_rn = np.random.random()
    
    if language_rn < 0.4:
        R0 = np.random.randint(len(kor_source))
        text_img = kor_source[R0][0]
        text_label = kor_source[R0][0]
        
        if text_length_rn > 0.02:
            R1 = np.random.randint(len(kor_source))
            T1 = kor_source[R1][0]
            if np.random.random()<0.8:
                text_img = text_img + random_blank(4) + T1
                text_label = text_label + T1
            else:
                text_img = text_img + random_special(1) + T1
                text_label = text_label + T1
                
        if text_length_rn > 0.05:
            R2 = np.random.randint(len(kor_source))
            T2 = kor_source[R2][0]
            if np.random.random()<0.8:
                text_img = text_img + random_blank(4) + T2
                text_label = text_label + T2
            else:
                text_img = text_img + random_special(1) + T2
                text_label = text_label + T2
                
        if text_length_rn > 0.20:
            R3 = np.random.randint(len(kor_source))
            T3 = kor_source[R3][0]
            if np.random.random()<0.8:
                text_img = text_img + random_blank(2) + T3
                text_label = text_label + T3
            else:
                text_img = text_img + random_special(1) + T3
                text_label = text_label + T3
        if text_length_rn > 0.40:
            R4 = np.random.randint(len(kor_source))
            T4 = kor_source[R4][0]
            if np.random.random()<0.8:
                text_img = text_img + random_blank(1) + T4
                text_label = text_label + T4
            else:
                text_img = text_img + random_special(1) + T4
                text_label = text_label + T4
            
        text_length = len(text_img)

    
    elif language_rn > 0.4 and language_rn <= 0.6:
        R0 = np.random.randint(len(kor_source))
        R1 = np.random.randint(len(eng_source))
        kor_text = kor_source[R0]
        eng_text = eng_source[R1]
        
      

        if language_rn >= 0.55:
            text_img = kor_text + eng_text
            text_label = text_img
            
        elif language_rn > 0.5 and language_rn <= 0.55:
            text_img = kor_text[0] + eng_text + kor_text[1]
            text_label = text_img
            
        elif language_rn > 0.45 and language_rn <= 0.5:
            text_img = eng_text[0:2] + kor_text + eng_text[2:4]
            text_label = text_img
            
        else:
            text_img = eng_text + kor_text
            text_label = text_img
            
    elif language_rn > 0.6 and language_rn <= 0.8:
        R0 = np.random.randint(len(eng_source))
        L0 = np.random.randint(1,3)
        text_img = eng_source[R0][0:L0]
        text_label = eng_source[R0][0:L0]
        if text_length_rn > 0.02 :
            R1 = np.random.randint(len(eng_source))
            L1 = np.random.randint(1,3)
            T1 = eng_source[R1][0:L1]
            if np.random.random()<0.8:
                text_img = text_img + random_blank(4) + T1
                text_label = text_label + T1
            else:
                text_img = text_img + random_special(1) + T1   
                text_label = text_label + T1
                
        if text_length_rn > 0.05 :
            R2 = np.random.randint(len(eng_source))
            L2 = np.random.randint(1,3)
            T2 = eng_source[R2][0:L2]
            
            if np.random.random()<0.8:
                text_img = text_img + random_blank(3) + T2
                text_label = text_label + T2
            else:
                text_img = text_img + random_special(1) + T2 
                text_label = text_label + T2
           
        if text_length_rn > 0.20:
            R3 = np.random.randint(len(eng_source))
            L3 = np.random.randint(1,3)
            T3 = eng_source[R3][0:L3]
            if np.random.random()<0.8:
                text_img = text_img + random_blank(2) + T3
                text_label = text_label + T3
            else:
                text_img = text_img + random_special(1) + T3
                text_label = text_label + T3
            
        if text_length_rn > 0.40:
            R4= np.random.randint(len(eng_source))
            L4= np.random.randint(1,3)
            T4= eng_source[R1][0:L4]
            if np.random.random()<0.8:
                text_img = text_img + random_blank(1) + T4
                text_label = text_label + T4
            else:
                text_img = text_img + random_special(1) + T4
                text_label = text_label + T4
        text_length = len(text_img)
        
    else:
        length = np.random.randint(3,11)
        text_img = str(np.random.randint(0,10))
        text_label = text_img
        for a in range(length):
            A = str(np.random.randint(0,10))
            text_img = text_img + A
            text_label = text_label + A
            if np.random.random() < 0.2:
                text_img = text_img + random_blank(2)
        
        
        
    if number_add_rn < 0.2:
        length = np.random.randint(0,3)
        num = str(np.random.randint(0,10))
        for a in range(length):
            num = num + str(np.random.randint(0,10))
            
        start = np.random.randint(0,max((len(text_img)-length),1))
        text_img = list(text_label[:-1])
        text_img[start:start + length] = num
        text_img = "".join(text_img)
        text_label = text_img

        
    font_type = font_list[font_rand_idx]
    img = Image.new('RGB', (width, height), color = (rand_bg_color, rand_bg_color, rand_bg_color))
    d = ImageDraw.Draw(img)
    if np.random.random() < 0.1:
        d.ellipse([(np.random.randint(0,64),np.random.randint(0,16)),(np.random.randint(64,128),np.random.randint(16,32))],
                fill = (np.random.randint(128,256),np.random.randint(128,256),np.random.randint(128,256)))
    

    margin_start = 5

    for char in text_img:
        font_size = np.random.randint(16,20)
        font = ImageFont.truetype(font_path+font_type, font_size)
        top_margin = np.random.randint(4,7)
        d.text((margin_start,top_margin), char, fill=(rand_text_color,rand_text_color,rand_text_color), font=font)
        text_shape = font.getsize(char)
        if " " in char:
            margin_start = margin_start + text_shape[0]
            continue
            
        if np.random.random()<0.04:
            d.rectangle([(margin_start-0.3,top_margin-0.3),(margin_start+text_shape[0]+0.3,top_margin+text_shape[1]+1)], 
            outline = (rand_text_color,rand_text_color,rand_text_color),width = 1)
        else:
            if np.random.random()<0.05:
                d.ellipse([(margin_start-0.3,top_margin-0.3),(margin_start+text_shape[0]+0.3,top_margin+text_shape[1]+1)],
                fill = None, outline = (rand_text_color,rand_text_color,rand_text_color))
        margin_start = margin_start + text_shape[0]

    
    d.line([(width_point1,height_point1), (width_point2,height_point2)],width = 1, fill = (rand_line_color,rand_line_color,rand_line_color))
    img = np.array(img.convert('L'))/255
    
    if np.random.random() < 0.05:
        img = rotate(img, 6)
    
    
    if np.random.random() < 0.05:
        img = shift(img, 6)

        
    if np.random.random() < 0.05:
        img = shrink(img, 0.3)
    
    if np.random.random() < 0.05:
        img = g_blur(img, 0.2)
    
    if np.random.random() < 0.05:
        img = tilt(img, 15)
    
    if np.random.random() < 0.05:
        img = reverse(img)

        
    img = np.reshape(img,(height,width))
    return(img, text_label)


def text_img_generator(width,height,kor_source,eng_source,font_path):

    
    font_list = os.listdir(font_path)
    n_font = len(font_list)
    font_rand_idx = np.random.randint(n_font)
    rand_text_color = np.random.randint(0,64)
    rand_bg_color = np.random.randint(192,256)
    rand_line_color = np.random.randint(64,128)
    width_point1, width_point2 = np.random.randint(0,width,2)
    height_point1, height_point2 = np.random.randint(0,height,2)
    language_rn =  np.random.random() 
    text_length_rn = np.random.random()
    number_add_rn = np.random.random()
    
    if language_rn < 0.4:
        R0 = np.random.randint(len(kor_source))
        text_img = kor_source[R0][0]
        text_label = kor_source[R0][0]
        
        if text_length_rn > 0.02:
            R1 = np.random.randint(len(kor_source))
            T1 = kor_source[R1][0]
            if np.random.random()<0.8:
                text_img = text_img + random_blank(5) + T1
                text_label = text_label + T1
            else:
                text_img = text_img + random_special(3) + T1
                text_label = text_label + T1
                
        if text_length_rn > 0.05:
            R2 = np.random.randint(len(kor_source))
            T2 = kor_source[R2][0]
            if np.random.random()<0.8:
                text_img = text_img + random_blank(5) + T2
                text_label = text_label + T2
            else:
                text_img = text_img + random_special(3) + T2
                text_label = text_label + T2
                
        if text_length_rn > 0.20:
            R3 = np.random.randint(len(kor_source))
            T3 = kor_source[R3][0]
            if np.random.random()<0.8:
                text_img = text_img + random_blank(3) + T3
                text_label = text_label + T3
            else:
                text_img = text_img + random_special(2) + T3
                text_label = text_label + T3
        if text_length_rn > 0.40:
            R4 = np.random.randint(len(kor_source))
            T4 = kor_source[R4][0]
            if np.random.random()<0.8:
                text_img = text_img + random_blank(1) + T4
                text_label = text_label + T4
            else:
                text_img = text_img + random_special(1) + T4
                text_label = text_label + T4
            
        text_length = len(text_img)

    
    elif language_rn > 0.4 and language_rn <= 0.6:
        R0 = np.random.randint(len(kor_source))
        R1 = np.random.randint(len(eng_source))
        kor_text = kor_source[R0]
        eng_text = eng_source[R1]
        
      

        if language_rn >= 0.55:
            text_img = kor_text + eng_text
            text_label = text_img
            
        elif language_rn > 0.5 and language_rn <= 0.55:
            text_img = kor_text[0] + eng_text + kor_text[1]
            text_label = text_img
            
        elif language_rn > 0.45 and language_rn <= 0.5:
            text_img = eng_text[0:2] + kor_text + eng_text[2:4]
            text_label = text_img
            
        else:
            text_img = eng_text + kor_text
            text_label = text_img
            
    elif language_rn > 0.6 and language_rn <= 0.8:
        R0 = np.random.randint(len(eng_source))
        L0 = np.random.randint(1,3)
        text_img = eng_source[R0][0:L0]
        text_label = eng_source[R0][0:L0]
        if text_length_rn > 0.02 :
            R1 = np.random.randint(len(eng_source))
            L1 = np.random.randint(1,3)
            T1 = eng_source[R1][0:L1]
            if np.random.random()<0.8:
                text_img = text_img + random_blank(5) + T1
                text_label = text_label + T1
            else:
                text_img = text_img + random_special(3) + T1   
                text_label = text_label + T1
                
        if text_length_rn > 0.05 :
            R2 = np.random.randint(len(eng_source))
            L2 = np.random.randint(1,3)
            T2 = eng_source[R2][0:L2]
            
            if np.random.random()<0.8:
                text_img = text_img + random_blank(4) + T2
                text_label = text_label + T2
            else:
                text_img = text_img + random_special(2) + T2 
                text_label = text_label + T2
           
        if text_length_rn > 0.20:
            R3 = np.random.randint(len(eng_source))
            L3 = np.random.randint(1,3)
            T3 = eng_source[R3][0:L3]
            if np.random.random()<0.8:
                text_img = text_img + random_blank(2) + T3
                text_label = text_label + T3
            else:
                text_img = text_img + random_special(1) + T3
                text_label = text_label + T3
            
        if text_length_rn > 0.40:
            R4= np.random.randint(len(eng_source))
            L4= np.random.randint(1,3)
            T4= eng_source[R1][0:L4]
            if np.random.random()<0.8:
                text_img = text_img + random_blank(1) + T4
                text_label = text_label + T4
            else:
                text_img = text_img + random_special(1) + T4
                text_label = text_label + T4
        text_length = len(text_img)
        
    else:
        length = np.random.randint(3,11)
        text_img = str(np.random.randint(0,10))
        text_label = text_img
        for a in range(length):
            A = str(np.random.randint(0,10))
            text_img = text_img + A
            text_label = text_label + A
            if np.random.random() < 0.2:
                text_img = text_img + random_blank(3)
        
        
        
    if number_add_rn < 0.2:
        length = np.random.randint(0,3)
        num = str(np.random.randint(0,10))
        for a in range(length):
            num = num + str(np.random.randint(0,10))
            
        start = np.random.randint(0,max((len(text_img)-length),1))
        text_img = list(text_label[:-1])
        text_img[start:start + length] = num
        text_img = "".join(text_img)
        text_label = text_img

        
    font_type = font_list[font_rand_idx]
    img = Image.new('RGB', (width, height), color = (rand_bg_color, rand_bg_color, rand_bg_color))
    d = ImageDraw.Draw(img)
    if np.random.random() < 0.2:
        d.ellipse([(np.random.randint(0,64),np.random.randint(0,16)),(np.random.randint(64,128),np.random.randint(16,32))],
                fill = (np.random.randint(128,256),np.random.randint(128,256),np.random.randint(128,256)))
    

    margin_start = 5

    for char in text_img:
        font_size = np.random.randint(12,22)
        font = ImageFont.truetype(font_path+font_type, font_size)
        top_margin = np.random.randint(3,8)
        d.text((margin_start,top_margin), char, fill=(rand_text_color,rand_text_color,rand_text_color), font=font)
        text_shape = font.getsize(char)
        if " " in char:
            margin_start = margin_start + text_shape[0]
            continue
            
        if np.random.random()<0.08:
            d.rectangle([(margin_start-0.3,top_margin-0.3),(margin_start+text_shape[0]+0.3,top_margin+text_shape[1]+1)], 
            outline = (rand_text_color,rand_text_color,rand_text_color))
        else:
            if np.random.random()<0.1:
                d.ellipse([(margin_start-0.3,top_margin-0.3),(margin_start+text_shape[0]+0.3,top_margin+text_shape[1]+1)],
                fill = None, outline = (rand_text_color,rand_text_color,rand_text_color))
        margin_start = margin_start + text_shape[0]

    
    d.line([(width_point1,height_point1), (width_point2,height_point2)],width = 1, fill = (rand_line_color,rand_line_color,rand_line_color))
    img = np.array(img.convert('L'))/255
    
    if np.random.random() < 0.2:
        img = rotate(img, 12)
    
    
    if np.random.random() < 0.2:
        img = shift(img, 8)

        
    if np.random.random() < 0.2:
        img = shrink(img, 0.4)
    
    if np.random.random() < 0.2:
        img = g_blur(img, 0.4)
    
    if np.random.random() < 0.2:
        img = tilt(img, 25)
    
    if np.random.random() < 0.2:
        img = reverse(img)

        
    img = np.reshape(img,(height,width))
    return(img, text_label)


def load_dictionary():
    char = ['-', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 
        'z', 'ㄱ', 'ㄲ','ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㄺ', 'ㄽ', 'ㄿ', 'ㄻ', 'ㄼ', 'ㄾ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅄ', 
        'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ', 'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ',
        'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ','ㅟ', 'ㅠ','ㅡ', 'ㅢ', 'ㅣ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' ]

    y_idx = range(len(char))
    dictionary = {}
    for i in range(len(char)):
        dictionary.update({char[i]:y_idx[i]})
    return(dictionary)

def decompose(label,text_length):
    decomposed_label = []
    count = 0
    for e in label:
        decomposed_word = ['-']*text_length
        for char in e:
            if ord('가') <= ord(char) <= ord('힣'):
                    element = hgtk.letter.decompose(char)
                    if element[2] == '':
                        element = element[0:2]
                        decomposed_word[count:count+2] = element
                        count = count + 2
                    else:
                        decomposed_word[count:count+3] = element
                        count = count + 3
                        
            elif ord('A') <= ord(char) <= ord('z'):
                decomposed_word[count] = char
                count = count + 1
                
            elif ord('0') <= ord(char) <= ord('9'):
                decomposed_word[count] = char
                count = count + 1
                
        decomposed_label.append(decomposed_word)
        count = 0
    return(np.array(decomposed_label))

def single_decompose(text,text_length):
        decomposed_word = ['-']*text_length
        count = 0
        for char in text:
            if ord('가') <= ord(char) <= ord('힣'):
                    element = hgtk.letter.decompose(char)
                    if element[2] == '':
                        element = element[0:2]
                        decomposed_word[count:count+2] = element
                        count = count + 2
                    else:
                        decomposed_word[count:count+3] = element
                        count = count + 3
                        
            elif ord('A') <= ord(char) <= ord('z'):
                decomposed_word[count] = char
                count = count + 1
                
                            
            elif ord('0') <= ord(char) <= ord('9'):
                decomposed_word[count] = char
                count = count + 1
                
        count = 0
        return(decomposed_word)

class DataGenerator():
    'Generates data for Keras'
    def __init__(self, batch_size, x_dim, kor_source, eng_source, text_length, dictionary, font_path, text_generator):
        'Initialization'
        self.text_generator = text_generator
        self.text_length = text_length
        self.dictionary = dictionary
        self.kor_source = kor_source
        self.eng_source = eng_source
        self.width = x_dim[1]
        self.height = x_dim[0]
        self.batch_size = batch_size
        self.font_path = font_path

    def data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.height, self.width))
        y = []
        
        # Generate data
        for i in range(self.batch_size):
            # Store sample
            X[i,], text = self.text_generator(self.width,self.height,self.kor_source,self.eng_source,self.font_path)
            
            y.append(text)
        y_hot = one_hot(y, self.text_length, self.dictionary)
        X = np.resize(X,(self.batch_size,self.height,self.width,1))

        return X, np.array(y_hot)

def decoder(pred,dictionary):
    char = set()
    decoded = []
    for e in pred:
        pred = []
        for f in e:
            idx = np.argmax(f, axis = 0)
            d = list(dictionary.keys())[list(dictionary.values()).index(idx)]
            pred.append(d)
        decoded.append(pred)
    return(decoded)

def load_images(path, width, height):
    names = os.listdir(path)
    imgset = []
    count = 0
    for e in names:
        img = cv2.imread(path+e,0)
        img = cv2.resize(img,(width,height))
        imgset.append(img)
        if count%50000 == 0:
            print(count)
            count = count +1
    return(np.array(imgset))


def test():
    print("Package is Loaded")