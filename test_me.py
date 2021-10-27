#from ttk import *
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from PIL import Image, ImageDraw, ImageFont
import os
import time
from Object_lib import object_detection


start = time.time()

#Define Path
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
test_path = 'data/alien_test'

#Load the pre-trained models
#model = load_model(model_path)
#model.load_weights(model_weights_path)

#Define image parameters
img_width, img_height = 150, 150

#Prediction Function
def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  print(array)
  result = array[0]
  #print(result)
  answer = np.argmax(result)
  

  return answer

def region_of_interest1(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255 # <-- This line altered for grayscale.
    
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    # Retrieve the number of color channels of the image.
    channel_count = img.shape[2]
    # Create a match color with the same color channel counts.
    match_mask_color = (255,) * channel_count
      
    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
    
    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    line_img = np.zeros(
	  (
		img.shape[0],
		img.shape[1],
		3
	  ),
	  dtype=np.uint8
    )
    img = np.copy(img)
    if lines is None:
       return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # Return the modified image.
    return img


modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'yolov3.weights'

labelsPath = 'coco.names'
labels = open(labelsPath).read().strip().split('\n')

np.random.seed(10)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

outputLayer = net.getLayerNames()
outputLayer = [outputLayer[i[0] - 1] for i in net.getUnconnectedOutLayers()]


global last_frame                                      #creating global              variable
last_frame = np.zeros((480, 640, 3), dtype=np.uint8)
global last_frame2                                      #creating global      variable
last_frame2 = np.zeros((480, 640, 3), dtype=np.uint8)

global last_frame3                                      #creating global      variable
last_frame3 = np.zeros((480, 640, 3), dtype=np.uint8)



global cap
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('harder_challenge_video.mp4')
global obj
obj=0
def show_vid():
    global obj
    global last_frame

    global last_frame2
    global last_frame3
    # Check if camera opened successfully
    cpt=0
    xc="Image"
    #while(cap.isOpened()):
    # Capture frame-by-frame
    ret, image = cap.read()
    
    #image=cv2.resize(image,(250,450))
  
    if ret == True:
        image = cv2.resize(image,(300,450))
        if obj==1:
            image=object_detection(image,labels,COLORS,net,outputLayer)
        print(obj)
    
        #print('This image is:', type(image), 'with dimensions:', image.shape)
        #cv2.imshow('Video Frame',image)
        #plt.imshow(image)
        #plt.show()
        #cv2.imwrite('1.jpg',image)
        #pic = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)     #we can change the display color of the frame gray,black&white here
        
        height = image.shape[0]
        width = image.shape[1]
        region_of_interest_vertices = [(0, height),(width / 2, height / 2),(width, height),]
        cropped_image = region_of_interest(image,np.array([region_of_interest_vertices], np.int32),)
        #cv2.imshow('croped Frame',cropped_image)
        
        #plt.figure()
        #plt.imshow(cropped_image)
        #plt.show()
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cannyed_image = cv2.Canny(gray_image, 100, 200)
        # Moved the cropping operation to the end of the pipeline.
        cropped_image = region_of_interest1(cannyed_image,np.array([region_of_interest_vertices], np.int32))
        #cv2.imshow('croped Frame2',cropped_image)
        last_frame2 = cropped_image.copy()
        cpt+=1
        #plt.figure()
        #plt.imshow(cropped_image)
        #plt.show()
        lines = cv2.HoughLinesP(
              cropped_image,
              rho=6,
              theta=np.pi / 60,
              threshold=160,
              lines=np.array([]),
              minLineLength=40,
              maxLineGap=25
          )
     
        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []
     
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
            if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
                continue
            if slope <= 0: # <-- If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else: # <-- Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
                
        min_y = int(image.shape[0] * (3 / 5))
        max_y = int(image.shape[0])
        print(left_line_y,left_line_x)
        line_image=image
        if len(left_line_y)!=0 :
            poly_left = np.poly1d(np.polyfit(left_line_y,left_line_x,deg=1))
     
            left_x_start = int(poly_left(max_y))
            left_x_end = int(poly_left(min_y))
            if len(right_line_y)!=0 :
                poly_right = np.poly1d(np.polyfit(
                      right_line_y,
                      right_line_x,
                      deg=1
                   ))
     
                right_x_start = int(poly_right(max_y))
                right_x_end = int(poly_right(min_y))
                #print(left_x_end,right_x_end)
        
                line_image = draw_lines(
                image,
                [[
                    [left_x_start, max_y, left_x_end, min_y],
                    [right_x_start, max_y, right_x_end, min_y],
                ]],
                thickness=5,
            )
                
        last_frame3 = line_image.copy()
        img = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_vid)
            # Press Q on keyboard to  exit
 
   


def show_vid2():
    #pic2 = cv2.cvtColor(last_frame2, cv2.COLOR_BGR2GRAY)
    img2 = Image.fromarray(last_frame2)
    img2tk = ImageTk.PhotoImage(image=img2)
    lmain2.img2tk = img2tk
    lmain2.configure(image=img2tk)
    lmain2.after(10, show_vid2)

def show_vid3():
    #pic3 = cv2.cvtColor(last_frame3, cv2.COLOR_BGR2GRAY)
    img3 = Image.fromarray(last_frame3)
    img3tk = ImageTk.PhotoImage(image=img3)
    lmain3.img3tk = img3tk
    lmain3.configure(image=img3tk)
    lmain3.after(10, show_vid3)


    
def lane():
        show_vid()
        show_vid2()
        show_vid3()
        #show_vid4()
        
def object1():
        global obj
        if obj==0:
            obj=1#show_vid4()
        else:
            obj=0
if __name__ == '__main__':
    root=tk.Tk()                                     #assigning root variable        for Tkinter as tk
    load = Image.open("logo.png")
    render = ImageTk.PhotoImage(load)
    
    lmain =tk.Label(root, image=render,borderwidth=15, highlightthickness=5, height=250, width=250, bg='white')
    lmain .image = render
    lmain .place(x=250, y=20)

    lmain2=tk.Label(root, image=render,borderwidth=15, highlightthickness=5, height=250, width=250, bg='white')
    lmain2.image = render
    lmain2.place(x=600, y=20)

    lmain3=tk.Label(root, image=render,borderwidth=15, highlightthickness=5, height=250, width=250, bg='white')
    lmain3.image = render
    lmain3.place(x=250, y=350)

    
    root.title("Real Time Lane And Object Detection")            #you can give any title
    root.geometry("1100x650") #size of window , x-axis, yaxis
    #exitbutton = Button(root, text='Quit',fg="red",command=   root.destroy).pack(side = BOTTOM,)
    quitButton2 = Button(root,command=root.destroy, text='Quit',fg="red",activebackground="dark red",width=20)
    quitButton2.place(x=10, y=250)
    
    quitButton = Button(root,command=lane, text="Lane Detection",fg="blue",activebackground="dark red",width=20)
    quitButton.place(x=10, y=50)
    quitButton1 = Button(root,command=object1, text="Object Detection",fg="blue",activebackground="dark red",width=20)
    quitButton1.place(x=10, y=150)
        
    #quitButton = Button(root,command=self.object, text="Object Detection",fg="blue",activebackground="dark red",width=20)
    #quitButton.place(x=10, y=100)
    
    root.mainloop()                                  #keeps the application in an infinite loop so it works continuosly
    cap.release()

