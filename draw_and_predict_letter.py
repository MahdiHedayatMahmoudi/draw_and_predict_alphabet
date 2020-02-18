from tkinter import *
from PIL import Image,ImageDraw
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf


letters_dict={'0': 'a' , '1': 'b' , '2': 'c' ,'3': 'd' , '4': 'e' , '5': 'f', 
              '6': 'g' , '7': 'h' , '8': 'i' ,'9': 'j' , '10': 'k' , '11': 'l',
              '12': 'm' , '13': 'n' , '14': 'o' ,'15': 'p' , '16': 'q' , '17': 'r',
              '18': 's' , '19': 't' , '20': 'u' ,'21': 'v' , '22': 'w' , '23': 'x',
              '24': 'y' , '25': 'z' }


letters_dict_capitals={'0': 'A' , '1': 'B' , '2': 'C' ,'3': 'D' , '4': 'E' , '5': 'F', 
              '6': 'G' , '7': 'H' , '8': 'I' ,'9': 'J' , '10': 'K' , '11': 'L',
              '12': 'M' , '13': 'N' , '14': 'O' ,'15': 'P' , '16': 'Q' , '17': 'R',
              '18': 'S' , '19': 'T' , '20': 'U' ,'21': 'V' , '22': 'W' , '23': 'X',
              '24': 'Y' , '25': 'Z' }

dir_model='./output/model.h5'
with tf.device('/cpu:0'):
    model=load_model(dir_model,compile=False)


class ImageGenerator:
    def __init__(self,parent,posx,posy,sizex,sizy,*kwargs):
        self.parent = parent
        self.posx = posx
        self.posy = posy
        self.sizex = sizex
        self.sizey = sizey
        self.b1 = "up"
        self.xold = None
        self.yold = None 
        top=Toplevel()
        top.title('Text')
        top.wm_geometry("794x370")
        self.drawing_area=Canvas(self.parent,width=self.sizex,height=self.sizey)
        self.show_text=Canvas(top)
        self.show_text.pack(fill=BOTH, expand=1)
        self.drawing_area.place(x=self.posx,y=self.posy)
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.b1up)
        
        self.parent.bind('<Return>', lambda event: self.save())
        self.parent.bind('<space>', lambda event: self.space())
        self.parent.bind('<BackSpace>', lambda event: self.delete())
        self.parent.bind('l', lambda event: self.new_line())
        self.parent.bind('c', lambda event: self.capital_letter())
        
        
        self.button=Button(text="Done!",command=self.save)
        
        #self.button.pack(side=BOTTOM)
        self.button.grid(column=0, row=1)
        
        self.button4=Button(text="Capital!",command=self.capital_letter)
        
        #self.button4.pack(side=BOTTOM)
        self.button4.grid(column=1, row=1)
        
        self.button3=Button(text="New line!",command=self.new_line)
        
        #self.button3.pack(side=BOTTOM)
        self.button3.grid(column=2, row=1)
    
        
        self.button2=Button(text="space!",command=self.space)
        #self.button2.pack(side=BOTTOM)
        self.button2.grid(column=3, row=1)
        
        self.button3=Button(text="delete!",command=self.delete)
        #self.button3.pack(side=BOTTOM)
        self.button3.grid(column=4, row=1)

        self.image=Image.new("RGB",(self.sizex,self.sizex),(255,255,255))
        self.draw=ImageDraw.Draw(self.image)
        self.sentence=None

    def space(self):
        self.sentence=self.sentence+' '
        self.drawing_area.delete("all")
        self.show_text.delete("all")
        self.image=Image.new("RGB",(self.sizex,self.sizey),(255,255,255))
        self.draw=ImageDraw.Draw(self.image)
        #self.drawing_area.create_text(20, 30, anchor=W, font=("Times",15),text=self.sentence )
        self.show_text.create_text(20, 30, anchor=W, font=("Times",15),text=self.sentence )
        
        
    def delete(self):
        list_char=list(self.sentence)
        list_char.pop(-1)
        self.sentence=''.join(list_char)
        self.drawing_area.delete("all")
        self.show_text.delete("all")
        self.image=Image.new("RGB",(self.sizex,self.sizey),(255,255,255))
        self.draw=ImageDraw.Draw(self.image)
        #self.drawing_area.create_text(20, 30, anchor=W, font=("Times",15),text=self.sentence )
        self.show_text.create_text(20, 30, anchor=W, font=("Times",15),text=self.sentence )

    def new_line(self):
        self.sentence=self.sentence+'\n'
    def save(self):
        filename = "temp.jpg"
        self.image.save(filename)
        
        img=np.array(self.image)
        img=img.astype(np.uint64)
        img=cv2.resize(img,(28,28) , interpolation=cv2.INTER_NEAREST)
        

        img=img[:,:,0]
        imgn=(img[:,:]==0)*255
        

        img=imgn/255.00
        
        with tf.device('/cpu:0'):
            label_prediction=model.predict(img.reshape(1,28,28,1))
        class_predicted=np.argmax(label_prediction)
        if self.sentence is None:
            self.sentence=letters_dict[str(class_predicted)]
        else:
            self.sentence=self.sentence+letters_dict[str(class_predicted)]
            
        self.drawing_area.delete("all")
        self.show_text.delete("all")
        self.image=Image.new("RGB",(self.sizex,self.sizey),(255,255,255))
        self.draw=ImageDraw.Draw(self.image)
        #self.drawing_area.create_text(20, 30, anchor=W, font=("Times",15),text=self.sentence )
        self.show_text.create_text(20, 30, anchor=W, font=("Times",15),text=self.sentence )
        
        
    def capital_letter(self):
        filename = "temp.jpg"
        self.image.save(filename)
        
        img=np.array(self.image)
        img=img.astype(np.uint64)
        img=cv2.resize(img,(28,28) , interpolation=cv2.INTER_NEAREST)
        

        img=img[:,:,0]
        imgn=(img[:,:]==0)*255
        

        img=imgn/255.00
        
        with tf.device('/cpu:0'):
            label_prediction=model.predict(img.reshape(1,28,28,1))
        class_predicted=np.argmax(label_prediction)
        if self.sentence is None:
            self.sentence=letters_dict_capitals[str(class_predicted)]
        else:
            self.sentence=self.sentence+letters_dict_capitals[str(class_predicted)]
            
        self.drawing_area.delete("all")
        self.show_text.delete("all")
        self.image=Image.new("RGB",(self.sizex,self.sizey),(255,255,255))
        self.draw=ImageDraw.Draw(self.image)
        #self.drawing_area.create_text(20, 30, anchor=W, font=("Times",15),text=self.sentence )
        self.show_text.create_text(20, 30, anchor=W, font=("Times",15),text=self.sentence )
        
        

    def clear(self):
        self.drawing_area.delete("all")
        self.image=Image.new("RGB",(self.sizex,self.sizey),(255,255,255))
        self.draw=ImageDraw.Draw(self.image)

    def b1down(self,event):
        self.b1 = "down"

    def b1up(self,event):
        self.b1 = "up"
        self.xold = None
        self.yold = None

    def motion(self,event):
        line_width=75
        paint_color = 'black'
        if self.b1 == "down":
            if self.xold is not None and self.yold is not None:
                self.drawing_area.create_line(self.xold, self.yold, event.x, event.y,
                               width=line_width, fill=paint_color, dash=(),capstyle=ROUND, smooth=TRUE, splinesteps=1)
                self.draw.line(((self.xold,self.yold),(event.x,event.y)),(0,0,0),width=75)
                


        self.xold = event.x
        self.yold = event.y

if __name__ == "__main__":
    sizex=650
    sizey=650
    root=Tk()
    root.title('Draw Here')
    root.wm_geometry("%dx%d+%d+%d" % (sizex, sizey, 10, 10))
    root.config(bg='white')
    ImageGenerator(root,10,10,sizex,sizey)
    root.mainloop() 
