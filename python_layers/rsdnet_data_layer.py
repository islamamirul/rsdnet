import random
import cv2
import numpy as np
from PIL import Image
import os
from scipy import misc
import matplotlib
import scipy.io
from Cython.Compiler.PyrexTypes import unique
matplotlib.use('TkAgg')

 
caffe_root = '/home/amirul/caffe-rsdnet/'  # MODIFY PATH for YOUR SETTING

import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

class SegImageDataLayer(caffe.Layer):

    def get_param(self):

            param = eval(self.param_str)
            
            # source txt folder
            self.source = param['source']
            
            # root folder
            if 'root_folder' in param.keys():
                self.root_folder = param['root_folder']
            else:
                self.root_folder = ''
                
            #batch size
            self.batch_size = param['batch_size']
            
            #phase
            
            #self.phase = param['phase']
            
            #shuffle each epoch
            if 'shuffle' in param.keys():
                self.shuffle = param['shuffle']
            else:
                self.shuffle = False
                
            #meal value       
        
            if 'mean_value' in param.keys():
                self.mean_value = param['mean_value']
            else:
                self.mean_value = ()
                
            #scale
            if 'scale' in param.keys():
                self.scale = param['scale']
            else:
                self.scale = ()
        
            #new height
        
            if 'new_height' in param.keys():
                self.new_height = param['new_height']
            else:
                self.new_height = 0
            
            #new width
            if 'new_width' in param.keys():
                self.new_width = param['new_width']
            else:   
                self.new_width = 0
        
            #crop height 
            
            if 'crop_height' in param.keys():
                self.crop_height = param['crop_height']
            else:
                self.crop_height = 0
        
            #crop width
            
            if 'crop_width' in param.keys():
                self.crop_width = param['crop_width']
            else:
                self.crop_width = 0
        
            #mirror
        
            if 'mirror' in param.keys():
                self.mirror = param['mirror']
            else:
                self.mirror = False        

    
    def setup(self,bottom,top):
            self.get_param()
  
            self.txtlines = open(self.source,'r').read().splitlines()
        
            self.linen = len(self.txtlines)
        
            if self.shuffle:
                random.shuffle(self.txtlines)
            self.idx = 0
            self.img_shape = self.img_shape()

    def reshape(self,bottom,top):
            self.load()
            self.transform()
            #self.label[self.label>0] = 1
            top[0].reshape(*self.data.shape)
            top[1].reshape(*self.label.shape)
            top[2].reshape(self.batch_size )
    def forward(self,bottom,top):

            top[0].data[...] = self.data
            top[1].data[...] = self.label
            top[2].data[...] = self.count

    def backward(self,top,propagate_down,bottom):
            pass

    def img_shape(self):

            im = Image.open(os.path.join(self.root_folder,self.txtlines[0].split(' ')[0])) 
            im = np.array(im,dtype=np.float32)
            
            if len(im.shape) == 2:
                c = 1
            else: 
                c = im.shape[2]
            
            if (self.crop_height>0) and (self.crop_width>0):
                return [self.crop_height, self.crop_width, c]
            elif (self.new_height>0) and (self.new_width>0):
                return [self.new_height,self.new_width,c]
            else:
                return [im.shape[0],im.shape[1],c]
         
    def load(self):
        
        self.data_buf  = []
        self.label_buf = []   
        self.count_buf = [] 
         
        for i in range(self.batch_size):
            if not os.path.exists(os.path.join(self.root_folder,self.txtlines[self.idx].split(' ')[0])):
                print 'img not exists!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!' 
            
            im = Image.open(os.path.join(self.root_folder,self.txtlines[self.idx].split(' ')[0])) 
            im = np.array(im,dtype=np.float32)
            
            #print im.shape
            if len(im.shape) == 2:
                im = im[np.newaxis,...] 
            else:
                im = im[:,:,::-1]
                im = im.transpose((2,0,1))
            
            #print im.shape
            self.data_buf.append(im)
            
            # Loading Stacks of label
            
            
            
            if not os.path.exists(os.path.join(self.root_folder,self.txtlines[self.idx].split(' ')[1])):
                print 'label not exists!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                
            #lb = Image.open(os.path.join(self.root_folder,self.txtlines[self.idx].split(' ')[1]))
              
            ## Load Labels as .mat files
            label = scipy.io.loadmat(os.path.join(self.root_folder,self.txtlines[self.idx].split(' ')[1]))
            lb = label ['stacked_masks'] 
            lb = np.array(lb,dtype=np.uint8)
            if len(lb.shape) == 2:
                lb = lb[np.newaxis,...]
            else:
                lb = lb.transpose((2,0,1))
            
            # Loading the label into label buffer

            self.label_buf.append(lb)
            
            # Loding the count into the buffer
            
            c = int (self.txtlines[self.idx].split(' ')[2])
            #count = np.array(c, dtype=np.uint8)

            self.count_buf.append(c)
            
            self.idx = self.idx + 1

            if self.idx == self.linen:  
                self.idx = 0
                if self.shuffle:
                    random.shuffle(self.txtlines)
         
    def transform(self):
        
        self.data = np.zeros((self.batch_size, self.img_shape[2], self.img_shape[0], self.img_shape[1]), dtype=np.float32)
        self.label = np.zeros((self.batch_size, 12,  self.img_shape[0], self.img_shape[1]), dtype = np.uint8)
        self.count = np.zeros((self.batch_size, 1), dtype = np.uint8)
        
        #resize
        if (self.new_height>0) or (self.new_width>0):
            rs_height = self.data.shape[2]
            rs_width  = self.data.shape[3]

            if self.new_height>0:
                rs_height = self.new_height
            if self.new_width>0:
                rs_width  = self.new_width

            
            for i in range(self.batch_size):
                
                new_data = cv2.resize(self.data_buf[i].transpose((1,2,0)),(rs_width,rs_height),interpolation = cv2.INTER_NEAREST)
                
                if len(new_data.shape) == 2:
                    new_data = new_data[np.newaxis,...]
                else:
                    new_data = new_data.transpose((2,0,1))
                
                self.data_buf[i] = new_data
                
                new_label = cv2.resize(self.label_buf[i].transpose((1,2,0)),(rs_width,rs_height), interpolation = cv2.INTER_NEAREST)
                
                #print np.unique(new_label)
                if len(new_label.shape) == 2:
                    new_label = new_label[np.newaxis,...]
                else:
                    new_label = new_label.transpose((2,0,1))
                
                #print np.unique(new_label)
                self.label_buf[i] = new_label
        
        
        #crop
        if (self.crop_height>0) or (self.crop_width>0):
            
            cr_height = self.img_shape[0]
            cr_width  = self.img_shape[1]

            if self.crop_height>0:
                cr_height = self.crop_height
            if self.crop_width>0:
                cr_width = self.crop_width

            #print cr_height,cr_width
            for i in range(self.batch_size):

                src_h = self.data_buf[i].shape[1]
                src_w = self.data_buf[i].shape[2]
                
                #print src_h, src_w
                
                
                pad_height = max(cr_height - src_h, 0);
                pad_width  = max(cr_width - src_w, 0);
                    
                if pad_height > 0 or pad_width > 0:
                        
                    
                    new_data2  = cv2.copyMakeBorder(self.data_buf[i].transpose((1,2,0)), 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, cv2.inRange(self.mean_value[0], self.mean_value[1], self.mean_value[2])) 
                    new_label2 = cv2.copyMakeBorder(self.label_buf[i].transpose((1,2,0)), 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, 0)
                    
                    #print new_data2.shape
                    #print new_label2.shape
                    
                    self.data_buf[i] = new_data2.transpose((2,0,1))
                    self.label_buf[i] = new_label2.transpose((2,0,1))
                    
                    src_h = self.data_buf[i].shape[1]
                    src_w = self.data_buf[i].shape[2]
                
                self.data_buf[i]  = self.data_buf[i] [ :, int((src_h-cr_height)/2) : int(int((src_h-cr_height)/2) + cr_height), int((src_w-cr_width)/2) : int(int((src_w-cr_width)/2) + cr_width)]
                self.label_buf[i] = self.label_buf[i][ :, int((src_h-cr_height)/2) : int(int((src_h-cr_height)/2) + cr_height), int((src_w-cr_width)/2) : int(int((src_w-cr_width)/2) + cr_width)]
                    
                #print self.label_buf[i].shape
        
        for i in range(self.batch_size):
            
            self.data[i, :, :, :] = self.data_buf[i]
            self.label[i, :, :,:] = self.label_buf[i]
            self.count[i,:] = self.count_buf[i]
        #sub mean        
        if len(self.mean_value)>0:
            if len(self.mean_value) != self.data.shape[1]:
                print
                print '*********************************************************'
                print '*****Mean value numbers mismatch with data channels!*****'
                print '*********************************************************'
                print 
                sys.exit(0)
            for i in range(self.data.shape[1]):
                self.data[:,i,:,:] = self.data[:,i,:,:] - self.mean_value[i]
        #mul scale
        if len(self.scale)>0:
            if len(self.scale) != self.data.shape[1]:
                print
                print '*********************************************************'
                print '********Scale numbers mismatch with data channels********'
                print '*********************************************************'
                print
                sys.exit(0)
            for i in range(self.data.shape[1]):
                self.data[:,i,:,:] = self.data[:,i,:,:] * self.scale[i]
        #mirror
        if self.mirror and random.random() > 0.5:
            self.data  = self.data[:, :, :, ::-1]
            self.label = self.label[:,:,:, ::-1] 

