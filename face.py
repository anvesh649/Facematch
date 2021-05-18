import sys
import cv2
import torch
import os
import numpy as np
import torch
from torch import nn

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def read(path):
    img=cv2.imread(path)
    try:
        assert img!=None,"Specify a proper path for the image"
    except:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            return img
class SiameseNN(nn.Module):
    def __init__(self):
        super(SiameseNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(14976, 128),
            nn.ReLU(),
            nn.Linear(128, 50)
        )
    def forward(self, im1, im2):
        feat1 = self.cnn(im1)
        feat2 = self.cnn(im2)
        return torch.norm(feat1 - feat2, dim=-1)
def face_extract(gray):
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	roi_gray=np.zeros((60,80))
	for (x,y,w,h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
	return cv2.resize(roi_gray,(80,60))

model=torch.load("78",map_location="cpu")
model=model.cpu()

with torch.no_grad():
	x1=read(sys.argv[1])
	x1=face_extract(x1)/255

	x2=read(sys.argv[2])
	x2=face_extract(x2)/255


	im1 = torch.from_numpy(x1.reshape(1,1,60,80)).float()
	im2 = torch.from_numpy(x2.reshape(1,1,60,80)).float()
	dist = model(im1, im2).numpy()
	dist=dist.flatten()
	dist=dist[0]
	thresh=0.8
	if(dist<thresh):
		print("match")
		conf=(thresh-dist)/thresh
		conf=round(conf, 2)
		print("confidence that it matches:",conf)
	else:
		print("no match")
		dist=dist**(0.5)
		conf=(dist-thresh)/thresh
		if(conf>1):
			conf=1
		conf=round(conf, 2)
		print("confidence that it doesn't match:",conf)
