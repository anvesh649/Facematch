Ensure the installation of all the libraries mentioned below to run face.py
1)pytorch-1.18
2)opencv
3)numpy

Steps to run face.py
if you are using anaconda or windows terminal navigating to the folder which contains all the files in repository
along with two images the passport images and the selfie
"arg1.jpg"-the location of the image which contains the passport image
"arg2.jpg"-the location of the image which contains the selfie image

ensure that haar.xml file and siamese_model are in the same folder as face.py file

command to run the code:
1)python face.py "arg1.jpg" "arg2.jpg"

if two images belong to the same person the console output prints that it's a match or if not it prints it as it's a no-match 
along with the confidence score.
