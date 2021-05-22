Ensure the installation of all the libraries mentioned below to run face.py<br>
1)pytorch-1.18<br>
2)opencv<br>
3)numpy<br>

Steps to run face.py
if you are using anaconda or windows terminal navigating to the folder which contains all the files in repository
along with two images the passport images and the selfie<br>
"arg1.jpg"-the location of the image which contains the passport image<br>
"arg2.jpg"-the location of the image which contains the selfie image<br>

ensure that haar.xml file and siamese_model are in the same folder as face.py file

command to run the code:
1)python face.py "arg1.jpg" "arg2.jpg"

if two images belong to the same person the console output prints that it's a match or if not it prints it as it's a no-match 
along with the confidence score.<br>
Updates-
siamese-model1 improves test accuracy from 76 to 81 from siamese_model
new model uses a architecure similar to vgg
