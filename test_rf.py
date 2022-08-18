import warnings
warnings.filterwarnings('ignore')
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cv2 
import numpy
import mahotas
from scipy.stats import skew,kurtosis
from skimage.feature import graycomatrix,graycoprops
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
import pickle
import matplotlib.pyplot as plt
color_feature=[]
texture_feature=[]
feature=[]

root=Tk()
filename=askopenfilename()
root.destroy()
image=cv2.imread(filename)
cv2.imshow('image',image)
cv2.waitKey(10)
blue_plane=image[:,:,0]
green_plane=image[:,:,1]
red_plane=image[:,:,2]
mean_blue=numpy.mean(blue_plane)
mean_green=numpy.mean(green_plane)
mean_red=numpy.mean(red_plane)
# print(mean_blue,mean_green,mean_red)

var_blue=numpy.var(blue_plane)
var_green=numpy.var(green_plane)
var_red=numpy.var(red_plane)
# print(var_blue,var_green,var_red)

skew_blue=skew(blue_plane.reshape(-1))
skew_green=skew(green_plane.reshape(-1))
skew_red=skew(red_plane.reshape(-1))
# print(skew_blue,skew_green,skew_red)

kurtosis_blue=kurtosis(blue_plane.reshape(-1))
kurtosis_green=kurtosis(green_plane.reshape(-1))
kurtosis_red=kurtosis(red_plane.reshape(-1))
# print(kurtosis_blue,kurtosis_green,kurtosis_red)
hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
cv2.imshow('hsv',hsv)
cv2.waitKey(10)
hue_plane=hsv[:,:,0]
saturation_plane=hsv[:,:,1]
value_plane=hsv[:,:,2]
mean_hue=numpy.mean(hue_plane)
mean_saturation=numpy.mean(saturation_plane)
mean_value=numpy.mean(value_plane)
# print(mean_hue,mean_saturation,mean_value)

var_hue=numpy.var(hue_plane)
var_saturation=numpy.var(saturation_plane)
var_value=numpy.var(value_plane)
# print(var_hue,var_saturation,var_value)

skew_hue=skew(hue_plane.reshape(-1))
skew_saturation=skew(saturation_plane.reshape(-1))
skew_value=skew(value_plane.reshape(-1))
# print(skew_hue,skew_saturation,skew_value)

kurtosis_hue=kurtosis(hue_plane.reshape(-1))
kurtosis_saturation=kurtosis(saturation_plane.reshape(-1))
kurtosis_value=kurtosis(value_plane.reshape(-1))
# print(kurtosis_hue,kurtosis_saturation,kurtosis_value)
lab=mahotas.colors.rgb2lab(image)
cv2.imshow('lab',lab)
cv2.waitKey(10)
l_plane=lab[:,:,0]
a_plane=lab[:,:,1]
b_plane=lab[:,:,2]
mean_l=numpy.mean(l_plane)
mean_a=numpy.mean(a_plane)
mean_b=numpy.mean(b_plane)
# print(mean_l,mean_a,mean_b)

var_l=numpy.var(l_plane)
var_a=numpy.mean(a_plane)
var_b=numpy.mean(b_plane)
# print(mean_l,mean_a,mean_b)

skew_l=skew(l_plane.reshape(-1))
skew_a=skew(a_plane.reshape(-1))
skew_b=skew(b_plane.reshape(-1))
# print(skew_l,skew_a,skew_b)

kurtosis_l=kurtosis(l_plane.reshape(-1))
kurtosis_a=kurtosis(a_plane.reshape(-1))
kurtosis_b=kurtosis(b_plane.reshape(-1))
# print(kurtosis_l,kurtosis_a,kurtosis_b)
color_feature=[mean_blue,mean_green,mean_red,var_blue,var_green,var_red,
skew_blue,skew_green,skew_red,kurtosis_blue,kurtosis_green,kurtosis_red,
mean_hue,mean_saturation,mean_value,var_hue,var_saturation,var_value,
skew_hue,skew_saturation,skew_value,kurtosis_hue,kurtosis_saturation,kurtosis_value,
mean_l,mean_a,mean_b,var_l,var_a,var_b,skew_l,skew_a,skew_b,kurtosis_l,kurtosis_a,kurtosis_b]
        
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
cv2.waitKey(10)
glcm=graycomatrix(gray,distances=[5],angles=[0],levels=256,symmetric=True,normed=True)
# print(glcm)
correlation=graycoprops(glcm,'correlation')[0,0]
energy=graycoprops(glcm,'energy')[0,0]
contrast=graycoprops(glcm,'contrast')[0,0]
homogeneity=graycoprops(glcm,'homogeneity')[0,0]
# print(correlation,energy,contrast,homogeneity)
texture_feature=[correlation,energy,contrast,homogeneity]
color_feature.extend(texture_feature)
feature.append(color_feature)

cv2.destroyAllWindows()
feature=numpy.array(feature)
print(feature.shape)

with open('label_name.pkl','rb') as f:
    label_name=pickle.load(f)
with open('rf_model.pkl','rb') as f:
    rf_model=pickle.load(f)
    
with open('rf_feature.pkl','rb') as f:
    rf_feature=pickle.load(f)
prediction=rf_model.predict(feature)
d={0:'airport',1:'bareland',2:'playground',3:'railwaystation'}
print(d[prediction[0]])
test_feature=rf_model.predict_proba(feature)
s=rf_feature.shape
distance=numpy.transpose(numpy.zeros([1,s[0]]))
for i in range(s[0]):
    f=rf_feature[i,:]
    distance[i]=numpy.sqrt(numpy.sum((f-test_feature)**2))
print(distance)
sort=numpy.argsort(distance,axis=0)
for i in range(0,10):
    path=label_name[int(sort[i])]
    print(path)
    img=cv2.imread(path)
    plt.subplot(2,5,i+1)
    plt.imshow(img)
plt.suptitle('retreived image')
plt.show()





    