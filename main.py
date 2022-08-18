import warnings
warnings.filterwarnings('ignore')
import os
import cv2
import numpy
import mahotas
from scipy.stats import skew,kurtosis
from skimage.feature import graycomatrix,graycoprops
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from keras.layers import Dense
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay,plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle
path=os.getcwd()

data_path=os.path.join(path,'ucmd')

data_list=os.listdir(data_path)
color_feature=[]
texture_feature=[]
feature=[]
label=[]
count=0
label_name=[]

for i in data_list:
    
    subfolder_path=os.path.join(data_path,i)
    
    subfolder_list=os.listdir(subfolder_path)
    for j in subfolder_list:
        
        image_path=os.path.join(subfolder_path,j)
        
        if j.endswith('.db'):
            continue
        print(image_path)
        label_name.append(image_path)
        image=cv2.imread(image_path)
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
        label.append(count)
    count=count+1
cv2.destroyAllWindows()
feature=numpy.array(feature)
print(feature.shape)
# print(label)
label=numpy.array(label)
xtrain,xtest,ytrain,ytest=train_test_split(feature,label,test_size=0.3,random_state=42)
print(xtrain.shape)
print(xtest.shape)
ytrain1=[]
ytest1=[]
for i in ytrain:
    empty_list=[0,0,0,0]
    empty_list[i]=1
    ytrain1.append(empty_list)
for i in ytest:
    empty_list=[0,0,0,0]  
    empty_list[i]=1
    ytest1.append(empty_list)
ytrain1=numpy.array(ytrain1)
ytest1=numpy.array(ytest1)
model=Sequential()
model.add(Dense(200,input_dim=40,activation='relu'))
model.add(Dense(4,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=optimizers.Adam(lr=0.018),metrics=['accuracy'])
history=model.fit(xtrain,ytrain1,validation_data=(xtest,ytest1),epochs=20,verbose=2,batch_size=8)
loss,accuracy=model.evaluate(xtest,ytest1)
print(loss,accuracy)
model.summary()
layer_name='dense_1'
new_model=Model(inputs=model.input,outputs=model.get_layer(layer_name).output)
ann_feature=new_model.predict(feature)
plt.title('training progress-Loss')
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='test')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()
plt.title('training progress-accuracy')
plt.plot(history.history['accuracy'],label='train')
plt.plot(history.history['val_accuracy'],label='test')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
model_json=model.to_json()
with open('ann_model.json','w') as json_file:
    json_file.write(model_json)
model.save_weights('ann_model.h5')

prediction=model.predict(xtest)
y_list=[]
for i in prediction:
    num_list=list(i)
    output=max(num_list)
    ind=num_list.index(output)
    y_list.append(ind)
cm=confusion_matrix(ytest,y_list)
display=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['airport','bareland','playground','railwaystation'])
display.plot()
plt.title('ann')
plt.show()
print(classification_report(ytest,y_list))

rf_model=RandomForestClassifier()
rf_model.fit(xtrain,ytrain)
prediction=rf_model.predict(xtest)
print(classification_report(ytest,prediction))
plot_confusion_matrix(rf_model,xtest,ytest)
plt.title('random forest')
plt.show()
pickle.dump(rf_model,open('rf_model.pkl','wb'))
rf_feature=rf_model.predict_proba(feature)
pickle.dump(rf_feature,open('rf_feature.pkl','wb'))
pickle.dump(label_name,open('label_name.pkl','wb'))
pickle.dump(ann_feature,open('ann_feature.pkl','wb'))


        
        




