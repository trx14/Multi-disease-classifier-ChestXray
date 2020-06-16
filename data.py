import cv2
import glob
import os
import numpy as np
import pandas as pd
from keras import backend as K
from sklearn import metrics
from keras.utils.np_utils import to_categorical


def load_inria_person(path):
    pos_path = os.path.join(path, "pos")
    neg_path = os.path.join(path, "neg")
    list1 = os.listdir(pos_path)
    list2 = os.listdir(neg_path)
    all_path = []
    for file in list1:
        if file.split('.')[-1] == 'png':
            filepath = os.path.join(pos_path,file)
            all_path.append(filepath)
    for file in list2:
        if file.split('.')[-1] == 'png':
            filepath = os.path.join(neg_path,file)
            all_path.append(filepath)
    arr = np.arange(len(all_path))
    np.random.shuffle(arr)
    y = []
    X = np.empty((0,128,128,3))
    number = 0
    for i in arr:
        print('*****************')
        number = number + 1
        print('%d / %d'%(number,len(all_path)))
        images = [cv2.resize(cv2.imread(all_path[i]), (128, 128))]
        if(i >= len(list1)):
            y = y+[0]
        else:
            y = y+[1]
        X = np.append(X,np.float32(images),axis=0)
    y = to_categorical(y, 2)
    return X, y

def get_im_cv2(paths, img_rows, img_cols, color_type, normalize, resize, dataset_path):
    # Load as grayscale
    imgs = []
    for path in paths:
        if color_type == 1:
            img = cv2.imread(os.path.join(dataset_path,path), 0)
        elif color_type == 3:
            img = cv2.imread(os.path.join(dataset_path,path))
        # Reduce size
        if resize:
            img = cv2.resize(img, (img_cols, img_rows))
        if normalize:
            img = img - np.sum(img)/(img_cols*img_rows)
        #augmentation
        r1 = np.random.randint(0,2,1)[0]
        r2 = np.random.randint(0,2,1)[0]
        
        #if r1 == 1:
        #    img = np.fliplr(img)
        #if r2 == 1:
        #    img = np.flipud(img)
        #
        imgs.append(img)
    
    return np.array(imgs)
def test(X_train,dataset_path):
    for i in range(len(X_train)):
        if(not os.path.exists(os.path.join(dataset_path,X_train[i]))):
            print(os.path.join(dataset_path,X_train[i]))
def get_train_batch(X_train, y_train, batch_size, is_resize, img_w, img_h, color_type, is_argumentation, dataset_path, channel):
    while 1:
        for i in range(0, len(X_train), batch_size):
            x = get_im_cv2(X_train[i:i+batch_size], img_w, img_h, color_type, True, is_resize, dataset_path)
            y = np.array(y_train[i:i+batch_size])
            x = np.expand_dims(x, axis=3)
            x = np.tile(x,(1,1,channel))
            #if is_argumentation:
            #x, y = img_augmentation(x, y)
            yield(x,y)

def readcsv(train,val):
    tra_name = pd.read_csv(train,usecols=['imgs'])
    tra_name = tra_name.values.tolist()
    tra_label = pd.read_csv(train,usecols=['lbls'])
    tra_label = tra_label.values.tolist()
    
    val_name = pd.read_csv(val,usecols=['imgs'])
    val_name = val_name.values.tolist()
    val_label = pd.read_csv(val,usecols=['lbls'])
    val_label = val_label.values.tolist()
    
    tra_Name = [];
    tra_Label = np.zeros((len(tra_name),1));
    val_Name = [];
    val_Label = np.zeros((len(val_name),1));
    
    
    
    for x in range(len(tra_name)):
        tra_Name.append(tra_name[x][0])
        #lbs = (tra_label[x][0]).split('|')
        lbs = (tra_label[x][0])
        for k in range(1):
        #tra_Label[x][k] = int(lbs[k])
            tra_Label[x][k] = lbs
    
    for x in range(len(val_name)):
        val_Name.append(val_name[x][0])
        #lbs = (val_label[x][0]).split('|')
        lbs = (val_label[x][0])
        for k in range(1):
            val_Label[x][k] = lbs
    return tra_Name,tra_Label,val_Name,val_Label

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer





