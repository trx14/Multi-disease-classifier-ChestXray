from keras.models import *
from keras.callbacks import *
import keras.backend as K
from keras.utils.np_utils import to_categorical
from model import *
from data import get_im_cv2,get_train_batch,readcsv,test,get_output_layer,test
import cv2
from sklearn.metrics import roc_auc_score,roc_curve
import argparse
import matplotlib.pyplot as plt
import math


def train(dataset_path):
    tra1 = '/SharedFolder/trx/net/keras-cam-master/csv/Duke_lung_Pneumonia/tra1.csv'
    val1 = '/SharedFolder/trx/net/keras-cam-master/csv/Duke_lung_Pneumonia/val1.csv'
    tra2 = '/SharedFolder/trx/net/keras-cam-master/csv/Duke_lung_Pneumonia/tra2.csv'
    val2 = '/SharedFolder/trx/net/keras-cam-master/csv/Duke_lung_Pneumonia/val2.csv'
    tra3 = '/SharedFolder/trx/net/keras-cam-master/csv/Duke_lung_Pneumonia/tra3.csv'
    val3 = '/SharedFolder/trx/net/keras-cam-master/csv/Duke_lung_Pneumonia/val3.csv'
    tra4 = '/SharedFolder/trx/net/keras-cam-master/csv/Duke_lung_Pneumonia/tra4.csv'
    val4 = '/SharedFolder/trx/net/keras-cam-master/csv/Duke_lung_Pneumonia/val4.csv'
    
    for j in range(5):
        for i in range(4):
            tra_Name = []
            tra_Label = []
            val_Name = []
            val_Label = []
            model = get_model('resnet50',False)
            if i == 0:
                tra_Name,tra_Label,val_Name,val_Label = readcsv(tra1,val1)
            if i == 1:
                tra_Name,tra_Label,val_Name,val_Label = readcsv(tra2,val2)
            if i == 2:
                tra_Name,tra_Label,val_Name,val_Label = readcsv(tra3,val3)
            if i == 3:
                tra_Name,tra_Label,val_Name,val_Label = readcsv(tra4,val4)
            ######
            #tra_Label = to_categorical(tra_Label)
            #val_Label = to_categorical(val_Label)
            ######
            print(len(tra_Name))
            print(tra_Label.shape)
            print(len(val_Name))
            print(val_Label.shape)
        
            #test(tra_Name,args.dataset_path)
        
            print("Training..")
            checkpoint_path="/SharedFolder/trx/net/keras-cam-master/checkpoint/dukelung_pneumonia/cross"+str(i)+"/weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}-DukelungPneresnet224.hdf5"
            checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
        
            #get_train_batch(X_train, y_train, batch_size, is_resize, img_w, img_h, color_type, is_argumentation, dataset_path):
            result = model.fit_generator(get_train_batch(tra_Name, tra_Label, 8, True, 224, 224, 1, False, args.dataset_path,3),
                                         steps_per_epoch=math.ceil(tra_Label.shape[0]/8.0),
                                         epochs=15, verbose=1,
                                         validation_data=get_train_batch(val_Name, val_Label, 8, True, 224, 224, 1, False, args.dataset_path,3),
                                         validation_steps=math.ceil(val_Label.shape[0]/8.0),
                                         callbacks=[checkpoint])

def probablity_result():
        tra_Name,tra_Label,val_Name,val_Label = readcsv(args.tra_path,args.val_path)
        test(val_Name,args.dataset_path)
        print(val_Label.shape)
        model = load_model('/Volumes/MyPassport/duke/test_chk1_model/ede/cross1/weights.20-0.22-0.94-Dukelungedema_.hdf5')
        result = model.predict_generator(get_train_batch(val_Name, val_Label, 8, True, 224, 224, 1, False, args.dataset_path,3),steps=math.ceil(val_Label.shape[0]/8.0),verbose=1)
        print(result.shape)
        AUC = roc_auc_score(val_Label,result)
        print('************')
        print(AUC)
        print(result)
        np.save("test_edema.hdf5",result)

def plot_roc_curve(y,scores):
        fpr_rf, tpr_rf, thresholds_rf = roc_curve(y, scores)
        auc_rf = auc(fpr_rf, tpr_rf)
        plt.figure()
        plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.savefig("roc.png")



def visualize_class_activation_map(model_path, img_path, output_path):
        model = load_model(model_path)
        img = cv2.imread(img_path)
        img = cv2.resize(img,(224,224))
        img = np.tile(img,(1,1,3))
        
        original_img = cv2.imread(img_path)
        original_img = np.tile(original_img,(1,1,3))
        print(original_img.shape)
        width, height, _ = original_img.shape
        
        #Get the 512 input weights to the softmax.
        class_weights = model.layers[-1].get_weights()[0]
        final_conv_layer = get_output_at(173)
        get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
        [conv_outputs, predictions] = get_output([img])
        conv_outputs = conv_outputs[0, :, :, :]

        #Create the class activation map.
        cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])
        print('********************')
        print('class_weights:')
        print(class_weights)
        print('class_weights_shape:')
        print(class_weights.shape)
        print('feature_map:')
        print(conv_outputs.shape)

        for i in range(512):
                cam += class_weights[i,1] * conv_outputs[:, :, i]
        print("predictions", predictions)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (height, width))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        heatmap[np.where(cam < 0.2)] = 0
        img = heatmap*0.5 + original_img
        print(img.shape)
        cv2.imwrite(output_path, img)
#model1 = load_model('/Volumes/MyPassport/duke/test_chk1_model/ate/cross0/weights.16-1.19-0.77-Dukelungate_.hdf5')
#model2 = load_model('/Volumes/MyPassport/duke/test_chk1_model/ate/cross1/weights.20-1.43-0.77-Dukelungate_.hdf5')
#model3 = load_model('/Volumes/MyPassport/duke/test_chk1_model/ate/cross2/weights.14-1.56-0.74-Dukelungate_.hdf5')
#model4 = load_model('/Volumes/MyPassport/duke/test_chk1_model/ate/cross3/weights.07-1.71-0.76-Dukelungate_.hdf5')

#model1 = load_model('/Volumes/MyPassport/duke/test_chk1_model/ede/cross0/weights.17-0.25-0.95-Dukelungedema_.hdf5')
#model2 = load_model('/Volumes/MyPassport/duke/test_chk1_model/ede/cross1/weights.20-0.22-0.94-Dukelungedema_.hdf5')
#model3 = load_model('/Volumes/MyPassport/duke/test_chk1_model/ede/cross2/weights.20-0.39-0.92-Dukelungedema_.hdf5')
#model4 = load_model('/Volumes/MyPassport/duke/test_chk1_model/ede/cross3/weights.19-0.50-0.89-Dukelungedema_.hdf5')

#model1 = load_model('/Volumes/MyPassport/duke/test_chk1_model/nod/cross0/weights.20-1.92-0.73-Dukelungnodule_.hdf5')
#model2 = load_model('/Volumes/MyPassport/duke/test_chk1_model/nod/cross1/weights.06-1.30-0.70-Dukelungnodule_.hdf5')
#model3 = load_model('/Volumes/MyPassport/duke/test_chk1_model/nod/cross2/weights.15-1.62-0.70-Dukelungnodule_.hdf5')
#model4 = load_model('/Volumes/MyPassport/duke/test_chk1_model/nod/cross3/weights.15-1.75-0.68-Dukelungnodule_.hdf5')

#model1 = load_model('/Volumes/MyPassport/duke/test_chk1_model/pne/cross0/weights.12-0.58-0.88-Dukelungpne_.hdf5')
#model2 = load_model('/Volumes/MyPassport/duke/test_chk1_model/pne/cross1/weights.08-1.02-0.83-Dukelungpne_.hdf5')
#model3 = load_model('/Volumes/MyPassport/duke/test_chk1_model/pne/cross2/weights.19-0.74-0.85-Dukelungpne_.hdf5')
#model4 = load_model('/Volumes/MyPassport/duke/test_chk1_model/pne/cross3/weights.16-1.20-0.81-Dukelungpne_.hdf5')

def visualize_class_activation_map2(img_path, output_path):
    print(os.listdir(img_path))
    model1 = load_model('/Volumes/MyPassport/duke/test_chk1_model/ede/cross0/weights.17-0.25-0.95-Dukelungedema_.hdf5')
    model2 = load_model('/Volumes/MyPassport/duke/test_chk1_model/ede/cross0/weights.17-0.25-0.95-Dukelungedema_.hdf5')
    model3 = load_model('/Volumes/MyPassport/duke/test_chk1_model/ede/cross0/weights.17-0.25-0.95-Dukelungedema_.hdf5')
    model4 = load_model('/Volumes/MyPassport/duke/test_chk1_model/ede/cross0/weights.17-0.25-0.95-Dukelungedema_.hdf5')

    for lists in os.listdir(img_path):
        if lists[-1] == 'g' and lists[0] != '.':
            print(lists)
            img = cv2.imread(os.path.join(img_path,lists))
            img = cv2.resize(img,(224,224))
            img = img -np.mean(img)
            original_img = cv2.imread(os.path.join(img_path,lists))
            print(original_img.shape)
            width, height, _ = original_img.shape
            
            cam = np.zeros(dtype = np.float32, shape = (7,7))
            avg_pro = 0
            for i in range(4):
                if i == 0:
                    model = model1
                if i == 1:
                    model = model2
                if i == 2:
                    model = model3
                if i == 3:
                    model = model4
                #Get the 512 input weights to the softmax.
                class_weights = model.layers[-1].get_weights()[0]
                final_conv_layer = model.layers[173]
                get_output = K.function([model.input], [final_conv_layer.output, model.layers[174].get_output_at(-1)])
                [conv_outputs, predictions] = get_output([np.array([img])])
                conv_outputs = conv_outputs[0, :, :, :]
                
                #Create the class activation map.
                for j in range(2048):
                    cam += class_weights[j] * conv_outputs[:, :, j]
                avg_pro = avg_pro + predictions
                print("predictions", predictions)
                print("model :",i)
        
            cam = cam/cam.max()
            cam = cv2.resize(cam, (height, width))
            cam[ cam<0 ] = 0
            
            heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            img = heatmap*0.5 + original_img*0.5
            cv2.imwrite(os.path.join(output_path,lists),img)

def visualize_class_activation_map3(img_path, output_path):
    print(os.listdir(img_path))
    model = load_model('/Volumes/MyPassport/duke/test_chk1_model/ate/cross0/weights.16-1.19-0.77-Dukelungate_.hdf5')
    
    for lists in os.listdir(img_path):
        if lists[-1] == 'g' and lists[0] != '.':
            print(lists)
            img = cv2.imread(os.path.join(img_path,lists))
            img = cv2.resize(img,(224,224))
            img = img -np.mean(img)
            original_img = cv2.imread(os.path.join(img_path,lists))
            print(original_img.shape)
            width, height, _ = original_img.shape
            
            cam = np.zeros(dtype = np.float32, shape = (7,7))

            #Get the 512 input weights to the softmax.
            class_weights = model.layers[-1].get_weights()[0]
            final_conv_layer = model.layers[173]
            get_output = K.function([model.input], [final_conv_layer.output, model.layers[174].get_output_at(-1)])
            [conv_outputs, predictions] = get_output([np.array([img])])
            conv_outputs = conv_outputs[0, :, :, :]
            
            #Create the class activation map.
            for j in range(2048):
                cam += class_weights[j] * conv_outputs[:, :, j]
            print("predictions", predictions)
            
            cam = cam/cam.max()
            cam = cv2.resize(cam, (height, width))
            cam[ cam<0 ] = 0
            
            heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            img = heatmap*0.5 + original_img*0.5
            cv2.imwrite(output_path+"/"+lists+"_"+str(predictions[0,0])+".png",img)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type = bool, default = False, help = 'Train the network or visualize a CAM')
    parser.add_argument("--evaluate", type = bool, default = False, help = 'Train the network or visualize a CAM')
    parser.add_argument("--visualize", type = bool, default = False, help = 'Train the network or visualize a CAM')
    parser.add_argument("--image_path", type = str, default = "heatmap.jpg", help = "Path of an image to run the network on")
    parser.add_argument("--output_path", type = str, default = "heatmap.jpg", help = "Path of an image to run the network on")
    parser.add_argument("--model_path", type = str, help = "Path of the trained model")
    parser.add_argument("--dataset_path", type = str, help = \
        'Path to image dataset. Should have pos/neg folders, like in the inria person dataset. \
        http://pascal.inrialpes.fr/data/human/')
    parser.add_argument("--tra_path", type = str, help = "Path of an image to run the network on")
    parser.add_argument("--val_path", type = str, help = "Path of an image to run the network on")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.train:
        train(args.dataset_path)
    elif args.evaluate:
        probablity_result()
    elif args.visualize:
        visualize_class_activation_map3(args.image_path, args.output_path)