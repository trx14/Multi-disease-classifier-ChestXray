from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D, GlobalMaxPooling2D, Input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras import applications
from keras import backend as K

K.set_image_dim_ordering('tf')
import h5py
from keras.optimizers import SGD,Adam


def VGG16_convolutions():
    model = applications.VGG19(include_top=None, weights='imagenet',input_shape=(224,224,3))
    
    top_model = Sequential()
    top_model.add(GlobalMaxPooling2D(data_format='channels_last',input_shape=model.output_shape[1:]))
    top_model.add(Dense(1, activation = 'sigmoid', init='uniform'))
    
    
    
    new_model = Model(inputs = model.input, outputs = top_model(model.output))
    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    new_model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics=['accuracy'])
    return new_model

def ResNet_convolutions(include_top):
    if include_top:
        print('load resnet include top')
        model = applications.ResNet50(include_top=None, weights=None,input_shape=(224,224,3),classes =2)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics=['accuracy'])
        return model
    else:
        print('load resnet exclude top')
        model = applications.ResNet50(include_top=None, weights='imagenet',input_shape=(224,224,3))
        top_model = Sequential()
        top_model.add(GlobalMaxPooling2D(data_format='channels_last',input_shape=model.output_shape[1:]))
        top_model.add(Dense(1, activation = 'sigmoid', init='uniform'))
    

    
        new_model = Model(inputs = model.input, outputs = top_model(model.output))

        adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        new_model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics=['accuracy'])
        return new_model

def DenseNet_convolutions():
    
    model = applications.densenet.DenseNet201(include_top=False, weights='imagenet',input_shape=(512,512,3))
    
    top_model = Sequential()
    top_model.add(GlobalMaxPooling2D(data_format='channels_last',input_shape=model.output_shape[1:]))
    top_model.add(Dense(1, activation = 'sigmoid', kernel_initializer='uniform'))
    
    
    
    new_model = Model(inputs = model.input, outputs = top_model(model.output))
    
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    new_model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics=['accuracy'])
    return new_model

def Xception_convolutions():
    
    model = applications.xception.Xception(include_top=False, weights='imagenet',input_shape=(224,224,3))
    
    top_model = Sequential()
    top_model.add(GlobalMaxPooling2D(data_format='channels_last',input_shape=model.output_shape[1:]))
    top_model.add(Dense(1, activation = 'sigmoid', kernel_initializer='uniform'))
    
    
    
    new_model = Model(inputs = model.input, outputs = top_model(model.output))
    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    new_model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics=['accuracy'])
    return new_model

def inception_resnet_v2():
    
    model = applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',input_shape=(224,224,3))
    
    top_model = Sequential()
    top_model.add(GlobalMaxPooling2D(data_format='channels_last',input_shape=model.output_shape[1:]))
    top_model.add(Dense(1, activation = 'sigmoid', kernel_initializer='uniform'))
    
    
    
    new_model = Model(inputs = model.input, outputs = top_model(model.output))
    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    new_model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics=['accuracy'])
    return new_model


def get_model(net,include_top):
    if net == 'vgg16':
        model = VGG16_convolutions()
        model.add(GlobalMaxPooling2D(data_format='channels_last'))
        model.add(Dense(1, activation = 'sigmoid', init='uniform'))
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics=['accuracy'])
        return model
    elif net == 'resnet50':
        model = ResNet_convolutions(include_top)
        return model
    elif net == 'densenet':
        model = DenseNet_convolutions()
        return model
    elif net == 'xception':
        print('load xception...')
        return Xception_convolutions()
    elif net == 'inception_resnet_v2':
        print('load inception_resnet_v2...')
        return inception_resnet_v2()

def load_model_weights(model, weights_path):
    print('Loading model.')
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
        model.layers[k].trainable = False
    f.close()
    print('Model loaded.')
    return model

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer