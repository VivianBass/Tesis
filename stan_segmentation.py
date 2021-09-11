!pip install imgaug==0.3.0

import colorsys
import random
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from skimage.measure import find_contours
import os
import math
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Activation, InputLayer, BatchNormalization, Concatenate, MaxPooling2D, Layer, Dropout, UpSampling2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator, Iterator, load_img, img_to_array, random_shift
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.data import Dataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
strategy = tf.distribute.get_strategy()
import warnings
warnings.filterwarnings("ignore")


def openImage(class_image):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    opening = cv2.morphologyEx(class_image.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    return opening
	
def getImgObjs(class_image,prob_image):

	n_objs, labels = cv2.connectedComponents(class_image)
	obj_imgs = []
	probs = []
	boxes = []

	for label in range(0,n_objs):
		
		obj_idx = np.where(labels == label)
		obj_img = np.zeros(class_image.shape)
		obj_img[obj_idx] = 1
		obj_imgs.append(obj_img)
		obj_prob = prob_image[obj_idx]
		probs.append(np.mean(obj_prob))
		
		x1 = np.min(obj_idx[1])
		y1 = np.min(obj_idx[0])
		x2 = np.max(obj_idx[1])
		y2 = np.max(obj_idx[0])
		boxes.append([x1, y1, x2, y2])
	
	return n_objs, boxes, probs, obj_imgs

def apply_mask(image, mask, color, alpha=0.0):

    for c in range(3):
        image[:, :, c] = np.where(mask == 1.0,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c]*255,
                                  image[:, :, c])
    return image
	
def visualize(test_image,n_objs,boxes,probs,obj_imgs,n_objs2,boxes2,probs2,obj_imgs2):

    masked_img = cv2.cvtColor(test_image,cv2.COLOR_GRAY2RGB)
    
    for i in range(1, n_objs2):
        color=(0.0, 1.0, 1.0)
        mask_img = obj_imgs2[i] 
        masked_img = apply_mask(masked_img, mask_img, color)
        padded_mask = np.zeros(
          (mask_img.shape[0] + 2, mask_img.shape[1] + 2), dtype=np.float32)
        padded_mask[1:-1, 1:-1] = mask_img
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
          verts = np.fliplr(verts) - 1
          p = Polygon(verts, facecolor="none", edgecolor=color)
          ax5.add_patch(p)

    for i in range(1, n_objs):
      color=(1.0, 0.0, 0.0)
      mask_img = obj_imgs[i] 
      masked_img = apply_mask(masked_img, mask_img, color)

      padded_mask = np.zeros((mask_img.shape[0] + 2, mask_img.shape[1] + 2), 
                             dtype=np.float32)
      padded_mask[1:-1, 1:-1] = mask_img
      contours = find_contours(padded_mask, 0.5)

      for verts in contours:
        verts = np.fliplr(verts) - 1
        p = Polygon(verts, facecolor="none", edgecolor=color)
        ax5.add_patch(p)

    return masked_img

from google.colab import drive
drive.mount('/content/drive')

"""### Leer imágenes"""

def load_image(path, id_name, image_size):
    
    image_path = os.path.join(path,id_name)
    mask_path=os.path.join("/content/drive/MyDrive/AP/Allimages/Investigacion/masks/",id_name)

    image = cv2.imread(image_path, 0)
    image = cv2.resize(image,(image_size,image_size))
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

    mask = cv2.imread(mask_path, 0)
    mask = cv2.resize(mask, (image_size, image_size),interpolation=cv2.INTER_CUBIC)
    th, mask = cv2.threshold(mask, 0.0001, 255, cv2.THRESH_BINARY)
    mask = np.expand_dims(mask, axis=-1)
    
    return image, mask

img_path="/content/drive/MyDrive/AP/Allimages/Autoencoder/"

img_ids = next(os.walk(img_path))[2]

tumor_images = []
mask_images = []

for img_id in img_ids:
    x, y = load_image(img_path,img_id,256,1)
    tumor_images.append(x)
    mask_images.append(y)

tumor_images = np.array(tumor_images)
mask_images = np.array(mask_images)

print(img_ids)

shuffled_indices = np.arange(tumor_images.shape[0])
np.random.shuffle(shuffled_indices)
shuffled_tumors = tumor_images[shuffled_indices]
shuffled_masks = mask_images[shuffled_indices]
tumor_images=shuffled_tumors
mask_images=shuffled_masks

"""### Visualizar imágenes"""

n_images = 5
n_cols = 2

fig, axs = plt.subplots(n_images,n_cols, figsize=(8, 20))

for i in range(0,n_images):
    
    random_img = random.randint(0,len(tumor_images)-1)
    img = tumor_images[random_img] 
    mask = mask_images[random_img]
    
    axs[i,0].imshow(img,cmap=plt.cm.bone)
    axs[i,1].imshow(mask[:,:,0],cmap="gray")

plt.show()

"""### 3. Definición del modelo

"""

def conv2D(n_filters, kernel_size, activation='relu', use_bn=False, **kwargs):
        def layer(x):
            x = Conv2D(n_filters, kernel_size, use_bias=(not use_bn),
                      padding='same', **kwargs)(x)
            if use_bn:
                x = BatchNormalization(x)
            x = Activation(activation)(x)
            return x
        return layer


    def encoder_block(n_filters):
        def layer(inputs):
            kernel3_inp, kernelconcat_inp = inputs
            x1 = conv2D(n_filters, kernel_size=1)(kernelconcat_inp)
            x1 = conv2D(n_filters, kernel_size=3)(x1)

            x5 = conv2D(n_filters, kernel_size=5)(kernelconcat_inp)
            x5 = conv2D(n_filters, kernel_size=3)(x5)

            concat = tf.concat([x1, x5], axis=3)
            concat_pool = MaxPooling2D(pool_size=(2, 2))(concat)

            x3 = x3_1 = conv2D(n_filters * 2, kernel_size=3)(kernel3_inp)
            #skip 1: segunda capa de filtro 3x3
            x3 = skip1 = conv2D(n_filters * 2, kernel_size=3)(x3)
            x3_pool = MaxPooling2D(pool_size=(2, 2))(x3)

            #skip 2: se unen la concatenada y la primera capa de 3x3
            skip2 = tf.add(x3_1, concat)
            return x3_pool, concat_pool, skip1, skip2
        return layer


    def decoder_block(n_filters, use_bn=False, mode='transpose'):
        def layer(inputs):
            inp, skip1, skip2 = inputs
            x = Conv2DTranspose(n_filters, kernel_size=3, strides=(2,2), padding='same')(inp)
            x = tf.nn.relu(x)
            concat = tf.concat([x, skip1], axis=3)
            x = conv2D(n_filters, kernel_size=3)(concat)
            concat = tf.concat([x, skip2], axis=3)
            x = conv2D(n_filters, kernel_size=3)(concat)
            return x
        return layer
        
    def build_stan(
        n_classes,
        input_shape=(None, None, 3),
        filters=[32, 64, 128, 256, 512],
        decode_mode='transpose',
        activation='sigmoid'):

        inp = Input(shape=input_shape)

        # codificador
        x3_pool, concat_pool, skip1_b1, skip2_b1 = encoder_block(filters[0])((inp, inp))
        x3_pool, concat_pool, skip1_b2, skip2_b2 = encoder_block(filters[1])((x3_pool, concat_pool))
        x3_pool, concat_pool, skip1_b3, skip2_b3 = encoder_block(filters[2])((x3_pool, concat_pool))
        x3_pool, concat_pool, skip1_b4, skip2_b4 = encoder_block(filters[3])((x3_pool, concat_pool))

        # cuello de botella
        x3_pool         = conv2D(n_filters=filters[4], kernel_size=3)(x3_pool)
        x3_pool         = conv2D(n_filters=filters[4], kernel_size=3)(x3_pool)
        concat_pool_1   = conv2D(n_filters=filters[4], kernel_size=1)(concat_pool)
        concat_pool_1   = conv2D(n_filters=filters[4], kernel_size=3)(concat_pool_1)
        concat_pool_5   = conv2D(n_filters=filters[4], kernel_size=5)(concat_pool)
        concat_pool_5   = conv2D(n_filters=filters[4], kernel_size=3)(concat_pool_5)

        mid = tf.concat([x3_pool, concat_pool_1, concat_pool_5], axis=3, name='encoded_fm_concat')

        # decodificador
        x = decoder_block(n_filters=filters[3], mode=decode_mode)((mid, skip1_b4, skip2_b4))
        x = decoder_block(n_filters=filters[2], mode=decode_mode)((x, skip1_b3, skip2_b3))
        x = decoder_block(n_filters=filters[1], mode=decode_mode)((x, skip1_b2, skip2_b2))
        x = decoder_block(n_filters=filters[0], mode=decode_mode)((x, skip1_b1, skip2_b1))

        # salida 
        x = conv2D(n_filters=n_classes, kernel_size=3, activation=activation)(x)

        return Model(inp, x)

    def freeze_model(model):
        for layer in model.layers:
            layer.trainable = True

    def STAN(
        n_classes, 
        input_shape=(None, None, 3),
        filters=[32, 64, 128, 256, 512],
        encoder_weights=None,
        output_activation='sigmoid',
        decode_mode='transpose',
        freeze_encoder=False,
        model_name='stan'):
          
        model = build_stan(
            n_classes=n_classes,
            input_shape=input_shape,
            filters=filters,
            activation=output_activation,
            decode_mode=decode_mode)
        
        if freeze_encoder:
            freeze_model(model)
        

        return model

"""### 4. Entrenamiento y aumento de datos

### Definir parametros de compilación
"""

def dice_coef1(y_true, y_pred, smooth=1e-6):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    
    return dice

def aer_stan(y_true, y_pred):
    #Qué tan probable es que el área resultante del método no sea el 
    #área segmentada por el experto.
    smooth = 1
    y_pos = K.round(K.clip(y_true, 0, 1)) #am
    y_pred_pos = K.round(K.clip(y_pred, 0, 1)) #ar
    intersection = K.sum(y_pos * y_pred_pos) + smooth
    union = K.sum(y_pos)+K.sum(y_pred_pos)-intersection + smooth
    return K.mean((union - intersection) / (K.sum(y_pos) + smooth))

def true_pos(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth) 
    return tp 

def false_pos(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    fp_ratio = (K.sum(y_neg * y_pred_pos) + smooth) / (K.sum(y_neg) + smooth)
    return fp_ratio

def true_neg(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn

def false_neg(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    fn_ratio = (K.sum(y_pos * y_pred_neg) + smooth) / (K.sum(y_pos) + smooth)
    return fn_ratio

def iou_coef(y_true, y_pred, smooth=1):
    #indica que tanto se parece el area segmentada por la red con el
    #area segmentada por el experto
    y_pos = K.round(K.clip(y_true, 0, 1)) #am
    y_pred_pos = K.round(K.clip(y_pred, 0, 1)) #ar
    intersection = K.sum(y_pos * y_pred_pos)
    union = K.sum(y_pos)+K.sum(y_pred_pos)-intersection
    iou = K.mean((intersection + smooth) / (union + smooth))
    return iou

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coef1(y_true, y_pred)
    return loss

def get_loss_by_name(name, **kwargs):
    if name == 'dice':
        return dice_loss

def compileModel(model):

    loss_name = 'dice'
    criterion = get_loss_by_name(loss_name)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

    optimizer = optimizer
    loss = criterion
    metrics=['binary_accuracy',dice_coef1]

    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

    return model

"""### Definir parámetros de entrenamiento"""

from datetime import datetime

def trainModel(train_images,train_masks,model,epochs=30):

    shuffled_indices = np.arange(train_images.shape[0])
    np.random.shuffle(shuffled_indices)
    shuffled_tumors = train_images[shuffled_indices]
    shuffled_masks = train_masks[shuffled_indices]

    samples_count = shuffled_tumors.shape[0]
    train_samples_count = int(0.8 * samples_count)
    validation_samples_count = int(0.2 * samples_count)

    train_images = shuffled_tumors[:train_samples_count]
    train_masks = shuffled_masks[:train_samples_count]
    validation_images = shuffled_tumors[train_samples_count:train_samples_count+validation_samples_count]
    validation_masks = shuffled_masks[train_samples_count:train_samples_count+validation_samples_count]

    batch_size=16

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=15) 

    #Tensorboard 
    logdir = os.path.join("/content/drive/MyDrive/AP/Allimages/logs/")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, 
                                                                  histogram_freq=1, 
                                                                  profile_batch = 100000000)
    
    checkpoint_path = "/content/drive/MyDrive/AP/Allimages/modeloStan.h5"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_best_only=True, 
                                                     monitor='val_loss',
                                                     mode='min',
                                                     save_weights_only=True,
                                                 verbose=1)
    
    train_images_norm = train_images/255
    train_masks_norm = train_masks/255

    validation_images_norm = validation_images/255
    validation_masks_norm = validation_masks/255

    model.fit(train_images_norm, train_masks_norm,
                         batch_size = batch_size,
                         epochs = epochs,
                         callbacks=[tensorboard_callback,cp_callback,early_stopping],
                         validation_data = (validation_images_norm,validation_masks_norm),
                         verbose = 1)        
    
    return model

"""### Entrenar modelo

"""

samples_count = len(tumor_images)
print("Total number of images")
print(samples_count)

train_samples_count = int(0.8 * samples_count)
test_samples_count = int(0.2 * samples_count)

train_images = tumor_images[:train_samples_count]
train_masks = mask_images[:train_samples_count]
test_images = tumor_images[train_samples_count:train_samples_count+test_samples_count]
test_masks = mask_images[train_samples_count:train_samples_count+test_samples_count]

print("Number of training images")
print(len(train_images))
print("Number of test images")
print(len(test_images))

"""### Aumento de datos
 
"""

train_images.shape, train_masks.shape,train_images.max(),train_masks.max()

import imgaug.augmenters as iaa

images_aug = train_images
masks_aug = train_masks

seq = iaa.Sequential([
    iaa.Sharpen((0.0, 0.5)),       
    iaa.Affine(rotate=(-45, 45)),  
    iaa.Affine(scale=(0.5,1.3))
    ], random_order=True)

n_aug = 40

for i in range(n_aug):
    images_aug_i, masks_aug_i = seq(images = images_aug, segmentation_maps = masks_aug)
    train_images = np.concatenate((train_images,images_aug_i),axis=0)
    train_masks = np.concatenate((train_masks,masks_aug_i),axis=0)

train_images.shape,train_masks.shape

n_images = 5
n_cols = 2

fig, axs = plt.subplots(n_images,n_cols, figsize=(8, 20))

for i in range(0,n_images):
    random_img = random.randint(0,len(train_images)-1)
    img = train_images[random_img] 
    mask = train_masks[random_img]

    axs[i,0].imshow(img,cmap=plt.cm.bone)
    axs[i,1].imshow(mask[:,:,0],cmap="gray")

plt.show()

model = STAN(1, input_shape=(256,256,3), decode_mode='transpose')
model = compileModel(model=model)
model = trainModel(train_images,train_masks,model,epochs=200)

!ls -R {'/content/drive/MyDrive/AP/Allimages/logs/'}


"""### Segmentar una nueva imagen

"""

def predictClassModel(image,model,image_size=256):

    test_image = image.reshape(1, image_size, image_size, 3)

    prob_image = model.predict(test_image)
    class_image = prob_image > 0.7

    class_image = class_image[0]
    prob_image = prob_image[0]

    return prob_image, class_image

"""## Probar una imagen del conjunto de prueba"""

test_masks = test_masks/255
test_images=test_images/255

print(test_masks.shape,test_images.shape,np.unique(test_masks),np.unique(test_images))

"""### Autoencoder"""

#validacion
modelo = STAN(1, input_shape=(256,256,3), decode_mode='transpose')
modelo.load_weights('/content/drive/MyDrive/AP/Allimages/modeloStan.h5')
modelo.compile(loss=get_loss_by_name('dice'), metrics=['binary_accuracy',dice_coef1, true_pos, false_pos, true_neg,false_neg,tf.keras.metrics.MeanAbsoluteError(),iou_coef,aer_stan])

loss, binary_Acc, dice, TPK,FPK,TNK,FNK,mae,ji,aer=modelo.evaluate(test_images,test_masks, verbose=0)

print(f'Loss: {loss}')
#Diferenciar correctamente sanos y enfermos
print(f'Accuracy: {(TPK+TNK)/(TPK+FPK+TNK+FNK)}')
print(f'Dice: {dice}')
print(f'TP: {TPK}')
print(f'FP: {FPK}')
print(f'TN: {TNK}')
print(f'FN: {FNK}')
#Qué tan lejos estuvieron las predicciones del valor real, en promedio.
print(f'MAE: {mae}')
#Qué tan probable es que el área resultante del método no sea el área segmentada por el experto. 
print(f'AER: {aer}')
#Probabilidad de que un enfermo efectivamente esté enfermo
print(f'SENSIBILIDAD: {TPK/(TPK+FNK)}')
#Probabilidad de que un sano  efectivamente esté sano
print(f'ESPECIFICIDAD: {TNK/(TNK+FPK)}')
print(f'PRECISION: {TPK/(TPK+FPK)}')
print(f'JI: {ji}')

"""## Visualización de resultados

### Autoencoder
"""

#validacion 

prueba=test_images*255
prueba=prueba.astype(np.uint8)

for test_idx in range(len(test_images)):
    test_image = test_images[test_idx]
    test_mask = test_masks[test_idx]

    prob_image, class_image = predictClassModel(test_image,model=modelo,image_size=256)
    fig = plt.figure(figsize=(18, 16))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    imagen = prueba[test_idx]
    open_image = openImage(class_image[:,:,0])
    n_objs, boxes, probs, obj_imgs = getImgObjs(open_image,prob_image[:,:,0])

    ax = fig.add_subplot(1, 5, 1)
    ax.imshow(test_image, cmap=plt.cm.bone)
    ax.set_title("Test image")
    ax.axis('off')
    ax = fig.add_subplot(1, 5, 2)
    ax.imshow(test_mask[:,:,0], cmap="gray")
    ax.set_title("Test mask")
    ax.axis('off')
    ax = fig.add_subplot(1, 5, 3)
    ax.imshow(prob_image[:,:,0], cmap=plt.cm.bone)
    ax.set_title("Probability image")
    ax.axis('off')
    ax = fig.add_subplot(1, 5, 4)
    ax.imshow(class_image[:,:,0], cmap="gray")
    ax.set_title("Class image")
    ax.axis('off')
    ax = fig.add_subplot(1, 5, 5)
    im=visualize(imagen[:,:,0],n_objs,boxes,probs,obj_imgs)
    ax.imshow(im, cmap="gray")
    
    ax.set_title("Predicted tumor")
    ax.axis('off')