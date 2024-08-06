import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from scipy import ndimage
from skimage.measure import regionprops
from skimage.filters import threshold_otsu  
import pandas as pd
from time import time
import keras
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

# Paths for genuine and forged images
genuine_image_paths = "C:\\Users\\arbaz\\Desktop\\signature_forgery\\Signature-Forgery-Detection\\real"
forged_image_paths = "C:\\Users\\arbaz\\Desktop\\signature_forgery\\Signature-Forgery-Detection\\forged"

# Image processing functions
def rgb2grey(img):
    return np.mean(img, axis=2)

def grey2bin(img):
    img = ndimage.gaussian_filter(img, 0.8)
    thres = threshold_otsu(img)
    binimg = img > thres
    return np.logical_not(binimg)

def preproc(path, display=True):
    img = mpimg.imread(path)
    if display:
        plt.imshow(img)
        plt.show()
    grey = rgb2grey(img)
    if display:
        plt.imshow(grey, cmap=matplotlib.cm.Greys_r)
        plt.show()
    binimg = grey2bin(grey)
    if display:
        plt.imshow(binimg, cmap=matplotlib.cm.Greys_r)
        plt.show()
    r, c = np.where(binimg)
    signimg = binimg[r.min():r.max(), c.min():c.max()]
    if display:
        plt.imshow(signimg, cmap=matplotlib.cm.Greys_r)
        plt.show()
    return signimg

# Feature extraction functions
def ratio(img):
    return np.sum(img) / (img.shape[0] * img.shape[1])

def centroid(img):
    rows, cols = np.where(img)
    return rows.mean() / img.shape[0], cols.mean() / img.shape[1]

def eccentricity_solidity(img):
    r = regionprops(img.astype("int8"))
    return r[0].eccentricity, r[0].solidity

def skew_kurtosis(img):
    h, w = img.shape
    x, y = np.arange(w), np.arange(h)
    xp, yp = img.sum(axis=0), img.sum(axis=1)
    cx, cy = np.sum(x * xp) / np.sum(xp), np.sum(y * yp) / np.sum(yp)
    sx, sy = np.sqrt(np.sum((x - cx)**2 * xp) / np.sum(img)), np.sqrt(np.sum((y - cy)**2 * yp) / np.sum(img))
    skewx, skewy = np.sum((x - cx)**3 * xp) / (np.sum(img) * sx**3), np.sum((y - cy)**3 * yp) / (np.sum(img) * sy**3)
    kurtx, kurty = np.sum((x - cx)**4 * xp) / (np.sum(img) * sx**4) - 3, np.sum((y - cy)**4 * yp) / (np.sum(img) * sy**4) - 3
    return (skewx, skewy), (kurtx, kurty)

def get_features(path, display=False):
    img = preproc(path, display=display)
    ratio_val = ratio(img)
    cent_y, cent_x = centroid(img)
    eccentricity, solidity = eccentricity_solidity(img)
    (skew_x, skew_y), (kurt_x, kurt_y) = skew_kurtosis(img)
    return ratio_val, cent_y, cent_x, eccentricity, solidity, skew_x, skew_y, kurt_x, kurt_y

def save_csv():
    if not os.path.exists('C:\\Users\\arbaz\\Desktop\\signature_forgery\\Features'):
        os.makedirs('C:\\Users\\arbaz\\Desktop\\signature_forgery\\Features/Training')
        os.makedirs('C:\\Users\\arbaz\\Desktop\\signature_forgery\\Features/Testing')
    
    for person in range(1, 13):
        per = f'{person:03d}'
        print(f'Saving features for person id-{per}')
        
        with open(f'C:\\Users\\arbaz\\Desktop\\signature_forgery\\Features\\Training/training_{per}.csv', 'w') as f:
            f.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
            for i in range(3):
                for g in [genuine_image_paths, forged_image_paths]:
                    path = os.path.join(g, f'{"021" if g == forged_image_paths else per}{per}_00{i}.png')
                    features = get_features(path)
                    output = 0 if g == forged_image_paths else 1
                    f.write(','.join(map(str, features)) + f',{output}\n')
        
        with open(f'C:\\Users\\arbaz\\Desktop\\signature_forgery\\Features\\Testing/testing_{per}.csv', 'w') as f:
            f.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y,output\n')
            for i in range(3, 5):
                for g in [genuine_image_paths, forged_image_paths]:
                    path = os.path.join(g, f'{"021" if g == forged_image_paths else per}{per}_00{i}.png')
                    features = get_features(path)
                    output = 0 if g == forged_image_paths else 1
                    f.write(','.join(map(str, features)) + f',{output}\n')

def process_test_image(path):
    features = get_features(path)
    if not os.path.exists('C:\\Users\\arbaz\\Desktop\\signature_forgery\\TestFeatures'):
        os.mkdir('C:\\Users\\arbaz\\Desktop\\signature_forgery\\TestFeatures')
    with open('C:\\Users\\arbaz\\Desktop\\signature_forgery\\TestFeatures/testcsv.csv', 'w') as f:
        f.write('ratio,cent_y,cent_x,eccentricity,solidity,skew_x,skew_y,kurt_x,kurt_y\n')
        f.write(','.join(map(str, features)) + '\n')

save_csv()

train_person_id = input("Enter person's id : ")
test_image_path = input("Enter path of signature image : ")
train_path = f'C:\\Users\\arbaz\\Desktop\\signature_forgery\\Features\\Training/training_{train_person_id}.csv'
process_test_image(test_image_path)
test_path = 'C:\\Users\\arbaz\\Desktop\\signature_forgery\\TestFeatures/testcsv.csv'

def read_csv(train_path, test_path, type2=False):
    df_train = pd.read_csv(train_path, usecols=range(n_input))
    train_input = df_train.values.astype(np.float32)
    correct_train = pd.read_csv(train_path, usecols=(n_input,)).values.ravel()
    corr_train = keras.utils.to_categorical(correct_train, 2)
    
    df_test = pd.read_csv(test_path, usecols=range(n_input))
    test_input = df_test.values.astype(np.float32)
    if type2:
        return train_input, corr_train, test_input
    
    correct_test = pd.read_csv(test_path, usecols=(n_input,)).values.ravel()
    corr_test = keras.utils.to_categorical(correct_test, 2)
    return train_input, corr_train, test_input, corr_test

# Network parameters
n_input = 9
n_hidden_1 = 7
n_hidden_2 = 10
n_hidden_3 = 30
n_classes = 2
learning_rate = 0.001
training_epochs = 1000

tf.reset_default_graph()
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], seed=1)),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes], seed=2))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], seed=3)),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes], seed=4))
}

def multilayer_perceptron(x):
    layer_1 = tf.tanh(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    out_layer = tf.tanh(tf.matmul(layer_1, weights['out']) + biases['out'])
    return out_layer

logits = multilayer_perceptron(X)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

train_input, corr_train, test_input = read_csv(train_path, test_path, type2=True)
start = time()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        _, cost = sess.run([train_op, loss_op], feed_dict={X: train_input, Y: corr_train})
    print("Optimization Finished!")
    
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: train_input, Y: corr_train}))
    output = sess.run(logits, feed_dict={X: test_input})
    print("Signature Verified" if np.argmax(output) == 1 else "Signature Forged")

print('Total time taken:', time() - start)
