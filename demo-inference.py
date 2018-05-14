# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:42:20 2017

@author: y00373036
"""

import tensorflow as tf
import os
import random
import tensorflow.contrib.slim as slim
import time
import numpy as np
import pickle
from PIL import Image


mode = "inference"
char_size = 3755
epochs = 5
batch_size = 128
checkpoint_dir = '/aiml/code/'
#train_data_dir = 'D:/Yang/softwares/Spider_ws/WordRecognition/data/train/'
#test_data_dir = 'D:/Yang/softwares/Spider_ws/WordRecognition/data/test/'

class DataIterator:
    def __init__(self, data_dir):
        self.image_names = []
        for root, sub_folder, file_list in os.walk(data_dir):
            self.image_names += [os.path.join(root, file_path) for file_path in file_list]
        random.shuffle(self.image_names)
        self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]

    @property
    def size(self):
        return len(self.labels)

    def input_pipeline(self, batch_size, num_epochs=None):
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)

        labels = input_queue[1]
        images_content = tf.read_file(input_queue[0])
        images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
        new_size = tf.constant([64, 64], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)
        return image_batch, label_batch


def build_graph(top_k):
    # with tf.device('/cpu:0'):
    images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='input_image')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')

    conv_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv1')
    max_pool_1 = slim.max_pool2d(conv_1, [2, 2], [2, 2], padding='SAME')
    conv_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv2')
    max_pool_2 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding='SAME')
    conv_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3')
    max_pool_3 = slim.max_pool2d(conv_3, [2, 2], [2, 2], padding='SAME')
    
    flatten = slim.flatten(max_pool_3)
    fc1 = slim.fully_connected(flatten, 1024, activation_fn=tf.nn.tanh, scope='fc1')
    logits = slim.fully_connected(fc1, char_size, activation_fn=None, scope='output_logit')
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))

    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    rate = tf.train.exponential_decay(2e-4, global_step, decay_steps=2000, decay_rate=0.97, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step = global_step)
    probabilities = tf.nn.softmax(logits)
    pred = tf.identity(probabilities, name = 'prediction')
    
    return {'images': images,
            'labels': labels,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'accuracy': accuracy}

def train():
    train_feeder = DataIterator(data_dir=train_data_dir)
    test_feeder = DataIterator(data_dir=test_data_dir)
    with tf.Session() as sess:
        train_images, train_labels = train_feeder.input_pipeline(batch_size)
        test_images, test_labels = test_feeder.input_pipeline(batch_size)
        graph = build_graph(top_k=1)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver = tf.train.Saver()

        print (':::Training Start:::')
        try:
            while not coord.should_stop():
                start_time = time.time()
                train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
                feed_dict = {graph['images']: train_images_batch,
                             graph['labels']: train_labels_batch}
                _, loss_val, step = sess.run(
                    [graph['train_op'], graph['loss'], graph['global_step']],
                    feed_dict=feed_dict)
                end_time = time.time()
                if step % 10 == 1:
                    print ("the step {0} takes {1} loss {2}".format(step, end_time - start_time, loss_val))
                if step > 200000:
                    break
                if step % 50 == 1:
                    test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                    feed_dict = {graph['images']: test_images_batch,
                                 graph['labels']: test_labels_batch}
                    accuracy_test = sess.run(
                        graph['accuracy'],
                        feed_dict=feed_dict)
                    print ('===============Eval a batch=======================')
                    print ('the step {0} test accuracy: {1}'.format(step, accuracy_test))
                    print ('===============Eval a batch=======================')
                if step % 200 == 1:
                    print ('Save the ckpt of {0}'.format(step))
                    saver.save(sess, os.path.join(checkpoint_dir, 'my-model'),
                               global_step=graph['global_step'])
        except tf.errors.OutOfRangeError:
            print ('==================Train Finished================')
            saver.save(sess, os.path.join(checkpoint_dir, 'my-model'), global_step=graph['global_step'])
        finally:
            coord.request_stop()
        coord.join(threads)

def new_inference(predict_dir):
    saver = tf.train.import_meta_graph( checkpoint_dir + "my-model-164152.meta", clear_devices=True)
    image_list = []
    new_file_list = []
    for root, _, file_list in os.walk(predict_dir):
        new_file_list += [file for file in file_list if ".nfs" not in file]
        new_file_list.sort(key= lambda x:int(x[:-4]))
        for file in new_file_list:
#            print (new_file_list)
            image = os.path.join(root, file)         
            temp_image = Image.open(image).convert('L')
            temp_image = temp_image.resize((64, 64), Image.ANTIALIAS)           
            temp_image = np.asarray(temp_image) / 255.0
            image_list.append(temp_image)
    image_list = np.asarray(image_list)       
    temp_image = image_list.reshape([len(new_file_list), 64, 64, 1])
    with tf.Session() as sess: 
        saver.restore(sess, checkpoint_dir + "my-model-164152") #读入模型参数
        graph = tf.get_default_graph()
        op = graph.get_tensor_by_name("prediction:0")
        input_tensor = graph.get_tensor_by_name('input_image:0')
        probs = sess.run(op,feed_dict = {input_tensor:temp_image})
        result = []
        for word in probs:
            result.append(np.argsort(-word)[:3])
        return result
        
def main():

    if mode == "train":
        train()
    if mode == 'inference':
        word_dict = pickle.load(open("/aiml/code/word_dict", "rb"))
        image_path = '/aiml/data/'
        index = new_inference(image_path)
        file = open("/aiml/result/result.txt", "w")        
#        print ("预测文字为: ")
        pred_list = []
        for i in index:
#            print ("最大几率三个:")
#            print (word_dict[str(i[0])],word_dict[str(i[1])],word_dict[str(i[2])])                
            pred_list.append(word_dict[str(i[0])])
            file.write(word_dict[str(i[0])])
        
if __name__ == "__main__":
#    tf.app.run()
    main()
