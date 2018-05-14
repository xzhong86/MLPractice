#!/usr/bin/env python3

import os 
import random
import tensorflow as tf 
import tensorflow.contrib.slim as slim 
import time 
import logging 
import numpy as np 
import pickle 
from PIL import Image 

FLAGS = None

def init_flags(paiml, mode):
    tf.app.flags.DEFINE_integer('charset_size', 3755, "Choose the first `charset_size` character to conduct our experiment.") 
    tf.app.flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.") 
    tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray") 
    tf.app.flags.DEFINE_integer('max_steps', 200002, 'the max training steps ') 
    tf.app.flags.DEFINE_integer('eval_steps', 100, "the step num to eval") 
    # save checkpoint for each 1000 steps
    tf.app.flags.DEFINE_integer('save_steps', 1000, "the steps to save") 

    tf.app.flags.DEFINE_string('aiml_dir', paiml, 'AIML root dir')
    tf.app.flags.DEFINE_string('checkpoint_dir', paiml + 'dfs/checkpoint/', 'the checkpoint dir') 
    tf.app.flags.DEFINE_string('train_data_dir', paiml + 'data/train/', 'the train dataset dir') 
    tf.app.flags.DEFINE_string('test_data_dir', paiml + 'data/test/', 'the test dataset dir')
    tf.app.flags.DEFINE_string('log_dir', paiml + 'dfs/checkpoint/', 'the logging dir') 

    tf.app.flags.DEFINE_string('mode', mode, 'Running mode. One of {"filter", "inference"}')

    tf.app.flags.DEFINE_string('wdict', 'word_dict', 'set word_dict file path')
    tf.app.flags.DEFINE_string('chkpt', None, 'set chkpt prefix path')
    tf.app.flags.DEFINE_string('pred_dir', None, 'set pred_dir path')

    global FLAGS
    FLAGS = tf.app.flags.FLAGS 


def get_file_list(predict_dir, recursive=False, do_sort=False):
    file_list = []
    for root, _, files in os.walk(predict_dir):
        files = [f for f in files if '.nfs' not in f]
        if do_sort:
            files.sort(key=lambda x: int(x[:-4]))
        for f in files:
            file_list.append(os.path.join(root, f))
        if not recursive:
            break
    return file_list

def do_inference(file_list, chkpt_prefix):
    saver = tf.train.import_meta_graph( chkpt_prefix + '.meta', clear_devices=True)
    image_list = []
    for _file in file_list:
        temp_image = Image.open(_file).convert('L')
        temp_image = temp_image.resize((64, 64), Image.ANTIALIAS)           
        temp_image = np.asarray(temp_image) / 255.0
        image_list.append(temp_image)
    
    image_list = np.asarray(image_list)       
    temp_image = image_list.reshape([len(file_list), 64, 64, 1])
    with tf.Session() as sess: 
        saver.restore(sess, chkpt_prefix) # load paramters
        graph = tf.get_default_graph()
        op = graph.get_tensor_by_name("prediction:0")
        input_tensor = graph.get_tensor_by_name('image_batch:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        probs = sess.run(op,feed_dict = {input_tensor:temp_image, keep_prob:1.0})
        result = []
        for word in probs:
            result.append(np.argsort(-word)[:5])
        return { 'results': result, 'images': file_list }

def do_big_inference(file_list, chkpt_prefix):
    saver = tf.train.import_meta_graph( chkpt_prefix + '.meta', clear_devices=True)
    with tf.Session() as sess: 
        saver.restore(sess, chkpt_prefix) # load paramters
        graph = tf.get_default_graph()
        op = graph.get_tensor_by_name("prediction:0")
        input_tensor = graph.get_tensor_by_name('image_batch:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')

        results = []
        for gp in range(0, len(file_list), 5000):
            image_list = []
            for _file in file_list[gp:gp+5000]:
                temp_image = Image.open(_file).convert('L')
                temp_image = temp_image.resize((64, 64), Image.ANTIALIAS)           
                temp_image = np.asarray(temp_image) / 255.0
                image_list.append(temp_image)

            print("inference group %d:%d" % (gp, gp+5000))
            num_files = len(image_list)
            image_list = np.asarray(image_list)       
            temp_image = image_list.reshape([num_files, 64, 64, 1])
            probs = sess.run(op,feed_dict = {input_tensor:temp_image, keep_prob:1.0})
            for word in probs:
                results.append(np.argsort(-word)[:5])

        return { 'results': results, 'images': file_list }


def do_estimate(wdict, pred_results):
    images = pred_results['images']
    cword = lambda c: wdict[str(c)]
    miss_cnt = 0
    for idx, pred in enumerate(pred_results['results']):
        img_path = images[idx]
        if os.path.islink(img_path):
            img_path = os.readlink(img_path)
        code = int(img_path.split(os.sep)[-2])
        pred = pred.tolist()
        hitn = pred.index(code) if code in pred else len(pred)
        if hitn > 0:
            miss_cnt += 1
            pred_str = ' '.join([cword(p) for p in pred])
            print("Out%d: expect %s, pred %s, %s" %
                  (hitn, cword(code), pred_str, img_path))
    total = len(images)
    right_rate = float(total - miss_cnt) / total
    print("Total %d/%d, %.4f" % (miss_cnt, total, right_rate))

def main(argv=None):
    paiml = FLAGS.aiml_dir
    word_dict = pickle.load(open(paiml + "code/word_dict", "rb"))
    chkpt_prefix = FLAGS.chkpt or paiml + 'code/model-0001'
    pred_dir = FLAGS.pred_dir or paiml + 'data/'

    if FLAGS.mode == "filter":
        print("run filter function")
        file_list = get_file_list(pred_dir, True, True)
        print("get %d files, start inference..." % len(file_list))
        if len(file_list) > 5000:
            pred_res  = do_big_inference(file_list, chkpt_prefix)
        else:
            pred_res  = do_inference(file_list, chkpt_prefix)
        do_estimate(word_dict, pred_res)
        
    elif FLAGS.mode == "inference":
        file_list = get_file_list(pred_dir, False, True)
        pred_res  = do_inference(file_list, chkpt_prefix)
        results = pred_res['results']
        pred_list = []
        for words in results: # get top 3 for each predict result
            possible = [word_dict[str(word)] for word in words[:3]]
            pred_list.append(possible)

        output = open(paiml + "result/result.txt", "w")        
        for pred in pred_list:
            output.write(pred[0])
        output.close()

    else:
        print('need to specify work mode')


# main start
if os.path.basename(__file__) == 'inference.py':
    init_flags('/aiml/', 'inference')
else:
    init_flags('./aiml/', 'train')

tf.app.run()

