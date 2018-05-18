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
LOGGER = None

def init_logger(filename):
    log = logging.getLogger('Training a chinese write char recognition') 
    log.setLevel(logging.DEBUG) 
    fmt = logging.Formatter('%(asctime)s: %(message)s', '%H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    if filename:
        ch = logging.FileHandler(filename, 'w')
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(fmt)
        log.addHandler(ch)
    global LOGGER
    LOGGER = log

def init_flags(paiml, mode):
    tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down") 
    tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness") 
    tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast") 

    tf.app.flags.DEFINE_integer('charset_size', 3755, "Choose the first `charset_size` character to conduct our experiment.") 
    tf.app.flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.") 
    tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray") 
    tf.app.flags.DEFINE_integer('max_steps', 200002, 'the max training steps ') 
    tf.app.flags.DEFINE_integer('eval_steps', 100, "the step num to eval") 
    # save checkpoint for each 1000 steps
    tf.app.flags.DEFINE_integer('save_steps', 10000, "the steps to save") 
    tf.app.flags.DEFINE_integer('chkpt_max_keep', 10, "max checkpoint to keep") 

    tf.app.flags.DEFINE_string('aiml_dir', paiml, 'AIML root dir')
    tf.app.flags.DEFINE_string('checkpoint_dir', paiml + 'dfs/checkpoint/', 'the checkpoint dir') 
    tf.app.flags.DEFINE_string('train_data_dir', paiml + 'data/train/', 'the train dataset dir') 
    tf.app.flags.DEFINE_string('test_data_dir', paiml + 'data/test/', 'the test dataset dir')
    tf.app.flags.DEFINE_string('log_dir', paiml + 'dfs/checkpoint/', 'the logging dir') 

    tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from checkpoint') 
    tf.app.flags.DEFINE_boolean('epoch', 1, 'Number of epoches') 
    tf.app.flags.DEFINE_integer('batch_size', 128, 'Validation batch size') 
    tf.app.flags.DEFINE_string('mode', mode, 'Running mode. One of {"train", "valid", "inference"}')

    tf.app.flags.DEFINE_float('acc_top', 0.85, 'set top accuracy')
    tf.app.flags.DEFINE_boolean('with_log', False, 'enable log file')
    tf.app.flags.DEFINE_string('chkpt', None, 'set chkpt prefix path')
    tf.app.flags.DEFINE_string('pred_dir', None, 'set pred_dir path')
    tf.app.flags.DEFINE_boolean('estimate', False, 'do estimate in inference mode')

    global FLAGS
    FLAGS = tf.app.flags.FLAGS 


class DataIterator:
    def __init__(self, data_dir): 
        # Set FLAGS.charset_size to a small value if available computation power is limited. 
        truncate_path = data_dir + ('%05d' % FLAGS.charset_size) 
        self.image_names = [] 
        for root, sub_folder, file_list in os.walk(data_dir): 
            if root < truncate_path: 
                self.image_names += [os.path.join(root, file_path) for file_path in file_list] 
        random.shuffle(self.image_names) 
        self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names] 
        print('truncate_path: ' + truncate_path)
        print('images: ' + str(len(self.image_names)))
        print('labels: ' + str(len(self.labels)))

    @property 
    def size(self): 
        return len(self.labels) 
 
    @staticmethod 
    def data_augmentation(images): 
        if FLAGS.random_flip_up_down: 
            images = tf.image.random_flip_up_down(images) 
        if FLAGS.random_brightness: 
            images = tf.image.random_brightness(images, max_delta=0.3) 
        if FLAGS.random_contrast: 
            images = tf.image.random_contrast(images, 0.8, 1.2) 
        return images 
 
    def input_pipeline(self, batch_size, num_epochs=None, aug=False): 
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string) 
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64) 
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs) 
 
        labels = input_queue[1] 
        images_content = tf.read_file(input_queue[0]) 
        images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32) 
        if aug: 
            images = self.data_augmentation(images) 
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32) 
        images = tf.image.resize_images(images, new_size) 
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000, 
                                                          min_after_dequeue=10000) 
        return image_batch, label_batch 


def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
         alphas = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
         pos = tf.nn.relu(_x)
         neg = alphas * (_x - abs(_x)) * 0.5
         return pos + neg   
 
def build_graph(top_k): 
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob') 
    images =    tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1], name='image_batch') 
    labels =    tf.placeholder(dtype=tf.int64,   shape=[None], name='label_batch')  

    batch_norm_params = {'decay':0.999,
                         'epsilon': 0.0001,                         
                         'updates_collections':tf.GraphKeys.UPDATE_OPS,
                         'variables_collections':{'beta': None, 'gamma': None, 'moving_mean':'moving_vars', 'moving_variance':'moving_vars',}}
    
    
    # 创建第一段卷积网络 -- outputs 112x112x64
    # 两个卷积层的卷积核都是3*3，卷积核数量（输出通道数）均为64，步长1*1，全像素扫描。
    #conv_1_1 = slim.conv2d(images, 64, [3, 3], [1, 1], padding='SAME', scope='conv1_1')
    #conv_1_2 = slim.conv2d(conv_1_1, 64, [3, 3], [1, 1], padding='SAME', scope='conv1_2')
    conv_1_1 = slim.conv2d(images, 64, [3, 3], [1, 1], padding='SAME', activation_fn=prelu, 
                           normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, scope='conv1_1')  #输出 64x64x64
    #conv_1_2 = slim.conv2d(conv_1_1, 64, [3, 3], [1, 1], padding='SAME', activation_fn=prelu, 
    #                       normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, scope='conv1_2')  #输出 64x64x64
    pool1   = slim.max_pool2d(conv_1_1, [2, 2], [2, 2], padding='SAME')
 
    
    # 创建第二段卷积网络 -- outputs 56x56x128
    #conv_2_1 = slim.conv2d(pool1, 128, [3, 3], [1, 1], padding='SAME', scope='conv2_1')
    #conv_2_2 = slim.conv2d(conv_2_1, 128, [3, 3], [1, 1], padding='SAME', scope='conv2_2')
    conv_2_1 = slim.conv2d(pool1, 128, [3, 3], [1, 1], padding='SAME', activation_fn=prelu, 
                           normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, scope='conv2_1')  
    #conv_2_2 = slim.conv2d(conv_2_1, 128, [3, 3], [1, 1], padding='SAME', activation_fn=prelu, 
    #                       normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, scope='conv2_2') 
    pool2   = slim.max_pool2d(conv_2_1, [2, 2], [2, 2], padding='SAME')    


    conv_3_1 = slim.conv2d(pool2, 256, [3, 3], [1, 1], padding='SAME', activation_fn=prelu, 
                           normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, scope='conv3_1')  
    #conv_2_2 = slim.conv2d(conv_2_1, 128, [3, 3], [1, 1], padding='SAME', activation_fn=prelu, 
    #                       normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, scope='conv2_2') 
    pool3   = slim.max_pool2d(conv_3_1, [2, 2], [2, 2], padding='SAME')    


    conv_4_1 = slim.conv2d(pool3, 512, [3, 3], [1, 1], padding='SAME', activation_fn=prelu, 
                           normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, scope='conv4_1')  
    conv_4_2 = slim.conv2d(conv_4_1, 512, [3, 3], [1, 1], padding='SAME', activation_fn=prelu, 
                           normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params, scope='conv4_2') 
    pool4   = slim.max_pool2d(conv_4_2, [2, 2], [2, 2], padding='SAME')    
   
    # flatten 将卷积网络的输出结果进行扁平化
    flatten = slim.flatten(pool4)
    
    fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024, activation_fn=tf.nn.tanh, scope='fc1') 
    logits = slim.fully_connected(slim.dropout(fc1, keep_prob), FLAGS.charset_size, activation_fn=None, scope='logits1')
        
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)) 
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32)) 
 
    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False) 
    
    rate = tf.train.exponential_decay(2e-4, global_step, decay_steps=2000, decay_rate=0.97, staircase=True)
    
    train_op = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step=global_step) 
    probabilities = tf.nn.softmax(logits) 
    pred = tf.identity(probabilities, name = 'prediction')

    tf.summary.scalar('loss', loss) 
    tf.summary.scalar('accuracy', accuracy) 
    merged_summary_op = tf.summary.merge_all() 
    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k) 
    #accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))
    accuracy_in_top_k = None
 
    return {'images': images, 
            'labels': labels, 
            'keep_prob': keep_prob, 
            'top_k': top_k, 
            'global_step': global_step, 
            'train_op': train_op, 
            'loss': loss, 
            'accuracy': accuracy, 
            'accuracy_top_k': accuracy_in_top_k, 
            'merged_summary_op': merged_summary_op, 
            'predicted_distribution': probabilities, 
            'predicted_index_top_k': predicted_index_top_k, 
            'predicted_val_top_k': predicted_val_top_k} 
 
def train(): 
    logger = LOGGER
    logger.info('Begin training')
    train_feeder = DataIterator(data_dir=FLAGS.train_data_dir) 
    test_feeder = DataIterator(data_dir=FLAGS.test_data_dir)
    with tf.Session() as sess: 
        logger.info("batch_size:" + str(FLAGS.batch_size))
        train_images, train_labels = train_feeder.input_pipeline(batch_size=FLAGS.batch_size, aug=True) 
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size) 
        
        graph = build_graph(top_k=1) 
        
        sess.run(tf.global_variables_initializer()) 
        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners(sess=sess, coord=coord) 
        saver = tf.train.Saver(max_to_keep=FLAGS.chkpt_max_keep) 
 
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph) 
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val') 
        start_step = 0 
        if FLAGS.restore: 
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir) 
            if ckpt: 
                saver.restore(sess, ckpt) 
                logger.info("restore from the checkpoint {0}".format(ckpt)) 
                start_step += int(ckpt.split('-')[-1])
            else:
                logger.info('restore from {0} failed.'.format(FLAGS.checkpoint_dir))
        else:
            logger.info("FLAGS.restore:" + str(FLAGS.restore))
 
        logger.info(':::Training Start:::') 
        acc_top = FLAGS.acc_top or 0.85
        try: 
            while not coord.should_stop(): 
                start_time = time.time() 
                train_images_batch, train_labels_batch = sess.run([train_images, train_labels]) 
                feed_dict = {graph['images']: train_images_batch, 
                             graph['labels']: train_labels_batch, 
                             graph['keep_prob']: 0.8} 
                _, loss_val, train_summary, step = sess.run( 
                    [graph['train_op'], graph['loss'], graph['merged_summary_op'], graph['global_step']], 
                    feed_dict=feed_dict) 
                train_writer.add_summary(train_summary, step) 
                end_time = time.time() 
                logger.debug("the step {0} takes {1} loss {2}".format(step, end_time - start_time, loss_val)) 
                if step > FLAGS.max_steps: 
                    break
                if step % FLAGS.eval_steps == 1: 
                    test_images_batch, test_labels_batch = sess.run([test_images, test_labels]) 
                    feed_dict = {graph['images']: test_images_batch, 
                                 graph['labels']: test_labels_batch, 
                                 graph['keep_prob']: 1.0} 
                    accuracy_test, test_summary = sess.run( 
                        [graph['accuracy'], graph['merged_summary_op']], 
                        feed_dict=feed_dict) 
                    test_writer.add_summary(test_summary, step) 
                    logger.debug('===============Eval a batch=======================') 
                    logger.info( 'the step {0} test accuracy: {1}' 
                                .format(step, accuracy_test)) 
                    logger.debug('===============Eval a batch=======================') 
                    if accuracy_test - acc_top >= 0.005:
                        logger.info('Save new top result of {0}, {1} => {2}'.format(
                            step, acc_top, accuracy_test))
                        acc_top = accuracy_test
                        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-top'),
                                    global_step=graph['global_step'])
                        continue
                
                if step % FLAGS.save_steps == 1: 
                    logger.info('Save the ckpt of {0}'.format(step)) 
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), 
                               global_step=graph['global_step']) 
        except tf.errors.OutOfRangeError: 
            logger.info('==================Train Finished================') 
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=graph['global_step']) 
        finally: 
            coord.request_stop() 
        coord.join(threads) 

def validation(): 
    logger = LOGGER
    logger.info('validation') 
    #test_feeder = DataIterator(data_dir='../data/test/') 
    test_feeder = DataIterator(data_dir=FLAGS.test_data_dir)
    
    #FLAGS.checkpoint_dir = paiml + 'dfs/checkpoint/v1/'
 
    final_predict_val = [] 
    final_predict_index = [] 
    groundtruth = [] 
 
    with tf.Session() as sess: 
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size, num_epochs=1) 
        graph = build_graph(3) 
 
        sess.run(tf.global_variables_initializer()) 
        sess.run(tf.local_variables_initializer())  # initialize test_feeder's inside state 
 
        coord = tf.train.Coordinator() 
        threads = tf.train.start_queue_runners(sess=sess, coord=coord) 
 
        saver = tf.train.Saver() 
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir) 
        
        if ckpt: 
            saver.restore(sess, ckpt) 
            logger.info("restore from the checkpoint {0}".format(ckpt))
        else:
            logger.warn('restore from {0} failed.'.format(FLAGS.checkpoint_dir))
 
        logger.info(':::Start validation:::') 
        try: 
            i = 0 
            acc_top_1, acc_top_k = 0.0, 0.0 
            while not coord.should_stop(): 
                i += 1 
                start_time = time.time() 
                test_images_batch, test_labels_batch = sess.run([test_images, test_labels]) 
                feed_dict = {graph['images']: test_images_batch, 
                             graph['labels']: test_labels_batch, 
                             graph['keep_prob']: 1.0} 
                batch_labels, probs, indices, acc_1, acc_k = sess.run([graph['labels'], 
                                                                       graph['predicted_val_top_k'], 
                                                                       graph['predicted_index_top_k'], 
                                                                       graph['accuracy'], 
                                                                       graph['accuracy_top_k']], feed_dict=feed_dict) 
                final_predict_val += probs.tolist() 
                final_predict_index += indices.tolist() 
                groundtruth += batch_labels.tolist() 
                acc_top_1 += acc_1 
                acc_top_k += acc_k 
                end_time = time.time() 
                logger.info("the batch {0} takes {1} seconds, accuracy = {2}(top_1) {3}(top_k)" 
                            .format(i, end_time - start_time, acc_1, acc_k))
 
        except tf.errors.OutOfRangeError: 
            logger.info('==================Validation Finished================') 
            acc_top_1 = acc_top_1 * FLAGS.batch_size / test_feeder.size 
            acc_top_k = acc_top_k * FLAGS.batch_size / test_feeder.size 
            logger.info('top 1 accuracy {0} top k accuracy {1}'.format(acc_top_1, acc_top_k)) 
        finally: 
            coord.request_stop() 
        coord.join(threads) 
    return {'prob': final_predict_val, 'indices': final_predict_index, 'groundtruth': groundtruth}         


def do_inference(predict_dir, chkpt_prefix):
    saver = tf.train.import_meta_graph( chkpt_prefix + '.meta', clear_devices=True)
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
        saver.restore(sess, chkpt_prefix) # load paramters
        graph = tf.get_default_graph()
        op = graph.get_tensor_by_name("prediction:0")
        input_tensor = graph.get_tensor_by_name('image_batch:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        probs = sess.run(op,feed_dict = {input_tensor:temp_image, keep_prob:1.0})
        result = []
        for word in probs:
            result.append(np.argsort(-word)[:3])
        return result


def do_estimate(predict_dir, wdict, results):
    wrong_cnt = 0
    word = lambda c: wdict[str(c)]
    for idx, pred in enumerate(results):
        ln = os.readlink(predict_dir + '/' + str(idx) + '.png')
        code = int(ln.split('/')[-2])
        #print('{0}: {1} {2} {3}'.format(idx, ln, code, word(pred[0])))
        if code != pred[0]:
            wrong_cnt += 1
            print("{0}.png: expect {1}, pred {2} {3} {4}"
                  .format(idx, word(code),
                          word(pred[0]), word(pred[1]), word(pred[2])))

    total = len(results)
    print("total %d/%d, %.4f" % (wrong_cnt, total, float(total - wrong_cnt) / total))


def main(argv=None):
    logname = None
    if argv and len(argv) > 0:
        print("Args possible wrong: " + " ".join(argv))

    if FLAGS.mode == "train" or FLAGS.with_log:
        date = time.strftime('%Y_%m%d_%H%M')
        logname = '{0}-{1}.log'.format(FLAGS.mode, date)
    init_logger(logname)

    if FLAGS.mode == "train":
        train()
    elif FLAGS.mode == "valid":
        validation()
    elif FLAGS.mode == "inference":
        paiml = FLAGS.aiml_dir
        word_dict = pickle.load(open(paiml + "code/word_dict", "rb"))
        chkpt_prefix = FLAGS.chkpt or paiml + 'code/model-0001'
        pred_dir = FLAGS.pred_dir or paiml + 'data/'
        results = do_inference(pred_dir, chkpt_prefix)
        if FLAGS.estimate:
            do_estimate(pred_dir, word_dict, results)

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

