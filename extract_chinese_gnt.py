#!/usr/bin/env python3

import os  
import numpy as np  
import struct  
import pickle
import PIL.Image  
import threading
   
def enumerate_gnt(gnt_file):
    with open(gnt_file, 'rb') as fh:
        header_size = 10  
        while True:  
            header = np.fromfile(fh, dtype='uint8', count=header_size)  
            if not header.size: break  
            sample_size = header[0] + (header[1]<<8) + (header[2]<<16) + (header[3]<<24)  
            tagcode = header[5] + (header[4]<<8)  
            width = header[6] + (header[7]<<8)  
            height = header[8] + (header[9]<<8)  
            if header_size + width*height != sample_size:  
                break  
            image = np.fromfile(fh, dtype='uint8', count=width*height).reshape((height, width))  
            yield image, tagcode

def read_from_gnt_dir(gnt_dir):  
    for file_name in os.listdir(gnt_dir):  
        if file_name.endswith('.gnt'):  
            file_path = os.path.join(gnt_dir, file_name)  
            for image, tagcode in enumerate_gnt(file_path):  
                yield image, tagcode, file_name

class CodeToID:
    def __init__(self):
        self.id_cnt = 0
        self.tag2id = {}
        self.tagarr = []
        self.lock = threading.Lock()

    def getId(self, tagcode):
        id = self.tag2id.get(tagcode)
        if id == None:
            self.lock.acquire()
            id = self.tag2id.get(tagcode)
            if id == None:
                id = len(self.tagarr)
                self.tagarr.append(tagcode)
                self.tag2id[tagcode] = id
            self.lock.release()
        return id

    def getUnicode(self, id):
        struct.pack('>H', self.tagarr[id]).decode('gb2312')

    def __getitem__(id):
        getUnicode(id)


def extract_gnts(idtab, gnt_dir, out_dir):
    counter = 0
    for image, tagcode, gntfile in read_from_gnt_dir(gnt_dir):  
        id = idtab.getId(tagcode)
        im = PIL.Image.fromarray(image) 
        sub = '%05d' % id
        dir = os.path.join(out_dir, sub)
        fname = sub + gntfile[0:4] + '.png'
        if not os.path.exists(dir):
            os.mkdir(dir)
        im.convert('RGB').save(os.path.join(dir, fname))
        counter += 1
        if counter > 0 and counter % 5000 == 0:
            print("extracted %d images" % counter)

#def get_gnt_list(gnt_dir):
#    gnts = []
#    for file_name in os.listdir(gnt_dir):  
#        if file_name.endswith('.gnt'):  
#            gnts.append({dir: gnt_dir,
#                         file: file_name,
#                         path: os.path.join(gnt_dir, file_name)})
#    gnts
#
#def extract_one_gnt():
#
#def mth_extract_gnts(idtab, gnt_dir, out_dir):
#    pool = threadpool.ThreadPool(12)
#    gnts = get_gnt_list(gnt_dir)
#    reqs = threadpool.makeRequests(extract_one_gnt, gnts)

#train_data_dir = "HWDB1.1trn_gnt"  
#test_data_dir = "HWDB1.1tst_gnt"  
train_data_dir = "gnt_train"  
test_data_dir = "gnt_test"  
idtab = CodeToID()
extract_gnts(idtab, train_data_dir, 'train')
extract_gnts(idtab, test_data_dir,  'test')
pickle.dump(idtab, 'idtable.bin', 2)

