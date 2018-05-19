#!/usr/bin/env python3

import sys
import os
import pickle

from PIL import Image, ImageDraw, ImageFont


def gen_images(fontpath, wdict):
    ft = ImageFont.truetype(fontpath, 48)
    for wkey in wdict.keys():
        idx = int(wkey)
        word = wdict[wkey]

        img = Image.new("RGB", [80,80], 'white')
        draw = ImageDraw.Draw(img)
        draw.text([2,2], word, font=ft, fill='black')

        sub = '%05d' % idx
        dir = os.path.join('out', sub)
        fname = sub + '_zh' + fontpath[-7:-4] + '.png'
        if not os.path.exists(dir):
            os.mkdir(dir)
        img.save(os.path.join(dir, fname))

    return

wdict = pickle.load(open('word_dict', "rb"))
for ttf in sys.argv[1:]:
    gen_images(ttf, wdict)

#show_font("zh_CN_109.ttf")
