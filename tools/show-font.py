#!/usr/bin/env python3

import sys
import os

from PIL import Image, ImageDraw, ImageFont

EXAMPLE_TEXT = u"模块定义了相同名称的类轮廓字体"

def show_font(fontpath, text=EXAMPLE_TEXT):
    try:
        img = Image.new("RGB", [1200,80], 'white')
        draw = ImageDraw.Draw(img)
        print("open font " + fontpath)
        ft = ImageFont.truetype(fontpath, 48)
        print("draw text")
        draw.text([10,2], text, font=ft, fill='black')
        #img.show()
        rname = os.path.splitext(fontpath)[0]
        img.save(rname + '.jpg', "jpeg")
    except IOError:
        print("get error when open font %s" % fontpath)

    return


for ttf in sys.argv[1:]:
    show_font(ttf)

#show_font("zh_CN_109.ttf")
