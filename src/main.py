import torch
import cv2
from torch import nn
import numpy as np
import vgamepad as vg
import mss.tools
import time
from PIL import ImageGrab


sLength = 918
sHeight = 516


def getScreenAsArray(sct, num):
    '''must pass in mss.mss() as sct'''
    #with mss.mss() as sct:
    monitor_number = 3
    mon = sct.monitors[monitor_number]
    monitor = {
        "top": mon["top"] + 35,
        "left": mon["left"],
        "width": sLength,
        "height": sHeight,
        "mon": monitor_number,
        "num": num
    }

    output = "sct-mon{mon}_{top}x{left}_{width}x{height}-{num}.png".format(**monitor)



    sct_img = np.array(sct.grab(monitor))

    cv2.imshow("hi",sct_img)

    # Save to the picture file
    #mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
    #print(output)






sctt = mss.mss()
num = 0
last_time = time.time()
while 1:
    last_time = time.time()
    getScreenAsArray(sctt, num)
    num+=1
    print(f"fps: {1/(time.time() - last_time)}")

'''
gamepad = vg.VX360Gamepad()
while 1:
    gamepad.update()
'''