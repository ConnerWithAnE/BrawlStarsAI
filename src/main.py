import torch
import cv2
from torch import nn
import numpy as np
import vgamepad as vg
import mss.tools
import time
from PIL import ImageGrab
import pytesseract


sLength = 918
sHeight = 516

pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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

    screenshot_img = sct.grab(monitor)
    gray_img = cv2.cvtColor(np.array(screenshot_img), cv2.COLOR_BGR2GRAY)

    teams_left = gray_img[15:45, 80:210]

    msk = cv2.inRange(teams_left, 240, 255)

    cv2.imshow('', msk)

    #print(pytesseract.image_to_string(teams_left))

    #cv2.imwrite

    # Save to the picture file
    #mss.tools.to_png(tmp.rgb, tmp.size, output=output)
    #print(output)
    cv2.waitKey(1)

    return msk




sctt = mss.mss()
num = 0
last_time = time.time()
while 1:
    last_time = time.time()
    teams_left = getScreenAsArray(sctt, num)
    num+=1
    #print(f"fps: {1/(time.time() - last_time)}")
    #print(num)
    if num % 10 == 0:
        teams = pytesseract.image_to_string(teams_left)
        if len(teams) > 1:
            print(teams.rstrip()[-1])
        #print(len(pytesseract.image_to_string(teams_left)))

'''
gamepad = vg.VX360Gamepad()
while 1:
    gamepad.update()
'''