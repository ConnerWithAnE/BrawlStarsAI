import torch
import cv2
from torch import nn
import numpy as np
import vgamepad as vg
import mss.tools
import time
import pyautogui
from PIL import ImageGrab
import pytesseract
import pygetwindow as gw
import pywinauto
import adb_shell
import keyboard
import random

from ahk import AHK



sLength = 918
sHeight = 516

pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
ahk = AHK(executable_path=r'C:\Program Files\AutoHotkey\AutoHotkey.exe')

keys = ['w','a','s','d']

win = ahk.find_window(title='LDPlayer')
win.activate()
while 1:
    k = random.randint(0,3)
    #win.send('w')
    ahk.key_down(keys[k])
    time.sleep(2)
    ahk.key_up(keys[k])
    ahk.key_press('e')
    time.sleep(1)
    #keyboard.press('w')


def getScreenAsArray(sct, num):
    '''must pass in mss.mss() as sct'''
   
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

    # Getting screen area showing teams remaining and masking it
    teams_left = cv2.inRange(gray_img[15:45, 80:210], 240, 255)
    cv2.imshow('', teams_left)



    #cv2.imwrite

    # Save to the picture file
    #mss.tools.to_png(tmp.rgb, tmp.size, output=output)
    #print(output)
    cv2.waitKey(1)

    return teams_left, screenshot_img




sctt = mss.mss()
num = 0
last_time = time.time()
teams_remaining = None
while 1:
    last_time = time.time()
    state = getScreenAsArray(sctt, num)
    num+=1
    #print(f"fps: {1/(time.time() - last_time)}")
    if num % 10 == 0:
        teams = pytesseract.image_to_string(state[0])
        if len(teams) > 1:
            teams_remaining = teams.rstrip()[-2:].strip(' ')
            print(teams_remaining)

'''
gamepad = vg.VX360Gamepad()
while 1:
    gamepad.update()
'''