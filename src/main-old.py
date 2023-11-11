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
#win.activate()

#ahk.click(459, 258)

#while 1:
    #print(ahk.get_mouse_position(coord_mode='Window'))
    #time.sleep(1)

'''
while 1:
    k = random.randint(0,3)
    #win.send('w')ede
    ahk.key_down(keys[k])
    time.sleep(2)
    ahk.key_up(keys[k])
    ahk.key_press('e')
    time.sleep(1)
    #keyboard.press('w')
'''


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

    # Convert the colour array to HSV, get the mean value of the luminance
    alive = int(cv2.cvtColor(np.array(screenshot_img), cv2.COLOR_BGR2HSV)[...,2].mean())
    

    # Getting screen area showing teams remaining and masking it
    teams_left = cv2.inRange(gray_img[15:45, 80:210], 240, 255)

    #cv2.imshow('', teams_left)
    #cv2.imwrite
    # Save to the picture file
    #mss.tools.to_png(tmp.rgb, tmp.size, output=output)
    #print(output)
    cv2.waitKey(1)

    return teamsRemaining(teams_left), screenshot_img, isAlive(alive)

def teamsRemaining(teams_arr):
    teams = pytesseract.image_to_string(teams_arr[0])
    if len(teams) > 1:
        teams_remaining = teams.rstrip()[-2:].strip(' ').strip(':')
        if teams_remaining.isdigit() and int(teams_remaining) <= 10:
            return teams
        
def isAlive(avg, lum_arr):
    new_avg = sum(lum_arr) / len(lum_arr)
    if new_avg+15 < avg:
        lum_arr.clear()
        return False, new_avg
    else:
        lum_arr.clear()
        return True, new_avg




sctt = mss.mss()
num = 0
last_time = time.time()
teams_remaining = None
avg = 0
alive_c = []
while 1:
    last_time = time.time()
    state, img, alive = getScreenAsArray(sctt, num)
    alive_c.append(alive)
    num+=1
    #print(avg)
    #print(f"fps: {1/(time.time() - last_time)}")
    if num % 10 == 0:
        print(teams_remaining(state))
    if num % 50 == 0:
        yalive, avg = isAlive(avg, alive_c)
        print(yalive)


my_dict = {}
for i in alive_c:
    my_dict[i] = alive_c.count(i)
    
print(my_dict)

'''
gamepad = vg.VX360Gamepad()
while 1:
    gamepad.update()
'''