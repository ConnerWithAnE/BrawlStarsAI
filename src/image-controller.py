from ahk import AHK
import cv2
import numpy as np
import time
import mss.tools
import pytesseract


class ImageController():

    def __init__(self):
        self.sct = mss.mss()
        self.tesseact = pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def teamsRemaining(self, teams_arr):
        teams = self.tesseact.image_to_string(teams_arr[0])
        if len(teams) > 1:
            teams_remaining = teams.rstrip()[-2:].strip(' ').strip(':')
            if teams_remaining.isdigit() and int(teams_remaining) <= 10:
                return teams
            
    def isAlive(self, avg, lum_arr):
        new_avg = sum(lum_arr) / len(lum_arr)
        if new_avg+15 < avg:
            lum_arr.clear()
            return False, new_avg
        else:
            lum_arr.clear()
            return True, new_avg



    def getScreenAsArray(self, sct, num):
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

        return self.teamsRemaining(teams_left), screenshot_img, self.isAlive(alive)

    





