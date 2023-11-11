from ahk import AHK
import cv2
import numpy as np
import time
import mss.tools
import pytesseract


class ImageController():

    pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    def __init__(self):
        self.sct = mss.mss()
        self.lum_arr = []
        self.avg = 0 

    def teamsRemaining(self, teams_arr):
        teams = pytesseract.image_to_string(teams_arr)
        if len(teams) > 1:
            teams_remaining = teams.rstrip()[-2:].strip(' ').strip(':')
            if teams_remaining.isdigit() and int(teams_remaining) <= 10:
                return teams_remaining
        return None
            
    def isAlive(self, alive):
        self.lum_arr.append(alive)
        new_avg = sum(self.lum_arr) / len(self.lum_arr)
        if new_avg+15 < self.avg:
            if len(self.lum_arr) >= 50:
                self.lum_arr.clear()
            self.avg = new_avg
            return False
        else:
            if len(self.lum_arr) >= 50:
                self.lum_arr.clear()
            self.avg = new_avg
            return True



    def getScreenAsArray(self):
        '''must pass in mss.mss() as sct'''
    
        monitor_number = 3
        mon = self.sct.monitors[monitor_number]
        monitor = {
            "top": mon["top"] + 35,
            "left": mon["left"],
            "width": 918,
            "height": 516,
            "mon": monitor_number,
        }

        screenshot_img = self.sct.grab(monitor)
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

        return self.teamsRemaining(teams_left), gray_img, self.isAlive(alive)

    





