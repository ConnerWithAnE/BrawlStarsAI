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
        self.lower_hsv_cubes = np.array([38, 158,97])
        self.higher_hsv_cubes = np.array([64,255,255])
        self.lower_hsv_final = np.array([100, 50, 50])
        self.higher_hsv_final = np.array([130, 255, 255])
        self.hsv_img = None

    def teamsRemaining(self, teams_arr):
        teams = pytesseract.image_to_string(teams_arr)
        if len(teams) > 1:
            teams_remaining = teams.rstrip()[-2:].strip(' ').strip(':')
            if teams_remaining.isdigit() and int(teams_remaining) <= 10:
                return int(teams_remaining)
        return None
    
    def winScreen(self):
        mask = cv2.inRange(self.hsv_img, self.lower_hsv_final, self.higher_hsv_final)
        percentage_in_range = (np.count_nonzero(mask) / (self.hsv_img.shape[0] * self.hsv_img.shape[1])) * 100
        print(percentage_in_range)
        if percentage_in_range >= 65.0:
            return True

    
    def powerCubesHeld(self, p_cubes):
        cubes = pytesseract.image_to_string(p_cubes, config="--psm 6")
        if len(cubes) > 1:
            cubes_held = cubes.rstrip().strip(' ')
            if cubes_held.isdigit():
                return int(cubes_held)
        return None
            
    def isAlive(self):
        new_avg = sum(self.lum_arr) / len(self.lum_arr)
        if new_avg+8 < self.avg:
            if len(self.lum_arr) >= 50:
                self.lum_arr.clear()
            self.avg = 0
            return False
        else:
            if len(self.lum_arr) >= 50:
                self.lum_arr.clear()
            self.avg = new_avg
            return True
    
    def addLumArr(self, lum):
        self.lum_arr.append(lum)

    def clearLumArr(self):
        self.lum_arr.clear()


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
        alive_lum = int(cv2.cvtColor(np.array(screenshot_img), cv2.COLOR_BGR2HSV)[...,2].mean())
        
        # Getting screen area showing teams remaining and masking it
        teams_left = cv2.inRange(gray_img[15:45, 80:210], 240, 255)



        # Getting the screen area, filtered, showing the number of power cubes held
        self.hsv_img = cv2.cvtColor(np.array(screenshot_img), cv2.COLOR_BGR2HSV)
        power_cube_location = np.array(screenshot_img)[155:185, 450:490]
        power_cubes = cv2.cvtColor(power_cube_location, cv2.COLOR_RGB2HSV)
        power_cubes_mask = cv2.inRange(power_cubes, self.lower_hsv_cubes, self.higher_hsv_cubes)
        power_cubes_filtered = cv2.inRange(cv2.cvtColor(cv2.bitwise_and(power_cube_location, power_cube_location, mask=power_cubes_mask),cv2.COLOR_BGR2GRAY),150, 180)

        #cv2.imshow('', teams_left)
        #cv2.imwrite
        # Save to the picture file
        #mss.tools.to_png(tmp.rgb, tmp.size, output=output)
        #print(output)
        #cv2.waitKey(1)

        gray_img = np.reshape(gray_img, (1,516,918))

        return teams_left, gray_img, alive_lum, power_cubes_filtered

    





