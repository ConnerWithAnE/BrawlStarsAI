import json
import os
from time import sleep
from ahk import AHK



class Controller():

    def __init__(self):
        self.start_up = True
        self.config = json.load(open('./config.json', 'r'))
        self.ahk = AHK(executable_path=r'C:\Program Files\AutoHotkey\AutoHotkey.exe')
        self.win = self.ahk.find_window(title='LDPlayer')

    def startGame(self):
        self.win.activate()
        if self.start_up == True:
            self.start_up = False
            for _, value in self.config["start_match"].items():
                self.ahk.click(value["x"], value["y"])
                sleep(0.1)
        else:
            for _, value in self.config["start_match"].items():
                self.ahk.click(value["x"], value["y"])
                sleep(0.1)
        sleep(7)

    def exitGameLose(self):
        for _, value in self.config["exit_match_showdown"].items():
                self.ahk.click(value["x"], value["y"])
                sleep(0.1)

    def exitGameWin(self):
        for _, value in self.config["exit_match_win_showdown"].items():
                self.ahk.click(value["x"], value["y"])
                sleep(0.1)

    def getAHK(self):
        return self.ahk
    
    def getWIN(self):
        return self.win
