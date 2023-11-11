import json
import os
from time import sleep
from ahk import AHK
from pynput import mouse

ahk = AHK(executable_path=r'C:\Program Files\AutoHotkey\AutoHotkey.exe')
mouselistener = None
brawler = None


''' Mouse Controls '''
def on_click(x, y, button, pressed):
    if button == mouse.Button.left:
        #return ahk.get_mouse_position(coord_mode='Window')
        return False

 
''' Config '''
def openConfig():
    config_function = input("Ovewrite existing config or append to existing? (Enter o or a) ")
    if os.path.isfile('./config.json') and config_function == "a":
        return json.load(open('./config.json'))
    else:
        return {}
    
def writeConfig(config):
    config_j = json.dumps(config)
    with open('./config.json', 'w') as file:
        file.write(config_j)

def setStartClicks(config, ahk_obj, brawler, steps_num):
    steps = steps_num
    start_clicks = {}
    print(f"Click on the {steps} desired regions of the Screen")
    for step in range (1, steps+1):
        print(f'Click number {step}: ', end='')
        with mouse.Listener(on_click=on_click) as listener:
            listener.join()
            pos = ahk_obj.get_mouse_position(coord_mode='Window')
            start_clicks[step] = {"x": pos[0], "y": pos[1]}
            print(pos)
        sleep(0.1)
    print(start_clicks)
    config[f"start_match_{brawler}"] = start_clicks
    
def setExitClicks(config, ahk_obj, mode, steps_num):
    steps = steps_num
    exit_clicks = {}
    print(f"Click on the {steps} desired regions of the Screen")
    for step in range (1, steps+1):
        print(f'Click number {step}: ', end='')
        with mouse.Listener(on_click=on_click) as listener:
            listener.join()
            pos = ahk_obj.get_mouse_position(coord_mode='Window')
            exit_clicks[step] = {"x": pos[0], "y": pos[1]}
            print(pos)
        sleep(0.1)
    print(exit_clicks)
    config[f"exit_match_{mode}"] = exit_clicks

if __name__ == '__main__':
    #option = None
    #while option != 0:
    config = openConfig()
    brawler = input("Please enter brawler name (lowercase): ")
    setStartClicks(config, ahk, brawler, 2)
    setExitClicks(config, ahk, "showdown", 1)
    writeConfig(config)
    print(config)