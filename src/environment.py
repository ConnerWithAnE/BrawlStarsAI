import time

class Environment():
    def __init__(self, image_controller, transform=None):
        self.actions = ['w','a','s','d']
        self.img_controller = image_controller
        self.teams = None
        _, self.image_state, _, _ = self.img_controller.getScreenAsArray()
        self.cubes = 0
        self.curr_step = 0
        self.time = 0
        #self.transform = transform
        '''
        self.action_state = {
            'w':0,
            'a':0,
            's':0,
            'd':0,
            'e':0,
            'f':0
        }
        '''
        self.action_state = [0,0,0,0]
        self.alive = True

    def move(self, action, ahk):
        for i in range(4):
            if self.action_state[i] != action[0][i]:
                if self.action_state[i] == 1:
                    print(f"up {self.actions[i]} ", end="")
                    ahk.key_up(self.actions[i])
                else:
                    print(f"down {self.actions[i]} ", end="")
                    ahk.key_down(self.actions[i])

        # Getting the e and f key pressed
        if action[0][4] == 1:
            ahk.key_press('e')
        if action[0][5] == 1:
            ahk.key_press('f')

        self.action_state = action[0]
        
    def step(self, action, ahk):
        #self.current_state = self.image_stream
        '''
        while self.teams not in [1,2,3,4,5,6,7,8,9,10]:
            self.teams, self.image_state = self.controller.getScreenAsArray()
        
        for key in set(self.action_state.keys()).intersection(action.keys()):
            if self.action_state[key] != action[key]:
                if self.action_state[key] == 1:
                    ahk.key_up(key)
                else:
                    ahk.key_down(key)
        '''
        self.curr_step+=1

        # Getting the wasd key up or down
        self.move(action, ahk)

        n_teams, n_image_state, n_alive, n_cubes = self.img_controller.getScreenAsArray()
        self.img_controller.addLumArr(n_alive)
        print(f"\nstep: {self.curr_step}")
        if self.curr_step % 4 == 0:
            self.alive = self.img_controller.isAlive()
        if self.curr_step % 10 == 0:
            self.img_controller.clearLumArr()
        n_teams = self.img_controller.teamsRemaining(n_teams)
        n_cubes = self.img_controller.powerCubesHeld(n_cubes)
        #print(f"Num Teams Saved: {self.teams}")
        #print(f"Num Teams new: {n_teams}")
        exit_type = 0
        reward = 0
        print(n_teams)
        if self.alive == False:
            reward -= 10
        if n_teams != None:
            if n_teams < self.teams:
                reward += 0.2
            self.teams = n_teams
            '''
            if self.teams <= 2:
                exit_type = 1
                if self.img_controller.winScreen():
                    exit_type = 1
                    reward+=15
                    self.alive = False
            '''
        if n_teams == None:
            if self.img_controller.winScreen() or self.teams == 1:
                    exit_type = 1
                    reward+=15
                    self.alive = False
        if n_cubes != None:
            print(f"Cubes: {n_cubes}")
            if n_cubes > self.cubes:
                reward += 10
            self.cubes = n_cubes
            '''
            else:
                reward = -0.5
            '''
        reward -= ((time.time()-self.time)/60)*0.1
        
        self.image_state = n_image_state
        
        return self.image_state, reward, self.alive, exit_type


    def reset(self, controller):
        controller.startGame()
        self.curr_step = 0
        self.cubes = 0
        self.img_controller.clearLumArr()
        #self.controller.avg = 0
        self.teams = None
        while self.teams not in [1,2,3,4,5,6,7,8,9,10]:
            teams, self.image_state, _, _ = self.img_controller.getScreenAsArray()
            self.teams = self.img_controller.teamsRemaining(teams)
            #print(self.teams)
        
        # Reseting if the match is a duo showdown ( only really applicable in practice match)
        
        if self.teams != 10:
            print("Duo game, resetting")
            controller.exitGameLose()
            time.sleep(2)
            self.reset(controller)
        '''
        self.current_state = {
            'w':0,
            'a':0,
            's':0,
            'd':0,
            'e':0,
            'f':0
        }
        '''
        self.action_state = [0,0,0,0]
        self.alive = True
        self.time = time.time()
        return self.image_state
