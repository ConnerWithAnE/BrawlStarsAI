
class Environment():
    def __init__(self, image_controller, transform=None):
        self.actions = ['w','a','s','d']
        self.controller = image_controller
        self.teams, self.image_state, _ = self.controller.getScreenAsArray()
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
        # Getting the wasd key up or down
        for i in range(4):
            if self.action_state[i] != action[0][i]:
                if self.action_state[i] == 1:
                    print(f"up {self.actions[i]}")
                    ahk.key_up(self.actions[i])
                else:
                    print(f"down {self.actions[i]}")
                    ahk.key_down(self.actions[i])

        # Getting the e and f key pressed
        if action[0][4] == 1:
            ahk.key_press('e')
        if action[0][5] == 1:
            ahk.key_press('f')

        self.action_state = action[0]
        n_teams, n_image_state, n_alive = self.controller.getScreenAsArray()

        if n_teams < self.teams and n_alive == True:
            reward = 1.0
        else:
            reward = 0
        self.teams = n_teams
        self.image_state = n_image_state
        self.alive = n_alive

        return self.image_state, reward, self.alive


    def reset(self, controller):
        controller.startGame()
        self.teams = None
        while self.teams not in ['1','2','3','4','5','6','7','8','9','10']:
            self.teams, self.image_state, _ = self.controller.getScreenAsArray()
            print('hit')
            print(self.teams)
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
        return self.image_state
