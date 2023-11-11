
class Environment(Dataset):
    def __init__(self, transform=None):
        self.controller = ImageController()
        self.teams, self.image_state = self.controller.getScreenAsArray()
        #self.transform = transform
        self.action_state = {
            'w':0,
            'a':0,
            's':0,
            'd':0,
            'e':0,
            'f':0
        }
        self.alive = True

    def step(self, action, ahk):
        #self.current_state = self.image_stream
        for key in set(self.action_state.keys()).intersection(action.keys()):
            if self.action_state[key] != action[key]:
                if self.action_state[key] == 1:
                    ahk.key_up(key)
                else:
                    ahk.key_down(key)
        self.action_state = action
        n_teams, n_image_state, n_alive = self.controller.getScreenAsArray()

        if n_teams < self.teams and n_alive == True:
            reward = 1.0
        else:
            reward = 0
        self.teams = n_teams
        self.image_state = n_image_state
        self.alive = n_alive

        return self.image_state, reward, self.alive


    def reset(self):
        self.teams, self.image_state = self.controller.getScreenAsArray()
        self.current_state = {
            'w':0,
            'a':0,
            's':0,
            'd':0,
            'e':0,
            'f':0
        }
        self.alive = True
        return self.image_state
