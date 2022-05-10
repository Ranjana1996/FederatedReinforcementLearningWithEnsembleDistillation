
#kl_div


import time


import os
import datetime

class ARGS():
    def __init__(self):
        self.env_name = 'PongDeterministic-v4'
        self.render = False
        self.episodes = 1500
        self.batch_size = 32
        self.epsilon_start = 1.0
        self.epsilon_final=0.01
        self.seed = 1773
        self.N = 40
        self.use_gpu =  True #torch.cuda.is_available()
        self.ensemble= True


        self.number_of_samples = 7
        self.number_of_prototypes = 2
        self.fraction = 0.5 ##--fraction * num Samples = num clients chosen
        self.local_steps = 40
        self.rounds = 6
        self.tennis = True
        self.max_buffer_size = 5000
        self.ensemble_epsilon = 0.5
        self.ensemble_batch_size = 32
        self.init_buffer_size = 200
        self.data_size=16
        self.max_epsilon_steps = self.local_steps*200
        self.sync_target_net_freq = self.max_epsilon_steps // 10
        self.eval_iter = 1
        self.folder_name = f"runs/ensemble/" + time.asctime(time.gmtime()).replace(" ", "_").replace(":", "_")

        self.replay_buffer_fill_len = 100


