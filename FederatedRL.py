#kl_div

import gym
import gym.spaces
import random
import torch
import torch.nn.functional as F
from ale_py import ALEInterface
from torch import nn
from  collections import  defaultdict

from ARGS import *
from Agent import *
from plotutils import *
from ReplayMemory import *

ale = ALEInterface()

devi = torch.device("cuda:0")
print("current device",torch.cuda.current_device())

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
args = ARGS()
set_seed(args.seed)

device = torch.device("cuda:0")
dtype = torch.float

os.makedirs('runs/', exist_ok=True)
os.makedirs(f'runs/ensemble/', exist_ok=True)
os.makedirs(args.folder_name, exist_ok=True)

# save the hyperparameters in a file
f = open(f'{args.folder_name}/args.txt', 'w')
for i in args.__dict__:
    f.write(f'{i}: {args.__dict__[i]}\n')
f.close()


# utils for atari


def make_env(env_name):
    env = gym.make(env_name)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env, env_name)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)


class FederatedLearning:
    
    def __init__(self, args):
        self.args = args
        self.create_prototypes()
        self.create_clients()
        self.logs = {}
        self.eval_rewards = defaultdict(list)
        self.eval_rewards_before_div = defaultdict(list)
        self.train_rewards = defaultdict(list)
        self.client_train_rewards = defaultdict(list)


        self.buffer = ReplayMemory(self.args.max_buffer_size)

    def create_prototypes(self):
        self.main_agents = {}
        self.main_agent_names = []
        self.updated_agents= {}
        self.client_mappings = dict()
        for i in range(self.args.number_of_prototypes):
            self.main_agent_names.append(f"prototype_{i}")
            self.main_agents[f"prototype_{i}"] = Agent(args, f"prototype_{i}",i)
            self.updated_agents[f"prototype_{i}"] = 0
            self.client_mappings[i] = []


    def create_clients(self):
        self.clients = {}
        self.client_names = []
        self.updated_clients = {}
        self.prototype_mapping ={}
        for i in range(self.args.number_of_samples):
            curr_prototype = i%args.number_of_prototypes
            self.client_names.append(f"client_{i}")
            self.clients[f"client_{i}"] = Agent(args, i, curr_prototype)
            self.client_mappings[curr_prototype].append(f"client_{i}")
            self.updated_clients[f"client_{i}"] = 0
            self.prototype_mapping[f"client_{i}"] = curr_prototype

    def update_clients(self):
        with torch.no_grad():
            def update(client_layer, main_layer):
                client_layer.weight.data = main_layer.weight.data.clone()
                client_layer.bias.data = main_layer.bias.data.clone()
                
            for i in range(self.args.number_of_samples):
                prototype = self.prototype_mapping[self.client_names[i]]
                update(self.clients[self.client_names[i]].dqn.conv1, self.main_agents[self.main_agent_names[prototype]].dqn.conv1)
                update(self.clients[self.client_names[i]].dqn.conv2, self.main_agents[self.main_agent_names[prototype]].dqn.conv2)
                update(self.clients[self.client_names[i]].dqn.conv3, self.main_agents[self.main_agent_names[prototype]].dqn.conv3)
                update(self.clients[self.client_names[i]].dqn.fc1, self.main_agents[self.main_agent_names[prototype]].dqn.fc1)
                update(self.clients[self.client_names[i]].dqn.fc2, self.main_agents[self.main_agent_names[prototype]].dqn.fc2)

                del self.clients[self.client_names[i]].buffer
                self.clients[self.client_names[i]].buffer = ReplayMemory(1000000 // 4)

                del self.clients[self.client_names[i]].target_dqn

                if prototype == 0:
                    self.clients[self.client_names[i]].target_dqn = DQN0(self.main_agents[self.main_agent_names[prototype]].num_actions)
                else:
                    self.clients[self.client_names[i]].target_dqn = DQN1(self.main_agents[self.main_agent_names[prototype]].num_actions)
                self.clients[self.client_names[i]].target_dqn.load_state_dict(self.clients[self.client_names[i]].dqn.state_dict()) 
                
                if self.args.use_gpu:
                    self.clients[self.client_names[i]].target_dqn.cuda()


    def create_fused_model(self, clients, prototype):
        conv1_mean_weight = torch.zeros(size=self.main_agents[self.main_agent_names[prototype]].dqn.conv1.weight.shape).to(device)
        conv1_mean_bias = torch.zeros(size=self.main_agents[self.main_agent_names[prototype]].dqn.conv1.bias.shape).to(device)

        conv2_mean_weight = torch.zeros(size=self.main_agents[self.main_agent_names[prototype]].dqn.conv2.weight.shape).to(device)
        conv2_mean_bias = torch.zeros(size=self.main_agents[self.main_agent_names[prototype]].dqn.conv2.bias.shape).to(device)

        conv3_mean_weight = torch.zeros(size=self.main_agents[self.main_agent_names[prototype]].dqn.conv3.weight.shape).to(device)
        conv3_mean_bias = torch.zeros(size=self.main_agents[self.main_agent_names[prototype]].dqn.conv3.bias.shape).to(device)

        linear1_mean_weight = torch.zeros(size=self.main_agents[self.main_agent_names[prototype]].dqn.fc1.weight.shape).to(device)
        linear1_mean_bias = torch.zeros(size=self.main_agents[self.main_agent_names[prototype]].dqn.fc1.bias.shape).to(device)

        linear2_mean_weight = torch.zeros(size=self.main_agents[self.main_agent_names[prototype]].dqn.fc2.weight.shape).to(device)
        linear2_mean_bias = torch.zeros(size=self.main_agents[self.main_agent_names[prototype]].dqn.fc2.bias.shape).to(device)

        num_clients = len(clients)
        with torch.no_grad():
            for i in clients:
                conv1_mean_weight += self.clients[self.client_names[i]].dqn.conv1.weight.clone()
                conv1_mean_bias += self.clients[self.client_names[i]].dqn.conv1.bias.clone()

                conv2_mean_weight += self.clients[self.client_names[i]].dqn.conv2.weight.clone()
                conv2_mean_bias += self.clients[self.client_names[i]].dqn.conv2.bias.clone()

                conv3_mean_weight += self.clients[self.client_names[i]].dqn.conv3.weight.clone()
                conv3_mean_bias += self.clients[self.client_names[i]].dqn.conv3.bias.clone()

                linear1_mean_weight += self.clients[self.client_names[i]].dqn.fc1.weight.clone()
                linear1_mean_bias += self.clients[self.client_names[i]].dqn.fc1.bias.clone()

                linear2_mean_weight += self.clients[self.client_names[i]].dqn.fc2.weight.clone()
                linear2_mean_bias += self.clients[self.client_names[i]].dqn.fc2.bias.clone()


            conv1_mean_weight = conv1_mean_weight / num_clients
            conv1_mean_bias = conv1_mean_bias / num_clients

            conv2_mean_weight = conv2_mean_weight / num_clients
            conv2_mean_bias = conv2_mean_bias / num_clients

            conv3_mean_weight = conv3_mean_weight / num_clients
            conv3_mean_bias = conv3_mean_bias / num_clients

            linear1_mean_weight = linear1_mean_weight / num_clients
            linear1_mean_bias = linear1_mean_bias / num_clients

            linear2_mean_weight = linear2_mean_weight / num_clients
            linear2_mean_bias = linear2_mean_bias / num_clients


            with torch.no_grad():
                def update(main_layer, averaged_layer_weight, averaged_layer_bias):
                    main_layer.weight.data = averaged_layer_weight.data.clone()
                    main_layer.bias.data = averaged_layer_bias.data.clone()

                update(self.main_agents[self.main_agent_names[prototype]].dqn.conv1, conv1_mean_weight, conv1_mean_bias)
                update(self.main_agents[self.main_agent_names[prototype]].dqn.conv2, conv2_mean_weight, conv2_mean_bias)
                update(self.main_agents[self.main_agent_names[prototype]].dqn.conv3, conv3_mean_weight, conv3_mean_bias)

                update(self.main_agents[self.main_agent_names[prototype]].dqn.fc1, linear1_mean_weight, linear1_mean_bias)
                update(self.main_agents[self.main_agent_names[prototype]].dqn.fc2, linear2_mean_weight, linear2_mean_bias)

            if self.args.use_gpu:
                try:
                    self.main_agents[self.main_agent_names[prototype]].dqn.cuda()
                except:
                    pass


    def fill_data(self, fill_length, prototype):
        state = self.main_agents[self.main_agent_names[prototype]].env.reset()
        for i in range(fill_length):
            action = self.main_agents[self.main_agent_names[prototype]].select_action(state, self.args.ensemble_epsilon)
            next_state, reward, done, _ = self.main_agents[self.main_agent_names[prototype]].env.step(action)
            self.buffer.add(state, action, reward, done, next_state)
            state = next_state
            if done:
                self.env.reset()


    def update_prototypes(self, round_no, idx_users):
        prototypes = {}
        for i in idx_users:
            curr_prototype = self.prototype_mapping[self.client_names[i]]
            if curr_prototype not in prototypes:
                prototypes[curr_prototype] = [i]
            else:
                prototypes[curr_prototype].append(i)

        for prototype, clients in prototypes.items():
            # averaging -- if prototype is only one fused model is just fed avg
            self.create_fused_model(clients, prototype)

            ##fill data in buffer
            if self.args.ensemble==True:
                self.fill_data(self.args.ensemble_batch_size, prototype)
                num_clients = len(clients)

                self.main_agents[self.main_agent_names[prototype]].optim.zero_grad()
                for i in range(self.args.N):
                    s_batch, _, _, _, _ = self.buffer.sample(self.args.ensemble_batch_size)
                    states = self.main_agents[self.main_agent_names[prototype]].to_var(torch.from_numpy(s_batch).float())
                    client_logits = 0
                    for client in clients:
                        actions = self.clients[self.client_names[client]].dqn(states)
                        client_logits += actions
                    client_logits /= num_clients

                    server_logits = self.main_agents[self.main_agent_names[prototype]].dqn(states)

                    kl_loss = nn.KLDivLoss(reduction="batchmean")
                    loss = kl_loss(F.log_softmax(client_logits,dim=1), F.softmax(server_logits,dim=1))
                    loss.backward()
                self.main_agents[self.main_agent_names[prototype]].optim.step()

        
    def step(self, idx_users, round_no):

        self.update_clients()
        
        for user in idx_users:
            print(f"Client {user}")
            
            rewards = self.clients[self.client_names[user]].train(
                replay_buffer_fill_len = self.args.replay_buffer_fill_len, 
                batch_size = self.args.batch_size, 
                episodes = self.args.local_steps,
                max_epsilon_steps = self.args.max_epsilon_steps,
                epsilon_start = self.args.epsilon_start - 0.03*(round_no - 1),
                epsilon_final = self.args.epsilon_final,
                sync_target_net_freq = self.args.sync_target_net_freq)
            
            print(f'LOCAL TRAIN: Avg Reward: {np.array(rewards).mean():.5f}')
            self.logs[f"{round_no}"]["train"]["rewards"].append(rewards)
            self.train_rewards[self.prototype_mapping[f"client_{user}"]].extend(rewards)
            self.client_train_rewards[user].extend(rewards)

        prev = self.args.ensemble
        self.args.ensemble = False
        self.update_prototypes(round_no, idx_users)

        # eval Rewards before kl divergence
        for i in range(self.args.number_of_prototypes):
            rewards= self.main_agents[self.main_agent_names[i]].play(self.args.eval_iter)
            self.eval_rewards_before_div[i].extend(rewards)

        self.args.ensemble = prev
        self.update_prototypes(round_no, idx_users)

        max_reward = [float('-inf') for i in range(self.args.eval_iter)]
        for i in range(self.args.number_of_prototypes):
            curr_reward = self.main_agents[self.main_agent_names[i]].play(self.args.eval_iter)
            self.eval_rewards[i].extend(curr_reward)
            if np.array(max_reward).mean() < np.array(curr_reward).mean():
                max_reward = curr_reward
        self.logs[f"{round_no}"]["eval"]["rewards"] = np.copy(max_reward)
        
        
        
    def run(self):
        
        m = max(int(self.args.fraction * self.args.number_of_samples), 1) 
        for round_no in range(self.args.rounds):
            self.logs[f"{round_no + 1}"] = {"train": {
                                            "rewards": [],
                                        },
                                        "eval": {
                                            "rewards": None
                                        }
                                       }
            idxs_users = np.random.choice(range(self.args.number_of_samples), m, replace=False)  #Get clients

            for user in idxs_users:
                self.updated_clients[f"client_{user}"] = round_no + 1
                
            self.step(idxs_users, round_no+1)
            print(f'{round_no + 1}/{self.args.rounds}')
            print(f'TRAIN: Avg Reward: {np.array(self.logs[f"{round_no + 1}"]["train"]["rewards"]).mean():.5f}')
            print(f'EVAL: Best Reward: {np.array(self.logs[f"{round_no + 1}"]["eval"]["rewards"]).mean():.5f}')
            if round_no % 3 == 0:
                for i in range(self.args.number_of_prototypes):
                    torch.save(self.main_agents[self.main_agent_names[i]].dqn.state_dict(), f'{args.folder_name}/model_prototype_{i}.pt')

            # if round_no % 4 ==0:
            #     print("Train Rewards Prototype",self.train_rewards)
            #     print("Train Rewards Client", self.client_train_rewards)
            #     print("Eval Rewards Prototype" , self.eval_rewards)
            #     print("Eval Rewards Before Divergence",self.eval_rewards)

        for i in range(self.args.number_of_prototypes):
            torch.save(self.main_agents[self.main_agent_names[i]].dqn.state_dict(), f'{args.folder_name}/model_prototype_{i}.pt')

        for key in self.train_rewards.keys():
            plotgraph(self.train_rewards[key],get_moving_average(self.train_rewards[key],50),'Episodes','Rewards',f'Prototype_{key}',f'Prototype_{key}',legend=['Rewards','Average Rewards'])
        for key in self.eval_rewards.keys():
            plotgraph(self.eval_rewards[key],get_moving_average(self.eval_rewards[key],50),'Episodes','Rewards',f'Prototype_Eval_{key}',f'Prototype_Eval_{key}',legend=['Rewards','Average Rewards'])

        for key in self.eval_rewards_before_div.keys():
            plotgraph(self.eval_rewards_before_div[key],get_moving_average(self.eval_rewards_before_div[key],50),'Episodes','Rewards',f'Prototype_Eval_Before_AvgLogits{key}',f'Prototype_Eval_Before_AvgLogits{key}',legend=['Rewards','Average Rewards'])

        for key in self.client_train_rewards.keys():
            plotgraph(self.client_train_rewards[key],get_moving_average(self.client_train_rewards[key],50),'Episodes','Rewards',f'Client_{key}',f'Client_{key}',legend=['Rewards','Average Rewards'])

fl = FederatedLearning(args)
fl.run()