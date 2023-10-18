import numpy as np  # Import the NumPy library
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim
import random

from output_estimator import Outcome_Estimater

# Define the DQN class
class DQN(nn.Module):
    def __init__(self, num_actions, embedding_dim, hidden_dim):
        super(DQN, self).__init__()
        self.embedding = nn.Embedding(num_actions, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, state):
        embedded_state = self.embedding(state)
        gru_out, _ = self.gru(embedded_state)
        state_representation = gru_out[:, -1, :]
        action_values = self.fc(state_representation)
        return action_values

class DQNAgent:
    def __init__(self, num_actions, embedding_dim, hidden_dim, learning_rate, epsilon, gamma, batch_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN(num_actions, embedding_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_memory = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, num_actions - 1)
        else:
            state = torch.tensor(state, dtype=torch.long).to(self.device)
            q_values = self.dqn(state)
            action = q_values.argmax().item()
            return action

    def update(self):
        if len(self.replay_memory) < self.batch_size:
            return

        mini_batch = random.sample(self.replay_memory, self.batch_size)
        for state, action, reward, next_state in mini_batch:
            self.optimizer.zero_grad()
            state = torch.tensor(state, dtype=torch.long).to(self.device)
            action = torch.tensor(action, dtype=torch.long).to(self.device)
            reward = torch.tensor(reward, dtype=torch.float).to(self.device)
            next_state = torch.tensor(next_state, dtype=torch.long).to(self.device)

            q_values = self.dqn(state)
            next_q_values = self.dqn(next_state)

            target_q_values = q_values.clone()
            target_q_values[0, action] = reward + self.gamma * torch.max(next_q_values)

            loss = torch.nn.MSELoss()(q_values, target_q_values)
            loss.backward()
            self.optimizer.step()
            
state_dim = 10  # Example state dimension
num_actions = 5  # Example number of actions
embedding_dim = 32
hidden_dim = 64
learning_rate = 0.001
epsilon = 0.1
gamma = 0.9
batch_size = 32

def adversary_pattern_generator(model,train_data,target_item,target_user):
    mid_result_recommender=model
    dataset=train_data
    item_amounts=dataset.shape[1]
    c1 = np.array(target_item)
    row_indices, col_indices = dataset.nonzero()
    item_ids = np.unique(col_indices)    
    c2 = dataset[target_user, :].nonzero()[1]
    I_set = set(item_ids)
    C1_set = set(c1)
    C2_set = set(c2)
    result_set = I_set.difference(C1_set.union(C2_set))
    remain_items = list(result_set)
    remain_items = np.array(remain_items)
  
    n_components = 128 #tempoary
    nmf_model = NMF(n_components=n_components)
    mat_W = nmf_model.fit_transform(dataset)  # User features (W matrix)
    mat_H = nmf_model.components_
    user_feature_vectors = mat_W
    mat_H=mat_H.transpose()
    item_feature_vectors = mat_H[remain_items,:]
    c1_item_feature_vectors=mat_H[c1,:]
    c2_item_feature_vectors=mat_H[c2,:]  
    
    c = 3
    kmeans = KMeans(n_clusters=c, random_state=0)
    kmeans.fit(item_feature_vectors)
    cluster_labels = kmeans.labels_
    clusters = []
    clusters_itemid = {i: [] for i in range(c+2)}
    for i, label in enumerate(cluster_labels):
        clusters_itemid[label].append(i)
    for i in range(c1.size):
        clusters_itemid[c].append(c1[i])
    for i in range(c2.size):
        clusters_itemid[c+1].append(c2[i])
    for cluster_id in range(c):
        cluster_center = kmeans.cluster_centers_[cluster_id]
        cluster_items = [item_feature_vectors[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        clusters.append([cluster_center] + cluster_items)
    item_feature_vectors = np.array(item_feature_vectors)
    c1_items=[c1_item_feature_vectors[i] for i in range(len(c1_item_feature_vectors))]
    c2_items=[c2_item_feature_vectors[i] for i in range(len(c2_item_feature_vectors))]
    clusters.append(c1_items)
    clusters.append(c2_items)
   
    num_actions=c+2

    agent = DQNAgent(num_actions, embedding_dim, hidden_dim, learning_rate, epsilon, gamma, batch_size)

    # Training process
    num_iterations = 100  # Number of training iterations
    #current_state = []  # Define an initial state as an empty list
    current_state=np.zeros(1)
    current_state=current_state.reshape(1,current_state.size)
    actions=[]
    for iteration in range(num_iterations):
        # Phase 1: Memory Generation Stage
        group_action = agent.select_action(current_state)
        actions.append(group_action)
        action_id=group_action
        
        random_index = random.randint(0, len(clusters[action_id]) - 1)
        sampled_item = clusters[action_id][random_index]
        sampled_item=[]
        sampled_item_vectors=[]
        for i in range(len(actions)):
            group_id=actions[i]
            random_index = random.randint(0, len(clusters_itemid[group_id]) - 1)
            sampled_item.append(clusters_itemid[group_id][random_index])
            sampled_item_vectors.append(clusters[group_id][random_index])
        upweight_delta=0.1
        reward=Outcome_Estimater(sampled_item,upweight_delta,mid_result_recommender,item_amounts,target_user)
        reward_score=torch.norm(reward[sampled_item[len(actions)-1]])
        # Simulate receiving a reward and transitioning to the next state
        next_state = current_state # Replace with the actual next state
        group_action_array=np.array([group_action])
        group_action_array=group_action_array.reshape(1,1)
        next_state=np.concatenate((next_state,group_action_array),axis=1)
        agent.replay_memory.append((current_state, group_action, reward_score, next_state))
        current_state = next_state

        # Phase 2: Parameter Update Stage
        agent.update()
            
    return actions, clusters_itemid