import numpy as np  # Import the NumPy library
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.optim as optim
import random

# Define the DQN class
class DQN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, num_actions):
        super(DQN, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(input_size, embedding_dim)

        # GRU layer
        self.gru = nn.GRU(embedding_dim, hidden_size)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        # Input 'x' is expected to be a sequence of cluster vectors
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = self.fc(x[-1, :, :])  # Use the final GRU output as the Q-values
        return x

# Define the DQN agent
class DQNAgent:
    def __init__(self, input_size, embedding_dim, hidden_size, num_actions, epsilon):
        self.dqn = DQN(input_size, embedding_dim, hidden_size, num_actions)
        self.epsilon = epsilon

    def select_action(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(0, self.dqn.fc.out_features, (1,))
        else:
            q_values = self.dqn(state)
            return q_values.max(1)[1]

# Example usage
input_size = 10  # Number of cluster vectors
embedding_dim = 64
hidden_size = 128
num_actions = 4  # Number of possible actions
epsilon = 0.1  # Exploration rate

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
  
    n_components = 10 #tempoary
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
    for cluster_id in range(c):
        cluster_center = kmeans.cluster_centers_[cluster_id]
        cluster_items = [item_feature_vectors[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        clusters.append([cluster_center] + cluster_items)
    item_feature_vectors = np.array(item_feature_vectors)
    c1_items=[c1_item_feature_vectors[i] for i in range(len(c1_item_feature_vectors))]
    c2_items=[c2_item_feature_vectors[i] for i in range(len(c2_item_feature_vectors))]
    clusters.append(c1_items)
    clusters.append(c2_items)
   
    iterate_rounds = 100
    target_update_frequency = 10
    done = 0
    dqn_agent = DQNAgent(input_size, embedding_dim, hidden_size, num_actions, epsilon)
    state = [0]
    state = torch.tensor(state, dtype=torch.float32, device='cuda')
    actions = []
    for i in range(iterate_rounds):
        action = dqn_agent.select_action(state)
        actions.append(action.item())
        print("Selected action:", action.item())
        
        action_id=action.item()
        random_index = random.randint(0, len(action_id) - 1)
        sampled_item = clusters[action_id][random_index]
        
        upweight_delta=0.1
        reward=Outcome_Estimater(sampled_item,upweight_delta,mid_result_recommender,item_amounts)
        
        next_state=state
        next_state.append(action)
        
        target_dqn_agent = DQNAgent(input_size, embedding_dim, hidden_size, num_actions, epsilon)
        optimizer = optim.Adam(dqn_agent.dqn.parameters(), lr=0.001)
        q_values = dqn_agent.dqn(state)
        q_value = q_values.gather(1, action.unsqueeze(1))
        with torch.no_grad():
            next_q_values_target = target_dqn_agent.dqn(next_state)
            max_next_q_value, _ = next_q_values_target.max(1, keepdim=True)
            target_q_value = reward + (1 - done) * max_next_q_value
        loss = torch.nn.functional.mse_loss(q_value, target_q_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % target_update_frequency == 0:
            target_dqn_agent.dqn.load_state_dict(dqn_agent.dqn.state_dict())

    return actions, clusters