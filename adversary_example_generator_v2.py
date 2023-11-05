import numpy as np  # Import the NumPy library
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy

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
        self.num_actions=num_actions
        self.replay_memory = []

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            self.dqn.eval()
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.long).to(self.device)
                q_values = self.dqn(state)
            action = q_values.argmax().item()
            return action

    def update(self):
        if len(self.replay_memory) < self.batch_size:
            return
        self.dqn.train()
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

def generate_fake_data(fake_user_amount,adversary_pattern,item_clusters,n_items):
    fake_users=np.zeros((fake_user_amount,n_items))
    for i in range(fake_user_amount):
        for j in range(len(adversary_pattern)):
            group_id=adversary_pattern[j]
            random_index = random.randint(0, len(item_clusters[group_id]) - 1)
            sampled_item = item_clusters[group_id][random_index]
            fake_users[i][sampled_item]=1
    fake_users_record=csr_matrix(fake_users)
    return fake_users_record

def adversary_pattern_generator(model,train_data,test_data,target_item,target_user,args):
    training_dataset=train_data
    testing_dataset=test_data
    user_amounts=training_dataset.shape[0]-args.n_fake_users
    item_amounts=training_dataset.shape[1]
    c1 = np.array(target_item)
    row_indices, col_indices = training_dataset.nonzero()
    item_ids = np.unique(col_indices)    
    c2 = np.array([item_amounts-1])
    I_set = set(item_ids)
    C1_set = set(c1)
    C2_set = set(c2)
    result_set = I_set.difference(C1_set.union(C2_set))
    remain_items = list(result_set)
    remain_items = np.array(remain_items)
  
    n_components = 128 #tempoary
    nmf_model = NMF(n_components=n_components)
    mat_W = nmf_model.fit_transform(training_dataset)  # User features (W matrix)
    mat_H = nmf_model.components_
    user_feature_vectors = mat_W
    mat_H=mat_H.transpose()
    item_feature_vectors = mat_H[remain_items,:]
    c1_item_feature_vectors=mat_H[c1,:]
    c2_item_feature_vectors=mat_H[c2,:]  
    
    c = args.clusters
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
    state_dim = 10  # Example state dimension
    embedding_dim = 32
    hidden_dim = 64
    #learning_rate = 0.001
    epsilon = 0.3
    gamma = 0.9
    #batch_size = 32
    fake_items = 10

    agent = DQNAgent(num_actions=num_actions, embedding_dim=embedding_dim, hidden_dim=hidden_dim, learning_rate=args.surrogate["lr"], epsilon=epsilon, gamma=gamma, batch_size=args.surrogate["batch_size"])

    # Training process
    num_iterations = args.adv_epochs  # Number of training iterations
    #current_state = []  # Define an initial state as an empty list
    rewards = []
    eval_window = 500
    reward_window = 100
    
    recommendation_of_normal_users=model.recommend(training_dataset[:user_amounts],item_amounts)
    training_dataset_copy = copy.deepcopy(training_dataset)           
    mid_result_recommender = copy.deepcopy(model)
    evaluation = mid_result_recommender.validate(train_data=training_dataset_copy, test_data=testing_dataset, target_items=target_item)
    del mid_result_recommender
    hit_ratio=evaluation.popitem()
    cur_perf = hit_ratio[1]
    print(f"iter 0: Evaluation: {np.mean(cur_perf):.4f}")
    for iteration in range(num_iterations):
        actions=[]
        current_state=np.array([c+1])
        current_state=current_state.reshape(1,current_state.size)
        # Phase 1: Memory Generation Stage
        for _ in range(fake_items):
            group_action = agent.select_action(current_state, epsilon)
            actions.append(group_action)
        
            sampled_item = []
            for i in range(len(actions)):
                group_id=actions[i]
                random_index = random.randint(0, len(clusters_itemid[group_id]) - 1)
                sampled_item.append(clusters_itemid[group_id][random_index])
            
            upweight_delta = 0.1
            mid_result_recommender = copy.deepcopy(model)

            reward = Outcome_Estimater(sampled_item, upweight_delta, mid_result_recommender, recommendation_of_normal_users,item_amounts, user_amounts,target_user,target_item)
            del mid_result_recommender
            torch.cuda.empty_cache()
            action_reward = reward[sampled_item[len(actions)-1]]
            reward_score = torch.norm(action_reward).detach().item()

            rewards.append(reward_score)
            
            # Simulate receiving a reward and transitioning to the next state
            next_state = current_state # Replace with the actual next state
            group_action_array = np.array([group_action])
            group_action_array = group_action_array.reshape(1,1)
            next_state = np.concatenate((next_state,group_action_array),axis=1)
            agent.replay_memory.append((current_state, group_action, reward_score, next_state))
            current_state = next_state

        # Phase 2: Parameter Update Stage
        agent.update()
        torch.cuda.empty_cache()

        if epsilon > 0.1:
            epsilon = epsilon - 0.0001

        if iteration % reward_window == reward_window-1:
            rewards = np.array(rewards)
            print(f"iter: {iteration+1}, Reward: {np.mean(rewards):.2e}")
            rewards = []
        
        if iteration % eval_window == eval_window-1:
            fake_data=generate_fake_data(fake_user_amount=args.n_fake_users, adversary_pattern=actions, item_clusters=clusters_itemid, n_items=item_amounts)
            training_dataset_copy = copy.deepcopy(training_dataset)
            for i in range(args.n_fake_users):
                training_dataset_copy[user_amounts+i]=fake_data[i]
            
            mid_result_recommender = copy.deepcopy(model)
            evaluation = mid_result_recommender.validate(train_data=training_dataset_copy, test_data=testing_dataset, target_items=target_item)
            del mid_result_recommender
            hit_ratio=evaluation.popitem()
            cur_perf = hit_ratio[1]
            print(f"iter: {iteration+1}, Evaluation: {np.mean(cur_perf):.4f}")
            
    return actions, clusters_itemid