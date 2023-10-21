Usage of the current version:

Generate fake data:
```shell script
srun -A r00066 python3 generate_attack.py -dataset Dataset_name -att_type Attack_method -pop populatiry -ratio fake_user_ratio -unroll unroll_epochs -tag tag_name
```
Parameters in the shell command:
- Dataset_name: The dataset used for attack (books/arts/ml-1m/steam)
- Attack_method: The attack model, (WRMF_ALS/WRMF_SGD/RandFilter/DQN), DQN is the new attack method.
- Popularity: Popularity of target items, (head/upper/lower/tail), which upper means upper_torso, lower meant lower_torso
- Fake_user_ratio: Ratio of fake users in percentage, 1/0.5/0.1/0.05
- Unroll_epochs: Unroll epochs for adversary training in WRMF method, 2 for books/arts, 5 for steam and 10 for ml-1m, large than the amount will cause CUDA memory overflow.
- Tag_name: The name for evaluate attack program to find the fake user data and model, you can name it arbitrary

For new attack method, the parameters included in the file generate_attack_args.py.
- adv_epochs is for the iteration of DQN network learning. 
- The structure of DQN network is embedding_dim, hidden_dim, epsilon and gamma.
- cluster is for the number of clusters for the items used in k-means cluster for group action. 
- batch_size and learning rate(lr) is in the surrogate model parameter of sur_wmf_dqn.

To evaluate, using the following command:
```shell script
srun -A r00066 python3 evaluate_attack.py -dataset ml-1m -att_type RandFilter -pop head -tag mrfh5
```
Where the parameter is same with generate_attack.py, the -tag parameter need to be same when you running generate_attack.py.  

