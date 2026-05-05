import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import get_human_preferences as get_human_preferences



def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net


class RewardModel:
    def __init__(self, ds, da, device,
                 ensemble_size=3, lr=3e-4, mb_size = 128, size_segment=1, 
                 train_batch_size=128,
                 env_maker=None, max_size=100, activation='tanh', capacity=5e5,  
                 large_batch=1, label_margin=0.0, 
                 teacher_beta=-1, teacher_gamma=1, 
                 teacher_eps_mistake=0, 
                 teacher_eps_skip=0, 
                 teacher_eps_equal=0,
                 mu=1,
                 weight_factor=1.0,
                 adv_mu=2,
                 path=None,
                 data_aug_ratio=1, 
                  pid="default_pid",
                  user_study=False, 
                  frame_size=[240,380],
                  max_feedback=1000, pref_fig=None):
        # train data is trajectories, must process to sa and s..
        self.frame_dim = frame_size
        self.frame_size =  frame_size[0]*frame_size[1]*3
        self.ds = ds
        self.da = da
        self.device = device
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment
        self.path = path
        self.data_aug_ratio = data_aug_ratio
        self.count = 0
        self.pref_fig = pref_fig
        
        self.capacity = int(capacity)
        self.train_batch_size = train_batch_size
        self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_mask = np.ones((self.capacity, 1), dtype=np.float32)
        self.frame_buffer = np.empty((self.max_size, 1000, self.frame_size), dtype=np.uint8)
        self.frame_ep_index = 0  # which episode slot we're writing into
        self.frame_step_index = 0  # which step within current episode
        self.frame_buffer_index = 0
        self.buffer_index = 0
        self.buffer_full = False
        self.construct_ensemble()
        self.inputs = []
        self.targets = []
        self.raw_actions = []
        self.img_inputs = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.CEloss = nn.CrossEntropyLoss()
        self.CEloss_ = nn.CrossEntropyLoss(reduction='none')
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.large_batch = large_batch
        self.mu = mu
        self.weight_factor = weight_factor
        self.adv_mu = adv_mu
        self.obs_l = 0
        self.action_l = 0
        
        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        self.label_margin = label_margin
        self.label_target = 1 - 2*self.label_margin
        self.number_feedback_sessions = int(max_feedback/mb_size)
        self.current_feedback_session = 1
        self.pid = pid                # <--- SAVE PID HERE
        self.user_study = user_study
    
        self.human_study_data = {
            'user_id': self.pid,      # <--- ADD TO DICT
            'sampled_trajectory1': [],
            'sampled_trajectory2': [], 
            'frames_sampled_trajectory1': [],
            'frames_sampled_trajectory2': [],
            'human_labels': [], 
            'ground_truth_labels': [],
            'time_taken': [],
            'rewatch_a': [],
            'rewatch_b': [],
            'rewatch_both': []
        }
    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]
    
    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size*new_frac)
    
    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)
        
    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
        
    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal
        
    def construct_ensemble(self):
        for i in range(self.de):
            model = nn.Sequential(*gen_net(in_size=self.ds+self.da, 
                                           out_size=1, H=256, n_layers=3, 
                                           activation=self.activation)).float().to(self.device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
            
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
    def add_data(self, obs, act, rew, done, frame):
        sa_t = np.concatenate([obs, act], axis=-1)
        r_t = rew
        
        flat_input = sa_t.reshape(1, self.da+self.ds)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)
        if frame is not None:
            frame = np.array(frame, dtype=np.uint8)
            flat_frame = frame.reshape(int(self.frame_size))
        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
            if frame is not None:
                self.frame_buffer[self.frame_ep_index, self.frame_step_index] = flat_frame
                self.frame_step_index += 1
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            if frame is not None:
                self.frame_buffer[self.frame_ep_index, self.frame_step_index] = flat_frame
                self.frame_step_index += 1
            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
            self.inputs.append([])
            self.targets.append([])
            if frame is not None:
                self.frame_ep_index = (self.frame_ep_index + 1) % self.max_size
                self.frame_step_index = 0
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            if frame is not None:
                self.frame_buffer[self.frame_ep_index, self.frame_step_index] = flat_frame
                self.frame_step_index += 1
                
    def add_data_batch(self, obses, rewards):
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])

    def r_hat_member(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](torch.from_numpy(x).float().to(self.device))

    def r_hat(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )
    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))
        
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len
                
            sa_t_1 = self.buffer_seg1[epoch*batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch*batch_size:last_index]
            labels = self.buffer_label[epoch*batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(self.device)
            total += labels.size(0)
            for member in range(self.de):
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)                
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)
    
    def get_queries(self, mb_size=20):
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        img_t_1, img_t_2 = None, None
        
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1
        
        # get train traj
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])
        # map inputs indices to circular frame_buffer indices
        start_ep = (self.frame_ep_index - max_len) % self.max_size
        frame_indices = [(start_ep + i) % self.max_size for i in range(max_len)]
        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_2 = train_inputs[batch_index_2] # Batch x T x dim of s&a
        r_t_2 = train_targets[batch_index_2] # Batch x T x 1
        sa_t_1 = train_inputs[batch_index_1] # Batch x T x dim of s&a
        r_t_1 = train_targets[batch_index_1] # Batch x T x 1
        # index frame_buffer directly — no reshape+take on giant arrays
        buf_idx_1 = [frame_indices[i] for i in batch_index_1]
        buf_idx_2 = [frame_indices[i] for i in batch_index_2]
        frames_t_1 = self.frame_buffer[buf_idx_1]  # (mb_size, max_episode_len, frame_size)
        frames_t_2 = self.frame_buffer[buf_idx_2]
        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1]) # (Batch x T) x dim of s&a
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1]) # (Batch x T) x 1
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1]) # (Batch x T) x dim of s&a
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1]) # (Batch x T) x 1

        # Generate time index 
        time_index = np.array([list(range(i*len_traj,
                                            i*len_traj+self.size_segment)) for i in range(mb_size)])
        time_index_2 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        
        
        time_index_1 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        
        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0) # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t_1, time_index_1, axis=0) # Batch x size_seg x 1
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0) # Batch x size_seg x dim of s&a
        r_t_2 = np.take(r_t_2, time_index_2, axis=0) # Batch x size_seg x 1
        # time_index here is within-episode so index axis=1 directly
        frames_t_1 = frames_t_1[np.arange(mb_size)[:, None], time_index_1 % len_traj]
        frames_t_2 = frames_t_2[np.arange(mb_size)[:, None], time_index_2 % len_traj]

        return sa_t_1, sa_t_2, r_t_1, r_t_2, frames_t_1, frames_t_2

    def get_queries_part(self, mb_size=20, part=10):
        self.count += 1
        len_traj, max_len = len(self.inputs[0]), part

        if len(self.inputs[-1]) < len_traj:
            train_inputs = np.array(self.inputs[-part-1:-1])
            train_targets = np.array(self.targets[-part-1:-1])
            start_ep = (self.frame_ep_index - part - 1) % self.max_size
        else:
            train_inputs = np.array(self.inputs[-part:])
            train_targets = np.array(self.targets[-part:])
            start_ep = (self.frame_ep_index - part) % self.max_size

        frame_indices = [(start_ep + i) % self.max_size for i in range(max_len)]

        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)

        sa_t_1 = train_inputs[batch_index_1]
        r_t_1 = train_targets[batch_index_1]
        sa_t_2 = train_inputs[batch_index_2]
        r_t_2 = train_targets[batch_index_2]

        buf_idx_1 = [frame_indices[i] for i in batch_index_1]
        buf_idx_2 = [frame_indices[i] for i in batch_index_2]
        frames_t_1 = self.frame_buffer[buf_idx_1]
        frames_t_2 = self.frame_buffer[buf_idx_2]

        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1])
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1])
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1])
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1])

        time_index = np.array([list(range(i*len_traj, i*len_traj+self.size_segment)) for i in range(mb_size)])
        time_index_1 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        time_index_2 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)

        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0)
        r_t_1 = np.take(r_t_1, time_index_1, axis=0)
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0)
        r_t_2 = np.take(r_t_2, time_index_2, axis=0)
        frames_t_1 = frames_t_1[np.arange(mb_size)[:, None], time_index_1 % len_traj]
        frames_t_2 = frames_t_2[np.arange(mb_size)[:, None], time_index_2 % len_traj]

        return sa_t_1, sa_t_2, r_t_1, r_t_2, frames_t_1, frames_t_2
    def put_queries(self, sa_t_1, sa_t_2, labels):
        labels = np.array(labels)
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index].reshape(-1, 1))

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:].reshape(-1, 1))


            self.buffer_index = remain
        else:
            if self.buffer_seg1.dtype == 'O':
                for i in range(sa_t_1.shape[0]):
                    self.buffer_seg1[self.buffer_index+i] = sa_t_1[i]
                for i in range(sa_t_2.shape[0]):
                    self.buffer_seg2[self.buffer_index+i] = sa_t_2[i]
            else:
                np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
                np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
                np.copyto(self.buffer_label[self.buffer_index:next_index], labels.reshape(-1, 1))
            self.buffer_index = next_index
            
    def get_scripted_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # skip the query
        if self.teacher_thres_skip > 0: 
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
        
        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size-1):
            temp_r_t_1[:,:index+1] *= self.teacher_gamma
            temp_r_t_2[:,:index+1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
            
        rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0: # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1), 
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat*self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels
        
        # making a mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]
 
        # equally preferable
        labels[margin_index] = -1

        if self.path:
            sa_t_1_path = self.path + f'{self.count}_sa_t_1.npy'
            r_t_1_path = self.path + f'{self.count}_r_t_1.npy'
            sa_t_2_path = self.path + f'{self.count}_sa_t_2.npy'
            r_t_2_path = self.path + f'{self.count}_r_t_2.npy'
            np.save(sa_t_1_path, sa_t_1)
            np.save(r_t_1_path, r_t_1)
            np.save(sa_t_2_path, sa_t_2)
            np.save(r_t_2_path, r_t_2)
        
        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels
    
    def save_human_study_data(self, sa_t_1, sa_t_2, frames_t_1, frames_t_2, human_labels, scripted_labels):
        # ── DIRECT ASSIGNMENT (NO APPEND) ──
        self.human_study_data['sampled_trajectory1'] = sa_t_1
        self.human_study_data['sampled_trajectory2'] = sa_t_2
        self.human_study_data['frames_sampled_trajectory1'] = frames_t_1
        self.human_study_data['frames_sampled_trajectory2'] = frames_t_2
        self.human_study_data['human_labels'] = human_labels
        self.human_study_data['ground_truth_labels'] = scripted_labels
        
        print(np.shape(sa_t_1), np.shape(sa_t_2), np.shape(scripted_labels), np.shape(human_labels))
        # ── 1. SAVE DATA LOCALLY (PER SESSION) ────────────────────────────────
        import pickle
        import os
        
        # Create a folder specifically for this participant
        save_dir = os.path.join(f"{os.getcwd()}/human_feedback_data", str(self.pid))
        os.makedirs(save_dir, exist_ok=True)
        
        # The current session was incremented in get_all_human_labels, so subtract 1
        session_num = self.current_feedback_session - 1 
        save_path = os.path.join(save_dir, f"session_{session_num:03d}.pkl")
        
        with open(save_path, "wb") as f:
            pickle.dump(self.human_study_data, f)
            
        print(f"Saved feedback session {session_num} to {save_path}")

        # ── 2. WIPE MEMORY ────────────────────────────────────────────────────
        # Reset the dictionary to empty lists so it doesn't blow up your RAM
        self.human_study_data = {
            'user_id': self.pid,
            'sampled_trajectory1': [],
            'sampled_trajectory2': [], 
            'frames_sampled_trajectory1': [],
            'frames_sampled_trajectory2': [],
            'human_labels': [], 
            'ground_truth_labels': [],
            'time_taken': [],
            'rewatch_a': [],
            'rewatch_b': [],
            'rewatch_both': []
        }
    def uniform_sampling(self, explore=False):
        # get queries
        if not explore:
            sa_t_1, sa_t_2, r_t_1, r_t_2, frames_t_1, frames_t_2= self.get_queries(
                mb_size=self.mb_size)
        else:
            sa_t_1, sa_t_2, r_t_1, r_t_2, frames_t_1, frames_t_2 = self.get_queries(
                mb_size=int(self.mb_size*explore))
            sa_t_1_, sa_t_2_, r_t_1_, r_t_2_, frames_t_1_, frames_t_2_ = self.get_queries_part(
                mb_size=int(self.mb_size*(1-explore)))
            sa_t_1 = np.concatenate([sa_t_1, sa_t_1_], axis=0)
            sa_t_2 = np.concatenate([sa_t_2, sa_t_2_], axis=0)
            r_t_1 = np.concatenate([r_t_1, r_t_1_], axis=0)
            r_t_2 = np.concatenate([r_t_2, r_t_2_], axis=0)
            frames_t_1 = np.concatenate([frames_t_1, frames_t_1_], axis=0)
            frames_t_2 = np.concatenate([frames_t_2, frames_t_2_], axis=0)       
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, scripted_labels = self.get_scripted_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2)
        
        if self.user_study:
            human_labels = self.get_all_human_labels(frames_t_1, frames_t_2)
            labels = human_labels
        else:
            human_labels = None
            labels = scripted_labels
        self.current_feedback_session += 1
        self.save_human_study_data(sa_t_1, sa_t_2, frames_t_1, frames_t_2, human_labels, scripted_labels)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)
    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        valid_mask = self.buffer_label[:max_len].flatten() != -2
        valid_indices = np.where(valid_mask)[0]  # actual buffer positions of valid labels
        max_len = len(valid_indices)
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = valid_indices[total_batch_index[member][epoch*self.train_batch_size:last_index]]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc, np.mean(ensemble_losses)
    
    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        valid_mask = self.buffer_label[:max_len].flatten() != -2
        valid_indices = np.where(valid_mask)[0]  # actual buffer positions of valid labels
        max_len = len(valid_indices)
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = valid_indices[total_batch_index[member][epoch*self.train_batch_size:last_index]]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), 1.0)
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc, np.mean(ensemble_losses)

    def shuffle_dataset(self, max_len):
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        return total_batch_index

    def get_cropping_mask(self, r_hat1, w):
        mask_1_, mask_2_ = [], []
        for i in range(w):
            B, S, _ = r_hat1.shape
            length = np.random.randint(S-15, S-5+1, size=B)
            start_index_1 = np.random.randint(0, S+1-length)
            start_index_2 = np.random.randint(0, S+1-length)
            mask_1 = torch.zeros((B,S,1)).to(self.device)
            mask_2 = torch.zeros((B,S,1)).to(self.device)
            for b in range(B):
                mask_1[b, start_index_1[b]:start_index_1[b]+length[b]]=1
                mask_2[b, start_index_2[b]:start_index_2[b]+length[b]]=1
            mask_1_.append(mask_1)
            mask_2_.append(mask_2)

        return torch.cat(mask_1_), torch.cat(mask_2_)

    def train_reward_iter(self, num_iters):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        valid_mask = self.buffer_label[:max_len].flatten() != -2
        valid_indices = np.where(valid_mask)[0]  # actual buffer positions of valid labels
        max_len = len(valid_indices)
        
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        total = 0
        
        start_index = 0
        for epoch in range(num_iters):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = start_index + self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs =valid_indices[total_batch_index[member][start_index:last_index]]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(self.device)
                if self.data_aug_ratio:
                    labels = labels.repeat(self.data_aug_ratio)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                if self.data_aug_ratio:
                    mask_1, mask_2 = self.get_cropping_mask(r_hat1, self.data_aug_ratio)
                    r_hat1 = r_hat1.repeat(self.data_aug_ratio,1,1)
                    r_hat2 = r_hat2.repeat(self.data_aug_ratio,1,1)
                    r_hat1 = (mask_1*r_hat1).sum(axis=1)
                    r_hat2 = (mask_2*r_hat2).sum(axis=1)
                else:
                    r_hat1 = r_hat1.sum(axis=1)
                    r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), 1.0)
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()

            start_index += self.train_batch_size
            if last_index == max_len:
                total_batch_index = self.shuffle_dataset(max_len)
                start_index = 0

            if np.mean(ensemble_acc / total) >= 0.98:
                break;
            #print(f'Iter {epoch}, Ensemble Acc: {ensemble_acc / total}, Loss: {np.mean(ensemble_losses)}')
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_acc, np.mean(ensemble_losses)

    


    
    def get_all_human_labels(self, frames_t_1, frames_t_2):
        labels, times, rw_a, rw_b, rw_both = [], [], [], [], []

        for trajectory_index in range(self.mb_size):
            segment1 = frames_t_1[trajectory_index]
            segment2 = frames_t_2[trajectory_index]
            
            # Unpack the dictionary returned by the UI
            feedback_data = self.get_single_human_label(segment1, segment2, trajectory_index)
            
            labels.append(feedback_data["value"])
            times.append(feedback_data["time_taken"])
            rw_a.append(feedback_data["rewatch_a"])
            rw_b.append(feedback_data["rewatch_b"])
            rw_both.append(feedback_data["rewatch_both"])

# --- NEW: Hide the window AFTER the loop is finished ---
        if self.pref_fig is not None:
            try:
                self.pref_fig.canvas.manager.window.hide()
            except AttributeError:
                pass # Handle cases where the window might have been closed manually
            
        # ── DIRECT ASSIGNMENT (NO APPEND) ──
        self.human_study_data['time_taken'] = times
        self.human_study_data['rewatch_a'] = rw_a
        self.human_study_data['rewatch_b'] = rw_b
        self.human_study_data['rewatch_both'] = rw_both
        
        return labels

    def get_single_human_label(self, segment1, segment2, trajectory_id):
        return get_human_preferences.get_single_human_label(
            segment1, segment2, trajectory_id, self.size_segment, self.pref_fig, self.frame_dim
        )
