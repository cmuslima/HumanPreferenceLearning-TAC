#!/usr/bin/env python3
import numpy as np
import torch
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, 'config/train_MRN.yaml')
import torch.nn.functional as F
sys.path.append(parent_dir)
from replay_buffer import ReplayBuffer
from reward_model_MRN import RewardModel
from collections import deque
from utils import MetaOptim
import utils
import hydra
import wandb
import warnings
import time
# Ignore specific DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Specifically target the Sentry/WandB warnings if they persist
os.environ["WANDB_SILENT"] = "true"
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="hydra")
class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False
        if 'metaworld' in cfg.env_suite:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)
        self.max_episode_len=self.env.spec.max_episode_steps
        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(cfg.agent)
        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device,
            max_episode_len=self.max_episode_len)
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # instantiating the reward model
        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation,
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch,
            large_batch=cfg.large_batch,
            label_margin=cfg.label_margin,
            teacher_beta=cfg.teacher_beta,
            teacher_gamma=cfg.teacher_gamma,
            teacher_eps_mistake=cfg.teacher_eps_mistake,
            teacher_eps_skip=cfg.teacher_eps_skip,
            teacher_eps_equal=cfg.teacher_eps_equal,
            alpha=cfg.alpha)
    def evaluate(self):
        average_predicted_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        
        for episode in range(self.cfg.num_eval_episodes):
            obs, _ = self.env.reset()
            self.agent.reset()
            done = False
            predicted_episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                predicted_episode_reward += self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))
                obs, reward, terminated, truncated, extra = self.env.step(action)
                done = terminated or truncated
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
                
            average_predicted_episode_reward += predicted_episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_predicted_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0
        return average_true_episode_reward, average_predicted_episode_reward, success_rate
    def learn_reward(self, first_flag=0):
        # get feedbacks
        labeled_queries, noisy_queries = 0, 0
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        train_acc = 0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc, reward_loss  = self.reward_model.train_reward_tac()
                total_acc = np.mean(train_acc)
                if total_acc > 0.97:
                    break;
            if self.cfg.wandb:
                wandb.log({'reward_model/train_acc':total_acc, 'reward_model/reward_loss':reward_loss, 'reward_model/budget':self.labeled_feedback}, step=self.step)
    def r_hat_critic_old(self, x):
        batch_size, segment_length, obsact = x.shape  # _ = obs+act
        assert obsact == self.env.observation_space.shape[0] + self.env.action_space.shape[0]
        obs = x[:, 0, :self.env.observation_space.shape[0]].reshape(batch_size, self.env.observation_space.shape[0])
        act = x[:, 0, self.env.observation_space.shape[0]:].reshape(batch_size, self.env.action_space.shape[0])
        obs = torch.from_numpy(obs).float().to(self.device)
        act = torch.from_numpy(act).float().to(self.device)

        q1, q2 = self.agent.critic_old(obs, act)
        assert q1.shape == (batch_size, 1)

        return q1, q2

    def bilevel_update(self):
        # sample from replay buffer and get meta reward from reward model (with grad)
        obs, action, reward, next_obs, not_done, not_done_no_max = self.replay_buffer.sample(self.agent.batch_size)

        inputs = np.concatenate([obs.cpu(), action.cpu()], axis=-1)
        reward = self.reward_model.r_hat_batch_grad(inputs)
        if self.cfg.wandb:
            wandb.log({'train/batch_reward':reward.detach().cpu().numpy().mean()}, step=self.step)


        # load parameters of critic_old from current critic
        self.agent.critic_old = hydra.utils.instantiate(self.cfg.agent.params.critic_cfg).to(self.device)
        self.agent.update_critic_old()

        # calculate target_Q for critic_old
        dist = self.agent.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.agent.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.agent.alpha.detach() * log_prob
        target_V = target_V.detach()
        target_Q = reward + (not_done * self.agent.discount * target_V)

        # get Q estimates of critic_old
        current_Q1, current_Q2 = self.agent.critic_old(obs, action)
        critic_old_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # optimize the critic_old
        pseudo_grads = torch.autograd.grad(critic_old_loss, self.agent.critic_old.parameters(), create_graph=True)
        critic_old_optimizer = MetaOptim(self.agent.critic_old, self.agent.critic_old.parameters(), lr=self.cfg.agent.params.critic_lr)
        critic_old_optimizer.load_state_dict(self.agent.critic_optimizer.state_dict())
        critic_old_optimizer.meta_step(pseudo_grads)
        del pseudo_grads

        # calculate loss using trajectory preferences
        index = self.reward_model.buffer_index
        num_eval_pref = self.cfg.reward_batch
        if index < self.cfg.reward_batch:
            idxs = np.append(np.arange(index), np.arange(self.reward_model.capacity - num_eval_pref + index, self.reward_model.capacity))
        else:
            idxs = np.arange(index - num_eval_pref, index)
        np.random.shuffle(idxs)

        sa_t_1 = self.reward_model.buffer_seg1[idxs]  # (B x len_segment x (obs+act))
        sa_t_2 = self.reward_model.buffer_seg2[idxs]  # (B x len_segment x (obs+act))
        labels = self.reward_model.buffer_label[idxs]  # (B x 1)
        labels = torch.from_numpy(labels.flatten()).long().to(self.device)  # (B) [1, 0, 0, 1, 0, 1, 0, 0, 1, 1]

        # get r_hat estimates from critic_old
        r_hat_critic1_q1, r_hat_critic1_q2 = self.r_hat_critic_old(sa_t_1)  # (B x 1)
        r_hat_critic2_q1, r_hat_critic2_q2 = self.r_hat_critic_old(sa_t_2)  # (B x 1)
        r_hat_critic_q1 = torch.cat([r_hat_critic1_q1, r_hat_critic2_q1], axis=-1)  # (B x 2)
        r_hat_critic_q2 = torch.cat([r_hat_critic1_q2, r_hat_critic2_q2], axis=-1)  # (B x 2)

        # compute loss CE((B x 2), (B)) + CE((B x 2), (B))
        outer_loss = (F.cross_entropy(r_hat_critic_q1, labels) + F.cross_entropy(r_hat_critic_q2, labels)) * self.cfg.outer_weight

        # optimize the reward function
        self.reward_model.opt.zero_grad()
        outer_loss.backward()
        self.reward_model.opt.step()

        # calculate target_Q for critic
        reward = self.reward_model.r_hat_batch(inputs)
        reward = torch.as_tensor(reward, device=self.device)
        target_Q = (reward + (not_done * self.agent.discount * target_V)).detach()
        current_Q1, current_Q2 = self.agent.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # optimize the critic
        self.agent.critic.zero_grad()
        critic_loss.backward()
        self.agent.critic_optimizer.step()

        if self.cfg.wandb:
            wandb.log({'train_critic/loss':critic_loss}, step=self.step)
        # update actor and alpha
        dist = self.agent.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.agent.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.agent.alpha.detach() * log_prob - actor_Q).mean()
        if self.cfg.wandb:
            wandb.log({'train_actor/loss':actor_loss, 'train_actor/target_entropy':self.agent.target_entropy, 'train_actor/entropy': -log_prob.mean() }, step=self.step)


        # optimize the actor
        self.agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.agent.actor_optimizer.step()

        self.agent.actor.log(self.logger, self.step)

        if self.agent.learnable_temperature:
            self.agent.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.agent.alpha * (-log_prob - self.agent.target_entropy).detach()).mean()

            if self.cfg.wandb:
                wandb.log({'train_alpha/loss':alpha_loss, 'train_alpha/value': self.agent.alpha}, step=self.step)


            alpha_loss.backward()
            self.agent.log_alpha_optimizer.step()

        if self.step % self.agent.critic_target_update_frequency == 0:  # critic_target_update_frequency = 2
            utils.soft_update_params(self.agent.critic, self.agent.critic_target, self.agent.critic_tau)

    def run(self):
        episode, train_predicted_episode_reward, done = 0, 0, True
        if self.log_success:
            training_episode_success = 0
        train_true_episode_reward = 0
        train_predicted_episode_reward=0
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10)

        interact_count = 0
        start_time = time.time()
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.log_success:
                    wandb.log({'train/true_episode_reward':train_true_episode_reward,
                            'train/predicted_episode_reward':train_predicted_episode_reward,
                            'train/success_rate': training_episode_success,
                            'train/duration': time.time()- start_time}, step=self.step)
                else:
                    wandb.log({'train/true_episode_reward':train_true_episode_reward,
                        'train/predicted_episode_reward':train_predicted_episode_reward,
                        'train/duration': time.time()- start_time}, step=self.step)
                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    eval_true_episode_reward, eval_predicted_episode_reward, eval_success_rate  = self.evaluate()
                    if self.cfg.wandb:
                        if self.log_success:
                            wandb.log({'eval/true_episode_reward':eval_true_episode_reward,
                                    'eval/predicted_episode_reward':eval_predicted_episode_reward,
                                    'eval/success_rate': eval_success_rate}, step=self.step)

                        else:
                            wandb.log({'eval/true_episode_reward':eval_true_episode_reward,
                                'eval/predicted_episode_reward':eval_predicted_episode_reward}, step=self.step)
                obs,_ = self.env.reset()
                self.agent.reset()
                done = False
                start_time = time.time()
                train_predicted_episode_reward = 0
                avg_train_true_return.append(train_true_episode_reward)
                train_true_episode_reward = 0
                if self.log_success:
                    training_episode_success = 0
                episode_step = 0
                episode += 1
            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)
                # update margin --> not necessary / will be updated soon
                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.max_episode_len)
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)
                # first learn reward
                self.learn_reward(first_flag=1)
                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                # reset Q due to unsupervised exploration
                self.agent.reset_critic()
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                # reset interact_count
                interact_count = 0
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)
                        # update margin --> not necessary / will be updated soon
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / self.max_episode_len)
                        self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)
                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0
                if self.step % self.cfg.num_meta_steps == 0 and self.total_feedback < self.cfg.max_feedback:
                    self.bilevel_update()
                    self.replay_buffer.relabel_with_predictor(self.reward_model)
                else:
                    self.agent.update(self.replay_buffer, self.step, 1)
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
                
            next_obs, reward, terminated, truncated, extra = self.env.step(action)

            done = terminated or truncated
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))


            train_predicted_episode_reward += reward_hat
            train_true_episode_reward += reward
            
            if self.log_success:
                training_episode_success = max(training_episode_success, extra['success'])
            # adding data to the reward training data
            self.reward_model.add_data(obs, action, reward, done)
            self.replay_buffer.add(
                obs, action, reward_hat, 
                next_obs, done, terminated)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1
        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)

@hydra.main(config_path=config_path, strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    wandb.init(
        project="soft_tac",
        config=utils.flatten_dict(dict(cfg)),
        name=f'MRN_TAC_{cfg.env}_{cfg.max_feedback}_{cfg.seed}',
        entity="musliman",
        mode='online',
    )
    workspace.run()

if __name__ == '__main__':
    main()