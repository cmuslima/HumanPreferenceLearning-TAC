import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np
import torch
import os
import sys
script_dir  = os.path.dirname(os.path.abspath(__file__))   # .../PEBBLE/human_study
pebble_dir  = os.path.dirname(script_dir)                   # .../PEBBLE
root_dir    = os.path.dirname(pebble_dir)                   # .../root

config_path = os.path.join(root_dir, 'config/train_PEBBLE.yaml')
sys.path.append(root_dir)

from replay_buffer import ReplayBuffer
from replay_buffer import ReplayBuffer
from reward_model_PEBBLE_human import RewardModel
from collections import deque
import utils
import hydra
import wandb
import warnings
import get_human_preferences as get_human_preferences
import time
import matplotlib
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["WANDB_SILENT"] = "true"
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="hydra")
os.environ["QT_QPA_PLATFORM"] = "xcb"
#matplotlib.use('Qt5Agg') 
PID=1
class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        self.cfg = cfg

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        if cfg.user_study:
            pref_fig = get_human_preferences.create_preference_window()
        else:
            pref_fig = None
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
            device=cfg.device,
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
            max_feedback=cfg.max_feedback,
            pid=PID, 
            user_study=cfg.user_study,
            frame_size=cfg.frame_size,
            pref_fig=pref_fig)
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
                train_acc, reward_loss  = self.reward_model.train_soft_reward()
                total_acc = np.mean(train_acc)
                if total_acc > 0.97:
                    break;
            if self.cfg.wandb:
                wandb.log({'reward_model/train_acc':total_acc, 'reward_model/reward_loss':reward_loss, 'reward_model/budget':self.labeled_feedback}, step=self.step)

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
                self.agent.update(self.replay_buffer, self.step, 1)
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
                
            next_obs, reward, terminated, truncated, extra = self.env.step(action)

            done = terminated or truncated
            if self.total_feedback < self.cfg.max_feedback and self.cfg.render_frames:
                frame = self.env.render()
            else:
                frame = None 
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))


            train_predicted_episode_reward += reward_hat
            train_true_episode_reward += reward
            
            if self.log_success:
                training_episode_success = max(training_episode_success, extra['success'])
            # adding data to the reward training data
            self.reward_model.add_data(obs, action, reward, done, frame)
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
        project="PbRL_Human_Preferences_Benchmarking",
        config=utils.flatten_dict(dict(cfg)),
        name=f'testingUI_PEBBLE_{cfg.experiment}_{cfg.env}_{cfg.max_feedback}_{cfg.seed}',
        entity="musliman",
        mode='online',
    )
    workspace.run()

if __name__ == '__main__':
    main()