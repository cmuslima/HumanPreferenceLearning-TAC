
import numpy as np
import torch
import os
import sys
import torch.nn.functional as F
import utils
from agent.sac_rime import compute_state_entropy
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(parent_dir, 'config/train_RIME.yaml')
sys.path.append(parent_dir)

from replay_buffer import ReplayBuffer
from reward_model_RIME import RIMERewardModel
from collections import deque
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
            k = 600
            tau = 0.001
        else:
            self.env = utils.make_env(cfg)
            k = 60
            tau = 0.001        
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
        self.reward_model = RIMERewardModel(
            device=self.device,
            k=k,
            threshold_variance=cfg.threshold_variance,
            threshold_alpha=cfg.threshold_alpha,
            threshold_beta_init=cfg.threshold_beta_init,
            threshold_beta_min=cfg.threshold_beta_min,
            flipping_tau=tau,
            num_warmup_steps=int(1/3*cfg.max_feedback/cfg.reward_batch*cfg.least_reward_update+0.5),
            ds=self.env.observation_space.shape[0],
            da=self.env.action_space.shape[0],
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
            teacher_eps_equal=cfg.teacher_eps_equal)
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
        labeled_queries = 0
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
                if epoch % 5 == 0 or epoch == self.cfg.reward_update - 1:
                    debug = True
                train_acc, reward_loss  = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                # early stop
                if total_acc > 0.98 and epoch > self.cfg.least_reward_update:
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

                # relabel buffer due to training of reward model during unsup steps
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                
                # first learn reward
                self.reward_model.set_lr_schedule()
                self.learn_reward(first_flag=1)
                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                # reset interact_count
                interact_count = 0
            
            # 3 differences from above: first_flag, corner case, update method (reset critic)
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
                
                # update reward model to fit with intrinsic reward
                for _ in range(5):
                    unsup_obs, full_obs, unsup_act, _, _, _, _ = self.replay_buffer.sample_state_ent(
                        self.agent.batch_size)
                    
                    state_entropy = compute_state_entropy(unsup_obs, full_obs, k=self.cfg.topK)
                    norm_state_entropy = (state_entropy - self.agent.state_ent.mean) / self.agent.state_ent.std
                    scale = ((self.agent.s_ent_stats - self.agent.state_ent.mean) / self.agent.state_ent.std).abs().max()
                    norm_state_entropy /= scale
                    
                    self.reward_model.opt.zero_grad()
                    unsup_rew_loss = 0.0
                    for member in range(self.reward_model.de):
                        rew_hat = self.reward_model.ensemble[member](torch.cat([unsup_obs, unsup_act], dim=-1).to(self.device))
                        unsup_rew_loss += F.mse_loss(rew_hat, norm_state_entropy.detach().to(self.device))
                    unsup_rew_loss.backward()
                    self.reward_model.opt.step()
                
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
        project="PbRL_Human_Preferences_Benchmarking",
        config=utils.flatten_dict(dict(cfg)),
        name=f'RIME_{cfg.env}_{cfg.max_feedback}_{cfg.seed}',
        entity="musliman",
        mode='online',
    )
    workspace.run()

if __name__ == '__main__':
    main()