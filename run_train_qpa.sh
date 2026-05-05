cd /home/callie/Projects/HumanPreferenceLearning-TAC/PEBBLE

# for seed in 0 1 2 3 4 5; do
# python train_PEBBLE.py env=walker_walk experiment=no_noise agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=500000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=2000 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 segment=50 max_reward_buffer_size=10 data_aug_ratio=20 ensemble_size=1 explore=false her_ratio=0.5 wandb=true device='cuda' seed=$seed
# done


# # cheetah_run:
# for seed in 0 1 2 3 4 5; do
# python train_PEBBLE.py env=cheetah_run experiment=no_noise agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=2000 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 segment=50 max_reward_buffer_size=10 data_aug_ratio=20 ensemble_size=1 explore=false her_ratio=0.5 wandb=true device='cuda' seed=$seed
# done

# for seed in 0 1 2 3 4 5; do
# python train_PEBBLE.py env=walker_walk experiment=noise20 agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=500000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=2000 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0.2 teacher_eps_skip=0 teacher_eps_equal=0 segment=50 max_reward_buffer_size=10 data_aug_ratio=20 ensemble_size=1 explore=false her_ratio=0.5 wandb=true device='cuda' seed=$seed
# done

# # cheetah_run:
# for seed in 0 1 2 3 4 5; do
# python train_PEBBLE.py env=cheetah_run experiment=noise20 agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=1000000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=2000 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0.2 teacher_eps_skip=0 teacher_eps_equal=0 segment=50 max_reward_buffer_size=10 data_aug_ratio=20 ensemble_size=1 explore=false her_ratio=0.5 wandb=true device='cuda' seed=$seed
# done

for seed in 0 1 2 3 4 5; do
python train_PEBBLE.py  env=walker_walk seed=$seed agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=500000 num_interact=20000 max_feedback=100 reward_batch=20 reward_update=50 feed_type=1 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 device=cuda
done