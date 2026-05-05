
# cd /home/callie/Projects/HumanPreferenceLearning/QPA/human_study

# for seed in 1 2 3 4 5; do
# python QPA_user_interface.py experiment=UI_no_frames env=walker_walk agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=500000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=2000 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 segment=50 max_reward_buffer_size=10 data_aug_ratio=20 ensemble_size=1 explore=false her_ratio=0.5 wandb=true device='cuda' render_frames=0 seed=$seed
# done

cd /home/callie/Projects/HumanPreferenceLearning/PEBBLE/human_study

for seed in 1 2 3 4 5; do

python PEBBLE_user_interface.py experiment=UI_no_frames env=walker_walk seed=$seed agent.params.actor_lr=0.0005 agent.params.critic_lr=0.0005 gradient_update=1 activation=tanh num_unsup_steps=9000 num_train_steps=500000 num_interact=20000 max_feedback=100 reward_batch=10 reward_update=50 feed_type=1 teacher_beta=-1 teacher_gamma=1 teacher_eps_mistake=0 teacher_eps_skip=0 teacher_eps_equal=0 wandb=true device='cuda' render_frames=0
done