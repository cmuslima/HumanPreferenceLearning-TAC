diff --git a/PEBBLE/reward_model_PEBBLE_TAC.py b/PEBBLE/reward_model_PEBBLE_TAC.py
index 2eabc44..d75af95 100644
--- a/PEBBLE/reward_model_PEBBLE_TAC.py
+++ b/PEBBLE/reward_model_PEBBLE_TAC.py
@@ -87,7 +87,7 @@ class RewardModel:
                  teacher_beta=-1, teacher_gamma=1, 
                  teacher_eps_mistake=0, 
                  teacher_eps_skip=0, 
-                 teacher_eps_equal=0):
+                 teacher_eps_equal=0, alpha=1):
         
         # train data is trajectories, must process to sa and s..   
         self.ds = ds
@@ -102,7 +102,7 @@ class RewardModel:
         self.max_size = max_size
         self.activation = activation
         self.size_segment = size_segment
-        self.alpha=1
+        self.alpha=alpha
 
         
         self.capacity = int(capacity)
@@ -598,6 +598,97 @@ class RewardModel:
         
         return len(labels)
     
+    def train_reward_tac_v2(self):
+        ensemble_losses = [[] for _ in range(self.de)]
+        ensemble_acc = np.array([0 for _ in range(self.de)])
+        
+        max_len = self.capacity if self.buffer_full else self.buffer_index
+        total_batch_index = []
+        for _ in range(self.de):
+            total_batch_index.append(np.random.permutation(max_len))
+        
+        num_epochs = int(np.ceil(max_len/self.train_batch_size))
+        list_debug_loss1, list_debug_loss2 = [], []
+        total = 0
+        
+        for epoch in range(num_epochs):
+            self.opt.zero_grad()
+            loss = 0.0
+            
+            last_index = (epoch+1)*self.train_batch_size
+            if last_index > max_len:
+                last_index = max_len
+                
+            for member in range(self.de):
+                
+                # get random batch
+                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
+                sa_t_1 = self.buffer_seg1[idxs]
+                sa_t_2 = self.buffer_seg2[idxs]
+                labels = self.buffer_label[idxs]
+                labels = torch.from_numpy(labels.flatten()).long().to(self.device)
+                
+                if member == 0:
+                    total += labels.size(0)
+                
+                # get logits
+                r_hat1 = self.r_hat_member(sa_t_1, member=member)
+                r_hat2 = self.r_hat_member(sa_t_2, member=member)
+                r_hat1 = r_hat1.sum(axis=1)
+                r_hat2 = r_hat2.sum(axis=1)
+                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
+
+
+# 1. Get Reward Difference
+                r_diff = r_hat1 - r_hat2  
+                r_diff = r_diff.view(-1)
+
+                # 2. Bradley-Terry "Soft" Probabilities
+                # prob_1: Model's probability that traj1 is better than traj2
+                prob_1 = torch.sigmoid(self.alpha * r_diff)
+                prob_2 = 1.0 - prob_1
+
+                # 3. Calculate Soft P (Agreement) and Q (Disagreement)
+                # labels: 0 if traj1 is ground-truth preferred, 1 if traj2 is preferred
+                is_1_preferred = (labels == 0).float()
+                is_2_preferred = (labels == 1).float()
+
+                # P is the sum of probabilities assigned to the correct trajectories
+                # Q is the sum of probabilities assigned to the incorrect trajectories
+                P = torch.sum(is_1_preferred * prob_1 + is_2_preferred * prob_2)
+                Q = torch.sum(is_1_preferred * prob_2 + is_2_preferred * prob_1)
+
+                # 4. Calculate Sigma_TAC (Equation 3)
+                # Assuming X0 and Y0 are 0 if there are no ties in your labels
+                numerator = P - Q
+                # Denominator uses (P+Q) which is effectively the batch size N
+                denominator = torch.sqrt((P + Q) * (P + Q) + 1e-8) 
+                
+                sigma_tac = numerator / denominator
+
+                # 5. Define Loss: We want to maximize correlation, so minimize (1 - sigma_tac)
+                curr_loss = 1.0 - sigma_tac
+
+                # compute loss
+                loss += curr_loss
+
+                ensemble_losses[member].append(curr_loss.item())
+                
+
+                # compute accuracy (agreement with preference)
+                pred = (r_hat1 > r_hat2).float()
+                pred = pred.view(-1)
+                
+                correct = (pred == (1.0 - labels)).sum().item()
+
+                ensemble_acc[member] += correct
+                
+            loss.backward()
+            self.opt.step()
+        
+        ensemble_acc = ensemble_acc / total
+        
+        return ensemble_acc, np.mean(ensemble_losses)
     def train_reward(self):
         ensemble_losses = [[] for _ in range(self.de)]
         ensemble_acc = np.array([0 for _ in range(self.de)])
