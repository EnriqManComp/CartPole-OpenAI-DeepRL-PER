import numpy as np
from sumTree import SumTree

class PER():
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
        self.PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    
        self.PER_b_increment_per_sampling = 0.001
    
        self.p_min = self.PER_e ** self.PER_a     
        
    
    def add(self, experience):
        # New entries get 1.0 of priority
        priority = (0.0 + self.PER_e) ** self.PER_a
        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority       
        
        """
            Save experience in HDD       
        
        """                   
        self.tree.add(priority, experience)   # set the max p for new p
        
    
    def sample(self, batch_size):
        # Minibatch        
        minibatch_ISWeights, minibatch = np.empty((batch_size, 1), dtype=np.float32), np.empty((batch_size, self.tree.data[0].size), dtype=object)
        minibatch_tree_idx = []
        # Calculating the priority segment        
        priority_segment = self.tree.total_priority / batch_size
        
        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
       
        # Calculating the max_weight

        p_min = self.p_min
        
        max_weight = (p_min * batch_size) ** (-self.PER_b)        
        
        for i in range(batch_size):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
           
            value = np.random.uniform(a, b)            
            
            """
            Experience that correspond to each value is retrieved
            """
            data, priority, tree_idx = self.tree.get_leaf(value)
            
            #P(j)
            sampling_probabilities = priority / self.tree.total_priority
            
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            minibatch_ISWeights[i, 0] = np.power(batch_size * sampling_probabilities, -self.PER_b)/ max_weight

            minibatch_tree_idx.append(tree_idx)            
            minibatch[i,:] = data        
        
        return minibatch_ISWeights, minibatch_tree_idx, minibatch
              
    
    
    """
    Update the priorities on the tree
    """
    def update(self, minibatch_tree_idx, abs_errors):        
        new_priorities = (abs_errors + self.PER_e) ** self.PER_a
        
        for tree_idx, new_priority in zip(minibatch_tree_idx, new_priorities):            
            if new_priority < self.p_min:
                self.p_min = new_priorities
            self.tree.update(tree_idx, new_priority)
     
                   
        
    
