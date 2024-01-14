import numpy as np
from sumTree import SumTree

class PER():
    def __init__(self, capacity):
        # Initializing SumTree
        self.tree = SumTree(capacity)
        # Setting hyperparameters, see README
        self.PER_e = 0.01  # Hyperparameter e 
        self.PER_a = 0.6  # Hyperparameter a 
        self.PER_b = 0.4  # Hyperparameter b
        # Hyperparameter b increment
        self.PER_b_increment_per_sampling = 0.001
        # Minimum priority
        self.p_min = self.PER_e ** self.PER_a     
        
    
    def add(self, experience):
        '''
            Add a new experience to the sumTree
            Args:
                experience (numpy array): Five Data of the transition step:
                    - current state (numpy array of type object) 
                    - reward
                    - action
                    - next state (numpy array of type object)
                    - done (True or False, end of the episode or loss the game)          

        '''
        # New entries get default priorities
        priority = (0.0 + self.PER_e) ** self.PER_a       
        # Adding the priority and experience to the SumTree 
        self.tree.add(priority, experience)         
    
    def sample(self, batch_size):
        '''
            Sample a minibatch of experiences from the SumTree
            Args:
                batch_size (int): Size of the minibatch.
            Returns:                
                minibatch_ISWeights (numpy array of type float32): weights for each sample of the minibatch.
                minibatch_tree_idx (list): indices of the sampled SumTree leaf.
                minibatch (numpy array of type object): minibatch of sampled experiences.

        '''
                
        minibatch_ISWeights, minibatch = np.empty((batch_size, 1), dtype=np.float32), np.empty((batch_size, self.tree.data[0].size), dtype=object)
        minibatch_tree_idx = []

        # Calculating the priority segment        
        priority_segment = self.tree.total_priority / batch_size
        
        # Increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])
       
        # Calculating the max_weight

        max_weight = (self.p_min * batch_size) ** (-self.PER_b)        
        
        for i in range(batch_size):

            # Choosing a value in the segment
            a, b = priority_segment * i, priority_segment * (i + 1)
           
            value = np.random.uniform(a, b)            
            
            # Getting the experience that correspond to the value

            data, priority, tree_idx = self.tree.get_leaf(value)
            
            # P(j) or sampling probability
            sampling_probabilities = priority / self.tree.total_priority
            
            # IS = (1/N * 1/P(i))**b /max w == (N*P(i))**-b  / max w 
            # weights for each sample or importance-sampling            
            minibatch_ISWeights[i, 0] = np.power(batch_size * sampling_probabilities, -self.PER_b)/ max_weight
            
            # Indices in the SumTree that correspond to the experience
            minibatch_tree_idx.append(tree_idx)            
            
            # Adding experience to the minibatch
            minibatch[i,:] = data        
        
        return minibatch_ISWeights, minibatch_tree_idx, minibatch
              
    def update(self, minibatch_tree_idx, abs_errors): 
        '''
            Update the priorities of the sampled experience  
            Args:
                minibatch_tree_idx (list): indices of the sampled SumTree leaf.
                abs_errors ()
            
                
                

        '''       
        new_priorities = (abs_errors + self.PER_e) ** self.PER_a
        
        for tree_idx, new_priority in zip(minibatch_tree_idx, new_priorities):            
            if new_priority < self.p_min:
                self.p_min = new_priorities
            self.tree.update(tree_idx, new_priority)
     
                   
        
    
