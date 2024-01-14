import numpy as np

class ReplayBuffer():
    # (s,a,r,s+1)

    def __init__(self, max_size):
        # Memory data
        self.storage = []
        # Data pointer
        self.ptr = 0
        # Memory capacity
        self.max_size = max_size        

    def add(self, experience):
        
        # If full memory restart ptr to first memory position
        self.ptr = self.ptr % self.max_size
        # Saving experinece
        self.storage.append(experience)
        self.ptr += 1        
        
            
    
    def sample(self, batch_size):        
        batch = np.random.choice(len(self.storage), batch_size, replace=False)
        minibatch = np.empty((batch_size, self.storage[0].size), dtype=object)
        count = 0
        for i in batch:            
            minibatch[count,:] = self.storage[i]
            count += 1
            
        
        return minibatch