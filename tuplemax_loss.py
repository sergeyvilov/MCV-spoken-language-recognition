import torch
from torch import nn

class TuplemaxLoss(nn.Module):
    
    '''
    Tuplemax loss pytorch implementation from
    
    Wan, Li, et al. "Tuplemax loss for language identification." 
    ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). 
    IEEE, 2019.
        
    '''
    
    def __init__(self, 
                 reduction='mean'):
        
        super().__init__()
        
        self.reduction = reduction
        

    def forward(self, logits, true_class_idx):
        
        '''
        logits [N_batch x N_classes]: raw output of the FC layer, without activation
        true_class_idx [N_batch x 1]: position of true class output (0..N_classes-1) for each element in the batch
        '''

        n,m = logits.shape

        true_label_pos = torch.zeros(n, m, dtype=torch.bool)
        true_label_pos[range(n),true_class_idx] = 1

        false_class_logits = logits[~true_label_pos].view(n,m-1)
        true_class_logits = logits[true_label_pos]

        #OLD way: log(exp) may explode for large exp, changed to logsumexp
        #tuplemax_loss = torch.mean(torch.log(torch.exp(false_class_logits)+torch.exp(true_class_logits.view(n,1))))-true_class_logits

        logsum = 0

        for k in range(m-1):
            tensor_concat = torch.concat((false_class_logits[:,k], true_class_logits)).view(2,n)
            logsum += torch.logsumexp(tensor_concat,0)
            
        tuplemax_loss = logsum/(m-1) - true_class_logits
        
        if self.reduction == 'mean':
            return tuplemax_loss.mean()
        else:
            return tuplemax_loss