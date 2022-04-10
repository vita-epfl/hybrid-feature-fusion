import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalLoss(torch.nn.Module):
    """Loss function for attribute fields.
    Args:
        head_net (AttributeField): Prediction head network corresponding to the
            attribute.
    """

    focal_gamma = 0.0


    def __init__(self, focal_gamma=2.0):
        super().__init__()

        self.focal_gamma = focal_gamma
        self.previous_loss = None



    @property
    def loss_function(self):
        loss_module = torch.nn.CrossEntropyLoss(reduction='none')
        
        return lambda x, t: loss_module(x, t.to(torch.long))


    def forward(self, *args):
        x, t = args
        loss = self.compute_loss(x, t)

        if (loss is not None) and (not torch.isfinite(loss).item()):
            raise Exception('found a loss that is not finite: {}, prev: {}'
                            ''.format(loss, self.previous_loss))
        self.previous_loss = float(loss.item()) if loss is not None else None

        return loss


    def compute_loss(self, x, t):
        if t is None:
            return None

        loss = self.loss_function(x, t)
        if self.focal_gamma != 0:
            focal = torch.nn.functional.softmax(x, dim=1)
            t_index =  t.to(torch.long).unsqueeze(0).t() # (batch_size, 1
            focal = 1. - focal.gather(1, t_index)
            focal = focal.view(1,-1).squeeze() # [,]
            loss = loss * focal.pow(self.focal_gamma)
          

        loss = loss.mean()
  
        return loss
