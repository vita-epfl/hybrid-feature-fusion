import torch
import sys
import os


def save_to_checkpoint(save_path, epoch, model, optimizer, scheduler=None, verbose=True):
    # save checkpoint to disk
    d_sche = None
    if scheduler is not None:
        d_sche = scheduler.state_dict()
    if save_path is not None:
        checkpoint = {'epoch': epoch + 1,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'scheduler': d_sche
                      }
        torch.save(checkpoint, '{}_epoch_{}.pt'.format(save_path, epoch))

    if verbose:
        print("saved model at epoch {}".format(epoch))

        
def load_from_checkpoint(checkpoint_path, model, optimizer = None, scheduler = None, verbose = True):
    """Loads model from checkpoint, loads optimizer and scheduler too if not None, 
       and returns epoch and iteration of the checkpoints
    """
    if not os.path.exists(checkpoint_path):
        raise ("File does not exist {}".format(checkpoint_path))
        
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,map_location='cpu')
        
    check_keys = list(checkpoint.keys())

    model.load_state_dict(checkpoint['model']) 
    
    if 'optimizer' in check_keys:
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])

    if 'scheduler' in check_keys:
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
  
    if 'epoch' in check_keys:
        epoch = checkpoint['epoch']
       
    if verbose: # optional printing
        print(f"Loaded model from checkpoint {checkpoint_path}")

    return epoch
    
# ---------------------------------------------------------------------
def reshape_bbox(bbox_list, device):
    new_bbox_list = []
    for j in range(len(bbox_list)):
        raw_bboxes = bbox_list[j]
        B = torch.stack(raw_bboxes).type(torch.FloatTensor).to(device, non_blocking=True)
        bboxes = []
        for i in range(B.shape[1]):
           bboxes.append(B[:,i].view(1,4))
        new_bbox_list.append(bboxes)
    
    return new_bbox_list


def batch_first(anns_list):
    anns_3d = []
    for i in range(len(anns_list[0])):
        anns_1d = []
        for t in range(len(anns_list)):
            anns = torch.squeeze(anns_list[t][i], dim=0)
            anns_1d.append(anns)
        anns_tensors_2d = torch.stack(anns_1d)
        anns_3d.append(anns_tensors_2d)
    # stack batch
    anns_tensors_3d = torch.stack(anns_3d)
    
    return anns_tensors_3d
    

def bbox_to_pv(bbox_list):
    pv_3d = []
    for i in range(len(bbox_list[0])):
        p_1d = []
        v_1d = []
        pv_1d = []
        for t in range(len(bbox_list)):
            bbox = torch.squeeze(bbox_list[t][i], dim=0)
            # float 
            b = list(map(lambda x: x.item(), bbox))
            # compute bbox center
            # xc = (b[0] + b[2]) / 2 - 960.0
            # c = abs(-(b[1] + b[3]) / 2 + 1080.0)
            xc = (b[0] + b[2]) / 2 
            yc = (b[1] + b[3]) / 2 
            
            # compute width, height
            w = abs(b[2] - b[0]) 
            h = abs(b[3] - b[1])
            p_1d.append([xc, yc, w, h])
        v_1d.append([0.0, 0.0, 0.0, 0.0])
        for t in range(1, len(bbox_list)):
            dx = abs(p_1d[t][0] -  p_1d[t-1][0])
            dy = abs(p_1d[t][1] -  p_1d[t-1][1])
            dw = abs(p_1d[t][2] -  p_1d[t-1][2])
            dh = abs(p_1d[t][3] -  p_1d[t-1][3])
            v_1d.append([dx, dy, dw, dh])
        for t in range(len(bbox_list)):
            pv_1d.append(torch.tensor(p_1d[t] + v_1d[t], dtype=torch.float32))
        pv_tensors_2d = torch.stack(pv_1d)
        pv_3d.append(pv_tensors_2d)
    # stack batch
    pv_tensors_3d = torch.stack(pv_3d)
    
    return pv_tensors_3d


def reshape_anns(anns_list, device):
    new_anns_list = []
    dim = len(anns_list[0])
    for j in range(len(anns_list)):
        raw_anns = anns_list[j]
        B = torch.stack(raw_anns).type(torch.FloatTensor).to(device, non_blocking=True)
        anns = []
        for i in range(B.shape[1]):
           anns.append(B[:,i].view(1,dim))
        new_anns_list.append(anns)
    
    return new_anns_list


