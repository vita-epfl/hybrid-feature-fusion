import argparse
import time
from sklearn.metrics import average_precision_score
from sympy import true
from zmq import device
from src.dataset.trans.data import *
from src.dataset.loader import *
from src.model.basenet import *
from src.model.baselines import *
from src.model.models import *
from src.transform.preprocess import *
from src.utils import *



def get_args():
    parser = argparse.ArgumentParser(description='Train hybrid model')
    parser.add_argument('--jaad', default=False, action='store_true',
                        help='use JAAD dataset')
    parser.add_argument('--pie', default=False, action='store_true',
                        help='use PIE dataset')
    parser.add_argument('--titan', default=False, action='store_true',
                        help='use TITAN dataset')
    parser.add_argument('--mode', default='GO', type=str,
                        help='transition mode, GO or STOP')
    parser.add_argument('--fps', default=5, type=int,
                        metavar='FPS', help='sampling rate(fps)')
    parser.add_argument('--max-frames', default=5, type=int,
                        help='maximum number of frames in histroy sequence')
    parser.add_argument('--pred', default=10, type=int,
                        help='prediction length, predicting-ahead time')
    parser.add_argument('--balancing-ratio', default=1.0, type=float,
                        help='ratio of balanced instances(1/0)')
    parser.add_argument('--seed', default=99, type=int,
                        help='random seed for sampling')
    parser.add_argument('--jitter-ratio', default=-1.0, type=float,
                        help='jitter bbox for cropping')
    parser.add_argument('--bbox-min', default=0, type=int,
                        help='minimum bbox size')
    
    parser.add_argument('--encoder-type', default='CC', type=str,
                        help='encoder for images, CC(crop-context) or RC(roi-context)')
    parser.add_argument('--encoder-pretrained', default=False, 
                        help='load pretrained encoder')
    parser.add_argument('--encoder-path', default='', type=str,
                        help='path to encoder checkpoint for loading the pretrained weights')
    parser.add_argument('-lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 4)')
    parser.add_argument('-e', '--epochs', default=10, type=int,
                        help='number of epochs to train')
    parser.add_argument('-wd', '--weight-decay', metavar='WD', type=float, default=1e-5,
                        help='Weight decay', dest='wd')
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    args = parser.parse_args()

    return args


def train_epoch(loader, model, criterion, optimizer, device):

    encoder_CNN = model['encoder']
    decoder_RNN = model['decoder']
    # freeze CNN-encoder during training
    encoder_CNN.train()
    for child in encoder_CNN.resnet.children():
        for para in child.parameters():
            para.requires_grad = False
    decoder_RNN.train()
    epoch_loss = 0.0
    for i, inputs in enumerate(loader):
        # compute output and loss
        targets = inputs['label'].to(device, non_blocking=True)
        images = inputs['image'].to(device, non_blocking=True)
        bboxes_ped = inputs['bbox_ped']
        seq_len = inputs['seq_length']
        behavior_list = reshape_anns(inputs['behavior'], device)
        behavior = batch_first(behavior_list).to(device, non_blocking=True)
        scene = inputs['attributes'].to(device, non_blocking=True)
        bbox_ped_list = reshape_bbox(bboxes_ped, device)
        pv = bbox_to_pv(bbox_ped_list).to(device, non_blocking=True)
        outputs_CNN = encoder_CNN(images, seq_len)
        outputs_RNN = decoder_RNN(xc_3d=outputs_CNN, xp_3d=pv, xb_3d=behavior, xs_2d=scene, x_lengths=seq_len)
        loss = criterion(outputs_RNN, targets.view(-1, 1))
        # record loss
        optimizer.zero_grad()
        epoch_loss += float(loss.item())
        # compute gradient and do SGD step, scheduler step
        loss.backward()
        optimizer.step()

    return epoch_loss / len(loader)



def val_epoch(loader, model, criterion, device):
    # swith to evaluate mode
    encoder_CNN = model['encoder']
    decoder_RNN = model['decoder']
    # freeze CNN-encoder 
    encoder_CNN.eval()
    decoder_RNN.eval()
    epoch_loss = 0.0
    n_p = 0.0
    n_n = 0.0
    n_tp = 0.0
    n_tn = 0.0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, inputs in enumerate(loader):
            # compute output and loss
            targets = inputs['label'].to(device, non_blocking=True)
            images = inputs['image'].to(device, non_blocking=True)
            bboxes_ped = inputs['bbox_ped']
            seq_len = inputs['seq_length']
            behavior_list = reshape_anns(inputs['behavior'], device)
            behavior = batch_first(behavior_list).to(device, non_blocking=True)
            scene = inputs['attributes'].to(device, non_blocking=True)
            bbox_ped_list = reshape_bbox(bboxes_ped, device)
            pv = bbox_to_pv(bbox_ped_list).to(device, non_blocking=True)
            outputs_CNN = encoder_CNN(images, seq_len)
            outputs_RNN = decoder_RNN(xc_3d=outputs_CNN, xp_3d=pv, xb_3d=behavior, xs_2d=scene, x_lengths=seq_len)
            loss = criterion(outputs_RNN, targets.view(-1, 1))
            epoch_loss += float(loss.item())
            for j in range(targets.size()[0]):
                y_true.append(int(targets[j].item()))
                y_pred.append(float(outputs_RNN[j].item()))
                if targets[j] > 0.5:
                    n_p += 1
                    if outputs_RNN[j] > 0.5:
                        n_tp += 1
                else:
                    n_n += 1
                    if outputs_RNN[j] < 0.5:
                        n_tn += 1

    AP_P = average_precision_score(y_true, y_pred)
    FP = n_n - n_tn
    acc_P = n_tp / (n_tp + FP) if n_tp + FP > 0 else 0.0
    recall_P = n_tp / n_p
    f1_p = 2 * (acc_P * recall_P) / (acc_P + recall_P) if acc_P + recall_P > 0 else 0.0
    print('------------------------------------------------')
    print(f'acc: {acc_P}')
    print(f'recall: {n_tp / n_p}')
    print(f'F1-score : {f1_p}')
    print(f"average precision for transition prediction: {AP_P}")
    print('\n')
    val_score = AP_P

    return epoch_loss / len(loader), val_score


def main():
    args = get_args()
    # loading data
    print('Start annotation loading -->', 'JAAD:', args.jaad, 'PIE:', args.pie, 'TITAN:', args.titan)
    print('------------------------------------------------------------------')
    anns_paths, image_dir = define_path(use_jaad=args.jaad, use_pie=args.pie, use_titan=args.titan)
    anns_paths_val, image_dir_val = define_path(use_jaad=args.jaad, use_pie=args.pie, use_titan=args.titan)
    train_data = TransDataset(data_paths=anns_paths, image_set="train", verbose=False)
    trans_tr = train_data.extract_trans_history(mode=args.mode, fps=args.fps, max_frames=None,
                                                verbose=True)
    non_trans_tr = train_data.extract_non_trans(fps=5, max_frames=None, verbose=True)
    print('-->>')
    val_data = TransDataset(data_paths=anns_paths_val, image_set="test", verbose=False)
    trans_val = val_data.extract_trans_history(mode=args.mode, fps=args.fps, max_frames=None, verbose=True)
    non_trans_val = val_data.extract_non_trans(fps=5, max_frames=None, verbose=True)
    print('-->>')
    sequences_train = extract_pred_sequence(trans=trans_tr, non_trans=non_trans_tr, pred_ahead=args.pred,
                                            balancing_ratio=1.0, neg_in_trans=True,
                                            bbox_min=args.bbox_min, max_frames=args.max_frames, seed=args.seed, verbose=True)
    print('-->>')
    sequences_val = extract_pred_sequence(trans=trans_val, non_trans=non_trans_val, pred_ahead=args.pred,
                                          balancing_ratio=1.0, neg_in_trans=True,
                                          bbox_min=args.bbox_min, max_frames=args.max_frames, seed=args.seed, verbose=True)
    print('------------------------------------------------------------------')
    print('Finish annotation loading', '\n')
    # construct and load model  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_res18 = build_encoder_res18(args)
    decoder_lstm = DecoderRNN_IMBS(CNN_embeded_size=256, h_RNN_0=256, h_RNN_1=64, h_RNN_2=16,
                                    h_FC0_dim=128, h_FC1_dim=64, h_FC2_dim=86, drop_p=0.2).to(device)
    model_gpu = {'encoder': encoder_res18, 'decoder': decoder_lstm}
    # training settings
    criterion = torch.nn.BCELoss().to(device)
    crnn_params = list(encoder_res18.fc.parameters()) + list(decoder_lstm.parameters())
    optimizer = torch.optim.Adam(crnn_params, lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    start_epoch = 0
    end_epoch = start_epoch + args.epochs
    # start training
    jitter_ratio = None if args.jitter_ratio < 0 else args.jitter_ratio
    crop_preprocess = CropBox(size=224, padding_mode='pad_resize', jitter_ratio=jitter_ratio)
    TRAIN_TRANSFORM = Compose([crop_preprocess,
                               ImageTransform(torchvision.transforms.ColorJitter(
                                   brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
                               ])
    VAL_TRANSFORM = crop_preprocess
    train_instances = PaddedSequenceDataset(sequences_train, image_dir=image_dir, padded_length=args.max_frames,
                                            hflip_p = 0.5, preprocess=TRAIN_TRANSFORM)
    val_instances = PaddedSequenceDataset(sequences_val, image_dir=image_dir_val, padded_length=args.max_frames,
                                            hflip_p = 0.0, preprocess=VAL_TRANSFORM)
    train_loader = torch.utils.data.DataLoader(train_instances, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_instances, batch_size=1, shuffle=False)
    ds = ''
    print(f'train loader : {len(train_loader)}')
    print(f'val loader : {len(val_loader)}')
    total_time = 0.0
    ap_min = 0.5
    print(f'Start training, PVIBS-lstm-model, neg_in_trans, initail lr={args.lr}, weight-decay={args.wd}, mf={args.max_frames}, training batch size={args.batch_size}')
    if args.output is None:
        Save_path = r'./checkpoints/PVIBS/Decoder_IMBS_lr{}_wd{}_{}_{}_bm{}_mf{}_bs{}'.format(args.lr, args.wd, ds, args.mode,args.bbox_min,args.max_frames,args.batch_size)
    else:
        Save_path = args.output
    for epoch in range(start_epoch, end_epoch):
        start_epoch_time = time.time()
        train_loss = train_epoch(train_loader, model_gpu, criterion, optimizer, device)
        val_loss, val_score = val_epoch(val_loader, model_gpu, criterion, device)
        scheduler.step(val_score)
        end_epoch_time = time.time() - start_epoch_time
        print('\n', '-----------------------------------------------------')
        print(f'End of epoch {epoch}')
        print('Training epoch loss: {:.4f}'.format(train_loss))
        print('Validation epoch loss: {:.4f}'.format(val_loss))
        print('Validation epoch score: {:.4f}'.format(val_score))
        print('Epoch time: {:.2f}'.format(end_epoch_time))
        print('--------------------------------------------------------', '\n')
        total_time += end_epoch_time
        if val_score > ap_min:
           save_to_checkpoint(Save_path , epoch, model_gpu['decoder'], optimizer, scheduler, verbose=True)
           ap_min = val_score
    print('\n', '**************************************************************')
    print(f'End training at epoch {end_epoch}')
    print('total time: {:.2f}'.format(total_time))


def main():
    if __name__ == '__main__':
        main()
