import argparse
import numpy as np
import torch
from sklearn.metrics import average_precision_score, classification_report
from src.dataset.trans.data import *
from src.dataset.loader import *
from src.model.basenet import *
from src.model.baselines import *
from src.model.models import *
from src.transform.preprocess import *
from src.utils import *


def get_args():
    parser = argparse.ArgumentParser(description='cropped frame model Training')

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
    parser.add_argument('--jitter-ratio', default=-1.0, type=float,
                        help='jitter bbox for cropping')
    parser.add_argument('--bbox-min', default=0, type=int,
                        help='minimum bbox size')
    parser.add_argument('-s', '--seed', type=int, default=99,
                        help='set random seed for sampling')

    parser.add_argument('--encoder-type', default='CC', type=str,
                        help='encoder for images, CC(crop-context) or RC(roi-context)')
    parser.add_argument('--encoder-pretrained', default=False, 
                        help='load pretrained encoder')
    parser.add_argument('--encoder-path', default='', type=str,
                        help='path to encoder checkpoint for loading the pretrained weights')
    parser.add_argument('--decoder-path', default='', type=str,
                        help='path to LSTM decoder checkpoints')

    args = parser.parse_args()

    return args



def eval_model(loader, model, device):
    # swith to evaluate mode
    encoder_CNN = model['encoder']
    decoder_RNN = model['decoder']
    encoder_CNN.eval()
    decoder_RNN.eval()
    y_true = []
    y_pred = []
    y_out = []
    scores = []
    with torch.no_grad():
        for i, inputs in enumerate(loader):
            # evaluate model
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
            outputs_RNN = decoder_RNN(xc_3d=outputs_CNN, xp_3d=pv, 
                                      xb_3d=behavior, xs_2d=scene, x_lengths=seq_len)
            for j in range(targets.size()[0]):
                y_true.append(int(targets[j].item()))
                y_out.append(float(outputs_RNN[j].item()))
                score = 1.0 - abs(float(outputs_RNN[j].item()) - float(targets[j].item()))
                scores.append(score)
                if outputs_RNN[j] >= 0.5:
                   y_pred.append(1)
                else:
                   y_pred.append(0)
                
    np_scores = np.array(scores)
    scores_mean = np.mean(np_scores)
    # AP  = average_precision_score(y_true, y_out)

    print(classification_report(y_true, y_pred))


    return  scores_mean



def main():
    args = get_args()
    # loading data
    print('Annotation loading-->', 'JAAD:', args.jaad, 'PIE:', args.pie, 'TITAN:', args.titan)
    print('------------------------------------------------------------------')
    anns_paths_eval, image_dir_eval = define_path(use_jaad=args.jaad, use_pie=args.pie, use_titan=args.titan)
    print('-->>')
    eval_data = TransDataset(data_paths=anns_paths_eval, image_set="test", verbose=False)
    trans_eval = eval_data.extract_trans_history(mode=args.mode, fps=args.fps, max_frames=None, verbose=True)
    non_trans_eval = eval_data.extract_non_trans(fps=5, max_frames=None, verbose=True)
    print('-->>')
    sequences_eval = extract_pred_sequence(trans=trans_eval, non_trans=non_trans_eval, pred_ahead=args.pred,
                                          balancing_ratio=1.0, neg_in_trans=True,
                                          bbox_min=args.bbox_min, max_frames=args.max_frames, seed=args.seed, verbose=True)
    print('------------------------------------------------------------------')
    print('Finish annotation loading', '\n')

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_res18 = build_encoder_res18(args)
    decoder_lstm = DecoderRNN_IMBS(CNN_embeded_size=256, h_RNN_0=256, h_RNN_1=64, h_RNN_2=16,
                                    h_FC0_dim=128, h_FC1_dim=64, h_FC2_dim=86, drop_p=0.2).to(device)
    decoder_path = args.decoder_path
    _ = load_from_checkpoint(decoder_path, decoder_lstm, optimizer=None, scheduler=None, verbose=True)
    model_gpu = {'encoder': encoder_res18, 'decoder': decoder_lstm}
    jitter_ratio = None if args.jitter_ratio < 0 else args.jitter_ratio
    crop_preprocess = CropBox(size=224, padding_mode='pad_resize', jitter_ratio=jitter_ratio)
    VAL_TRANSFORM = crop_preprocess
    val_instances = PaddedSequenceDataset(sequences_eval, image_dir=image_dir_eval, padded_length=args.max_frames,
                                            hflip_p = 0.0, preprocess=VAL_TRANSFORM)
    test_loader = torch.utils.data.DataLoader(val_instances, batch_size=1, shuffle=False)

    print(f'Test loader : {len(test_loader)}')
    print(f'Start evaluation on balanced test set, PVIBS, jitter={jitter_ratio}, bbox-min={args.bbox_min}')
    test_score = eval_model(test_loader, model_gpu, device)
    print('\n', '-----------------------------------------------------')
    print('----->')
    print('Model Evaluation score: {:.4f}'.format(test_score))
    print('--------------------------------------------------------', '\n')
    
    

if __name__ == '__main__':
    main()
