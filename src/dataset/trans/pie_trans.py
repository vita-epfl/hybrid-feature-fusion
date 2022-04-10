import pickle
import numpy as np
import copy


def get_ped_ids_pie(annotations, image_set="all") -> list:
    """
    Returns all pedestrian ids
    :return: A list of pedestrian ids
    """
    pids = []
    image_set_nums = {'train': ['set01', 'set02', 'set04'],
                      'val': ['set05', 'set06'],
                      'test': ['set03'],
                      'all': ['set01', 'set02', 'set03',
                              'set04', 'set05', 'set06']}
    set_ids = image_set_nums[image_set]
    for sid in set_ids:
        for vid in sorted(annotations[sid]):
            pids.extend(annotations[sid][vid]['ped_annotations'].keys())

    return pids


def get_ped_info_pie(annotations, image_set="all") -> dict:
    """
        Get pedestrians' information,i.e. frames,bbox,occlusion, actions(walking or not),cross behavior.
        :param: annotations: PIE annotations in dictionary form
                image_set : str: train,val.test set split of PIE
        :return: information of all pedestrians in one video
    """
    assert image_set in ['train', 'test', 'val', "all"], "Image set should be train, test or val"
    pids = get_ped_ids_pie(annotations, image_set)
    dataset = annotations
    ped_info = {}
    for idx in pids:
        sn, vn, _ = idx.split("_")
        sid = "set{:02d}".format(int(sn))
        vid = "video_{:04d}".format(int(vn))
        ped_info[idx] = {}
        ped_info[idx]["set_number"] = sid
        ped_info[idx]["video_number"] = vid
        ped_info[idx]['frames'] = copy.deepcopy(dataset[sid][vid]['ped_annotations'][idx]['frames'])
        ped_info[idx]['bbox'] = copy.deepcopy(dataset[sid][vid]['ped_annotations'][idx]['bbox'])
        ped_info[idx]['occlusion'] = copy.deepcopy(dataset[sid][vid]['ped_annotations'][idx]['occlusion'])
        ped_info[idx]['action'] = copy.deepcopy(dataset[sid][vid]['ped_annotations'][idx]['behavior']['action'])
        ped_info[idx]['cross'] = copy.deepcopy(dataset[sid][vid]['ped_annotations'][idx]['behavior']['cross'])
        ped_info[idx]['behavior'] = []
        look = copy.deepcopy(dataset[sid][vid]['ped_annotations'][idx]['behavior']['look'])
        gesture = copy.deepcopy(dataset[sid][vid]['ped_annotations'][idx]['behavior']['gesture'])
        for i in range(len(gesture)):
            beh_vec = [0, 0, 0, 0]
            beh_vec[0] = ped_info[idx]['action'][i]
            beh_vec[1] = look[i]
            hg = gesture[i]
            if hg == 4:
                beh_vec[2] = 1  # nod
            elif hg == 0:
                beh_vec[3] = 0  # undefined
            else:
                beh_vec[3] = 1  # hand gestures
            ped_info[idx]['behavior'].append(beh_vec)

            # attribute vector
        atr_vec = [0, 0, 0, 0, 0, 0]
        atr_vec[0] = dataset[sid][vid]['ped_annotations'][idx]['attributes']['num_lanes']
        if dataset[sid][vid]['ped_annotations'][idx]['attributes']['intersection'] > 0:
            atr_vec[1] = 1
        # atr_vec[2] = dataset[sid][vid]['ped_annotations'][idx]['attributes']['designated']
        if dataset[sid][vid]['ped_annotations'][idx]['attributes']['signalized'] > 0:
            atr_vec[3] = 1
        atr_vec[4] = dataset[sid][vid]['ped_annotations'][idx]['attributes']['traffic_direction']
        # atr_vec[5] = dataset[sid][vid]['ped_annotations'][idx]['attributes']['motion_direction']
        ped_info[idx]['attributes'] = copy.deepcopy(atr_vec)

        # process traffic light

    return ped_info


def filter_None(x):
    # small help function to filter None in list
    if x is None:
        return False
    else:
        return True


def ped_info_clean_pie(annotations, image_set="all") -> dict:
    """
     Remove all frames has occlusion tag = 2 (fully occluded)
    :param: annotations: PIE annotations in dictionary form
            image_set : image_set : str: train,val.test set split of PIE
    :return: cleaned information of all pedestrians in given set
    """
    assert image_set in ['train', 'test', 'val', "all"], "Image set should be train, test or val"
    ped_info = get_ped_info_pie(annotations, image_set)
    ids = list(ped_info.keys())
    # remove all frames with occlusion tag=2
    for idx in ids:
        occ = np.array(ped_info[idx]['occlusion'])
        full_occ = np.flatnonzero(occ == 2)
        # set fully occluded frames to None
        for i in range(len(full_occ)):
            ped_info[idx]['frames'][full_occ[i]] = None
            ped_info[idx]['bbox'][full_occ[i]] = None
            ped_info[idx]['action'][full_occ[i]] = None
            ped_info[idx]['occlusion'][full_occ[i]] = None
            ped_info[idx]['cross'][full_occ[i]] = None
            ped_info[idx]['behavior'][full_occ[i]] = None

        # filter all None values
        ped_info[idx]['frames'] = list(filter(filter_None, ped_info[idx]['frames']))
        ped_info[idx]['bbox'] = list(filter(filter_None, ped_info[idx]['bbox']))
        ped_info[idx]['action'] = list(filter(filter_None, ped_info[idx]['action']))
        ped_info[idx]['occlusion'] = list(filter(filter_None, ped_info[idx]['occlusion']))
        ped_info[idx]['cross'] = list(filter(filter_None, ped_info[idx]['cross']))
        ped_info[idx]['behavior'] = list(filter(filter_None, ped_info[idx]['behavior']))

    return ped_info


def add_trans_label_pie(dataset, verbose=False) -> None:
    """
    Add labels to show the time (number of frames)
    away from next action transition
    """
    all_wts = 0  # walking to standing
    all_stw = 0  # standing to walking
    ped_ids = list(dataset.keys())
    for idx in ped_ids:
        action = dataset[idx]['action']
        frames = dataset[idx]['frames']
        n_frames = len(frames)
        dataset[idx]['next_transition'] = []
        stw_time = []
        wts_time = []
        for j in range(len(action) - 1):
            if action[j] == 0 and action[j + 1] == 1:
                all_stw += 1
                stw_time.append(frames[j + 1])
            elif action[j] == 1 and action[j + 1] == 0:
                all_wts += 1
                wts_time.append(frames[j + 1])
        # merge
        trans_time_ped = np.array(sorted(stw_time + wts_time))
        # set transition tag
        for i in range(n_frames):
            t = frames[i]
            future_trans_ped = trans_time_ped[trans_time_ped >= t]
            if future_trans_ped.size > 0:
                next_trans_ped = future_trans_ped[0]
                dataset[idx]['next_transition'].append(next_trans_ped - t)
            else:
                dataset[idx]['next_transition'].append(None)
    if verbose:
        print('----------------------------------------------------------------')
        print('PIE:')
        print(f'Total number of standing to walking transitions (raw): {all_stw}')
        print(f'Total number of walking to standing transitions  (raw): {all_wts}')

    return None


def build_ped_dataset_pie(pie_anns_path, image_set="all", verbose=False) -> dict:
    """
    Build pedestrian dataset from PIE annotations
    """
    assert image_set in ['train', 'test', 'val', "all"], "Image set should be train, test or val"
    pie_anns = pickle.load(open(pie_anns_path, 'rb'))
    ped_dataset = ped_info_clean_pie(pie_anns, image_set)
    add_trans_label_pie(ped_dataset, verbose)

    return ped_dataset


# -----------------------------------------------
class PieTransDataset:
    # dataset class for pedestrian samples
    def __init__(self, pie_anns_path, image_set="all", verbose=False):
        assert image_set in ['train', 'test', 'val', "all"]
        self.dataset = build_ped_dataset_pie(pie_anns_path, image_set=image_set, verbose=verbose)
        self.name = image_set

    def __repr__(self):
        return f"PieTransDataset(image_set={self.name})"

    def extract_trans_frame(self, mode="GO", frame_ahead=0, fps=30, verbose=False) -> dict:
        """
        Extract the frame when the action transition happens
        """
        dataset = self.dataset
        assert mode in ["GO", "STOP"], "Transition type should be STOP or GO"
        ids = list(dataset.keys())
        samples = {}
        j = 0
        step = 30 // fps
        t_ahead = step * frame_ahead
        for idx in ids:
            sid = copy.deepcopy(dataset[idx]['set_number'])
            vid = copy.deepcopy(dataset[idx]['video_number'])
            frames = copy.deepcopy(dataset[idx]['frames'])
            bbox = copy.deepcopy(dataset[idx]['bbox'])
            action = copy.deepcopy(dataset[idx]['action'])
            cross = copy.deepcopy(dataset[idx]['cross'])
            behavior = copy.deepcopy(dataset[idx]['behavior'])
            attributes = copy.deepcopy(dataset[idx]['attributes'])
            next_transition = copy.deepcopy(dataset[idx]["next_transition"])
            for i in range(len(frames)):
                key = None
                old_id = None
                d1 = min(i, 5)
                d2 = min(len(frames) - i - 1, 5)
                if mode == "GO":
                    if next_transition[i] == 0 and action[i] == 1 and action[i - d1] == 0 and action[i + d2] == 1:
                        j += 1
                        new_id = "{:04d}".format(j) + "_" + self.name
                        key = "PG_" + new_id
                        old_id = idx
                if mode == "STOP":
                    if next_transition[i] == 0 and action[i] == 0 and action[i - d1] == 1 and action[i + d2] == 0:
                        j += 1
                        new_id = "{:04d}".format(j) + "_" + self.name
                        key = "PS_" + new_id
                        old_id = idx
                if key is not None and i - t_ahead >= 0:
                    samples[key] = {}
                    samples[key]["source"] = "PIE"
                    samples[key]['set_number'] = sid
                    samples[key]['video_number'] = vid
                    samples[key]["old_id"] = old_id
                    samples[key]['frame'] = frames[i - t_ahead]
                    samples[key]['bbox'] = bbox[i - t_ahead]
                    samples[key]['action'] = action[i - t_ahead]
                    samples[key]['cross'] = cross[i - t_ahead]
                    samples[key]['behavior'] = behavior[i - t_ahead]
                    samples[key]['attributes'] = attributes
                    samples[key]['frame_ahead'] = frame_ahead
                    samples[key]['type'] = mode
                    samples[key]['fps'] = fps
        if verbose:
            print(f"Extract {len(samples.keys())} {mode} sample frames from PIE {self.name} set")

        return samples

    def extract_trans_history(self, mode="GO", fps=30, max_frames=None, post_frames=0, verbose=False) -> dict:
        """
        Extract the whole history of pedestrian up to the frame when transition happens
        :params: mode: target transition type, "GO" or "STOP"
                fps: frame-per-second, sampling rate of extracted sequences, default 30
                verbose: optional printing of sample statistics
        """
        dataset = self.dataset
        assert mode in ["GO", "STOP"], "Transition type should be STOP or GO"
        ids = list(dataset.keys())
        samples = {}
        j = 0
        step = 30 // fps
        assert isinstance(step, int)
        for idx in ids:
            sid = copy.deepcopy(dataset[idx]['set_number'])
            vid = copy.deepcopy(dataset[idx]['video_number'])
            frames = copy.deepcopy(dataset[idx]['frames'])
            bbox = copy.deepcopy(dataset[idx]['bbox'])
            action = copy.deepcopy(dataset[idx]['action'])
            behavior = copy.deepcopy(dataset[idx]['behavior'])
            attributes = copy.deepcopy(dataset[idx]['attributes'])
            next_transition = copy.deepcopy(dataset[idx]["next_transition"])
            for i in range(len(frames)):
                key = None
                old_id = None
                d1 = min(i, 5)
                d2 = min(len(frames) - i - 1, 5)
                if mode == "GO":
                    if next_transition[i] == 0 and action[i] == 1 and action[i - d1] == 0 and action[i + d2] == 1:
                        j += 1
                        new_id = "{:04d}".format(j) + "_" + self.name
                        key = "PG_" + new_id
                        old_id = idx
                        ae = np.array(action[i::-step])
                        ce = np.array(np.nonzero(ae == 1))
                        d_pre = ce[0][1] - 1 if ce.size > 1 else len(ae) - 1
                        ap = np.array(action[i::step])
                        cp = np.array(np.nonzero(ap == 0))
                        d_pos = cp[0][0] if cp.size > 0 else len(ap)
                if mode == "STOP":
                    if next_transition[i] == 0 and action[i] == 0 and action[i - d1] == 1 and action[i + d2] == 0:
                        j += 1
                        new_id = "{:04d}".format(j) + "_" + self.name
                        key = "PS_" + new_id
                        old_id = idx
                        ae = np.array(action[i::-step])
                        ce = np.array(np.nonzero(ae == 0))
                        d_pre = ce[0][1] - 1 if ce.size > 1 else len(ae) - 1
                        ap = np.array(action[i::step])
                        cp = np.array(np.nonzero(ap == 1))
                        d_pos = cp[0][0] if cp.size > 0 else len(ap)
                if key is not None:
                    if max_frames is None:
                        t = None
                    else:
                        t = i - max_frames * step if (i - max_frames * step >= 0) else None
                    i = i + min(post_frames, d_pos) * step
                    samples[key] = {}
                    samples[key]["source"] = "PIE"
                    samples[key]["old_id"] = old_id
                    samples[key]['set_number'] = sid
                    samples[key]['video_number'] = vid
                    samples[key]['frame'] = frames[i:t:-step]
                    samples[key]['frame'].reverse()
                    samples[key]['bbox'] = bbox[i:t:-step]
                    samples[key]['bbox'].reverse()
                    samples[key]['action'] = action[i:t:-step]
                    samples[key]['action'].reverse()
                    samples[key]['behavior'] = behavior[i:t:-step]
                    samples[key]['behavior'].reverse()
                    samples[key]['attributes'] = attributes
                    samples[key]['pre_state'] = d_pre
                    samples[key]['post_state'] = d_pos
                    samples[key]['type'] = mode
                    samples[key]['fps'] = fps
        if verbose:
            keys = list(samples.keys())
            pids = []
            num_frames = 0
            for k in keys:
                pids.append(samples[k]['old_id'])
                num_frames += len(samples[k]['frame'])
            print(f"Extract {len(pids)} {mode} history samples from {self.name} dataset in PIE ,")
            print(f"samples contain {len(set(pids))} unique pedestrians and {num_frames} frames.")

        return samples

    def extract_non_trans(self, fps=30, max_frames=None, verbose=False):
        dataset = self.dataset
        ids = list(dataset.keys())
        samples = {'walking': {}, 'standing': {}}
        step = 30 // fps
        assert isinstance(step, int)
        jw = 0
        js = 0
        for idx in ids:
            sid = copy.deepcopy(dataset[idx]['set_number'])
            vid_id = copy.deepcopy(dataset[idx]['video_number'])
            frames = copy.deepcopy(dataset[idx]['frames'])
            bbox = copy.deepcopy(dataset[idx]['bbox'])
            action = copy.deepcopy(dataset[idx]['action'])
            behavior = copy.deepcopy(dataset[idx]['behavior'])
            attributes = copy.deepcopy(dataset[idx]['attributes'])
            a = np.array(action)  # action array
            key = None
            action_type = None
            old_id = None
            if a[a < 0.5].size == 0:  # all walking
                jw += 1
                new_id = "{:04d}".format(jw) + "_" + self.name
                key = "PW_" + new_id
                old_id = idx
                action_type = 'walking'
            elif a[a > 0.5].size == 0:  # all standing
                js += 1
                new_id = "{:04d}".format(js) + "_" + self.name
                key = "PN_" + new_id
                old_id = idx
                action_type = 'standing'
            if max_frames is None:
                t = None
            else:
                t = len(frames) - max_frames * step if (len(frames) - max_frames * step >= 0) else None
            if key is not None:
                samples[action_type][key] = {}
                samples[action_type][key]["source"] = "PIE"
                samples[action_type][key]["old_id"] = old_id
                samples[action_type][key]['video_number'] = vid_id
                samples[action_type][key]['set_number'] = sid
                samples[action_type][key]['frame'] = frames[-1:t:-step]
                samples[action_type][key]['frame'].reverse()
                samples[action_type][key]['bbox'] = bbox[-1:t:-step]
                samples[action_type][key]['bbox'].reverse()
                samples[action_type][key]['action'] = action[-1:t:-step]
                samples[action_type][key]['action'].reverse()
                samples[action_type][key]['behavior'] = behavior[-1:t:-step]
                samples[action_type][key]['behavior'].reverse()
                samples[action_type][key]['attributes'] = attributes
                samples[action_type][key]['action_type'] = action_type
                samples[action_type][key]['fps'] = fps

        if verbose:
            keys_w = list(samples['walking'].keys())
            keys_s = list(samples['standing'].keys())
            pid_w = []
            pid_s = []
            n_w = 0
            n_s = 0
            for kw in keys_w:
                pid_w.append(samples['walking'][kw]['old_id'])
                n_w += len(samples['walking'][kw]['frame'])
            for ks in keys_s:
                pid_s.append(samples['standing'][ks]['old_id'])
                n_s += len(samples['standing'][ks]['frame'])
            print(f"Extract Non-transition samples from {self.name} dataset in PIE :")
            print(f"Walking: {len(pid_w)} samples,  {len(set(pid_w))} unique pedestrians and {n_w} frames.")
            print(f"Standing: {len(pid_s)} samples,  {len(set(pid_s))} unique pedestrians and {n_s} frames.")

        return samples
