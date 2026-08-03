import os
import pickle
import os.path as osp
import random
import torch.utils.data as tordata
import json
from utils import get_msg_mgr

import torch
import torch.nn.functional as F

def remove_black_border_batch(batch_imgs, threshold=10, target_h=256, target_w=128):
    """
    批量去除黑边，支持 B x C x H x W 的输入。
    输入: batch_imgs - B x C x H x W 的 PyTorch Tensor，值范围[0,255]或[0,1]
    输出: 去除黑边并调整大小到 target_h x target_w 的 Tensor - B x C x target_h x target_w
    """
    assert batch_imgs.dim() == 4, "Input must be a 4D tensor (B x C x H x W)"
    B, C, H, W = batch_imgs.shape
    batch_imgs = batch_imgs.float()

    # 批量灰度化：对 RGB 取均值，得到 B x H x W
    if C > 1:
        gray = batch_imgs.mean(dim=1)  # B x H x W
    else:
        gray = batch_imgs.squeeze(1)  # B x H x W

    # 批量生成掩码：判断非黑区域
    mask = gray > threshold  # B x H x W

    # 处理全黑图像：直接返回原始图像（或填充为目标尺寸）
    result = torch.zeros(B, C, target_h, target_w, device=batch_imgs.device, dtype=batch_imgs.dtype)
    for i in range(B):
        if not mask[i].any():
            result[i] = resize_with_padding(batch_imgs[i], target_h, target_w)
            continue

        # 批量边界检测
        y_nonzero = torch.any(mask[i], dim=1)  # H
        x_nonzero = torch.any(mask[i], dim=0)  # W

        y_indices = torch.where(y_nonzero)[0]
        x_indices = torch.where(x_nonzero)[0]

        if y_indices.numel() == 0 or x_indices.numel() == 0:
            result[i] = resize_with_padding(batch_imgs[i], target_h, target_w)
            continue

        y_min, y_max = y_indices[[0, -1]]
        x_min, x_max = x_indices[[0, -1]]

        # 裁剪单张图像
        cropped = batch_imgs[i, :, y_min:y_max+1, x_min:x_max+1]
        result[i] = resize_with_padding(cropped, target_h, target_w)

    return result

def resize_with_padding(img, target_h, target_w):
    """
    等比例缩放 + padding，支持单张或批量图像。
    输入: img - C x H x W 或 B x C x H x W
    输出: C x target_h x target_w 或 B x C x target_h x target_w
    """
    is_batch = img.dim() == 4
    if not is_batch:
        img = img.unsqueeze(0)  # 转换为 B x C x H x W

    B, C, H, W = img.shape
    scale = torch.min(torch.tensor([target_h / H, target_w / W], device=img.device))
    new_h, new_w = (H * scale).long(), (W * scale).long()

    # 批量缩放
    img_resized = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)

    # 计算 padding
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # 批量 padding
    img_padded = F.pad(img_resized, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    return img_padded.squeeze(0) if not is_batch else img_padded

# 替换原来的 batch_remove_black_border
def batch_remove_black_border(batch_imgs, threshold=10, target_h=256, target_w=128):
    return remove_black_border_batch(batch_imgs, threshold, target_h, target_w)


class DataSet(tordata.Dataset):
    def __init__(self, data_cfg, training):
        """
            seqs_info: the list with each element indicating 
                            a certain gait sequence presented as [label, type, view, paths];
        """
        self.training = training
        self.video_sample_ratio = data_cfg['video_sample_ratio'] if 'video_sample_ratio' in data_cfg.keys() else None
        self.sync_video_modalities = data_cfg.get('sync_video_modalities', False)
        self.__dataset_parser(data_cfg, training)
        self.cache = data_cfg['cache']
        self.label_list = [seq_info[0] for seq_info in self.seqs_info]
        self.types_list = [seq_info[1] for seq_info in self.seqs_info]
        self.views_list = [seq_info[2] for seq_info in self.seqs_info]

        self.label_set = sorted(list(set(self.label_list)))
        self.types_set = sorted(list(set(self.types_list)))
        self.views_set = sorted(list(set(self.views_list)))
        self.seqs_data = [None] * len(self)
        self.indices_dict = {label: [] for label in self.label_set}
        for i, seq_info in enumerate(self.seqs_info):
            self.indices_dict[seq_info[0]].append(i)
        if self.cache:
            self.__load_all_data()

    def __len__(self):
        return len(self.seqs_info)

    def __loader__(self, paths):
        from torchvision.io import read_video
        import random
        paths = sorted(paths)
        data_list = []
        video_frame_indices = None
        video_source_length = None
        for pth in paths:
            if pth.endswith('.pkl'):
                with open(pth, 'rb') as f:
                    _ = pickle.load(f)
                f.close()
                if self.training and self.sync_video_modalities and video_frame_indices is not None:
                    if len(_) != video_source_length:
                        raise ValueError(
                            'Cannot synchronize video modalities: {} has {} frames, but the source video has {}.'.format(
                                pth, len(_), video_source_length))
                    _ = [_[frame_idx] for frame_idx in video_frame_indices]
            elif pth.endswith('.avi'):
                video, _, _ = read_video(pth, output_format="TCHW", pts_unit='sec')

                if self.training:
                    random_idx = sorted(random.sample(range(video.size(0)), min(40, video.size(0))), key=int)
                    if self.sync_video_modalities:
                        video_frame_indices = random_idx
                        video_source_length = video.size(0)
                    video = video[random_idx, :, :, :]

                # ratios = torch.full((video.size(0),), video.size(-1) / video.size(-2), device=video.device)
                _ = batch_remove_black_border(video)
                # _ = ratio_resize(_, ratios, 256, 128) # ratio-resize
                _ = _.numpy()
            else:
                raise ValueError('- Loader - just support .pkl !!!')
            data_list.append(_)
        for idx, data in enumerate(data_list):
            if len(data) != len(data_list[0]):
                raise ValueError(
                    'Each input data({}) should have the same length.'.format(paths[idx]))
            if len(data) == 0:
                raise ValueError(
                    'Each input data({}) should have at least one element.'.format(paths[idx]))
        return data_list

    def __getitem__(self, idx):
        if not self.cache:
            data_list = self.__loader__(self.seqs_info[idx][-1])
        elif self.seqs_data[idx] is None:
            data_list = self.__loader__(self.seqs_info[idx][-1])
            self.seqs_data[idx] = data_list
        else:
            data_list = self.seqs_data[idx]
        seq_info = self.seqs_info[idx]
        return data_list, seq_info

    def __load_all_data(self):
        for idx in range(len(self)):
            self.__getitem__(idx)

    def __dataset_parser(self, data_config, training):
        dataset_root = data_config['dataset_root']
        try:
            data_in_use = data_config['data_in_use']  # [n], true or false
        except:
            data_in_use = None

        with open(data_config['dataset_partition'], "rb") as f:
            partition = json.load(f)
        train_set = partition["TRAIN_SET"]
        test_set = partition["TEST_SET"]
        label_list = os.listdir(dataset_root)
        train_set = [label for label in train_set if label in label_list]
        test_set = [label for label in test_set if label in label_list]
        miss_pids = [label for label in label_list if label not in (
            train_set + test_set)]
        msg_mgr = get_msg_mgr()

        def log_pid_list(pid_list):
            if len(pid_list) >= 3:
                msg_mgr.log_info('[%s, %s, ..., %s]' %
                                 (pid_list[0], pid_list[1], pid_list[-1]))
            else:
                msg_mgr.log_info(pid_list)

        if len(miss_pids) > 0:
            msg_mgr.log_debug('-------- Miss Pid List --------')
            msg_mgr.log_debug(miss_pids)
        if training:
            msg_mgr.log_info("-------- Train Pid List --------")
            log_pid_list(train_set)
        else:
            msg_mgr.log_info("-------- Test Pid List --------")
            log_pid_list(test_set)

        def get_seqs_info_list(label_set):
            seqs_info_list = []
            for lab in label_set:
                for typ in sorted(os.listdir(osp.join(dataset_root, lab))):
                    for vie in sorted(os.listdir(osp.join(dataset_root, lab, typ))):
                        seq_info = [lab, typ, vie]
                        seq_path = osp.join(dataset_root, *seq_info)
                        seq_dirs = sorted(os.listdir(seq_path))
                        if seq_dirs != []:
                            seq_dirs = [osp.join(seq_path, dir)
                                        for dir in seq_dirs]
                            if data_in_use is not None:
                                seq_dirs = [dir for dir, use_bl in zip(
                                    seq_dirs, data_in_use) if use_bl]
                            seqs_info_list.append([*seq_info, seq_dirs])
                        else:
                            msg_mgr.log_debug(
                                'Find no .pkl file in %s-%s-%s.' % (lab, typ, vie))
            return seqs_info_list

        def build_balanced_eval_subset(seqs_info_list):
            subset_cfg = data_config.get('eval_subset', {}) or {}
            if not subset_cfg.get('enabled', False):
                return seqs_info_list

            strategy = str(subset_cfg.get(
                'strategy', 'ccgr_gallery_view')).lower()

            if strategy == 'casiab_view_balanced':
                seed = int(subset_cfg.get('seed', 7890))
                num_pids = int(subset_cfg.get('num_pids', 8))
                sequence_types = [str(value) for value in subset_cfg.get(
                    'sequence_types', ['nm-05'])]
                requested_views = [str(value) for value in subset_cfg.get(
                    'views',
                    ['000', '018', '036', '054', '072', '090',
                     '108', '126', '144', '162', '180'])]
                if num_pids <= 0 or not sequence_types or not requested_views:
                    raise ValueError(
                        'CASIA-B eval_subset requires positive num_pids and '
                        'non-empty sequence_types/views.')

                def normalize_casiab_view(raw_view):
                    try:
                        return '{:03d}'.format(int(float(str(raw_view))))
                    except (TypeError, ValueError):
                        return str(raw_view)

                def sequence_key(seq_info):
                    return '-'.join(seq_info[:3])

                per_pid = {}
                for seq_info in seqs_info_list:
                    pid, seq_type, raw_view = seq_info[:3]
                    if seq_type not in sequence_types:
                        continue
                    view = normalize_casiab_view(raw_view)
                    if view not in requested_views:
                        continue
                    per_pid.setdefault(pid, {}).setdefault(
                        seq_type, {}).setdefault(view, []).append(seq_info)

                eligible_pids = []
                for pid, type_map in per_pid.items():
                    complete = all(
                        seq_type in type_map
                        and all(view in type_map[seq_type]
                                for view in requested_views)
                        for seq_type in sequence_types
                    )
                    if complete:
                        eligible_pids.append(pid)
                eligible_pids = sorted(eligible_pids)
                if len(eligible_pids) < num_pids:
                    raise ValueError(
                        'CASIA-B eval_subset requested {} identities, but only '
                        '{} contain every requested type/view.'.format(
                            num_pids, len(eligible_pids)))

                pid_rng = random.Random(seed)
                pid_rng.shuffle(eligible_pids)
                selected_pids = sorted(eligible_pids[:num_pids])
                selected_keys = set()
                for pid in selected_pids:
                    for seq_type in sequence_types:
                        for view in requested_views:
                            candidates = sorted(
                                per_pid[pid][seq_type][view],
                                key=sequence_key)
                            selected_keys.add(sequence_key(candidates[0]))

                subset = [
                    seq for seq in seqs_info_list
                    if sequence_key(seq) in selected_keys]
                expected_size = (
                    num_pids * len(sequence_types) * len(requested_views)
                )
                if len(subset) != expected_size:
                    raise RuntimeError(
                        'CASIA-B view-balanced subset produced {} sequences; '
                        'expected {}.'.format(len(subset), expected_size))
                msg_mgr.log_info(
                    '-------- CASIA-B View-Balanced Eval Subset --------')
                msg_mgr.log_info(
                    'Selected {} identities x {} types x {} views = {} '
                    'sequences with seed {}.'.format(
                        num_pids, len(sequence_types),
                        len(requested_views), len(subset), seed))
                log_pid_list(selected_pids)
                return subset

            if strategy == 'ccpg_protocol_balanced':
                seed = int(subset_cfg.get('seed', 7890))
                num_pids = int(subset_cfg.get('num_pids', 99))
                sequences_per_pid = int(
                    subset_cfg.get('sequences_per_pid', 30))
                base_views_per_type = int(
                    subset_cfg.get('base_views_per_type', 4))
                protocol_types = [str(value) for value in subset_cfg.get(
                    'protocol_types',
                    ['U0_D0_BG', 'U0_D0', 'U1_D1', 'U2_D2',
                     'U3_D3', 'U0_D3', 'U1_D0'])]
                camera_views = [str(value) for value in subset_cfg.get(
                    'camera_views',
                    ['01', '02', '03', '04', '05',
                     '06', '07', '08', '09', '10'])]

                if (num_pids <= 0 or sequences_per_pid <= 0
                        or base_views_per_type <= 1):
                    raise ValueError(
                        'CCPG eval_subset counts must be positive and '
                        'base_views_per_type must exceed one.')
                if len(protocol_types) != len(set(protocol_types)):
                    raise ValueError(
                        'CCPG eval_subset.protocol_types contains duplicates.')
                if len(camera_views) != len(set(camera_views)):
                    raise ValueError(
                        'CCPG eval_subset.camera_views contains duplicates.')
                minimum_per_pid = len(protocol_types) * base_views_per_type
                if sequences_per_pid < minimum_per_pid:
                    raise ValueError(
                        'CCPG sequences_per_pid={} is smaller than the {} '
                        'sequences required by {} types x {} base views.'
                        .format(
                            sequences_per_pid, minimum_per_pid,
                            len(protocol_types), base_views_per_type))

                def sequence_key(seq_info):
                    return '-'.join(seq_info[:3])

                def normalized_camera(raw_view):
                    for camera in sorted(
                            camera_views, key=len, reverse=True):
                        if (raw_view == camera
                                or raw_view.startswith(camera + '_')):
                            return camera
                    return None

                per_pid = {}
                for seq_info in seqs_info_list:
                    pid, seq_type, raw_view = seq_info[:3]
                    camera = normalized_camera(str(raw_view))
                    if seq_type not in protocol_types or camera is None:
                        continue
                    per_pid.setdefault(pid, {}).setdefault(
                        seq_type, {}).setdefault(camera, []).append(seq_info)

                eligible_pids = []
                common_cameras_by_pid = {}
                for pid, type_map in per_pid.items():
                    if any(seq_type not in type_map
                           for seq_type in protocol_types):
                        continue
                    common_cameras = set(camera_views)
                    for seq_type in protocol_types:
                        common_cameras &= set(type_map[seq_type].keys())
                    if len(common_cameras) < base_views_per_type:
                        continue
                    eligible_pids.append(pid)
                    common_cameras_by_pid[pid] = sorted(common_cameras)

                eligible_pids = sorted(eligible_pids)
                if len(eligible_pids) < num_pids:
                    raise ValueError(
                        'CCPG eval_subset requested {} identities, but only '
                        '{} contain every protocol type and at least {} common '
                        'camera views.'.format(
                            num_pids, len(eligible_pids),
                            base_views_per_type))

                pid_rng = random.Random(seed)
                pid_rng.shuffle(eligible_pids)
                selected_pids = sorted(eligible_pids[:num_pids])
                processing_pids = list(selected_pids)
                random.Random(seed + 1).shuffle(processing_pids)

                selected_keys = set()
                selected_counts_by_pid = {pid: 0 for pid in selected_pids}
                type_counts = {seq_type: 0 for seq_type in protocol_types}
                camera_counts = {camera: 0 for camera in camera_views}
                type_camera_counts = {
                    (seq_type, camera): 0
                    for seq_type in protocol_types
                    for camera in camera_views
                }

                def choose_sequence(candidates):
                    # Prefer the canonical *_0 recording; otherwise use the
                    # lexicographically first available repeat.
                    return sorted(candidates, key=sequence_key)[0]

                for pid in processing_pids:
                    type_map = per_pid[pid]
                    tie_cameras = list(common_cameras_by_pid[pid])
                    random.Random('{}:{}:base'.format(
                        seed, pid)).shuffle(tie_cameras)
                    tie_rank = {
                        camera: index
                        for index, camera in enumerate(tie_cameras)}
                    chosen_cameras = sorted(
                        common_cameras_by_pid[pid],
                        key=lambda camera: (
                            camera_counts[camera], tie_rank[camera]))[
                                :base_views_per_type]

                    for seq_type in protocol_types:
                        for camera in chosen_cameras:
                            chosen = choose_sequence(
                                type_map[seq_type][camera])
                            key = sequence_key(chosen)
                            selected_keys.add(key)
                            selected_counts_by_pid[pid] += 1
                            type_counts[seq_type] += 1
                            camera_counts[camera] += 1
                            type_camera_counts[(seq_type, camera)] += 1

                    extras_needed = sequences_per_pid - minimum_per_pid
                    extra_candidates = []
                    for seq_type in protocol_types:
                        for camera, candidates in type_map[seq_type].items():
                            chosen = choose_sequence(candidates)
                            if sequence_key(chosen) not in selected_keys:
                                extra_candidates.append(
                                    (seq_type, camera, chosen))
                    extra_ties = list(extra_candidates)
                    random.Random('{}:{}:extra'.format(
                        seed, pid)).shuffle(extra_ties)
                    extra_tie_rank = {
                        sequence_key(item[2]): index
                        for index, item in enumerate(extra_ties)}

                    for _ in range(extras_needed):
                        if not extra_candidates:
                            raise ValueError(
                                'Identity {} has insufficient distinct CCPG '
                                'protocol sequences for {} samples.'.format(
                                    pid, sequences_per_pid))
                        extra_candidates.sort(key=lambda item: (
                            type_counts[item[0]],
                            camera_counts[item[1]],
                            type_camera_counts[(item[0], item[1])],
                            extra_tie_rank[sequence_key(item[2])]))
                        seq_type, camera, chosen = extra_candidates.pop(0)
                        selected_keys.add(sequence_key(chosen))
                        selected_counts_by_pid[pid] += 1
                        type_counts[seq_type] += 1
                        camera_counts[camera] += 1
                        type_camera_counts[(seq_type, camera)] += 1

                subset = [
                    seq for seq in seqs_info_list
                    if sequence_key(seq) in selected_keys]
                expected_size = num_pids * sequences_per_pid
                if len(subset) != expected_size:
                    raise RuntimeError(
                        'CCPG balanced subset produced {} sequences; expected '
                        '{}.'.format(len(subset), expected_size))
                if any(count != sequences_per_pid
                       for count in selected_counts_by_pid.values()):
                    raise RuntimeError(
                        'CCPG balanced subset did not retain exactly {} '
                        'sequences per identity.'.format(sequences_per_pid))

                # Every selected probe must retain a same-identity gallery at
                # a different camera under each of the four CCPG conditions.
                protocol_pairs = [
                    (['U0_D0_BG', 'U0_D0'],
                     ['U1_D1', 'U2_D2', 'U3_D3']),
                    (['U3_D3'], ['U0_D3']),
                    (['U1_D0'], ['U1_D1']),
                    (['U0_D0_BG'], ['U0_D0']),
                ]
                selected_per_pid = {}
                for seq in subset:
                    selected_per_pid.setdefault(seq[0], []).append(seq)
                for pid, pid_seqs in selected_per_pid.items():
                    for probe_types, gallery_types in protocol_pairs:
                        gallery_cameras = {
                            normalized_camera(str(seq[2]))
                            for seq in pid_seqs if seq[1] in gallery_types}
                        probes = [
                            seq for seq in pid_seqs
                            if seq[1] in probe_types]
                        if not probes or not gallery_cameras:
                            raise RuntimeError(
                                'CCPG subset protocol coverage is incomplete '
                                'for identity {}.'.format(pid))
                        for probe in probes:
                            probe_camera = normalized_camera(str(probe[2]))
                            if not any(camera != probe_camera
                                       for camera in gallery_cameras):
                                raise RuntimeError(
                                    'CCPG subset has no cross-camera gallery '
                                    'for identity {} probe {}.'.format(
                                        pid, sequence_key(probe)))

                msg_mgr.log_info(
                    '-------- CCPG Protocol-Balanced Eval Subset --------')
                msg_mgr.log_info(
                    'Selected {} identities x {} sequences = {} sequences '
                    'with seed {}.'.format(
                        num_pids, sequences_per_pid, len(subset), seed))
                msg_mgr.log_info(
                    'Protocol type counts: {}'.format(type_counts))
                msg_mgr.log_info(
                    'Camera counts: {}'.format(camera_counts))
                log_pid_list(selected_pids)
                return subset

            if strategy != 'ccgr_gallery_view':
                raise ValueError(
                    'Unknown eval_subset strategy: {}'.format(strategy))

            gallery_set = set(partition.get('GALLERY_SET', []))
            if not gallery_set:
                raise ValueError(
                    'eval_subset requires GALLERY_SET in dataset_partition.')

            seed = int(subset_cfg.get('seed', 7890))
            num_pids = int(subset_cfg.get('num_pids', 150))
            probes_per_pid = int(subset_cfg.get('probes_per_pid', 5))
            gallery_per_pid = int(subset_cfg.get('gallery_per_pid', 2))
            min_sequences_per_pid = int(
                subset_cfg.get('min_sequences_per_pid', 48))
            probe_views = [str(view) for view in subset_cfg.get(
                'probe_views',
                ['0', '22_5', '45', '67_5', '90',
                 '112_5', '135', '157_5', '180', 'over'])]
            allow_multiple_probes_per_view = bool(subset_cfg.get(
                'allow_multiple_probes_per_view', False))
            if num_pids <= 0 or probes_per_pid <= 0 or gallery_per_pid <= 0:
                raise ValueError(
                    'eval_subset counts must all be positive integers.')
            if (not allow_multiple_probes_per_view
                    and len(probe_views) < probes_per_pid):
                raise ValueError(
                    'eval_subset.probe_views must contain at least as many '
                    'views as probes_per_pid.')

            def sequence_key(seq_info):
                return '-'.join(seq_info[:3])

            def normalized_probe_view(raw_view):
                for view in sorted(probe_views, key=len, reverse=True):
                    if raw_view == view or raw_view == view + '.avi':
                        return view
                    if raw_view.startswith(view + '_'):
                        return view
                return None

            per_pid = {}
            for seq_info in seqs_info_list:
                per_pid.setdefault(seq_info[0], []).append(seq_info)

            eligible_pids = []
            for pid, pid_seqs in per_pid.items():
                pid_gallery = [
                    seq for seq in pid_seqs
                    if sequence_key(seq) in gallery_set]
                pid_gallery_keys = {
                    sequence_key(seq) for seq in pid_gallery}
                available_probe_views = {
                    normalized_probe_view(seq[2])
                    for seq in pid_seqs
                    if sequence_key(seq) not in pid_gallery_keys}
                available_probe_count = sum(
                    normalized_probe_view(seq[2]) is not None
                    for seq in pid_seqs
                    if sequence_key(seq) not in pid_gallery_keys)
                has_all_probe_views = all(
                    view in available_probe_views for view in probe_views)
                if (len(pid_seqs) >= min_sequences_per_pid
                        and len(pid_gallery) >= gallery_per_pid
                        and has_all_probe_views
                        and (not allow_multiple_probes_per_view
                             or available_probe_count >= probes_per_pid)):
                    eligible_pids.append(pid)
            eligible_pids = sorted(eligible_pids)
            if len(eligible_pids) < num_pids:
                raise ValueError(
                    'eval_subset requested {} identities, but only {} satisfy '
                    'the completeness/gallery requirements.'.format(
                        num_pids, len(eligible_pids)))

            pid_rng = random.Random(seed)
            pid_rng.shuffle(eligible_pids)
            selected_pids = sorted(eligible_pids[:num_pids])
            selected_keys = set()
            selected_probe_view_counts = {view: 0 for view in probe_views}

            # Rotate requested views across identities for global view balance.
            # The legacy path selects at most one probe from each view.  The
            # opt-in repeated-view path supports larger subsets while still
            # selecting distinct sequences and retaining every requested view.
            view_stride = max(1, len(probe_views) // probes_per_pid)
            for pid_index, pid in enumerate(selected_pids):
                pid_seqs = per_pid[pid]
                pid_gallery = sorted(
                    [seq for seq in pid_seqs
                     if sequence_key(seq) in gallery_set],
                    key=sequence_key)
                chosen_gallery = pid_gallery[:gallery_per_pid]
                selected_keys.update(sequence_key(seq) for seq in chosen_gallery)

                gallery_keys = {sequence_key(seq) for seq in pid_gallery}
                probe_by_view = {view: [] for view in probe_views}
                for seq in pid_seqs:
                    if sequence_key(seq) in gallery_keys:
                        continue
                    view = normalized_probe_view(seq[2])
                    if view is not None:
                        probe_by_view[view].append(seq)

                if allow_multiple_probes_per_view:
                    # Shuffle candidates deterministically within each view,
                    # then consume them in a rotating round-robin order.  If a
                    # view is exhausted, move to the next available view.
                    for view, candidates in probe_by_view.items():
                        candidates.sort(key=sequence_key)
                        random.Random('{}:{}:{}'.format(
                            seed, pid, view)).shuffle(candidates)
                    view_cursors = {view: 0 for view in probe_views}
                    chosen_probe_count = 0
                    while chosen_probe_count < probes_per_pid:
                        preferred = (pid_index + chosen_probe_count) % len(probe_views)
                        chosen_view = None
                        for view_offset in range(len(probe_views)):
                            view = probe_views[
                                (preferred + view_offset) % len(probe_views)]
                            if view_cursors[view] < len(probe_by_view[view]):
                                chosen_view = view
                                break
                        if chosen_view is None:
                            raise ValueError(
                                'Identity {} has only {} usable non-gallery '
                                'probes; requested {}.'.format(
                                    pid, chosen_probe_count, probes_per_pid))
                        cursor = view_cursors[chosen_view]
                        chosen_probe = probe_by_view[chosen_view][cursor]
                        view_cursors[chosen_view] += 1
                        selected_keys.add(sequence_key(chosen_probe))
                        selected_probe_view_counts[chosen_view] += 1
                        chosen_probe_count += 1
                else:
                    target_views = [
                        probe_views[(pid_index + offset * view_stride)
                                    % len(probe_views)]
                        for offset in range(probes_per_pid)]
                    if len(set(target_views)) != probes_per_pid:
                        raise ValueError(
                            'eval_subset view rotation produced duplicate views; '
                            'adjust probe_views or probes_per_pid.')
                    for view in target_views:
                        candidates = sorted(
                            probe_by_view.get(view, []), key=sequence_key)
                        if not candidates:
                            raise ValueError(
                                'Identity {} has no non-gallery probe for requested '
                                'view {}.'.format(pid, view))
                        choice_rng = random.Random('{}:{}:{}'.format(
                            seed, pid, view))
                        chosen_probe = candidates[
                            choice_rng.randrange(len(candidates))]
                        selected_keys.add(sequence_key(chosen_probe))
                        selected_probe_view_counts[view] += 1

            subset = [
                seq for seq in seqs_info_list
                if sequence_key(seq) in selected_keys]
            expected_size = num_pids * (gallery_per_pid + probes_per_pid)
            if len(subset) != expected_size:
                raise RuntimeError(
                    'Balanced eval subset produced {} sequences; expected {}.'
                    .format(len(subset), expected_size))

            msg_mgr.log_info('-------- Balanced Eval Subset --------')
            msg_mgr.log_info(
                'Selected {} identities x ({} gallery + {} probe) = {} '
                'sequences with seed {}.'.format(
                    num_pids, gallery_per_pid, probes_per_pid,
                    len(subset), seed))
            msg_mgr.log_info(
                'Probe view counts: {}'.format(selected_probe_view_counts))
            log_pid_list(selected_pids)
            return subset

        if training:
            self.seqs_info = get_seqs_info_list(train_set)
        else:
            self.seqs_info = build_balanced_eval_subset(
                get_seqs_info_list(test_set))
