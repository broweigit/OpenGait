import os
from time import strftime, localtime
import numpy as np
from utils import get_msg_mgr, mkdir

from .metric import mean_iou, cuda_dist, compute_ACC_mAP, evaluate_rank, evaluate_many
from .re_rank import re_ranking

def de_diag(acc, each_angle=False):
    # Exclude identical-view cases
    dividend = acc.shape[1] - 1.
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / dividend
    if not each_angle:
        result = np.mean(result)
    return result


def cross_view_gallery_evaluation(feature, label, seq_type, view, dataset, metric):
    '''More details can be found: More details can be found in 
        [A Comprehensive Study on the Evaluation of Silhouette-based Gait Recognition](https://ieeexplore.ieee.org/document/9928336).
    '''
    probe_seq_dict = {'CASIA-B': {'NM': ['nm-01'], 'BG': ['bg-01'], 'CL': ['cl-01']},
                      'OUMVLP': {'NM': ['00']}}

    gallery_seq_dict = {'CASIA-B': ['nm-02', 'bg-02', 'cl-02'],
                        'OUMVLP': ['01']}

    msg_mgr = get_msg_mgr()
    acc = {}
    mean_ap = {}
    view_list = sorted(np.unique(view))
    for (type_, probe_seq) in probe_seq_dict[dataset].items():
        acc[type_] = np.zeros(len(view_list)) - 1.
        mean_ap[type_] = np.zeros(len(view_list)) - 1.
        for (v1, probe_view) in enumerate(view_list):
            pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                view, probe_view)
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]
            gseq_mask = np.isin(seq_type, gallery_seq_dict[dataset])
            gallery_y = label[gseq_mask]
            gallery_x = feature[gseq_mask, :]
            dist = cuda_dist(probe_x, gallery_x, metric)
            eval_results = compute_ACC_mAP(
                dist.cpu().numpy(), probe_y, gallery_y, view[pseq_mask], view[gseq_mask])
            acc[type_][v1] = np.round(eval_results[0] * 100, 2)
            mean_ap[type_][v1] = np.round(eval_results[1] * 100, 2)

    result_dict = {}
    msg_mgr.log_info(
        '===Cross View Gallery Evaluation (Excluded identical-view cases)===')
    out_acc_str = "========= Rank@1 Acc =========\n"
    out_map_str = "============= mAP ============\n"
    for type_ in probe_seq_dict[dataset].keys():
        avg_acc = np.mean(acc[type_])
        avg_map = np.mean(mean_ap[type_])
        result_dict[f'scalar/test_accuracy/{type_}-Rank@1'] = avg_acc
        result_dict[f'scalar/test_accuracy/{type_}-mAP'] = avg_map
        out_acc_str += f"{type_}:\t{acc[type_]}, mean: {avg_acc:.2f}%\n"
        out_map_str += f"{type_}:\t{mean_ap[type_]}, mean: {avg_map:.2f}%\n"
    # msg_mgr.log_info(f'========= Rank@1 Acc =========')
    msg_mgr.log_info(f'{out_acc_str}')
    # msg_mgr.log_info(f'========= mAP =========')
    msg_mgr.log_info(f'{out_map_str}')
    return result_dict

# Modified From https://github.com/AbnerHqC/GaitSet/blob/master/model/utils/evaluator.py


def single_view_gallery_evaluation(feature, label, seq_type, view, dataset, metric):
    probe_seq_dict = {'CASIA-B': {'NM': ['nm-05', 'nm-06'], 'BG': ['bg-01', 'bg-02'], 'CL': ['cl-01', 'cl-02']},
                      'OUMVLP': {'NM': ['00']},
                      'CASIA-E': {'NM': ['H-scene2-nm-1', 'H-scene2-nm-2', 'L-scene2-nm-1', 'L-scene2-nm-2', 'H-scene3-nm-1', 'H-scene3-nm-2', 'L-scene3-nm-1', 'L-scene3-nm-2', 'H-scene3_s-nm-1', 'H-scene3_s-nm-2', 'L-scene3_s-nm-1', 'L-scene3_s-nm-2', ],
                                  'BG': ['H-scene2-bg-1', 'H-scene2-bg-2', 'L-scene2-bg-1', 'L-scene2-bg-2', 'H-scene3-bg-1', 'H-scene3-bg-2', 'L-scene3-bg-1', 'L-scene3-bg-2', 'H-scene3_s-bg-1', 'H-scene3_s-bg-2', 'L-scene3_s-bg-1', 'L-scene3_s-bg-2'],
                                  'CL': ['H-scene2-cl-1', 'H-scene2-cl-2', 'L-scene2-cl-1', 'L-scene2-cl-2', 'H-scene3-cl-1', 'H-scene3-cl-2', 'L-scene3-cl-1', 'L-scene3-cl-2', 'H-scene3_s-cl-1', 'H-scene3_s-cl-2', 'L-scene3_s-cl-1', 'L-scene3_s-cl-2']
                                  },
                      'SUSTech1K': {'Normal': ['01-nm'], 'Bag': ['bg'], 'Clothing': ['cl'], 'Carrying':['cr'], 'Umberalla': ['ub'], 'Uniform': ['uf'], 'Occlusion': ['oc'],'Night': ['nt'], 'Overall': ['01','02','03','04']}
                      }
    gallery_seq_dict = {'CASIA-B': ['nm-01', 'nm-02', 'nm-03', 'nm-04'],
                        'OUMVLP': ['01'],
                        'CASIA-E': ['H-scene1-nm-1', 'H-scene1-nm-2', 'L-scene1-nm-1', 'L-scene1-nm-2'],
                        'SUSTech1K': ['00-nm'],}
    msg_mgr = get_msg_mgr()
    acc = {}
    view_list = sorted(np.unique(view))
    num_rank = 1
    if dataset == 'CASIA-E':
        view_list.remove("270")
    if dataset == 'SUSTech1K':
        num_rank = 5 
    view_num = len(view_list)

    for (type_, probe_seq) in probe_seq_dict[dataset].items():
        acc[type_] = np.zeros((view_num, view_num, num_rank)) - 1.
        for (v1, probe_view) in enumerate(view_list):
            pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                view, probe_view)
            pseq_mask = pseq_mask if 'SUSTech1K' not in dataset   else np.any(np.asarray(
                        [np.char.find(seq_type, probe)>=0 for probe in probe_seq]), axis=0
                            ) & np.isin(view, probe_view) # For SUSTech1K only
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]

            for (v2, gallery_view) in enumerate(view_list):
                gseq_mask = np.isin(seq_type, gallery_seq_dict[dataset]) & np.isin(
                    view, [gallery_view])
                gseq_mask = gseq_mask if 'SUSTech1K' not in dataset  else np.any(np.asarray(
                            [np.char.find(seq_type, gallery)>=0 for gallery in gallery_seq_dict[dataset]]), axis=0
                                ) & np.isin(view, [gallery_view]) # For SUSTech1K only
                gallery_y = label[gseq_mask]
                gallery_x = feature[gseq_mask, :]
                dist = cuda_dist(probe_x, gallery_x, metric)
                idx = dist.topk(num_rank, largest=False)[1].cpu().numpy()
                acc[type_][v1, v2, :] = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                                                     0) * 100 / dist.shape[0], 2)

    result_dict = {}
    msg_mgr.log_info('===Rank-1 (Exclude identical-view cases)===')
    out_str = ""
    for rank in range(num_rank):
        out_str = ""
        for type_ in probe_seq_dict[dataset].keys():
            sub_acc = de_diag(acc[type_][:,:,rank], each_angle=True)
            if rank == 0:
                msg_mgr.log_info(f'{type_}@R{rank+1}: {sub_acc}')
                result_dict[f'scalar/test_accuracy/{type_}@R{rank+1}'] = np.mean(sub_acc)
            out_str += f"{type_}@R{rank+1}: {np.mean(sub_acc):.2f}%\t"
        msg_mgr.log_info(out_str)
    return result_dict


def evaluate_indoor_dataset(data, dataset, metric='euc', cross_view_gallery=False):
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view = np.array(view)

    if dataset not in ('CASIA-B', 'OUMVLP', 'CASIA-E', 'SUSTech1K'):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    if cross_view_gallery:
        return cross_view_gallery_evaluation(
            feature, label, seq_type, view, dataset, metric)
    else:
        return single_view_gallery_evaluation(
            feature, label, seq_type, view, dataset, metric)


def evaluate_real_scene(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    feature, label, seq_type = data['embeddings'], data['labels'], data['types']
    label = np.array(label)

    gallery_seq_type = {'0001-1000': ['1', '2'],
                        "HID2021": ['0'], '0001-1000-test': ['0'],
                        'GREW': ['01'], 'TTG-200': ['1']}
    probe_seq_type = {'0001-1000': ['3', '4', '5', '6'],
                      "HID2021": ['1'], '0001-1000-test': ['1'],
                      'GREW': ['02'], 'TTG-200': ['2', '3', '4', '5', '6']}

    num_rank = 20
    acc = np.zeros([num_rank]) - 1.
    gseq_mask = np.isin(seq_type, gallery_seq_type[dataset])
    gallery_x = feature[gseq_mask, :]
    gallery_y = label[gseq_mask]
    pseq_mask = np.isin(seq_type, probe_seq_type[dataset])
    probe_x = feature[pseq_mask, :]
    probe_y = label[pseq_mask]

    dist = cuda_dist(probe_x, gallery_x, metric)
    idx = dist.topk(num_rank, largest=False)[1].cpu().numpy()
    acc = np.round(np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                          0) * 100 / dist.shape[0], 2)
    msg_mgr.log_info('==Rank-1==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[0])))
    msg_mgr.log_info('==Rank-5==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[4])))
    msg_mgr.log_info('==Rank-10==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[9])))
    msg_mgr.log_info('==Rank-20==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[19])))
    return {"scalar/test_accuracy/Rank-1": np.mean(acc[0]), "scalar/test_accuracy/Rank-5": np.mean(acc[4])}


def GREW_submission(data, dataset, metric='euc'):
    get_msg_mgr().log_info("Evaluating GREW")
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']
    label = np.array(label)
    view = np.array(view)
    gallery_seq_type = {'GREW': ['01', '02']}
    probe_seq_type = {'GREW': ['03']}
    gseq_mask = np.isin(seq_type, gallery_seq_type[dataset])
    gallery_x = feature[gseq_mask, :]
    gallery_y = label[gseq_mask]
    pseq_mask = np.isin(seq_type, probe_seq_type[dataset])
    probe_x = feature[pseq_mask, :]
    probe_y = view[pseq_mask]

    num_rank = 20
    dist = cuda_dist(probe_x, gallery_x, metric)
    idx = dist.topk(num_rank, largest=False)[1].cpu().numpy()

    save_path = os.path.join(
        "GREW_result/"+strftime('%Y-%m%d-%H%M%S', localtime())+".csv")
    mkdir("GREW_result")
    with open(save_path, "w") as f:
        f.write("videoId,rank1,rank2,rank3,rank4,rank5,rank6,rank7,rank8,rank9,rank10,rank11,rank12,rank13,rank14,rank15,rank16,rank17,rank18,rank19,rank20\n")
        for i in range(len(idx)):
            r_format = [int(idx) for idx in gallery_y[idx[i, 0:num_rank]]]
            output_row = '{}'+',{}'*num_rank+'\n'
            f.write(output_row.format(probe_y[i], *r_format))
        print("GREW result saved to {}/{}".format(os.getcwd(), save_path))
    return


def HID_submission(data, dataset, rerank=True, metric='euc'):
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info("Evaluating HID")
    feature, label, seq_type = data['embeddings'], data['labels'], data['views']
    label = np.array(label)
    seq_type = np.array(seq_type)
    probe_mask = (label == "probe")
    gallery_mask = (label != "probe")
    gallery_x = feature[gallery_mask, :]
    gallery_y = label[gallery_mask]
    probe_x = feature[probe_mask, :]
    probe_y = seq_type[probe_mask]
    if rerank:
        feat = np.concatenate([probe_x, gallery_x])
        dist = cuda_dist(feat, feat, metric).cpu().numpy()
        msg_mgr.log_info("Starting Re-ranking")
        re_rank = re_ranking(
            dist, probe_x.shape[0], k1=6, k2=6, lambda_value=0.3)
        idx = np.argsort(re_rank, axis=1)
    else:
        dist = cuda_dist(probe_x, gallery_x, metric)
        idx = dist.cpu().sort(1)[1].numpy()

    save_path = os.path.join(
        "HID_result/"+strftime('%Y-%m%d-%H%M%S', localtime())+".csv")
    mkdir("HID_result")
    with open(save_path, "w") as f:
        f.write("videoID,label\n")
        for i in range(len(idx)):
            f.write("{},{}\n".format(probe_y[i], gallery_y[idx[i, 0]]))
        print("HID result saved to {}/{}".format(os.getcwd(), save_path))
    return


def evaluate_segmentation(data, dataset):
    labels = data['mask']
    pred = data['pred']
    miou = mean_iou(pred, labels)
    get_msg_mgr().log_info('mIOU: %.3f' % (miou.mean()))
    return {"scalar/test_accuracy/mIOU": miou}


def evaluate_Gait3D(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()

    features, labels, cams, time_seqs = data['embeddings'], data['labels'], data['types'], data['views']
    import json
    probe_sets = json.load(
        open('./datasets/Gait3D/Gait3D.json', 'rb'))['PROBE_SET']
    probe_mask = []
    for id, ty, sq in zip(labels, cams, time_seqs):
        if '-'.join([id, ty, sq]) in probe_sets:
            probe_mask.append(True)
        else:
            probe_mask.append(False)
    probe_mask = np.array(probe_mask)

    # probe_features = features[:probe_num]
    probe_features = features[probe_mask]
    # gallery_features = features[probe_num:]
    gallery_features = features[~probe_mask]
    # probe_lbls = np.asarray(labels[:probe_num])
    # gallery_lbls = np.asarray(labels[probe_num:])
    probe_lbls = np.asarray(labels)[probe_mask]
    gallery_lbls = np.asarray(labels)[~probe_mask]

    results = {}
    msg_mgr.log_info(f"The test metric you choose is {metric}.")
    dist = cuda_dist(probe_features, gallery_features, metric).cpu().numpy()
    cmc, all_AP, all_INP = evaluate_rank(dist, probe_lbls, gallery_lbls)

    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    for r in [1, 5, 10]:
        results['scalar/test_accuracy/Rank-{}'.format(r)] = cmc[r - 1] * 100
    results['scalar/test_accuracy/mAP'] = mAP * 100
    results['scalar/test_accuracy/mINP'] = mINP * 100

    # print_csv_format(dataset_name, results)
    msg_mgr.log_info(results)
    return results


def evaluate_CCPG(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()

    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']

    label = np.array(label)
    for i in range(len(view)):
        view[i] = view[i].split("_")[0]
    view_np = np.array(view)
    view_list = list(set(view))
    view_list.sort()

    view_num = len(view_list)

    probe_seq_dict = {'CCPG': [["U0_D0_BG", "U0_D0"], [
        "U3_D3"], ["U1_D0"], ["U0_D0_BG"]]}

    gallery_seq_dict = {
        'CCPG': [["U1_D1", "U2_D2", "U3_D3"], ["U0_D3"], ["U1_D1"], ["U0_D0"]]}
    if dataset not in (probe_seq_dict or gallery_seq_dict):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    num_rank = 5
    acc = np.zeros([len(probe_seq_dict[dataset]),
                   view_num, view_num, num_rank]) - 1.

    ap_save = []
    cmc_save = []
    minp = []
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        # for gallery_seq in gallery_seq_dict[dataset]:
        gallery_seq = gallery_seq_dict[dataset][p]
        gseq_mask = np.isin(seq_type, gallery_seq)
        gallery_x = feature[gseq_mask, :]
        # print("gallery_x", gallery_x.shape)
        gallery_y = label[gseq_mask]
        gallery_view = view_np[gseq_mask]

        pseq_mask = np.isin(seq_type, probe_seq)
        probe_x = feature[pseq_mask, :]
        probe_y = label[pseq_mask]
        probe_view = view_np[pseq_mask]

        msg_mgr.log_info(
            ("gallery length", len(gallery_y), gallery_seq, "probe length", len(probe_y), probe_seq))
        distmat = cuda_dist(probe_x, gallery_x, metric).cpu().numpy()
        # cmc, ap = evaluate(distmat, probe_y, gallery_y, probe_view, gallery_view)
        cmc, ap, inp = evaluate_many(
            distmat, probe_y, gallery_y, probe_view, gallery_view)
        ap_save.append(ap)
        cmc_save.append(cmc[0])
        minp.append(inp)

    # print(ap_save, cmc_save)

    msg_mgr.log_info(
        '===Rank-1 (Exclude identical-view cases for Person Re-Identification)===')
    msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' % (
        cmc_save[0]*100, cmc_save[1]*100, cmc_save[2]*100, cmc_save[3]*100))

    msg_mgr.log_info(
        '===mAP (Exclude identical-view cases for Person Re-Identification)===')
    msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' % (
        ap_save[0]*100, ap_save[1]*100, ap_save[2]*100, ap_save[3]*100))

    msg_mgr.log_info(
        '===mINP (Exclude identical-view cases for Person Re-Identification)===')
    msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' %
                     (minp[0]*100, minp[1]*100, minp[2]*100, minp[3]*100))

    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        # for gallery_seq in gallery_seq_dict[dataset]:
        gallery_seq = gallery_seq_dict[dataset][p]
        for (v1, probe_view) in enumerate(view_list):
            for (v2, gallery_view) in enumerate(view_list):
                gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                    view, [gallery_view])
                gallery_x = feature[gseq_mask, :]
                gallery_y = label[gseq_mask]

                pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                    view, [probe_view])
                probe_x = feature[pseq_mask, :]
                probe_y = label[pseq_mask]

                dist = cuda_dist(probe_x, gallery_x, metric)
                idx = dist.sort(1)[1].cpu().numpy()
                # print(p, v1, v2, "\n")
                acc[p, v1, v2, :] = np.round(
                    np.sum(np.cumsum(np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]], 1) > 0,
                           0) * 100 / dist.shape[0], 2)
    result_dict = {}
    for i in range(1):
        msg_mgr.log_info(
            '===Rank-%d (Include identical-view cases)===' % (i + 1))
        msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' % (
            np.mean(acc[0, :, :, i]),
            np.mean(acc[1, :, :, i]),
            np.mean(acc[2, :, :, i]),
            np.mean(acc[3, :, :, i])))
    for i in range(1):
        msg_mgr.log_info(
            '===Rank-%d (Exclude identical-view cases)===' % (i + 1))
        msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' % (
            de_diag(acc[0, :, :, i]),
            de_diag(acc[1, :, :, i]),
            de_diag(acc[2, :, :, i]),
            de_diag(acc[3, :, :, i])))
    result_dict["scalar/test_accuracy/CL"] = acc[0, :, :, i]
    result_dict["scalar/test_accuracy/UP"] = acc[1, :, :, i]
    result_dict["scalar/test_accuracy/DN"] = acc[2, :, :, i]
    result_dict["scalar/test_accuracy/BG"] = acc[3, :, :, i]
    np.set_printoptions(precision=2, floatmode='fixed')
    for i in range(1):
        msg_mgr.log_info(
            '===Rank-%d of each angle (Exclude identical-view cases)===' % (i + 1))
        msg_mgr.log_info('CL: {}'.format(de_diag(acc[0, :, :, i], True)))
        msg_mgr.log_info('UP: {}'.format(de_diag(acc[1, :, :, i], True)))
        msg_mgr.log_info('DN: {}'.format(de_diag(acc[2, :, :, i], True)))
        msg_mgr.log_info('BG: {}'.format(de_diag(acc[3, :, :, i], True)))
    return result_dict

import torch
def evaluate_CCPG_part(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    
    # =======================================================
    # 0. 先跑一次标准评测 (以便对比 Baseline: 82.560%)
    # =======================================================
    msg_mgr.log_info("\n" + "#" * 60)
    msg_mgr.log_info(">>> Running Standard CCPG Evaluation First...")
    msg_mgr.log_info("#" * 60)
    try:
        evaluate_CCPG(data.copy(), dataset, metric)
    except Exception as e:
        msg_mgr.log_warning(f"Standard evaluate_CCPG failed: {e}")

    msg_mgr.log_info("\n" + "#" * 60)
    msg_mgr.log_info(">>> Running Part-based Evaluation (Strict Alignment: 3D Input)...")
    msg_mgr.log_info("#" * 60)

    # 1. 数据解包
    feature = data['embeddings']
    label = data['labels']
    seq_type = data['types']
    view = data['views']

    if isinstance(feature, torch.Tensor): feature = feature.detach().cpu().numpy()
    if isinstance(label, torch.Tensor): label = label.detach().cpu().numpy()
    else: label = np.array(label)

    # 4. 形状检查与动态配置 (合并到这里)
    try:
        if feature.ndim != 3:
            msg_mgr.log_info(f"Error: Feature shape {feature.shape} is not [N, C, P]!")
            return {}
        N, C, P = feature.shape
    except ValueError:
        return {}

    # ================= [结构配置区] =================
    TEST_PARTS = True
    NUM_FPN_HEADS = 4  
    # ===============================================

    # [关键修改]：动态计算每个 Head 的 Part 数量
    if P % NUM_FPN_HEADS != 0:
        msg_mgr.log_warning(f"Warning: Total parts {P} cannot be evenly divided by {NUM_FPN_HEADS} heads!")
    
    PARTS_PER_HEAD = P // NUM_FPN_HEADS  # 动态计算，例如 128->32, 240->60
    
    msg_mgr.log_info(f"Dynamic Config: Total Parts={P}, Heads={NUM_FPN_HEADS}, Parts/Head={PARTS_PER_HEAD}")

    # 2. View 处理
    if len(view) > 0 and "_" in view[0]:
        view = [v.split("_")[0] for v in view]
    
    view_np = np.array(view)
    view_list = sorted(list(set(view))) 
    view_num = len(view_list)
    
    # 3. 序列定义
    probe_seq_dict = {'CCPG': [["U0_D0_BG", "U0_D0"], ["U3_D3"], ["U1_D0"], ["U0_D0_BG"]]}
    gallery_seq_dict = {'CCPG': [["U1_D1", "U2_D2", "U3_D3"], ["U0_D3"], ["U1_D1"], ["U0_D0"]]}

    # === [关键修改 1] 动态 de_diag (不硬编码除以10) ===
    def de_diag(acc, each_angle=False):
        # 1. 标记有效数据 (非 -1)
        valid_mask = (acc != -1)
        # 2. 排除对角线
        diag_indices = np.diag_indices_from(acc)
        valid_mask[diag_indices] = False
        
        # 3. 计算分子和分母
        sum_acc = np.sum(acc * valid_mask, axis=1)
        count_valid = np.sum(valid_mask, axis=1)
        
        # 防止除以0
        count_valid[count_valid == 0] = 1.0
        result = sum_acc / count_valid
        
        if not each_angle:
            # 同样只对有有效数据的行求平均
            valid_rows = (count_valid > 0)
            if np.sum(valid_rows) > 0:
                result = np.mean(result[valid_rows])
            else:
                result = 0.0
        return result

    # === [关键修改 2] 评测核心 (支持 3D 输入，不 Flatten) ===
    def run_evaluation_core(feat_input, title_suffix=""):
        # feat_input: 期望是 [N, C, P]
        
        # 如果是单 Part [N, C]，需要增加维度变成 [N, C, 1] 以适配 cuda_dist
        if feat_input.ndim == 2:
            feat_input = feat_input[:, :, np.newaxis]

        final_results = []

        for p, probe_seq in enumerate(probe_seq_dict[dataset]):
            gallery_seq = gallery_seq_dict[dataset][p]
            
            g_seq_mask = np.isin(seq_type, gallery_seq)
            p_seq_mask = np.isin(seq_type, probe_seq)
            
            if np.sum(g_seq_mask) == 0 or np.sum(p_seq_mask) == 0:
                final_results.append(0.0)
                continue

            feat_p = feat_input[p_seq_mask]
            feat_g = feat_input[g_seq_mask]
            
            # [关键点] 传入 3D Tensor，cuda_dist 计算 Sum(L2)，与 Standard 对齐
            distmat = cuda_dist(feat_p, feat_g, metric).cpu().numpy()

            view_p_sub = view_np[p_seq_mask]
            view_g_sub = view_np[g_seq_mask]
            label_p_sub = label[p_seq_mask]
            label_g_sub = label[g_seq_mask]

            # [关键修改 3] 初始化为 -1
            acc_matrix = np.full((view_num, view_num), -1.0)

            for v1, probe_view in enumerate(view_list):
                for v2, gallery_view in enumerate(view_list):
                    p_idx = np.where(view_p_sub == probe_view)[0]
                    g_idx = np.where(view_g_sub == gallery_view)[0]
                    
                    if len(p_idx) == 0 or len(g_idx) == 0:
                        continue 

                    dist_subset = distmat[np.ix_(p_idx, g_idx)]
                    
                    sorted_indices = np.argsort(dist_subset, axis=1)
                    rank1_idx = sorted_indices[:, 0]
                    
                    pred_labels = label_g_sub[g_idx][rank1_idx]
                    gt_labels = label_p_sub[p_idx]
                    
                    correct = (pred_labels == gt_labels)
                    acc_val = np.mean(correct) * 100
                    acc_matrix[v1, v2] = acc_val

            avg_acc = de_diag(acc_matrix)
            final_results.append(avg_acc)

        r1_str = ', '.join([f'{x:.1f}' for x in final_results])
        msg_mgr.log_info(f'{title_suffix:<25} | Rank-1: {r1_str}')

    # 5. 执行评估

    # (A) Full Feature [N, C, 128] -> 3D Input -> Sum Logic
    msg_mgr.log_info("\n" + "="*40)
    msg_mgr.log_info("1. Full Feature Evaluation (Sum of Parts)")
    msg_mgr.log_info("="*40)
    # [FIX] 不要 reshape! 直接传 3D Tensor
    run_evaluation_core(feature, title_suffix="[ALL Combined]")

    # (B) FPN Head & Part Evaluation
    if TEST_PARTS:
        msg_mgr.log_info("\n" + "="*40)
        msg_mgr.log_info(f"2. FPN Branch Evaluation")
        msg_mgr.log_info("="*40)

        # 硬编码：单个 Branch 在一个 FPN Head 里的 Part 数量
        PARTS_PER_SINGLE_BRANCH = 16
        # 自动计算当前模型总共拼接了几个 Branch
        num_branches = PARTS_PER_HEAD // PARTS_PER_SINGLE_BRANCH
        msg_mgr.log_info(f"Detected {num_branches} branches per Head (Parts per branch: {PARTS_PER_SINGLE_BRANCH})")
        
        for head_idx in range(NUM_FPN_HEADS):
            msg_mgr.log_info(f"\n>>> [FPN Head {head_idx}]")
            
            start = head_idx * PARTS_PER_HEAD
            end = start + PARTS_PER_HEAD
            
            # [FIX] Head 整体 [N, C, 32] -> 3D Input -> Sum Logic
            # 同样不要 flatten
            head_feat_chunk = feature[:, :, start:end] 
            run_evaluation_core(head_feat_chunk, title_suffix=f"[Head-{head_idx} Full]")

            # 2. 分别评估每一个 Branch (子组的 Full 评估)
            if num_branches > 1:
                for b_idx in range(num_branches):
                    b_start = start + (b_idx * PARTS_PER_SINGLE_BRANCH)
                    b_end = b_start + PARTS_PER_SINGLE_BRANCH
                    branch_feat_chunk = feature[:, :, b_start:b_end]
                    run_evaluation_core(branch_feat_chunk, title_suffix=f"  *[Branch-{b_idx} Full]")

            # 3. 细粒度到单个 Part 
            for part_idx in range(PARTS_PER_HEAD):
                abs_idx = start + part_idx
                part_feat = feature[:, :, abs_idx : abs_idx+1]
                
                # 为了可读性，如果是多 branch，可以在标题里标明属于哪个 branch
                b_id = part_idx // PARTS_PER_SINGLE_BRANCH
                inner_p_id = part_idx % PARTS_PER_SINGLE_BRANCH
                if num_branches > 1:
                    title = f"    - Branch{b_id} Part {inner_p_id:02d}"
                else:
                    title = f"    - Part {part_idx:02d}"
                    
                run_evaluation_core(part_feat, title_suffix=title)

    return {}

def evaluate_CCPG_part_FPN6(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    
    # =======================================================
    # 0. 先跑一次标准评测 (以便对比 Baseline: 82.560%)
    # =======================================================
    msg_mgr.log_info("\n" + "#" * 60)
    msg_mgr.log_info(">>> Running Standard CCPG Evaluation First...")
    msg_mgr.log_info("#" * 60)
    try:
        evaluate_CCPG(data.copy(), dataset, metric)
    except Exception as e:
        msg_mgr.log_warning(f"Standard evaluate_CCPG failed: {e}")

    msg_mgr.log_info("\n" + "#" * 60)
    msg_mgr.log_info(">>> Running Part-based Evaluation (Strict Alignment: 3D Input)...")
    msg_mgr.log_info("#" * 60)

    # 1. 数据解包
    feature = data['embeddings']
    label = data['labels']
    seq_type = data['types']
    view = data['views']

    if isinstance(feature, torch.Tensor): feature = feature.detach().cpu().numpy()
    if isinstance(label, torch.Tensor): label = label.detach().cpu().numpy()
    else: label = np.array(label)

    # 4. 形状检查与动态配置 (合并到这里)
    try:
        if feature.ndim != 3:
            msg_mgr.log_info(f"Error: Feature shape {feature.shape} is not [N, C, P]!")
            return {}
        N, C, P = feature.shape
    except ValueError:
        return {}

    # ================= [结构配置区] =================
    TEST_PARTS = True
    NUM_FPN_HEADS = 6  
    # ===============================================

    # [关键修改]：动态计算每个 Head 的 Part 数量
    if P % NUM_FPN_HEADS != 0:
        msg_mgr.log_warning(f"Warning: Total parts {P} cannot be evenly divided by {NUM_FPN_HEADS} heads!")
    
    PARTS_PER_HEAD = P // NUM_FPN_HEADS  # 动态计算，例如 128->32, 240->60
    
    msg_mgr.log_info(f"Dynamic Config: Total Parts={P}, Heads={NUM_FPN_HEADS}, Parts/Head={PARTS_PER_HEAD}")

    # 2. View 处理
    if len(view) > 0 and "_" in view[0]:
        view = [v.split("_")[0] for v in view]
    
    view_np = np.array(view)
    view_list = sorted(list(set(view))) 
    view_num = len(view_list)
    
    # 3. 序列定义
    probe_seq_dict = {'CCPG': [["U0_D0_BG", "U0_D0"], ["U3_D3"], ["U1_D0"], ["U0_D0_BG"]]}
    gallery_seq_dict = {'CCPG': [["U1_D1", "U2_D2", "U3_D3"], ["U0_D3"], ["U1_D1"], ["U0_D0"]]}

    # === [关键修改 1] 动态 de_diag (不硬编码除以10) ===
    def de_diag(acc, each_angle=False):
        # 1. 标记有效数据 (非 -1)
        valid_mask = (acc != -1)
        # 2. 排除对角线
        diag_indices = np.diag_indices_from(acc)
        valid_mask[diag_indices] = False
        
        # 3. 计算分子和分母
        sum_acc = np.sum(acc * valid_mask, axis=1)
        count_valid = np.sum(valid_mask, axis=1)
        
        # 防止除以0
        count_valid[count_valid == 0] = 1.0
        result = sum_acc / count_valid
        
        if not each_angle:
            # 同样只对有有效数据的行求平均
            valid_rows = (count_valid > 0)
            if np.sum(valid_rows) > 0:
                result = np.mean(result[valid_rows])
            else:
                result = 0.0
        return result

    # === [关键修改 2] 评测核心 (支持 3D 输入，不 Flatten) ===
    def run_evaluation_core(feat_input, title_suffix=""):
        # feat_input: 期望是 [N, C, P]
        
        # 如果是单 Part [N, C]，需要增加维度变成 [N, C, 1] 以适配 cuda_dist
        if feat_input.ndim == 2:
            feat_input = feat_input[:, :, np.newaxis]

        final_results = []

        for p, probe_seq in enumerate(probe_seq_dict[dataset]):
            gallery_seq = gallery_seq_dict[dataset][p]
            
            g_seq_mask = np.isin(seq_type, gallery_seq)
            p_seq_mask = np.isin(seq_type, probe_seq)
            
            if np.sum(g_seq_mask) == 0 or np.sum(p_seq_mask) == 0:
                final_results.append(0.0)
                continue

            feat_p = feat_input[p_seq_mask]
            feat_g = feat_input[g_seq_mask]
            
            # [关键点] 传入 3D Tensor，cuda_dist 计算 Sum(L2)，与 Standard 对齐
            distmat = cuda_dist(feat_p, feat_g, metric).cpu().numpy()

            view_p_sub = view_np[p_seq_mask]
            view_g_sub = view_np[g_seq_mask]
            label_p_sub = label[p_seq_mask]
            label_g_sub = label[g_seq_mask]

            # [关键修改 3] 初始化为 -1
            acc_matrix = np.full((view_num, view_num), -1.0)

            for v1, probe_view in enumerate(view_list):
                for v2, gallery_view in enumerate(view_list):
                    p_idx = np.where(view_p_sub == probe_view)[0]
                    g_idx = np.where(view_g_sub == gallery_view)[0]
                    
                    if len(p_idx) == 0 or len(g_idx) == 0:
                        continue 

                    dist_subset = distmat[np.ix_(p_idx, g_idx)]
                    
                    sorted_indices = np.argsort(dist_subset, axis=1)
                    rank1_idx = sorted_indices[:, 0]
                    
                    pred_labels = label_g_sub[g_idx][rank1_idx]
                    gt_labels = label_p_sub[p_idx]
                    
                    correct = (pred_labels == gt_labels)
                    acc_val = np.mean(correct) * 100
                    acc_matrix[v1, v2] = acc_val

            avg_acc = de_diag(acc_matrix)
            final_results.append(avg_acc)

        r1_str = ', '.join([f'{x:.1f}' for x in final_results])
        msg_mgr.log_info(f'{title_suffix:<25} | Rank-1: {r1_str}')

    # 5. 执行评估

    # (A) Full Feature [N, C, 128] -> 3D Input -> Sum Logic
    msg_mgr.log_info("\n" + "="*40)
    msg_mgr.log_info("1. Full Feature Evaluation (Sum of Parts)")
    msg_mgr.log_info("="*40)
    # [FIX] 不要 reshape! 直接传 3D Tensor
    run_evaluation_core(feature, title_suffix="[ALL Combined]")

    # (B) FPN Head & Part Evaluation
    if TEST_PARTS:
        msg_mgr.log_info("\n" + "="*40)
        msg_mgr.log_info(f"2. FPN Branch Evaluation")
        msg_mgr.log_info("="*40)
        
        for head_idx in range(NUM_FPN_HEADS):
            msg_mgr.log_info(f"\n>>> [FPN Head {head_idx}]")
            
            start = head_idx * PARTS_PER_HEAD
            end = start + PARTS_PER_HEAD
            
            # [FIX] Head 整体 [N, C, 32] -> 3D Input -> Sum Logic
            # 同样不要 flatten
            head_feat_chunk = feature[:, :, start:end] 
            run_evaluation_core(head_feat_chunk, title_suffix=f"[Head-{head_idx} Full]")

            # 单个 Part [N, C, 1]
            for part_idx in range(PARTS_PER_HEAD):
                abs_idx = start + part_idx
                # 切片保持 3D 维度
                part_feat = feature[:, :, abs_idx : abs_idx+1]
                run_evaluation_core(part_feat, title_suffix=f"  - Part {part_idx:02d}")

    return {}

import pandas as pd
def evaluate_simple_split(data, dataset, metric='euc'):
    """
    一个简单的评估器，忽略序列类型和视角，仅根据人员ID进行评估。
    对于每一个ID，将其所有样本数据随机均分为两半，分别作为 Probe 和 Gallery。

    Args:
        data (dict): 包含 'embeddings', 'labels' 等信息的字典。
        dataset (str): 数据集名称 (此函数中未使用，为保持接口一致性)。
        metric (str, optional): 使用的距离度量。默认为 'euc' (欧氏距离)。

    Returns:
        dict: 包含 Rank-1, Rank-5, Rank-10, mAP, mINP 的评估结果字典。
    """
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info("--- Starting Evaluation with Simple 50/50 Split per ID ---")

    # 1. 从输入数据中提取特征和标签(人员ID)
    features = data['embeddings']
    pids = np.array(data['labels'])
    
    # 2. 使用 Pandas DataFrame 方便地按ID分组和采样
    # 创建一个包含人员ID和其在原始特征数组中索引的DataFrame
    df = pd.DataFrame({
        'pid': pids,
        'original_index': np.arange(len(pids))
    })

    probe_indices = []
    gallery_indices = []

    # 3. 对每个ID进行数据划分
    # a. 按 'pid' 分组
    for pid, group in df.groupby('pid'):
        # b. 获取该ID下所有样本的原始索引
        pid_indices = group['original_index'].values
        
        # c. 如果样本数少于2，则无法划分，全部放入gallery以保证gallery不为空
        if len(pid_indices) < 2:
            gallery_indices.extend(pid_indices)
            continue
            
        # d. 随机打乱索引
        np.random.shuffle(pid_indices)
        
        # e. 找到分割点，进行均分
        split_point = len(pid_indices) // 2
        
        # f. 前一半作为 gallery，后一半作为 probe
        gallery_indices.extend(pid_indices[:split_point])
        probe_indices.extend(pid_indices[split_point:])

    msg_mgr.log_info(f"Total samples processed: {len(pids)}")
    msg_mgr.log_info(f"Gallery set size: {len(gallery_indices)}")
    msg_mgr.log_info(f"Probe set size: {len(probe_indices)}")

    if not probe_indices or not gallery_indices:
        msg_mgr.log_error("Probe or Gallery set is empty. Evaluation cannot proceed.")
        return {}
        
    # 4. 根据划分好的索引，构建 Probe 和 Gallery 集
    probe_x = features[probe_indices]
    probe_y = pids[probe_indices]
    
    gallery_x = features[gallery_indices]
    gallery_y = pids[gallery_indices]

    # 5. 计算距离矩阵 (复用您已有的函数)
    # 注意：cuda_dist 返回的是GPU张量，需要转到CPU并变为numpy数组
    dist = cuda_dist(probe_x, gallery_x, metric).cpu().numpy()

    # 6. 计算评估指标 (复用您已有的函数)
    cmc, all_AP, all_INP = evaluate_rank(dist, probe_y, gallery_y)

    # 7. 整理并打印结果
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    
    results = {}
    msg_mgr.log_info("--- Simple Split Evaluation Results ---")
    for r in [1, 5, 10]:
        rank_acc = cmc[r - 1] * 100
        results[f'scalar/test_accuracy/Rank-{r}'] = rank_acc
        msg_mgr.log_info(f"Rank-{r:<2}: {rank_acc:.2f}%")
        
    results['scalar/test_accuracy/mAP'] = mAP * 100
    results['scalar/test_accuracy/mINP'] = mINP * 100
    
    msg_mgr.log_info(f"mAP  : {mAP * 100:.2f}%")
    msg_mgr.log_info(f"mINP : {mINP * 100:.2f}%")
    msg_mgr.log_info("-" * 40)
    
    return results

def evaluate_scoliosis(data, dataset, metric='euc'):

    msg_mgr = get_msg_mgr()
    
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

    logits = np.array(data['embeddings'])
    labels = data['types']
    
    # Label mapping: negative->0, neutral->1, positive->2  
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    true_ids = np.array([label_map[status] for status in labels])
    
    pred_ids = np.argmax(logits.mean(-1), axis=-1)
    
    # Calculate evaluation metrics
    # Total Accuracy: proportion of correctly predicted samples among all samples
    accuracy = accuracy_score(true_ids, pred_ids)
    
    # Macro-average Precision: average of precision scores for each class
    precision = precision_score(true_ids, pred_ids, average='macro', zero_division=0)
    
    # Macro-average Recall: average of recall scores for each class  
    recall = recall_score(true_ids, pred_ids, average='macro', zero_division=0)
    
    # Macro-average F1: average of F1 scores for each class
    f1 = f1_score(true_ids, pred_ids, average='macro', zero_division=0)
    
    # Confusion matrix (for debugging)
    # cm = confusion_matrix(true_ids, pred_ids, labels=[0, 1, 2])
    # class_names = ['Negative', 'Neutral', 'Positive']
    
    # Print results
    msg_mgr.log_info(f"Total Accuracy: {accuracy*100:.2f}%")
    msg_mgr.log_info(f"Macro-avg Precision: {precision*100:.2f}%") 
    msg_mgr.log_info(f"Macro-avg Recall: {recall*100:.2f}%")
    msg_mgr.log_info(f"Macro-avg F1 Score: {f1*100:.2f}%")
    
    return {
        "scalar/test_accuracy/": accuracy,
        "scalar/test_precision/": precision, 
        "scalar/test_recall/": recall,
        "scalar/test_f1/": f1
    }

def evaluate_FreeGait(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()

    features, labels, cams, time_seqs = data['embeddings'], data['labels'], data['types'], data['views']
    import json
    probe_sets = json.load(
        open('./datasets/FreeGait/FreeGait.json', 'rb'))['PROBE_SET']
    
    probe_mask = []
    for id, ty, sq in zip(labels, cams, time_seqs):
        if '-'.join([id, ty, sq]) in probe_sets:
            probe_mask.append(True)
        else:
            probe_mask.append(False)
    probe_mask = np.array(probe_mask)

    # probe_features = features[:probe_num]
    probe_features = features[probe_mask]
    # gallery_features = features[probe_num:]
    gallery_features = features[~probe_mask]
    # probe_lbls = np.asarray(labels[:probe_num])
    # gallery_lbls = np.asarray(labels[probe_num:])
    probe_lbls = np.asarray(labels)[probe_mask]
    gallery_lbls = np.asarray(labels)[~probe_mask]

    results = {}
    msg_mgr.log_info(f"The test metric you choose is {metric}.")
    dist = cuda_dist(probe_features, gallery_features, metric).cpu().numpy()
    cmc, all_AP, all_INP = evaluate_rank(dist, probe_lbls, gallery_lbls)

    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    for r in [1, 5, 10]:
        results['scalar/test_accuracy/Rank-{}'.format(r)] = cmc[r - 1] * 100
    results['scalar/test_accuracy/mAP'] = mAP * 100
    results['scalar/test_accuracy/mINP'] = mINP * 100

    # print_csv_format(dataset_name, results)
    msg_mgr.log_info(results)
    return results



import copy
def evaluate_CCGR_MINIV2(data, dataset, metric='euc'):
    # name = 'CCGR_MINI__BiggerGait__SmallDINOv2_Gaitbase_84Frame30_448224_6432HPP32_NoAlign_Sep12B_WiMask.pkl'
    # pickle.dump(data, open(name, 'wb'))

    msg_mgr = get_msg_mgr()
    keys_with_embeddings = sorted([k for k in data.keys() if 'embeddings' in k])
    data_tmp = copy.deepcopy(data)
    for i in keys_with_embeddings:
        msg_mgr.log_info('========= %s =========' % i)
        data_tmp['embeddings'] = data[i]
        evaluate_CCGR_MINI(data_tmp, dataset, metric='euc')

def evaluate_CCGR_MINI(data, dataset, metric='euc'):
    assert 'CCGR' in dataset 
    msg_mgr = get_msg_mgr()
    features, labels, cams, time_seqs = data['embeddings'], data['labels'], data['types'], data['views']

    import json
    gallery_sets = json.load(
        open('./datasets/CCGR-MINI/CCGR-MINI.json', 'rb'))['GALLERY_SET']
    probe_mask = []
    for id, ty, sq in zip(labels, cams, time_seqs):
        if '-'.join([id, ty, sq]) in gallery_sets:
            probe_mask.append(False)
        else:
            probe_mask.append(True)
    probe_mask = np.array(probe_mask)
    probe_features = features[probe_mask]
    gallery_features = features[~probe_mask]
    probe_lbls = np.asarray(labels)[probe_mask]
    gallery_lbls = np.asarray(labels)[~probe_mask]

    results = {}
    msg_mgr.log_info(f"The test metric you choose is {metric}.")
    dist = cuda_dist(probe_features, gallery_features, metric).cpu().numpy()
    cmc, all_AP, all_INP = evaluate_rank(dist, probe_lbls, gallery_lbls)

    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    for r in [1, 5, 10]:
        results['scalar/test_accuracy/Rank-{}'.format(r)] = cmc[r - 1] * 100
    results['scalar/test_accuracy/mAP'] = mAP * 100
    results['scalar/test_accuracy/mINP'] = mINP * 100

    msg_mgr.log_info(results)
    return results