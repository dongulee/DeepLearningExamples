from src.katech import KATECHDetection
from src.utils import COCODetection
from argparse import ArgumentParser
import numpy as np

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def bbox_mul(bbox, width, height):
    return (bbox[0]*width, bbox[1]*height, bbox[2]*width, bbox[3]*height)


def eval_katech_result(inf, gt, categories, conf_thresholds=None, iou_threshold=0.5):
    ### evaluation
    # args:
    # gt[img_id] = (filename, (width, height), bbox_sizes, bbox_labels, fullpath) LTRB sizes
    # inf[img_id] = [(bbox, prob, label), ...] LTRB
    # categories: [cat_name: cat_id]

    if conf_thresholds==None:
        conf_thresholds = np.append(np.arange(0.1, 1, 0.1), 0.95)
    # Normalizae to ltrb, lbl_id
    gt_norm = {}
    for k, v in gt.items():
        bboxes_norm = [bbox_mul(bbox,v[1][0], v[1][1]) for bbox in v[2]]
        lbls_norm = [categories[lbl] for lbl in v[3]]
        assert len(bboxes_norm) == len(lbls_norm)
        gt_norm[k] = [(bboxes_norm[i], lbls_norm[i]) for i in range(len(bboxes_norm))]


    # initialize evaluation table
    eval_class = {} # class id: {'GT':0, pth: [TP FP FN], pth2: {...}}
    for cid in categories.values():
        eval_class[cid]={}
        eval_class[cid]['GT'] = 0
        for pth in conf_thresholds:
            eval_class[cid][pth] = [0, 0, 0]
    
    # evaluate for each image
    for img_id, inf_bbs in inf.items():
        # record ground truth information
        img_gt = gt_norm[img_id]
        for gtbox in img_gt: 
            eval_class[gtbox[1]]['GT'] += 1     # num of gt ojects for categories
   
        # low th -> high th: high recall -> low recall
        for p_th in conf_thresholds:
            # eval for each bbox
            for bbox in [bbox for bbox in inf_bbs if bbox[1] >= p_th]:
                
                lbl = bbox[2]
                cands = [get_iou(bbox[0], gtbbox[0]) for gtbbox in img_gt if gtbbox[1]==lbl]
                if len([iou for iou in cands if iou >= iou_threshold]) > 0:
                    # TP
                    eval_class[lbl][p_th][0] += 1
                else:
                    # FP
                    eval_class[lbl][p_th][1] += 1
            eval_class[lbl][p_th][2] += eval_class[lbl]['GT'] - eval_class[lbl][p_th][0]
        
            
        
    mAP_sum = 0
    for cid in categories.values():
        AP = 0
        PRs = []
        for p_th in conf_thresholds:
            TP = eval_class[cid][pth][0]
            FP = eval_class[cid][pth][1]
            FN = eval_class[cid][pth][2]
            precision = TP/(TP+FP)
            recall = TP/(TP+FN)
            PRs.append((precision, recall))
        PRs.sort(PRs, key=lambda x:x[0])
        PRs.sort(PRs, key=lambda x:x[1])
        r_prev = PRs[0][0]
        p_prev = PRs[0][1]
        dr = 0
        for (p, r) in PRs:
            dr += r-r_prev
            r_prev = r
            if p < p_prev:
                AP += p_prev * dr
            p_prev = p
        mAP_sum += AP
    mAP = mAP_sum / len(categories.keys())
    print(mAP)
                
            
        
        
    
def main(args):
    katech = KATECHDetection.load('katech_ckpt.pkl')
    # katech.load('katech_dataset.pkl')
    print(katech.img_folder)
    coco = COCODetection('/data/coco/coco_2017/', '/data/coco/coco_2017/annotations/instances_val2017.json')
    coco_lbl_map = coco.label_map
    coco_lbl_info = coco.label_info
    
    # idx_map = [katech.img_keys]

    results = np.load('katech_infer_test.npy')
    print("katech infer results are loaded")
    result_dict = {}
    for r in results:
        img_id = int(r[0])
        x, y, w, h = r[1:5]
        prob = r[5]
        label = r[6]
        if img_id in result_dict.keys():
            result_dict[img_id].append(((x, y, w+x, h+y), prob, label))
        else:
            result_dict[img_id] = [((x, y, w+x, h+y), prob, label)]

    print("pre-precessing of results")
    
    eval_katech_result(result_dict, katech.images, katech.label_name_map)
    
    
        


if __name__=="__main__":
    parser = ArgumentParser(description="Program for an analysis of inference result")
    args = parser.parse_args()
    args.data = None
    main(args)