from src.katech import KATECHDetection
from argparse import ArgumentParser
import numpy as np


def main(args):
    katech = KATECHDetection.load('katech_dataset.pkl')
    # katech.load('katech_dataset.pkl')
    print(katech.img_folder)

    # idx_map = [katech.img_keys]

    results = np.load('katech_infer_test.npy')
    result_dict = {}
    for r in results:
        img_id = katech.img_keys[int(r[0])]
        x, y, w, h = r[1:5]
        prob = r[5]
        label = r[6]
        if img_id in result_dict.keys():
            result_dict[img_id].append((x, y, w, h, prob, label))
        else:
            result_dict[img_id] = [(x, y, w, h, prob, label)]

    ### evaluation
    # answer for img_id
    # katech.images[img_id]
    # =(dict_data['filename'], (width, height), bbox_sizes, bbox_labels, fullpath)
    
    for imgid in result_dict.keys():
        


if __name__=="__main__":
    parser = ArgumentParser(description="Program for an analysis of inference result")
    args = parser.parse_args()
    args.data = None
    main(args)