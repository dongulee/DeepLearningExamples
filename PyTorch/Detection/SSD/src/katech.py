import json
import time
import os
import bz2
import pickle
from PIL import Image
from collections import namedtuple
import torch
import torch.utils.data as data
import xmltodict

def analyze_key(key):
    rootdir = key.split('/')[-1]
    date = rootdir[:8]
    time = rootdir[9:15]
    if len(rootdir)>15:
        scenario = rootdir[15:]
    else:
        scenario = 'NULL'
    timestamp = date[:4]+'-'+date[4:6]+'-'+date[6:8]+ ' ' + time[:2]+':'+time[2:4]+':'+time[4:6]
    return timestamp, scenario

def to_COCO_bbox(bboxes, width, height):
    # KATECH bbox (parsed to dictionary from xml) style
    # from List of object: {'name': ABC, ... , 'bndbox':{'xmin', 'ymin', 'xmax', 'ymax'} (LBRT) style }
    # to  List of bbox_size and list of bbox_label
    bbox_sizes = []
    bbox_labels = []
    width = int(width)
    height = int(height)
    for bbox in bboxes:
        l, b, r, t = bbox['bndbox']['xmin'], bbox['bndbox']['ymin'], bbox['bndbox']['xmax'], bbox['bndbox']['ymax']
        l = int(l)
        b = int(b)
        r = int(r)
        t = int(t)
        bbox_size = (l/width, t/height, r/width, b/height)
        bbox_sizes.append(bbox_size)
        bbox_labels.append(bbox['name'])
    return bbox_sizes, bbox_labels


# Implement a datareader for KATECH dataset
class KATECHDetection(data.Dataset):
    def __init__(self, img_folder, transform=None):
        self.img_folder = img_folder
        # self.annotate_file = None

        # Start processing annotation
        self.imgs = []
        self.labels = []
        for cur_dir, subdirs, files in os.walk(self.img_folder):
            if len(files) > 0:
                # when the root is a leaf directory
                if files[0].endswith('jpg'):
                    self.imgs.append({'parent': cur_dir, 'files': files})
                elif files[0].endswith('xml'):
                    self.labels.append({'parent': cur_dir, 'files': files})
                else:
                    pass

        self.datapoints = {}

        for img in self.imgs:
            key_dir = os.path.join('/', *img['parent'].split('/')[:-1])
            item_key = img['parent'].split('/')[-1]
            items = img['files']
            if key_dir in self.datapoints.keys():
                if item_key in self.datapoints[key_dir].keys():
                    #Error
                    print("key duplicates: {}".format(item_key))
                else:
                    self.datapoints[key_dir][item_key]=items
            else:
                self.datapoints[key_dir]={}
                self.datapoints[key_dir][item_key]=items

        for lbl in self.labels:
            key_dir = os.path.join('/', *lbl['parent'].split('/')[:-1])
            item_key = lbl['parent'].split('/')[-1]
            items = lbl['files']
            if key_dir in self.datapoints.keys():
                if item_key in self.datapoints[key_dir].keys():
                    #Error
                    print("key duplicates: {}".format(item_key))
                else:
                    self.datapoints[key_dir][item_key]=items
            else:
                self.datapoints[key_dir]={}
                self.datapoints[key_dir][item_key]=items
        
        self.canonical_datapoints = []
        Datapoint = namedtuple('Datapoint', ['vid', 'timestamp', 'imgfilepath', 'lblfilepath', 'scenario'])

        for (k,v) in self.datapoints.items(): 
            # e.g. 
            # k:'.../20200114_155214', 
            # v: {'1': [images ...],
            #     '2': [images ...],
            #     '3': [images ...],
            #     '1_annotations_v001_1': [labels ...],
            #     ...}
            
            timestamp, scenario = analyze_key(k)
            vid = k

            # simply check for the validate labels
            if '1' in v.keys() and '2' in v.keys() and '3' in v.keys() and '1_annoatations_v001_1' in v.keys() and '2_annotations_v001_1' in v.keys() and '3_annotations_v001_1' in v.keys():
                check1 = (len(v['1']) == len(v['1_annotations_v001_1']))
                check2 = (len(v['2']) == len(v['2_annotations_v001_1']))
                check3 = (len(v['3']) == len(v['3_annotations_v001_1']))
                if check1 == True and check2 == True and check3 == True:
                    pass
                else:
                    print("3 cameras are not complete: {}".format(k))
                    continue
                pass
            else:
                print("data points are not complete: {}".format(k))
                #print(v.keys())
                #continue

            for cam_num in [1, 2, 3]:
                imgdir = str(cam_num)
                for imagefile in v[imgdir]:
                    labelfile_key = imagefile.split('.')[0] + '_v001_1.xml'
                    lbldir = imgdir + '_annotations_v001_1'
                    if labelfile_key in v[lbldir]:
                        labelfile = labelfile_key
                        lblfilepath = os.path.join(k, lbldir, labelfile)
                    else:
                        lblfilepath = 'NULL'
                    imgfilepath = os.path.join(k, imgdir, imagefile)
                    
                    # imgfilepath = os.path.abspath(imgfilepath)
                    # lblfilepath = os.path.abspath(lblfilepath)
                    self.canonical_datapoints.append(
                        Datapoint(vid, timestamp, imgfilepath, lblfilepath, scenario)
                    )


        
        # images[key][0] = image file name (fn)
        # images[key][1] = image size
        # images[key][2] = objects in image

        self.images = {}
        labels = {}
        self.label_name_map = {}  # label name -> label number (no label id in KATECH)
        self.label_map = {}
        self.label_info = {} # label number -> label name
        remove_dps = []
        for dp in self.canonical_datapoints:
            if dp.lblfilepath=='NULL':
                continue
            with open(dp.lblfilepath, 'r') as lbl:
                ordered = xmltodict.parse(lbl.read())
                dict_data = json.loads(json.dumps(ordered))['annotation']
                if 'object' not in dict_data.keys():
                    remove_dps.append(dp)                    
                    continue
                
                width = int(dict_data['size']['width'])
                height = int(dict_data['size']['height'])
                
                if type(dict_data['object'])==type({}):
                    dict_data['object'] = [dict_data['object']]
                elif type(dict_data['object'])==type([]):
                    pass
                else:
                    print("object data in label file is wrong: {}".format(lblfilepath))
                bbox_sizes, bbox_labels = to_COCO_bbox(dict_data['object'], width, height)
                for lbl in bbox_labels:
                    labels[lbl]=True
                fullpath = os.path.join(dp.vid, dict_data['folder'], dict_data['filename'])
                self.images[dict_data['filename']] = (dict_data['filename'], (width, height), bbox_sizes, bbox_labels, fullpath)
        for x in remove_dps:
            self.canonical_datapoints.remove(x)

        cnt = 0
        self.label_info[cnt] = "background"
        for cat in labels.keys():
            cnt += 1
            self.label_name_map[cat] = cnt
            self.label_map[cnt] = cnt # Identical map
            self.label_info[cnt] = cat
        
        # # build inference for images
        # for img in self.data["images"]:
        #     img_id = img["id"]
        #     img_name = img["file_name"]
        #     img_size = (img["height"],img["width"])
        #     if img_id in self.images: raise Exception("dulpicated image record")
        #     self.images[img_id] = (img_name, img_size, [])

        # # read bboxes
        # for bboxes in self.data["annotations"]:
        #     img_id = bboxes["image_id"]
        #     category_id = bboxes["category_id"]
        #     bbox = bboxes["bbox"]
        #     bbox_label = self.label_name_map[bboxes["category_id"]]
        #     self.images[img_id][2].append((bbox, bbox_label))

        for k, v in list(self.images.items()):
            if len(v[2]) == 0:
                self.images.pop(k)

        self.img_keys = list(self.images.keys())
        self.transform = transform

    @property
    def labelnum(self):
        return len(self.label_info)

    @staticmethod
    def load(pklfile):
        with bz2.open(pklfile, "rb") as fin:
            ret = pickle.load(fin)
        return ret

    def save(self, pklfile):
        with bz2.open(pklfile, "wb") as fout:
            pickle.dump(self, fout)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # img_data
        # [0]: img filename
        # [1]: (w, h)
        # [2]: bboxes (LTRB/wh) style
        # [3]: labels
        # [4]: img filepath

        img_id = self.img_keys[idx]
        img_data = self.images[img_id]
        img = Image.open(img_data[4]).convert("RGB")

        htot, wtot = img_data[1]
        bbox_sizes = torch.tensor(img_data[2])
        bbox_labels = [self.label_name_map[x] for x in img_data[3]]
        bbox_labels = torch.tensor(bbox_labels)
        

        #for (xc, yc, w, h), bbox_label in img_data[2]:
        # for (l,t,w,h), bbox_label in img_data[2]:
        #     r = l + w
        #     b = t + h
        #     #l, t, r, b = xc - 0.5*w, yc - 0.5*h, xc + 0.5*w, yc + 0.5*h
        #     bbox_size = (l/wtot, t/htot, r/wtot, b/htot)
        #     bbox_sizes.append(bbox_size)
        #     bbox_labels.append(bbox_label)

        # bbox_sizes = torch.tensor(bbox_sizes)
        # bbox_labels =  torch.tensor(bbox_labels)


        if self.transform != None:
            img, (htot, wtot), bbox_sizes, bbox_labels = \
                self.transform(img, (htot, wtot), bbox_sizes, bbox_labels)
        else:
            pass

        return img, img_id, (htot, wtot), bbox_sizes, bbox_labels