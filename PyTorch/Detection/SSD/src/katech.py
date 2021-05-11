import json
from os.path import isfile
import time
import os
import bz2
import pickle
from PIL import Image
from collections import namedtuple
import torch
import torch.utils.data as data
import xmltodict
import re

def analyze_key(key):
    rootdir = key.split('/')[-1]
    date = rootdir[:8]
    time = rootdir[9:15]
    if len(rootdir)>15:
        scenario = rootdir[15:]
    else:
        scenario = 'NULL'
    timestamp = '-'.join([date[:4], date[4:6], date[6:8]]) + ' ' + ':'.join([time[:2], time[2:4], time[4:6]])
    return rootdir, timestamp, scenario

def convert_timestamp(timestamp):
    year, month, day = timestamp[:4], timestamp[4:6], timestamp[6:8]
    hh, mm, ss = timestamp[9:11], timestamp[11:13], timestamp[13:15]
    sec_bias = int(timestamp[16:])/30   #FIXME: fractional seconds
    ss = str(int(ss)+sec_bias)
    ts = '-'.join([year, month, day])+' '+':'.join([hh, mm, ss])
    return ts
    
def to_COCO_bbox(bboxes, width, height):
    # KATECH bbox (parsed to dictionary from xml) style
    # from List of object: {'name': ABC, ... , 'bndbox':{'xmin', 'ymin', 'xmax', 'ymax'} (LTRB) style }
    # to  List of bbox_size and list of bbox_label
    bbox_sizes = []
    bbox_labels = []
    width = int(width)
    height = int(height)
    for bbox in bboxes:
        l, t, r, b = bbox['bndbox']['xmin'], bbox['bndbox']['ymin'], bbox['bndbox']['xmax'], bbox['bndbox']['ymax']
        l = int(l)
        t = int(t)
        r = int(r)
        b = int(b)
        bbox_size = (l/width, t/height, r/width, b/height)
        bbox_sizes.append(bbox_size)
        bbox_labels.append(bbox['name'])
    return bbox_sizes, bbox_labels


# Implement a datareader for KATECH dataset
class KATECHDetection(data.Dataset):
    
    def __init__(self, img_folder: str, re_folder:'pattern text'=None, transform=None, ckpt: str=None):
        '''
        self.img_folder: root directory of dataset
        
        self.imgs: image files (dir, filename)
        self.labels: label files (dir, filename)
        self.datapoints: {dir: list of {key: image, label}}
        
        self.canonical_datapoints: list of (id, path, timestamp, imgfilepath, lblfilepath, scenario)

        self.images: img_id -> (imgpath, (w, h), box_sizes, box_labels)
        self.label_name_map: label name -> label number
        self.label_info: label number -> label name
        self.img_keys = list(self.images.keys())
        self.transform: data transformation
        '''
        if ckpt is None:
            self.canonical_datapoints = []
            self.images = {}
            self.label_name_map = {}    # label name -> label number
            self.label_map = {}
            self.label_info = {}        # label number -> label name
        else:
            self.load_ckpt(ckpt)
        
        self.img_folder = img_folder
        self.front_cams = ['1', '2', '3']
        self.side_cams = ['4', '5']
        self.rear_cams = ['6', '7']
        self.annotation_version = 'annotations_v001_1'  # default for 2021

        if re_folder == None:
            # default rule: yyyymmdd_hhmmss 
            self.re_folder = re.compile(r'^\d{8}_\d{6}$')
        else:
            self.re_folder = re.compile(re_folder)

        # Start processing annotation
        self.parse_rootdir()
        
        labels = {}
        
        remove_dps = []
        
        for dp in self.canonical_datapoints:
            if dp['lblfilepath']=='NULL':
                # remove_dps.append(dp)
                continue
            with open(dp['lblfilepath'], 'r') as lbl:
                ordered = xmltodict.parse(lbl.read())
                dict_data = json.loads(json.dumps(ordered))['annotation']
                if 'object' not in dict_data.keys():
                    # remove_dps.append(dp)                    
                    continue
                
                width = int(dict_data['size']['width'])
                height = int(dict_data['size']['height'])
                
                if type(dict_data['object'])==type({}):
                    dict_data['object'] = [dict_data['object']]
                elif type(dict_data['object'])==type([]):
                    pass
                else:
                    print("object data in label file seems to be wrong: {}".format(dp['lblfilepath']))
                bbox_sizes, bbox_labels = to_COCO_bbox(dict_data['object'], width, height)
                for lbl in bbox_labels:
                    labels[lbl]=True
                fullpath = os.path.join(dp['path'], dict_data['folder'], dict_data['filename'])
                self.images[dp['id']] = (fullpath, (width, height), bbox_sizes, bbox_labels)
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

        self._split_train_val()
        self.images_all = self.images
        self.status = 'all'
    
    def parse_rootdir(self):
        imgs = []
        labels = []
        for cur_dir, subdirs, files in os.walk(self.img_folder):
            if len(files) > 0:
                # when the root is a leaf directory
                if files[0].endswith('jpg'):
                    imgs.append({'parent': cur_dir, 'files': files})
                elif files[0].endswith('xml'):
                    labels.append({'parent': cur_dir, 'files': files})
                else:
                    pass

        datapoints = {} # key: path to video, value: images, labels

        for img in imgs:
            key_dir = os.path.join('/', *img['parent'].split('/')[:-1]) # path to one image source (video)
            item_key = img['parent'].split('/')[-1] # name of the sub directory
            items = img['files']                    # images

            # FIXME: change to utilize a master database
            if self.re_folder.match(key_dir) == False:
                print("violation of a directory naming rule: {}".format(item_key))
                continue
            if key_dir in datapoints.keys():
                if item_key in datapoints[key_dir].keys():
                    # Error which cannot happen because 'item_keys' are in same dir
                    print("key duplicates: {}".format(item_key))
                else:
                    datapoints[key_dir][item_key]=items
            else:
                datapoints[key_dir]={}
                datapoints[key_dir][item_key]=items

        for lbl in labels:
            key_dir = os.path.join('/', *lbl['parent'].split('/')[:-1])
            item_key = lbl['parent'].split('/')[-1]
            items = lbl['files']
            if key_dir in datapoints.keys():
                if item_key in datapoints[key_dir].keys():
                    #Error
                    print("key duplicates: {}".format(item_key))
                else:
                    datapoints[key_dir][item_key]=items
            else:
                datapoints[key_dir]={}
                datapoints[key_dir][item_key]=items

        cnt = 0 # identifier variable for each datapoint
        for (k,v) in datapoints.items(): 
            # e.g. 
            # k:'.../20200114_155214', 
            # v: {'1': [images ...],
            #     '2': [images ...],
            #     '3': [images ...],
            #     '1_annotations_v001_1': [labels ...],
            #     ...}
            
            vid, timestamp, scenario = analyze_key(k)
            path = k
            err_flag = False
            # simply check for the validate labels
            subdirs = list(v.keys())
            cams = [d for d in subdirs if d.find(self.annotation_version) < 0]
            cams.sort()
            anns = [d for d in subdirs if d.find(self.annotation_version) > 0]
            anns.sort()

            if len(cams)!=len(anns):
                print("camera and label are mis-matching in a path: {}".format(k))
                continue
            # Matching image and label file
            for i in range(len(cams)):
                if '_'.join(cams[i], self.annotation_version) not in anns:
                    print("annotation dir for {} does not exist: {}".format(cams[i], k))
                    err_flag = True
                if len(v[cams[i]]) != len(v[anns[i]]):
                    print("annotation files for {}/{} are not complete".format(k,cams[i]))
            
            if err_flag == True:
                continue
                
            for i in range(len(cams)):
                imgfiles = v[cams[i]]
                lblfiles = v[anns[i]]
                for imgfile in imgfiles:
                    lblfile_name = imgfile.split('.')[0] + '_v001_1.xml'
                    if lblfile_name in lblfiles:
                        lblfile = lblfile_name
                        lblfilepath = os.path.join(k, anns[i], lblfile)
                    else:
                        lblfilepath = 'NULL'
                    imgfilepath = os.path.join(k, cams[i], imgfile)

                    self.canonical_datapoints.append(
                        {
                            'id': cnt,
                            'vid': vid,
                            'path': path,
                            'timestamp': timestamp,
                            'imgfile': imgfile,
                            'lblfile': lblfile,
                            'imgfilepath': imgfilepath,
                            'lblfilepath': lblfilepath,
                            'scenario': scenario
                        }
                    )
                    cnt += 1
  
            # (old) Matching image and label file
            # for cam_num in self.front_cams:
            #     imgdir = str(cam_num)
            #     for imagefile in v[imgdir]:
            #         labelfile_key = imagefile.split('.')[0] + '_v001_1.xml'
            #         lbldir = imgdir + self.annotation_version
            #         if labelfile_key in v[lbldir]:
            #             labelfile = labelfile_key
            #             lblfilepath = os.path.join(k, lbldir, labelfile)
            #         else:
            #             lblfilepath = 'NULL'
            #         imgfilepath = os.path.join(k, imgdir, imagefile)

            #         self.canonical_datapoints.append(
            #             {'id':cnt, 'vid':vid, 'path': path, 'timestamp':timestamp, 'imgfilepath':imgfilepath, 'lblfilepath': lblfilepath, 'scenario': scenario}
            #         )
            #         cnt += 1

        print("{} images are added to dataset".format(cnt))

    @property
    def labelnum(self):
        return len(self.label_info)

    @staticmethod
    def load(pklfile):
        with bz2.open(pklfile, "rb") as fin:
            ret = pickle.load(fin)
        return ret

    def load_ckpt(self, ckpt):
        # load canonical datapoints from db or file
        if isfile(ckpt):
            # load from file
            data = self.load(ckpt)
            self.canonical_datapoints = data.canonical_datapoints
            self.images = data.images
            self.label_name_map = data.images
            self.label_map = data.label_map
            self.label_info = data.label_info
        else:
            #load from db
            pass #TODO:

    def save(self, pklfile):
        with bz2.open(pklfile, "wb") as fout:
            pickle.dump(self, fout)

    def _split_train_val(self, val_ratio=0.1):
        num = len(self.images)
        train_num = int(num*(1-val_ratio))
    
        self.image_train = {self.img_keys[i]:self.images[self.img_keys[i]] \
        for i in range(train_num)}
        self.image_val = {self.img_keys[i]:self.images[self.img_keys[i]] \
        for i in range(train_num, num)}
        
    def set_val(self):        
        self.status = 'val'
        self.images = self.image_val
        self.img_keys = list(self.image_val.keys())

    def set_train(self):        
        self.status = 'train'
        self.images = self.image_train
        self.img_keys = list(self.image_train.keys())

    def set_all(self):        
        self.status = 'all'
        self.images = self.images_all
        self.img_keys = list(self.images_all.keys())

    def to_coco(self, filename):
        info = {
            "description": "KATECH Dataset 2021",
            "url": "n.a.",
            "version": "0.1",
            "year": 2021,
            "contributor": "ROK MOTIE Autonomous Vehicle Consortium",
            "date_created": "2021/04/22"
        }
        licenses = [
            {
                "url":"n.a.",
                "id": 1,
                "name": "Private License"
        }
        ]
        images = [{'license': 1, 
                    'file_name': self.images[idx][0].split('/')[-1],
                    'coco_url': 'n.a.',
                    'height': self.images[idx][1][1],
                    'width': self.images[idx][1][0],
                    'date_captured': convert_timestamp(self.images[idx][0].split('/')[-1][2:-4]),
                    'flickr_url': 'n.a.',
                    'id': idx
                    } 
                    for idx in self.img_keys]

        categories = [{'supercategory': 'TBD', 'id': k, 'name': v} for k, v in self.label_info.items() ]
        
        annotations = []
        cnt = 0
        for idx in self.img_keys:
            w, h = self.images[idx][1]
            bboxes = self.images[idx][2] # LTRB?
            labels = self.images[idx][3]
            for bbox_size, label in zip(bboxes, labels):
                annotations.append(
                    {
                        "iscrowd": 0,
                        "image_id": idx,
                        "area": bbox_size[0]*bbox_size[1]*w*h*0.9,
                        "bbox": bbsize_to_xywh(bbox_size, w, h),
                        "category_id": self.label_name_map[label],
                        "id": cnt
                    }
                )
                cnt += 1
        coco_annotation = {
            'info': info,
            'licenses': licenses,
            'images': images,
            'annotations': annotations,
            'categories': categories
        }
        with open(filename, 'w') as f:
            json.dump(coco_annotation, f)

    def change_imgdir(self, imgdir):
        sub = self.img_folder
        self.set_all()
        print(len(self))
        self.images = {k:(v[0].replace(sub,imgdir), v[1], v[2], v[3])
                        for k, v in self.images.items()}
        self._split_train_val()
        self.img_folder = imgdir

    def create_symbol_links(self, dir):
        #TODO: inspect 'dir'
        for idx, img in self.images.items():
            filepath = img[0]
            dstfilename = filepath.split('/')[-1]
            dstfilename = os.path.join(dir, dstfilename)
            os.symlink(filepath, dstfilename)
    def __len__(self):
        return len(self.images)

    def get_patch(self, obj: 'coco annotation'):
        img_id = obj['image_id']
        x, y, w, h = obj['bbox']
        bbox = (x, y, x+w, y+h)
        img_data = self.images[img_id]
        img = Image.open(img_data[0]).convert("RGB")
        return img, img.crop(bbox)

    def __getitem__(self, idx):
        # img_data
        # [0]: img filepath
        # [1]: (w, h)
        # [2]: bboxes (LTRB/wh) style
        # [3]: labels

        img_id = self.img_keys[idx]
        img_data = self.images[img_id]
        img = Image.open(img_data[0]).convert("RGB")

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

def bbsize_to_xywh(bbox_size, w, h):
    # bbox_size: LTRB/w,h
    l = bbox_size[0]*w
    t = bbox_size[1]*h
    r = bbox_size[2]*w
    b = bbox_size[3]*h
    return (l, t, r-l, b-t)

