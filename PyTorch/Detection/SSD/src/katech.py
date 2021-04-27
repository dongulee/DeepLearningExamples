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
def convert_timestamp(timestamp):
    year, month, day = timestamp[:4], timestamp[4:6], timestamp[6:8]
    hh, mm, ss = timestamp[9:11], timestamp[11:13], timestamp[13:15]
    sec_bias = int(timestamp[16:])/30
    ss = str(int(ss)+sec_bias)
    ts = year+'-'+month+'-'+day+' '+hh+':'+mm+':'+ss
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
    def __init__(self, img_folder, transform=None):
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

        self.datapoints = {} # key: path to video, value: images, labels

        for img in self.imgs:
            key_dir = os.path.join('/', *img['parent'].split('/')[:-1]) # path to one image source (video)
            item_key = img['parent'].split('/')[-1] # cam number
            items = img['files']    # images
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
        # Datapoint = namedtuple('Datapoint', ['vid', 'timestamp', 'imgfilepath', 'lblfilepath', 'scenario'])


        cnt = 0 # identifier variable for each datapoint
        for (k,v) in self.datapoints.items(): 
            # e.g. 
            # k:'.../20200114_155214', 
            # v: {'1': [images ...],
            #     '2': [images ...],
            #     '3': [images ...],
            #     '1_annotations_v001_1': [labels ...],
            #     ...}
            
            timestamp, scenario = analyze_key(k)
            path = k

            # simply check for the validate labels
            if '1' in v.keys() and '2' in v.keys() and '3' in v.keys() and '1_annotations_v001_1' in v.keys() and '2_annotations_v001_1' in v.keys() and '3_annotations_v001_1' in v.keys():
                check1 = (len(v['1']) == len(v['1_annotations_v001_1']))
                check2 = (len(v['2']) == len(v['2_annotations_v001_1']))
                check3 = (len(v['3']) == len(v['3_annotations_v001_1']))
                if check1 == True and check2 == True and check3 == True:
                    pass
                else:
                    print("3 cameras are not complete: {}/{},{},{}".format(k,check1, check2, check3))
                    # continue
                pass
            else:
                print("data points are not complete: {}".format(k))

            
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
                        {'id':cnt, 'path': path, 'timestamp':timestamp, 'imgfilepath':imgfilepath, 'lblfilepath': lblfilepath, 'scenario': scenario}
                    )
                    cnt += 1


        print("{} images are added to dataset".format(cnt))

        
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
                    print("object data in label file is wrong: {}".format(lblfilepath))
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
    def create_symbol_links(self, dir):
        #TODO: inspect 'dir'
        for idx, img in self.images.items():
            filepath = img[0]
            dstfilename = filepath.split('/')[-1]
            dstfilename = os.path.join(dir, dstfilename)
            os.symlink(filepath, dstfilename)
    def __len__(self):
        return len(self.images)

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

