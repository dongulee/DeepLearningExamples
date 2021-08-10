from src.katech import KATECHDetection
import argparse
import os
from db_connect import dbcon
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='test for a katech data managing functions')
    parser.add_argument('--data', type=str,
            help = 'data root directory')
    parser.add_argument('--ckpt', type=str,
            help = 'dataset ckpt file')
    parser.add_argument('--train-dir', type=str,
            help = 'path of coco-style train data directory')
    parser.add_argument('--val-dir', type=str,
            help = 'path of coco-stye validation data directory')
    parser.add_argument('--ann-dir', type=str,
            help = 'path of coco-style annotation directory')
    parser.add_argument('--save', type=str,
            help = 'path of save file for the class object')
    parser.add_argument('--dbinsert', action='store_true',
            help = 'insert into a database')

    args = parser.parse_args()
    
    katech:KATECHDetection
    if args.ckpt == None:
        katech = KATECHDetection(img_folder=args.data, re_folder=r'^\d{8}_\d{6}(_I1)?$')
        print('{cardinality} images are process to be used'.format(cardinality=len(katech)))
    
    else:
        katech = KATECHDetection.load(args.ckpt)
        if katech.img_folder != args.data:
            katech.change_imgdir(args.data)
        print('Existing dataset class is loaded ({} images)'.format(len(katech)))

    if args.save != None:
        katech.save(args.save)
    
    # COCO Annotation Format
    if args.ann_dir != None:
        annfile_train = os.path.join(args.ann_dir, 'instances_train2021.json')
        annfile_val   = os.path.join(args.ann_dir, 'instances_val2021.json')
        if not os.path.exists(args.ann_dir):
            os.mkdir(args.ann_dir)
        katech.set_train()
        katech.to_coco(annfile_train)
        katech.set_val()
        katech.to_coco(annfile_val)

    # COCO-style train image directory
    if args.train_dir != None:
        if not os.path.exists(args.train_dir):
            os.mkdir(args.train_dir)
        katech.set_train()
        katech.create_symbol_links(args.train_dir)
        
    # COCO-style validation image directory
    if args.val_dir != None:
        if not os.path.exists(args.val_dir):
            os.mkdir(args.val_dir)
        katech.set_val()
        katech.create_symbol_links(args.val_dir)

        
    # Insert this Data Version into DB Table
    if args.dbinsert == True:
        # Connect to the database and check a sanity
        DB = dbcon("dbname='postgres' user='postgres' host='147.47.200.145' password='abcd1234!'")
        tables = DB.get_tables()
        tnames = [t[1] for t in tables]
        sanity_list = ['katech_datapoints', 'katech_bboxes', 'categories', 'data_catalog']
        for tbl in sanity_list:
            if tbl in tnames:
                ret = DB.adhoc_query("SELECT count(*) FROM {tname};".format(tname=tbl))
                if ret[0][0] != 0:   # first column of the first record
                    raise ValueError("Table {} is not empty. Please check the DB".format(tbl))

        # list of rawvideos (temporary) FIXME:
        rawvideos = list(dict.fromkeys([cdp['vid'] for cdp in katech.canonical_datapoints]))
        rawvideos = [x[:15] for x in rawvideos]
        cnt = 0
        with open('fake_rawvideos.list', 'w') as f:
            for raw in rawvideos:
                f.write(str(cnt) + ', ' + raw)
                cnt += 1
        
        #TODO: dataset_id logic
        dsid = 0
        # categories
        cats = [(dsid, cat[0], str(cat[1])) for cat in katech.label_info.items()]
        DB.insert_rows(tname='categories', data=cats)
        
        # datapoints
        # dataset_id, datapoint_id, rawvideo_id, imagefilepath, labelfilepath,
        # width,height,depth,file_format, file_size
        # version 0
        katech.set_train()
        data_ver = 0 # TODO: data version control
        rows = [(
            dsid,
            img['id'],
            rawvideos.index('_'.join(img['vid'].split('_')[:2])),
            img['imgfilepath'],
            img['lblfilepath'],
            *katech.images[img['id']][1],
            3, # TODO: hardcoded depth (3; COLOR)
            img['imgfile'].split('.')[-1],
            os.stat(img['imgfilepath']).st_size,
            data_ver
        ) for img in tqdm(katech.canonical_datapoints) if img['id'] in katech.img_keys
        ]
        DB.insert_rows('katech_datapoints', rows)

        # version 1
        katech.set_val()
        data_ver = 1
        rows = [(
            dsid,
            img['id'],
            rawvideos.index('_'.join(img['vid'].split('_')[:2])),
            img['imgfilepath'],
            img['lblfilepath'],
            *katech.images[img['id']][1],
            3, # TODO: hardcoded depth (3; COLOR)
            img['imgfile'].split('.')[-1],
            os.stat(img['imgfilepath']).st_size,
            data_ver
        ) for img in tqdm(katech.canonical_datapoints) if img['id'] in katech.img_keys
        ]
        DB.insert_rows('katech_datapoints', rows)
        
        # bboxes: dataset_id, dp_id, bbox_id, v0, v1, v2, v3, cat_id, area
        katech.set_all()
        coco_anns = katech.to_coco()
        rows = [
            (dsid, bbox['image_id'], bbox['id'], *bbox['bbox'], bbox['category_id'], bbox['area']) 
            for bbox in coco_anns['annotations']]
        DB.insert_rows('katech_bboxes', rows)
        
if __name__=="__main__":
    main()
