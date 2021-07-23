from src.katech import KATECHDetection
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='test for a katech data managing functions')
    parser.add_argument('--data', type=str,
            help = 'data root directory')
    parser.add_argument('--ckpt', type=str, default='none',
            help = 'dataset ckpt file')
    parser.add_argument('--train-dir', type=str, default='none',
            help = 'path to coco-style train data directory')
    parser.add_argument('--val-dir', type=str, default='none',
            help = 'path to coco-stye validation data directory')
    parser.add_argument('--ann-dir', type=str, default='none',
            help = 'path to coco-style annotation directory')
#     parser.add_argument('--dirname-rule', type=str, default=None,
#             help = 'directory naming rule of source_dir')
    parser.add_argument('--save', type=str,
            help = 'save the KATECHDetection object into a file')
    args = parser.parse_args()
    if args.ckpt == 'none':
        katech = KATECHDetection(img_folder=args.data, re_folder=r'^\d{8}_\d{6}(_I1)?$')
    else:
        katech = KATECHDetection.load(args.ckpt)


    if args.save is not None:
        katech.save(args.save)

    # creating symbolic links into train, val directory
    # creating annotation files into ann_dir
    annfile_train = os.path.join(args.ann_dir, 'instances_train2021.json')
    annfile_val   = os.path.join(args.ann_dir, 'instances_val2021.json')
    # annfile_all   = os.path.join(args.ann_dir, 'instances_val2021.json')
    katech.set_train()
    
    if not os.path.exists(args.ann_dir):
        os.mkdir(args.ann_dir)
    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)
    if not os.path.exists(args.val_dir):
        os.mkdir(args.val_dir)
    
    katech.create_symbol_links(args.train_dir)
    katech.to_coco(annfile_train)
    
    katech.set_val()
    katech.create_symbol_links(args.val_dir)
    katech.to_coco(annfile_val)
    
#     katech.set_all()
#     katech.to_coco(annfile_all)
    
     
    print('{cardinality} images are process to be used'.format(cardinality=len(katech)))

    
        
if __name__=="__main__":
    main()
