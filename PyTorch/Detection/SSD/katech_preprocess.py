from src.katech import KATECHDetection
import argparse
import os

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

        

    

    
        
if __name__=="__main__":
    main()
