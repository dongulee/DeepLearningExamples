from src.katech import KATECHDetection
import argparse

def main():
    parser = argparse.ArgumentParser(description='test for a katech data managing functions')
    parser.add_argument('--data', type=str,
            help = 'data root directory')
    parser.add_argument('--ckpt', type=str, default='none',
            help = 'dataset ckpt file')
    
    args = parser.parse_args()
    if args.ckpt == 'none':
        katech = KATECHDetection(args.data)
    else:
        katech = KATECHDetection.load(args.ckpt)

    katech.to_coco('instances_katech2021.json')
    
    print(len(katech))
if __name__=="__main__":
    main()
