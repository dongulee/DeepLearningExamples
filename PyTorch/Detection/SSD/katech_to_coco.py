from src.katech import KATECHDetection

katech = KATECHDetection.load('katech_ckpt5.pkl')
katech.set_train()
katech.to_coco('/data/KATECH/annotations/instances_train2021.json')
katech.set_val()
katech.to_coco('/data/KATECH/annotations/instances_val2021.json')
katech.set_all()
katech.to_coco('/data/KATECH/annotations/instances_all2021.json')