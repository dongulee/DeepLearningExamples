import torch
import numpy as np


def infer(model, data, encoder, inv_map, args):
    if args.distributed:
        N_gpu = torch.distributed.get_world_size()
    else:
        N_gpu = 1

    model.eval()
    if not args.no_cuda:
        model.cuda()
    ret = []
    for nbatch, (img, img_id, img_size, _, _) in enumerate(data):
        print("Parsing batch: {}/{}".format(nbatch, len(data)), end='\r')
        with torch.no_grad():
            inp = img.cuda()
            if args.amp:
                inp = inp.half()
            # Get predictions
            ploc, plabel = model(inp)
            ploc, plabel = ploc.float(), plabel.float()
            for idx in range(ploc.shape[0]):
                # ease-of-use for specific predictions
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)

                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, 0.50, 200)[0]
                except:
                    # raise
                    print("")
                    print("No object detected in idx: {}".format(idx))
                    continue

                htot, wtot = img_size[0][idx].item(), img_size[1][idx].item()
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    ret.append([img_id[idx], 
                                loc_[0] * wtot, \
                                loc_[1] * htot,
                                (loc_[2] - loc_[0]) * wtot,
                                (loc_[3] - loc_[1]) * htot,
                                prob_,
                                label_])
                                # inv_map[label_]])
    
    ret = np.array(ret).astype(np.float32)
    return ret

    