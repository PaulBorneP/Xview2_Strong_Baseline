from utils import *
from zoo.models import SeResNext50_Unet_Loc
import timeit
from tqdm import tqdm
from torch.autograd import Variable
import torch
import random
import numpy as np
import os
import sys
from os import path, makedirs, listdir

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


np.random.seed(1)

random.seed(1)


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

if __name__ == '__main__':
    t0 = timeit.default_timer()

    seed = int(sys.argv[1])
    sub_folder = sys.argv[2]

    test_dir = '/local_storage/datasets/sgerard/xview2/test/images'
    pred_folder = f'/local_storage/users/paulbp/xview2/predictions/{sub_folder}/pred50_loc_tuned'
    models_folder = f'/local_storage/users/paulbp/xview2/weights{sub_folder}'

    use_all_flips = True

    makedirs(pred_folder, exist_ok=True)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

    # cudnn.benchmark = True

    models = []

    for seed in [seed]:

        snap_to_load = 'res50_loc_{}_tuned_best'.format(seed)
        model = SeResNext50_Unet_Loc().cuda()
        print("=> loading checkpoint '{}'".format(snap_to_load))
        checkpoint = torch.load(
            path.join(models_folder, snap_to_load), map_location='cpu')
        loaded_dict = checkpoint['state_dict']
        sd = model.state_dict()
        for k in model.state_dict():
            if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
                sd[k] = loaded_dict[k]
        loaded_dict = sd
        model.load_state_dict(loaded_dict)
        print("loaded checkpoint '{}' (epoch {}, best_score {})"
              .format(snap_to_load, checkpoint['epoch'], checkpoint['best_score']))
        model.eval()
        models.append(model)

    with torch.no_grad():
        for f in tqdm(sorted(listdir(test_dir))):
            if '_pre_' in f:
                fn = path.join(test_dir, f)

                img = cv2.imread(fn, cv2.IMREAD_COLOR)
                img = normalize_image(img)

                inp = []
                inp.append(img)
                if use_all_flips:
                    inp.append(img[::-1, ...])
                    inp.append(img[:, ::-1, ...])
                    inp.append(img[::-1, ::-1, ...])
                inp = np.asarray(inp, dtype='float')
                inp = torch.from_numpy(inp.transpose((0, 3, 1, 2))).float()
                inp = Variable(inp).cuda()

                pred = []
                for model in models:
                    msk = model(inp)
                    msk = torch.sigmoid(msk)
                    msk = msk.cpu().numpy()

                    pred.append(msk[0, ...])
                    if use_all_flips:
                        pred.append(msk[1, :, ::-1, :])
                        pred.append(msk[2, :, :, ::-1])
                        pred.append(msk[3, :, ::-1, ::-1])

                pred_full = np.asarray(pred).mean(axis=0)

                msk = pred_full * 255
                msk = msk.astype('uint8').transpose(1, 2, 0)
                cv2.imwrite(path.join(pred_folder, f.replace(
                    '.png', '_part1.png')), msk[..., 0],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
