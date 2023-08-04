from utils import *
from zoo.models import Dpn92_Unet_Double
import timeit
from tqdm import tqdm
from torch.autograd import Variable
from torch import nn
import torch
import random
import sys
from os import path, makedirs, listdir

import numpy as np

np.random.seed(1)

random.seed(1)


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

test_dir = '/local_storage/datasets/sgerard/xview2/test/images'
models_folder = '/local_storage/users/paulbp/xview2/weights/winner/'
pred_folder_root = "/local_storage/users/paulbp/xview2/winner/predictions/"

use_all_flips = True

if __name__ == '__main__':
    t0 = timeit.default_timer()

    seed = int(sys.argv[1])
    # vis_dev = sys.argv[2]

    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ["CUDA_VISIBLE_DEVICES"] = vis_dev

    pred_folder = path.join(
        pred_folder_root, 'dpn92cls_cce_{}_tuned'.format(seed))
    makedirs(pred_folder, exist_ok=True)

    # cudnn.benchmark = True

    models = []

    snap_to_load = 'dpn92_cls_cce_{}_tuned_best'.format(seed)

    model = Dpn92_Unet_Double(pretrained=None).cuda()

    model = nn.DataParallel(model).cuda()

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
                img2 = cv2.imread(fn.replace(
                    '_pre_', '_post_'), cv2.IMREAD_COLOR)

                img = np.concatenate([img, img2], axis=2)
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
                    msk = torch.softmax(msk[:, :, ...], dim=1)
                    msk = msk.cpu().numpy()

                    msk[:, 0, ...] = 1 - msk[:, 0, ...]

                    pred.append(msk[0, ...])
                    if use_all_flips:
                        pred.append(msk[1, :, ::-1, :])
                        pred.append(msk[2, :, :, ::-1])
                        pred.append(msk[3, :, ::-1, ::-1])

                pred_full = np.asarray(pred).mean(axis=0)

                msk = pred_full * 255
                msk = msk.astype('uint8').transpose(1, 2, 0)
                cv2.imwrite(path.join(pred_folder, f.replace(
                    '.png', '_part1.png')), msk[..., :3],
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
                cv2.imwrite(path.join(pred_folder, f.replace('.png', '_part2.png')), msk[..., 2:],
                            [cv2.IMWRITE_PNG_COMPRESSION, 9])

    elapsed = timeit.default_timer() - t0
    print('Time: {:.3f} min'.format(elapsed / 60))
