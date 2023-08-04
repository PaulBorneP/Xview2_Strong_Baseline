import os
import random
from typing import Dict, Sequence

import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch import nn
import torch
import umap
import umap.plot


from utils import *
from refactor_utils import load_snapshot
from zoo.models import Res34_Unet_Double

np.random.seed(1)
random.seed(1)
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def get_umap_embedding(embeddings: Dict[str, np.ndarray], umap_args: Dict, labeling: str = "disaster", cmap: str = "Spectral") -> np.ndarray:
    """Get UMAP embedding for a list of embeddings

        Args:
            embeddings: list of embeddings
            cmap: color map for UMAP
            umap_args: arguments for UMAP
            labeling: either "disaster" or "event"
        Returns:
            umap_embedding: UMAP embedding
    """

    labels = np.array([x["disaster"] for x in embeddings])
    event_mapping = {"lower-puna-volcano": "volcano",
                     "palu-tsunami": "earthquake",
                     "mexico-earthquake": "earthquake",
                     "socal-fire": "fire",
                     "woolsey-fire": "fire",
                     "portugal-wildfire": "fire",
                     "pinery-bushfire": "fire",
                     "nepal-flooding": "flood",
                     "midwest-flooding": "flood",
                     "moore-tornado": "wind",
                     "joplin-tornado": "wind",
                     "hurricane-florence": "flood",
                     "hurricane-harvey": "flood",
                     "hurricane-michael": "wind",
                     "tuscaloosa-tornado": "wind",
                     "guatemala-volcano": "volcano",
                     "sunda-tsunami": "earthquake",
                     "santa-rosa-wildfire": "fire",
                     "hurricane-matthew": "wind"}
    if labeling == "disaster":
        pass
    elif labeling == "event":

        labels = np.array([event_mapping[x] for x in labels])
    else:
        raise ValueError("labeling must be either 'disaster' or 'event'")

    embeddings = np.array([x["embedding"] for x in embeddings])
    embeddings = np.concatenate(embeddings, axis=0)
    embeddings = np.squeeze(embeddings)

    mapper = umap.UMAP(**umap_args)
    umap_embedding = mapper.fit_transform(embeddings)
    # if isinstance(cmap, str):
    #     umap.plot.points(mapper, color_key_cmap=cmap,
    #                      labels=labels, height=1000, width=1600)
    # elif isinstance(cmap, dict):
    #     if labeling == "event":
    #         raise ValueError("cmap must be a string if labeling is 'disaster")
    #     umap.plot.points(mapper, color_key=cmap,
    #                      labels=labels, height=1000, width=1600)
    # else:
    #     raise ValueError("cmap must be either a string or a dict")
    umap.plot.output_notebook()
    hover_data = pd.DataFrame({'labels': labels, 'umap_x': umap_embedding[:, 0], 'umap_y': umap_embedding[:, 1], "event" :np.array([event_mapping[x] for x in labels])})
    if isinstance(cmap, str):
        p = umap.plot.interactive(
            mapper, labels=labels, hover_data=hover_data, point_size=3, color_key_cmap=cmap)
    elif isinstance(cmap, dict):
        p = umap.plot.interactive(
            mapper, labels=labels, hover_data=hover_data, point_size=3, color_key=cmap)
    else:
        raise ValueError("cmap must be either a string or a dict")
    umap.plot.show(p)
    return umap_embedding, labels


def get_embeddings(img_dirs: str, models_folder: str, pred_embedding_folder: str, seed: int) -> Sequence[np.ndarray]:
    """Get embeddings for all images in img_dir.

        Args:
            img_dir: directory containing images
            models_folder: directory containing models
            pred_embedding_folder: directory to save embeddings
            seed: seed for model

        Returns:
            embeddings: list of embeddings
    """

    os.makedirs(pred_embedding_folder, exist_ok=True)
    models = []
    print(os.path.join(models_folder, 'res34_cls2_{}_0_tuned_best'.format(seed)))
    snap_to_load = 'res34_cls2_{}_0_tuned_best'.format(seed) if os.path.exists(
        os.path.join(models_folder, 'res34_cls2_{}_0_tuned'.format(seed))) else 'res34_cls2_{}_0_best'.format(seed)
    model = Res34_Unet_Double().cuda()
    model = nn.DataParallel(model).cuda()
    load_snapshot(model, snap_to_load, models_folder)
    model.eval()
    models.append(model)
    embeddings = []
    with torch.no_grad():
        for img_dir in img_dirs:
            for f in tqdm(sorted(os.listdir(img_dir))):
                if '_pre_' in f:
                    fn = os.path.join(img_dir, f)

                    img = cv2.imread(fn, cv2.IMREAD_COLOR)
                    img2 = cv2.imread(fn.replace(
                        '_pre_', '_post_'), cv2.IMREAD_COLOR)

                    img = np.concatenate([img, img2], axis=2)
                    inp = normalize_image(img)
                    inp = np.asarray(inp, dtype='float')
                    inp = torch.from_numpy(inp.transpose(
                        (2, 0, 1))).unsqueeze(0).float()

                    inp = Variable(inp).cuda()
                    embedding = model.module.get_embeddings(inp)
                    embedding = embedding.cpu().numpy()
                    embeddings.append({"disaster": f.split(
                        "/")[-1].split("_")[0], "embedding": embedding})

        return embeddings


def save_embeddings(embeddings: Sequence[np.ndarray], pred_embedding_folder: str) -> None:
    """Save embeddings to pred_embedding_folder.

        Args:
            embeddings: list of embeddings
            pred_embedding_folder: directory to save embeddings
    """
    # create folder if it doesn't exist
    for i, embedding in enumerate(embeddings):
        np.save(os.path.join(pred_embedding_folder,
                '{}.npy'.format(i)), embedding)
