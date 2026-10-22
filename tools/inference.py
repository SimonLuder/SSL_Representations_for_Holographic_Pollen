import os
import numpy as np
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pollen_datasets.poleno import PairwiseHolographyImageFolder

from ssl_poleno.model import SelfSupervisedLearner
from ssl_poleno.model.backbones import get_backbone, set_single_channel_input, update_linear_layer
from ssl_poleno.utils import config


def load_ssl_model_weights(model, ckpt_path):
    """
    Load weights from either:
      - a Lightning checkpoint (.ckpt)
      - a plain PyTorch model (.pth)
    """

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Detect Lightning checkpoint
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        print(f"[INFO] Detected Lightning checkpoint: {ckpt_path}")
        lightning_state_dict = ckpt["state_dict"]
        state_dict = {
            k.replace("model.", "", 1): v for k, v in lightning_state_dict.items()
            }
    # Detect standard PyTorch state_dict
    elif isinstance(ckpt, dict):
        print(f"[INFO] Detected standard PyTorch state_dict: {ckpt_path}")
        state_dict = ckpt
    # Detect model object (rare, but supported)
    else:
        print(f"[INFO] Detected full model object in checkpoint.")
        model = ckpt
        return model

    model.load_state_dict(state_dict, strict=False)
    print("[INFO] Model weights successfully loaded.")
    return model


def inference(ckpt_path, conf, save_as="inference.npz"):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_conf = conf["dataset"]
    model_conf = conf["ssl"]

    # Transformations
    transforms_list = [transforms.ToTensor()]
    if dataset_conf.get("img_interpolation"):
        transforms_list.append(
            transforms.Resize(
                (dataset_conf["img_interpolation"], 
                    dataset_conf["img_interpolation"]),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        )
    transforms_list.append(
        transforms.Normalize(
            [0.5,] * dataset_conf["img_channels"],
            [0.5,] * dataset_conf["img_channels"],
        )
    )
    transform = transforms.Compose(transforms_list)

    print(dataset_conf.get("labels_test"))
    print(os.path.exists(dataset_conf.get("labels_test")))

    # Dataset
    dataset_test = PairwiseHolographyImageFolder(
        root=dataset_conf["root"],
        transform=transform,
        dataset_cfg=dataset_conf,
        labels=dataset_conf.get("labels_test"),
        verbose=True,
    )

    # DataLoader
    dataloader = DataLoader(
        dataset_test,
        batch_size=4,
        shuffle=False,
        num_workers=4,
    )

    # ----------------------------
    # 2. Rebuild BYOL model
    # ----------------------------

    # Backbone
    backbone = get_backbone(model_conf["backbone"], pretrained=False)
    backbone = set_single_channel_input(backbone)
    if model_conf["embedding_size"] is not None:
        backbone = update_linear_layer(
            backbone, layer=model_conf["hidden_layer"], 
            out_features=model_conf["embedding_size"]
        )

    use_prediction_head = True if str(model_conf.get("objective", None)).lower() in ["byol", "simsiam"] else False
    use_momentum = True if str(model_conf.get("objective", None)).lower() == "byol" else False

    # Recreate BYOL model
    model = SelfSupervisedLearner(
        backbone,
        image_size=dataset_conf.get("img_interpolation", dataset_conf["img_size"]),
        image_channels=dataset_conf.get("img_channels", 1),
        hidden_layer=model_conf.get("hidden_layer", "avgpool"),
        projection_size=model_conf.get("projection_size", 256),
        projection_hidden_size=model_conf.get("projection_hidden_size", 4096),
        augment_fn=torch.nn.Identity(),
        augment_fn2=torch.nn.Identity(),
        moving_average_decay=model_conf.get("moving_average_decay", 0.99),
        use_momentum=use_momentum,
        use_prediction_head=use_prediction_head,
    )

    # ----------------------------
    # 3. Load weights (fix Lightning prefixes)
    # ----------------------------
    model = load_ssl_model_weights(model, ckpt_path)
    model.to(device)
    model.eval()

    all_proj1, all_proj2 = [], []
    all_emb1, all_emb2 = [], []
    all_filenames1, all_filenames2 = [], []

    for (images, _, filenames) in tqdm(dataloader):
        img1, img2 = images
        file1, file2 = filenames

        img1 = img1.to(device)
        img2 = img2.to(device)
        
        with torch.no_grad():
            ((proj1, emb1), (proj2, emb2)) = model(img1, img2, return_embedding=True, return_projection=True)

            # Move to CPU and convert to numpy
            all_proj1.append(proj1.cpu().numpy())
            all_proj2.append(proj2.cpu().numpy())
            all_emb1.append(emb1.cpu().numpy())
            all_emb2.append(emb2.cpu().numpy())
            all_filenames1.extend(file1)
            all_filenames2.extend(file2)

    # Stack everything into single arrays
    all_proj1 = np.concatenate(all_proj1, axis=0)
    all_proj2 = np.concatenate(all_proj2, axis=0)
    all_emb1 = np.concatenate(all_emb1, axis=0)
    all_emb2 = np.concatenate(all_emb2, axis=0)
    all_filenames1 = np.array(all_filenames1)
    all_filenames2 = np.array(all_filenames2)

    # Save in a compressed npz file
    save_dir = os.path.split(ckpt_path)[0]
    filename = os.path.join(save_dir, save_as)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset_file = os.path.basename(dataset_conf.get("labels_test"))
    
    np.savez_compressed(
        filename,
        dataset=dataset_file,
        proj1=all_proj1,
        proj2=all_proj2,
        emb1=all_emb1,
        emb2=all_emb2,
        files1=all_filenames1,
        files2=all_filenames2
    )


if __name__ == "__main__":

    ckpt_file = None

    parser = argparse.ArgumentParser(description='Path of the configuration file.')
    parser.add_argument('--ckpt', default=ckpt_file, type=str)
    parser.add_argument('--config', default=None, type=str)
    parser.add_argument('--outfile', default="inference.npz", type=str)
    args = parser.parse_args()

    if args.config is None:
        args.config = config.get_ckpt_config_file(args.ckpt)

    print(f"Running inference with checkpoint: {args.ckpt} and config: {args.config}")
    
    # config_updates = {
    #     "dataset": {
    #         "root": "Z:/marvel/marvel-fhnw/data/",
    #         "labels_test": "data/final/poleno/basic_test_20.csv",
    #     }
    # }

    conf = config.load(args.config)
    # conf = config.deep_update(conf, config_updates)

    inference(args.ckpt, conf, save_as=args.outfile)