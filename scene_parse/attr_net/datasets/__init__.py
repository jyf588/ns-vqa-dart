from torch.utils.data import DataLoader
from .dash_object_loader import DashTorchDataset
from .clevr_object import ClevrObjectDataset
from .clevr_dart_object import ClevrDartObjectDataset


def get_dataset(opt, split):
    if opt.dataset == "dash":
        if split == "val":
            min_img_id = opt.end_id
            max_img_id = None
        elif split in ["train", "test"]:
            min_img_id = opt.start_id
            max_img_id = opt.end_id
        else:
            raise ValueError(f"Invalid split: {split}.")

        ds = DashTorchDataset(
            dataset_dir=opt.dataset_dir,
            height=opt.height,
            width=opt.width,
            min_img_id=min_img_id,
            max_img_id=max_img_id,
            use_attr=opt.pred_attr,
            use_size=opt.pred_size,
            use_position=opt.pred_position,
            use_up_vector=opt.pred_up_vector,
            coordinate_frame=opt.coordinate_frame,
            split=split,
        )

    elif opt.dataset == "clevr":
        if split == "train":
            ds = ClevrObjectDataset(
                opt.clevr_mini_ann_path,
                opt.clevr_mini_img_dir,
                "mini",
                max_img_id=opt.split_id,
                concat_img=opt.concat_img,
            )
        elif split == "val":
            ds = ClevrObjectDataset(
                opt.clevr_mini_ann_path,
                opt.clevr_mini_img_dir,
                "mini",
                min_img_id=opt.split_id,
                concat_img=opt.concat_img,
            )
        elif split == "test":
            ds = ClevrObjectDataset(
                opt.clevr_val_ann_path,
                opt.clevr_val_img_dir,
                "val",
                concat_img=opt.concat_img,
            )
        else:
            raise ValueError("Invalid dataset split: %s" % split)
    elif opt.dataset == "clevr_dart":
        if split == "train":
            ds = ClevrDartObjectDataset(
                opt,
                opt.clevr_mini_ann_path,
                opt.clevr_mini_img_dir,
                "mini",
                max_img_id=opt.split_id,
            )
        elif split == "val":
            ds = ClevrDartObjectDataset(
                opt,
                opt.clevr_mini_ann_path,
                opt.clevr_mini_img_dir,
                "mini",
                min_img_id=opt.split_id,
            )  # TODO: tmp
        elif split == "test":
            ds = ClevrDartObjectDataset(
                opt,
                opt.clevr_mini_ann_path,
                opt.clevr_mini_img_dir,
                "mini",
                min_img_id=opt.split_id,
            )  # TODO: tmp
        else:
            raise ValueError("Invalid dataset split: %s" % split)
    else:
        raise ValueError("Invalid datsaet %s" % opt.dataset)
    return ds


def get_dataloader(opt, split):
    ds = get_dataset(opt, split)
    loader = DataLoader(
        dataset=ds,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=opt.shuffle_data,
    )
    return loader
