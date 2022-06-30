# Copyright 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
from tqdm import tqdm

import monai.transforms as mt
from monai.apps.datasets import DecathlonDataset
from monai.apps.utils import download_and_extract
from monai.utils.enums import PostFix


class SliceWithMaxNumLabelsd(mt.MapTransform):
    """Get a 2D slice of a 3D volume with the maximum number of non-zero voxels in the label."""

    def __init__(self, keys, label_key):
        self.keys = keys
        self.label_key = label_key

    def __call__(self, data):
        d = dict(data)
        im = d[self.label_key]
        q = (im > 0).reshape(-1, im.shape[-1]).sum(0)
        _slice = q.argmax(0)
        for key in self.keys:
            d[key] = d[key][..., _slice]
        return d


class SaveSliced(mt.MapTransform):
    """Save the 2D slice to file."""

    def __init__(self, keys, path):
        self.keys = keys
        self.path = path

    def __call__(self, data):
        d = {}
        for key in self.keys:
            fname = os.path.basename(data[PostFix.meta(key)]["filename_or_obj"])
            path = os.path.join(self.path, key, fname)
            nib.save(nib.Nifti1Image(np.asarray(data[key]), np.eye(4)), path)
            d[key] = path
        return d


def download_data(task, download_path):
    """Download data (if necessary) and return a list of images and corresponding labels."""
    resource, md5 = DecathlonDataset.resource[task], DecathlonDataset.md5[task]
    compressed_file = os.path.join(download_path, task + ".tar")
    data_dir = os.path.join(download_path, task)
    if not os.path.exists(data_dir):
        download_and_extract(resource, compressed_file, download_path, hash_val=md5)
    images = sorted(glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    labels = sorted(glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [{"image": image, "label": label} for image, label in zip(images, labels)]
    return data_dicts


def main(task, path, download_path):
    data_dicts = download_data(task, download_path)
    keys = list(data_dicts[0].keys())
    for key in keys:
        os.makedirs(os.path.join(path, key), exist_ok=True)

    # list of transforms to convert to 2d slice
    transform_2d_slice = mt.Compose(
        [
            mt.LoadImaged(keys),
            mt.AsChannelFirstd("image"),
            mt.AddChanneld("label"),
            SliceWithMaxNumLabelsd(keys, "label"),
            SaveSliced(keys, path),
        ]
    )

    for data in tqdm(data_dicts):
        # skip the 2d extraction if possible
        if len(glob(os.path.join(path, "*", os.path.basename(data["image"])))) == 2:
            continue
        # extract the slice
        _ = transform_2d_slice(data)


def print_input_args(args):
    data = dict(sorted(args.items()))
    col_width = max(len(i) for i in data.keys())
    for k, v in data.items():
        print(f"\t{k:<{col_width}}: {v if v is not None else 'None'}")


if __name__ == "__main__":
    default_download_root_dir = os.environ.get("MONAI_DATA_DIRECTORY")
    if default_download_root_dir is None:
        default_download_root_dir = tempfile.mkdtemp()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # add arguments
    parser.add_argument(
        "-t",
        "--task",
        help="Task to generate the 2d dataset from.",
        type=str,
        choices=["Task01_BrainTumour"],
        default="Task01_BrainTumour",
    )
    parser.add_argument(
        "-d", "--download_path", help="Path for downloading full dataset.", type=str, default=default_download_root_dir
    )
    parser.add_argument("-p", "--path", help="Path for output. Default: download_path/{task}2D", type=str)

    # parse input arguments
    args = vars(parser.parse_args())

    # set default output path if necessary
    if args["path"] is None:
        args["path"] = os.path.join(args["download_path"], args["task"] + "2D")

    # print args and run the 2d extraction
    print_input_args(args)
    main(**args)
