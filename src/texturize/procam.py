from __future__ import annotations
import math
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F

import psutil               # type: ignore
import h5py                 # type: ignore
import numpy as np          # type: ignore


class Procam:
    def __init__(self, matrix, metadata, crop=None):
        self.matrix = matrix
        self.basis_size = (metadata.basis_height, metadata.basis_width)
        self.projection_size = (metadata.image_height, metadata.image_width)
        self.crop = (
            crop
            if crop is not None
            else (0, 0, self.projection_size[1], self.projection_size[0])
        )

    def __call__(self, image):
        # resize image to basis size
        orig_size = image.shape[2:]
        image_input = F.interpolate(image, size=self.basis_size, mode='area')

        # project image
        n, c, h, w = image_input.shape
        image_input_reshaped = image_input.permute(0, 3, 2, 1).reshape(n, h * w, 3)
        image_projected = (1.0 * torch.einsum(
            'hwcb,nbc->nhwc', self.matrix, image_input_reshaped
        )).clamp(0.0, 1.0)

        # crop projection and resize it back to original image size
        image_projected_cropped = (
            image_projected[:, self.crop[1]:self.crop[3], self.crop[0]:self.crop[2], :]
        )
        return F.interpolate(
            image_projected_cropped.permute(0, 3, 1, 2),
            size=orig_size,
            mode='area'
        )


class ProcamSimple:
    def __init__(self, background=None, brightness=1.0):
        self.background = background
        self.brightness = brightness
        if background is None:
            self.render_fn = lambda img: img
        else:
            self.render_fn = (
                lambda img: img * F.interpolate(background, size=img.shape[2:], mode='area')
            )

    def __call__(self, image):
        return self.brightness * self.render_fn(image)


class Metadata:
    def __init__(
        self, files: Optional[List[dict]], basis_width: int, basis_height: int,
        image_width: int, image_height: int
    ):
        if files is None:
            self.files = [{}]   # type: List[dict]
        else:
            self.files = files

        self.basis_width = basis_width
        self.basis_height = basis_height
        self.image_width = image_width
        self.image_height = image_height

        # derived quantities
        self.n_bases = basis_width * basis_height
        self.n_pixels = image_width * image_height

    def __str__(self) -> str:
        return ('{{[{}, {}] [{}, {}]}}'.format(
            self.basis_width, self.basis_height,
            self.image_width, self.image_height
        ))

    def get_numpy_array(self) -> np.ndarray:
        return np.array(
            [
                self.basis_width,
                self.basis_height,
                self.image_width,
                self.image_height
            ], dtype=np.int32
        )

    @staticmethod
    def create_from_array(array: np.ndarray) -> Metadata:
        return Metadata(None, array[0], array[1], array[2], array[3])

    @staticmethod
    def create_from_matrix_file(path: str) -> Metadata:
        with h5py.File(path, 'r') as matrix_file:
            return Metadata.create_from_array(
                matrix_file['metadata']
            )


class HDF5Params:
    def __init__(
        self, matrix_shape: Tuple[int, int, int, int], insert_dim: int = 3,
        cache_size: int = 100, dtype: type = np.float32,
        rdcc_nslots: int = -1, quiet: bool = False
    ):
        self.rdcc_nbytes = cache_size * 1024 * 1024
        self.rdcc_w0 = 1

        if psutil.virtual_memory().available < self.rdcc_nbytes:
            raise ValueError(
                'Requested HDF5 cache size is larger than available memory!'
            )

        w_size = np.dtype(dtype).itemsize
        self.matrix_shape = matrix_shape
        self.matrix_size = np.prod(self.matrix_shape, dtype=np.uint64) * w_size
        chunk_shape = [x for x in matrix_shape]    # tuple to list
        chunk_shape[insert_dim] = 1
        self.chunk_shape = tuple(chunk_shape)      # list to tuple
        self.chunk_size = np.prod(self.chunk_shape, dtype=np.uint64) * w_size

        if self.chunk_size > self.rdcc_nbytes:
            raise ValueError(
                'HDF5 chunk size is larger than requested cache size!'
            )

        self.n_chunks = matrix_shape[insert_dim]
        self.n_chunks_in_cache = min(
            math.floor(self.rdcc_nbytes / self.chunk_size), self.n_chunks
        )

        # setting this according to h5py docs
        # (http://docs.h5py.org/en/stable/high/file.html#chunk-cache)
        self.rdcc_nslots = (rdcc_nslots
                            if rdcc_nslots != -1
                            else math.ceil(
                                self.rdcc_nbytes / self.chunk_size) * 100
                            )

        if not quiet:
            print(
                (
                    'HDF5 parameters configured to be the following:\n'
                    '\trdcc_nbytes: {}\n'
                    '\trdcc_w0: {}\n'
                    '\trdcc_nslots: {}\n'
                    '\tmatrix_shape: {}\n'
                    '\tmatrix_size: {}\n'
                    '\tchunk_shape: {}\n'
                    '\tchunk_size: {}\n'
                ).format(
                    self.rdcc_nbytes, self.rdcc_w0, self.rdcc_nslots,
                    self.matrix_shape, self.matrix_size,
                    self.chunk_shape, self.chunk_size
                )
            )

    @staticmethod
    def create_from_metadata(
        metadata: Metadata, cache_size: int, rdcc_nslots: int = -1
    ) -> HDF5Params:
        return HDF5Params(
            (metadata.image_height, metadata.image_width, 3, metadata.n_bases),
            cache_size=cache_size, rdcc_nslots=rdcc_nslots
        )


def load_matrix_wrap(path: str, cache_size: int = 1000) -> np.ndarray:
    metadata = Metadata.create_from_matrix_file(path)
    hdf5_params = HDF5Params.create_from_metadata(metadata, cache_size)
    return load_matrix(path, hdf5_params)


def load_matrix(
    path: str, hdf5_params: HDF5Params, start: int = 0,
    end: Optional[int] = None, matrix_name: Optional[str] = None
) -> np.ndarray:
    with h5py.File(
        path, 'r', rdcc_nbytes=hdf5_params.rdcc_nbytes,
        rdcc_w0=hdf5_params.rdcc_w0, rdcc_nslots=hdf5_params.rdcc_nslots
    ) as matrix_file:
        return matrix_file['matrix'][:, :, :, start:end]
