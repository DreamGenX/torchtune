# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.data._utils import truncate
from torchtune.datasets._packed import PackedDataset


class PreTokenizedDataset(Dataset):
    """
    Freeform dataset for any pre-tokenized corpus.

    Args:
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
            for more details.
        tokens_column (str): name of column in the sample that contains the pre-tokenized tokens. Default is "tokens".
        mask_column (str): name of column in the sample that contains the pre-tokenized mask. Default is "mask".
        max_seq_len (int): The maximum sequence length to truncate the tokens to. Default is None, in which case no truncation is done.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.
    """

    def __init__(
        self,
        source: str,
        tokens_column: str = "tokens",
        mask_column: str = "mask",
        max_seq_len: int = None,
        filter_fn: Optional[Callable] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._data = load_dataset(source, **load_dataset_kwargs)
        self._tokens_column = tokens_column
        self._mask_column = mask_column
        self.max_seq_len = max_seq_len

        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        tokens = sample[self._tokens_column]
        mask = sample[self._mask_column] if self._mask_column else None

        if mask is not None and len(mask) != len(tokens):
            raise ValueError("Mask and tokens must be the same length")

        # Truncate if needed, but don't coerce EOS id
        if self.max_seq_len is not None:
            tokens = truncate(tokens, self.max_seq_len - 1)
            if mask is not None:
                mask = mask[:len(tokens)]

        if mask is not None:
            labels = list(np.where(
                mask,
                CROSS_ENTROPY_IGNORE_IDX,
                tokens,
            ))
        else:
            labels = tokens.copy()

        return {"tokens": tokens, "labels": labels}


def pre_tokenized_dataset(
    tokenizer: any,
    *,
    source: str,
    tokens_column: str = "tokens",
    mask_column: str = "mask",
    max_seq_len: int = None,
    packed: bool = False,
    split_across_pack: bool = True,
    split: str = "train",
    filter_fn: Optional[Callable] = None,
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[PreTokenizedDataset, PackedDataset]:
    """
    Build a configurable dataset from a freeform, unstructured text corpus similar
    to datasets used in pre-training. This method should be
    used to configure a custom text dataset from the yaml config instead of
    using :class:`~torchtune.datasets.PreTokenizedDataset` directly, as it is made to be config friendly.

    Args:
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
            for more details.
        tokens_column (str): name of column in the sample that contains the pre-tokenized tokens. Default is "tokens".
        mask_column (str): name of column in the sample that contains the pre-tokenized mask. Default is "mask".
        max_seq_len (int): The maximum sequence length to truncate the tokens to. Default is None, in which case no truncation is done.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        split_across_pack (bool): if the last sample in a pack does not fit in ``max_seq_len``,
            split the sample into the next pack, or move it entirely to the beginning of the next pack.
            For pre-training, typically this is set to True for general text completion. For
            fine-tuning, typically this is set to False to avoid truncating sentences in instruct
            tuning. This argument is ignored if ``packed=False``. Default is True.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.

    Examples:
        >>> from torchtune.datasets import pre_tokenized_dataset
        >>> dataset = pre_tokenized_dataset(
        ...   source="allenai/c4",
        ...   data_dir="realnewslike",
        ...   packed=False,
        ...   split="train",
        ... )

    This can also be accomplished via the yaml config::

        dataset:
            _component_: torchtune.datasets.pre_tokenized_dataset
            source: allenai/c4
            data_dir: realnewslike
            packed: False
            split: train

    Returns:
        Union[PreTokenizedDataset, PackedDataset]: the configured :class:`~torchtune.datasets.TextCompletionDataset`
            or :class:`~torchtune.datasets.PackedDataset` if ``packed=True``

    Raises:
        ValueError: If ``packed=True`` and ``max_seq_len`` is not set.
    """
    ds = PreTokenizedDataset(
        source=source,
        tokens_column=tokens_column,
        mask_column=mask_column,
        max_seq_len=max_seq_len,
        split=split,
        filter_fn=filter_fn,
        **load_dataset_kwargs,
    )
    if packed:
        if max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set."
            )
        return PackedDataset(
            ds, max_seq_len=max_seq_len, split_across_pack=split_across_pack
        )
    return ds
