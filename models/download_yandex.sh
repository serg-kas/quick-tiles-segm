#!/bin/bash

# ЗАГРУЗКА с yandex диска
# Можно скачать вручную по ссылке https://disk.yandex.ru/d/cMsiauLhyvhsgg

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Use either wget or curl to download the checkpoints
if command -v wget &> /dev/null; then
#    CMD="wget"
    CMD="wget -O sam2.1_hiera_large.pt"
else
    echo "Please install wget to download the checkpoints."
    exit 1
fi

# 
sam2p1_hiera_l_url="https://getfile.dokpub.com/yandex/get/https://disk.yandex.ru/d/cMsiauLhyvhsgg"

echo "Downloading sam2.1_hiera_large.pt checkpoint..."
$CMD $sam2p1_hiera_l_url || { echo "Failed to download checkpoint from $sam2p1_hiera_l_url"; exit 1; }

echo "All checkpoints are downloaded successfully."
