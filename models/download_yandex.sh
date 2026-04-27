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
sam2p1_hiera_l_url="https://downloader.disk.yandex.ru/disk/bad2e17fccdf119c056a76a786484695945432f6c43ef4f7d9baad73d432460f/69effbe6/LzFnBVSI_fpmpGWRTbxW1kbcvbyj02tXOEBISnQytpdE4ANkUDkcjQfDUjGrlD4VzbiO5DwvR6ED737ps8TGIQ%3D%3D?uid=0&filename=sam2.1_hiera_large.pt&disposition=attachment&hash=kj2do0x3QFfdkK4JUq8wfyEHcgB1GNaZOhZLApFOhah48Dxz0Jf1kAxqi6z7dwQtq/J6bpmRyOJonT3VoXnDag%3D%3D%3A&limit=0&content_type=application%2Fzip&owner_uid=271816632&fsize=898083611&hid=cd874b14be76c4ea8e50faaa1dd57bbf&media_type=compressed&tknv=v3"

echo "Downloading sam2.1_hiera_large.pt checkpoint..."
$CMD $sam2p1_hiera_l_url || { echo "Failed to download checkpoint from $sam2p1_hiera_l_url"; exit 1; }

echo "All checkpoints are downloaded successfully."
