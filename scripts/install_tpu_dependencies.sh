#!/bin/bash

echo "sudo add-apt-repository ppa:alessandro-strada/ppa"
echo -n "Execute?"; read
sudo add-apt-repository ppa:alessandro-strada/ppa

echo "sudo apt update && sudo apt install -y python3-opencv google-drive-ocamlfuse"
echo -n "Execute?"; read
sudo apt update && sudo apt install -y python3-opencv google-drive-ocamlfuse

echo "pip install \"jax[tpu]>=0.2.16\" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
echo -n "Execute?"; read
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

echo "pip install flax gin-config pydrive"
echo -n "Execute?"; read
pip install flax gin-config

echo "python3 -c \"import jax; print(jax.device_count()); print(jax.numpy.add(1, 1))\""
echo -n "Execute?"; read
python3 -c "import jax; print(jax.device_count()); print(jax.numpy.add(1, 1))"

echo "Mount Google Drive folder. Click on link and authenticate to provide Google Drive access."
echo -n "Execute?"; read
mkdir ~/gdfuse-mount
env PATH=`pwd`:$PATH google-drive-ocamlfuse -config gdfuse-config -browser show-gauth-link ~/gdfuse-mount

echo "Google Drive folder is now available on ~/gdfuse-mount"
