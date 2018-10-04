#!/bin/bash

cd dataset; mkdir glove
cd glove

ZIPFILE=radglove.800M.100d.zip

echo "==> Downloading glove vectors trained on 4.5M radiology reports..."
wget http://nlp.stanford.edu/zyh/$ZIPFILE

echo "==> Unzipping glove vectors..."
unzip $ZIPFILE
rm $ZIPFILE

echo "==> Done."

