#!/usr/bin/env bash
wget https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/stsa.binary.train -O train.txt
wget https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/stsa.binary.dev -O validation.txt
wget https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/stsa.binary.test -O test.txt