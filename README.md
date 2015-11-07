# Study_Chanier
Chainerの勉強用．MNISTを学習する．

# まず，Chainerを動かす．
http://chainer.org/
環境はPython 2.7系なので注意が必要．
私の環境では，Python2.7.10でChanier1.4.0の動作確認をしている．（2015/11/7）

## Pythonの環境設定
Python環境として2.7.10の環境をインストールする．
おすすめはanaconda．
この辺が詳しい．
http://qiita.com/knao124/items/edb6a1b5a62410768f05

## インストール

```
pip install chainer
```

## ChainerでMNISTを学習する．

```
wget https://github.com/pfnet/chainer/archive/v1.4.0.tar.gz
tar xzf v1.4.0.tar.gz
python chainer-1.4.0/examples/mnist/train_mnist.py
```

CUDAを使うときは，

```
python chainer-1.4.0/examples/mnist/train_mnist.py --gpu 0
```

--gpuの引数は，どのGPUを使うか選択する．


