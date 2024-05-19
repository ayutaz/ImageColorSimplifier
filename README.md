# ImageColorSimplifier

画像の色を簡略化するツールです。

<!-- TOC -->

* [ImageColorSimplifier](#imagecolorsimplifier)
    * [使い方](#使い方)
        * [K-means法による色の簡略化](#k-means法による色の簡略化)
    * [ライセンス](#ライセンス)

<!-- TOC -->

## 使い方

### K-means法による色の簡略化

**インストール**

```sh
pip install opencv-python-headless numpy matplotlib scikit-learn
```

**実行方法**

```sh
python K-means-clustering.py --image_path /path/to/your/image.png --num_colors 5 --attempts 10
```

引数を指定しない場合は、デフォルト値が使用されます。

## ライセンス

[Apache License 2.0](LICENSE)
