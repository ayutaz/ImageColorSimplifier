**Image Color Simplifier**

このプロジェクトは、さまざまなクラスタリング手法を使用して画像の色を量子化するツールを提供します。以下の方法が実装されています：

1. K-meansクラスタリング
2. DBSCAN（Density-Based Spatial Clustering of Applications with Noise）
3. GMM（Gaussian Mixture Model）

<!-- TOC -->
  * [必要なライブラリ](#必要なライブラリ)
  * [K-meansクラスタリング](#k-meansクラスタリング)
  * [DBSCANクラスタリング](#dbscanクラスタリング)
  * [GMMクラスタリング](#gmmクラスタリング)
  * [ライセンス](#ライセンス)
<!-- TOC -->

## 必要なライブラリ

以下のコマンドを使用して、必要なライブラリをインストールします：

```sh
pip install -r requirements.txt
```

## K-meansクラスタリング
K-meansクラスタリングは、以下のコマンドで実行できます：

```sh
python K-means-clustering.py --image_path <画像のパス> --num_colors <色の数> --attempts <試行回数>
```
例：
```sh
python K-means-clustering.py --image_path test.png --num_colors 5 --attempts 10
```
パラメータ：
--image_path: 入力画像のパス
--num_colors: 量子化する色の数
--attempts: K-meansクラスタリングの試行回数

## DBSCANクラスタリング
DBSCANクラスタリングは、以下のコマンドで実行できます：

```sh
python DBSCAN-clustering.py --image_path <画像のパス> --eps <eps値> --min_samples <最小サンプル数> --scale_factor <縮小率>
```
例：
```sh
python DBSCAN-clustering.py --image_path test.png --eps 10.0 --min_samples 10 --scale_factor 0.5
```

パラメータ：
--image_path: 入力画像のパス
--eps: 2つのサンプルが同じクラスタに属するための最大距離
--min_samples: コアポイントと見なされるための近傍内の最小サンプル数
--scale_factor: 画像を縮小する割合

## GMMクラスタリング
GMMクラスタリングは、以下のコマンドで実行できます：

```shell
python GMM-clustering.py --image_path <画像のパス> --n_components <コンポーネント数>

```

例：
```sh
python GMM-clustering.py --image_path test.png --n_components 5
```
パラメータ：
--image_path: 入力画像のパス
--n_components: GMMのコンポーネント（クラスタ）の数

## ライセンス
[Apache License 2.0](LICENSE)