import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


def main(image_path, k):
    # 画像の読み込み
    image = cv2.imread(image_path)

    # 画像をBGRからRGBに変換（matplotlibで表示するため）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 画像の形状を取得
    h, w, _ = image.shape

    # 画像を2次元配列に変換
    data = image.reshape((-1, 3))
    data = np.float32(data)

    # K-meansクラスタリングによる色の数を減らす
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # クラスタリングの結果を元の形に戻す
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape((h, w, 3))

    # 画像を表示
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image_rgb)
    plt.subplot(1, 2, 2)
    plt.title(f'Segmented Image with {k} Colors')
    plt.imshow(segmented_image)
    plt.show()

    # 結果の保存
    output_path = '/mnt/data/segmented_image.png'
    cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

    print(f'Segmented image saved to: {output_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-means clustering color quantization")
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('num_colors', type=int, help='Number of colors for quantization')

    args = parser.parse_args()

    main(args.image_path, args.num_colors)
