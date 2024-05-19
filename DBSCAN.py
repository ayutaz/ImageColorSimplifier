import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN


def apply_dbscan(image, eps, min_samples):
    # 画像を2次元配列に変換
    data = image.reshape((-1, 3))
    data = np.float32(data)

    # DBSCANクラスタリングの適用
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = db.labels_

    # ノイズとして識別されたピクセルに対処
    unique_labels = np.unique(labels)
    centers = []
    for label in unique_labels:
        if label == -1:  # ノイズ
            centers.append([0, 0, 0])  # 黒に設定
        else:
            centers.append(np.mean(data[labels == label], axis=0))

    centers = np.uint8(centers)
    segmented_image = centers[labels]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image

def main(image_path, eps, min_samples, scale_factor):
    # 画像の読み込み
    image = cv2.imread(image_path)

    # 画像のサイズを縮小
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # 画像をBGRからRGBに変換（matplotlibで表示するため）
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # DBSCANクラスタリングの適用
    segmented_image = apply_dbscan(image_rgb, eps, min_samples)

    # 画像を表示
    plt.figure(figsize=(8, 4))  # ウィンドウのサイズを変更
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(cv2.resize(image, (new_width, new_height)), cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.title('Segmented Image using DBSCAN')
    plt.imshow(segmented_image)
    plt.tight_layout()  # レイアウトを自動調整
    plt.show()

    # 結果の保存
    output_path = 'segmented_image_dbscan.png'
    cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

    print(f'Segmented image saved to: {output_path}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DBSCAN clustering for image color quantization")
    parser.add_argument('--image_path', type=str, default='test.png', help='Path to the input image')
    parser.add_argument('--eps', type=float, default=10.0, help='The maximum distance between two samples for one to be considered as in the neighborhood of the other.')
    parser.add_argument('--min_samples', type=int, default=10, help='The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.')
    parser.add_argument('--scale_factor', type=float, default=0.1, help='Factor to scale the image down by.')

    args = parser.parse_args()

    main(args.image_path, args.eps, args.min_samples, args.scale_factor)
