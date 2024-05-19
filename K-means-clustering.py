import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def apply_kmeans(image, k, attempts):
    # 画像を2次元配列に変換
    data = image.reshape((-1, 3))
    data = np.float32(data)

    # K-meansクラスタリングを複数回実行して最適な結果を選択
    best_labels = None
    best_centers = None
    best_inertia = float('inf')
    for _ in range(attempts):
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=300)
        kmeans.fit(data)
        if kmeans.inertia_ < best_inertia:
            best_inertia = kmeans.inertia_
            best_labels = kmeans.labels_
            best_centers = kmeans.cluster_centers_

    return best_labels, best_centers

def main(image_path, k, attempts):
    # 画像の読み込み
    image = cv2.imread(image_path)

    # 画像をBGRからRGBに変換（matplotlibで表示するため）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # クラスタリングの適用
    labels, centers = apply_kmeans(image_rgb, k, attempts)

    # クラスタリングの結果を元の形に戻す
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image_rgb.shape)

    # 元の画像の色を使用して単色化
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = (labels == label).reshape(image_rgb.shape[:2])
        mean_color = np.mean(image_rgb[mask], axis=0)
        segmented_image[mask] = mean_color

    # 画像を表示
    plt.figure(figsize=(8, 4))  # ウィンドウのサイズを変更
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image_rgb)
    plt.subplot(1, 2, 2)
    plt.title(f'Segmented Image with {k} Colors')
    plt.imshow(segmented_image)
    plt.tight_layout()  # レイアウトを自動調整
    plt.show()

    # 結果の保存
    output_path = 'segmented_image.png'
    cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

    print(f'Segmented image saved to: {output_path}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="K-means clustering color quantization with parameter tuning")
    parser.add_argument('--image_path', type=str, default='test.png', help='Path to the input image')
    parser.add_argument('--num_colors', type=int, default=5, help='Number of colors for quantization')
    parser.add_argument('--attempts', type=int, default=10, help='Number of attempts for K-means clustering')

    args = parser.parse_args()

    main(args.image_path, args.num_colors, args.attempts)
