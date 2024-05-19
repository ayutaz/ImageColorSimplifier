import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture


def apply_gmm(image, n_components):
    # 画像を2次元配列に変換
    data = image.reshape((-1, 3))
    data = np.float32(data)

    # ガウシアン混合モデルの適用
    gmm = GaussianMixture(n_components=n_components).fit(data)
    labels = gmm.predict(data)
    centers = gmm.means_

    centers = np.uint8(centers)
    segmented_image = centers[labels]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image

def main(image_path, n_components):
    # 画像の読み込み
    image = cv2.imread(image_path)

    # 画像をBGRからRGBに変換（matplotlibで表示するため）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # GMMクラスタリングの適用
    segmented_image = apply_gmm(image_rgb, n_components)

    # 画像を表示
    plt.figure(figsize=(8, 4))  # ウィンドウのサイズを変更
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image_rgb)
    plt.subplot(1, 2, 2)
    plt.title('Segmented Image using GMM')
    plt.imshow(segmented_image)
    plt.tight_layout()  # レイアウトを自動調整
    plt.show()

    # 結果の保存
    output_path = 'segmented_image_gmm.png'
    cv2.imwrite(output_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

    print(f'Segmented image saved to: {output_path}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GMM clustering for image color quantization")
    parser.add_argument('--image_path', type=str, default='test.png', help='Path to the input image')
    parser.add_argument('--n_components', type=int, default=5, help='Number of components for GMM.')

    args = parser.parse_args()

    main(args.image_path, args.n_components)
