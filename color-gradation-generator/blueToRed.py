from PIL import Image

def generate_blue_to_red_gradient(width=256, height=256, filename='blue_to_red_gradient.png'):
    """
    青から赤への水平グラデーション画像を生成します。

    Parameters:
    - width (int): 画像の幅（ピクセル）
    - height (int): 画像の高さ（ピクセル）
    - filename (str): 保存する画像ファイル名
    """
    # 新しいRGB画像を作成
    image = Image.new("RGB", (width, height))
    pixels = image.load()

    for x in range(width):
        # グラデーションの進行度合いを0から1に正規化
        t = x / (width - 1)

        # 青から赤への線形補間
        # 青 (0, 0, 255) から赤 (255, 0, 0) への遷移
        r = int(t * 255)
        g = 0
        b = int((1 - t) * 255)

        color = (r, g, b)

        # 現在のx列の全てのピクセルに色を設定
        for y in range(height):
            pixels[x, y] = color

    # 画像を保存
    image.save(filename)
    print(f"{width}x{height} の青から赤へのカラースケール画像 '{filename}' を生成しました。")

if __name__ == "__main__":
    generate_blue_to_red_gradient()

