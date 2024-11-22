from PIL import Image

# 画像のサイズを定義
width, height = 256, 1

# 新しい画像を作成（RGBカラーモード）
image = Image.new("RGB", (width, height))

# グラデーションを設定
for i in range(width):
    if i < width // 2:
        # 青から緑へのグラデーション
        r = 0
        g = int(i * 255 / (width // 2 - 1))
        b = int(255 - g)
    else:
        # 緑から赤へのグラデーション
        r = int((i - width // 2) * 255 / (width // 2 - 1))
        g = int(255 - r)
        b = 0
    
    # ピクセルに色を設定
    image.putpixel((i, 0), (r, g, b))

# 画像を保存
image.save("blue_green_red_gradient.png")

# 画像を表示（任意）
image.show()
