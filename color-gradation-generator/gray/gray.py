from PIL import Image

# 画像のサイズを指定
width, height = 256, 256

# RGB値 (237, 237, 237) を設定
color = (237, 237, 237)

# 新しいRGB画像を作成
image = Image.new("RGB", (width, height), color)

# 画像を保存
image.save("256x256_image.png")
print("256x256ピクセルの画像を '256x256_image.png' として保存しました。")

# 画像を表示（オプション）
image.show()
