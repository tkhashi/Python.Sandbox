from PIL import Image

# 元の画像を開く（256x1のPNGファイル）
input_image = Image.open("input_image.png")

# 256x256の新しい画像を作成（RGBカラーモード）
output_image = Image.new("RGB", (256, 256))

# 元の画像の行を新しい画像の全ての行にコピー
for y in range(256):
    output_image.paste(input_image, (0, y))

# 新しい画像を保存
output_image.save("expanded_image.png")

# 画像を表示（任意）
output_image.show()

