from PIL import Image

# 入力および出力ファイルのパスを指定
# input_path = 'input.png'          # 入力画像のパス
input_path = 'texture.png'
final_output_path = 'output.png'  # 256x256に変換した画像の保存先

# 258x2 ピクセルの画像を 256x1 ピクセルに変換
with Image.open(input_path) as img:
    width, height = img.size
    print(f"入力画像のサイズ: {width}x{height}")

    # 画像サイズの確認（258x2）
    if width != 258 or height != 2:
        raise ValueError(f"入力画像のサイズが258x2ではありません。現在のサイズ: {width}x{height}")

    # 最初の行（0行目）を抽出し、幅を256ピクセルにクロップ
    # (左, 上, 右, 下) のボックスを指定
    crop_box = (0, 0, 256, 1)
    first_row_cropped = img.crop(crop_box)
    
    # クロップ後のサイズを確認
    cropped_width, cropped_height = first_row_cropped.size
    print(f"クロップ後のサイズ: {cropped_width}x{cropped_height}")

    # 256x1 ピクセルの画像を 256x256 ピクセルに変換
    if cropped_width != 256 or cropped_height != 1:
        raise ValueError(f"クロップ後の画像のサイズが256x1ではありません。現在のサイズ: {cropped_width}x{cropped_height}")

    # 各行にコピーして256x256に拡大
    final_img = Image.new(first_row_cropped.mode, (256, 256))

    for y in range(256):
        final_img.paste(first_row_cropped, (0, y))

    # 最終画像のサイズを確認
    print(f"最終画像のサイズ: {final_img.size}")

    # 最終画像を保存
    final_img.save(final_output_path)
    print(f"256x256 ピクセルの画像を {final_output_path} として保存しました。")
