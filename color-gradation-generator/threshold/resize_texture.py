from PIL import Image

def resize(originalMin, originalMax, thresholdMin, thresholdMax, inputPath, outputPath):
    try:
        axis =  (originalMax - originalMin) / (thresholdMax - thresholdMin)
        print(f"axis: {axis}")
    except ZeroDivisionError:
        print("エラー: thresholdMaxとthresholdMinが同じ値です。")
        return

    try:
        with Image.open(inputPath) as img:
            original_width, original_height = img.size
            print(f"original image size: {original_width}x{original_height}")
            
            resized_width = int(original_width * axis)
            resized_img = img.resize((resized_width, original_height), Image.BICUBIC)
            
            padding_width = abs(original_width - resized_width)
            print(f"padding_width: {padding_width}")
            
            # 最右端1ピクセル
            # TODO: thresholdによって左右どちらを伸ばすか決める
            right_color = resized_img.crop((resized_width - 1, 0, resized_width, original_height))
            padding_right = right_color.resize((padding_width, original_height))

            result_img = concat(resized_img, padding_right)
            result_img = result_img.resize((original_width, original_height))
            
            result_img.save(outputPath)
            print(f"画像が {outputPath} に保存されました。")

    except FileNotFoundError:
        print(f"エラー: ファイル {inputPath} が存在しません。")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

def concat(left_img, right_img):
    l_width, l_height = left_img.size
    r_width, _ = right_img.size
    result = Image.new(left_img.mode, (l_width + r_width, l_height))

    result.paste(left_img, (0,0))
    result.paste(right_img, (l_width, 0))

    return result


if __name__ == "__main__":
    inputPath = "input.png"
    inputPath = "input-blue-white-red.png"
    outputPath = "resize_texture.png"

    original_min = 0.0
    original_max = 13.260545217332858

    threshold_min = 0.0
    threshold_max = 4.4201817391

    resize(original_min, original_max, threshold_min, threshold_max, inputPath, outputPath)
