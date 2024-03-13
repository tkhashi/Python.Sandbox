import cv2
import numpy as np
import math
import glob
 
##
# @brief AKAZEによる画像特徴量取得
# @param img 特徴量を取得したい画像（RGB順想定）
# @param pt1 特徴量を求める開始座標 tuple (default 原点)
# @param pt2 特徴量を求める終了座標 tuple (default None=画像の終わり位置)
# @return key points
def get_keypoints(img, pt1=(0, 0), pt2=None):
    if pt2 is None:
        pt2 = (img.shape[1], img.shape[0])
 
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = cv2.rectangle(np.zeros_like(gray), pt1, pt2, color=1, thickness=-1)
    sift = cv2.AKAZE_create()
 
    # find the key points and descriptors with AKAZE
    return sift.detectAndCompute(gray, mask=mask)
 
 
##
# @brief 特徴記述子kp2/des2にマッチするような pointを求める
# @param kp1 合わせたい画像のkeypoint
# @param des1 合わせたい画像の特徴記述
# @param kp2 ベースとなる画像のkeypoint
# @param des2 ベースとなる画像の特徴記述
# @return apt1 kp1の座標　apt2 それに対応するkp2
def get_matcher(kp1, des1, kp2, des2):
    if len(kp1) == 0 or len(kp2) == 0:
        return None
 
    # Brute-Force Matcher生成
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
 
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
 
    if len(good) == 0:
        return None
 
    target_position = []
    base_position = []
    # x,y座標の取得
    for g in good:
        target_position.append([kp1[g.queryIdx].pt[0], kp1[g.queryIdx].pt[1]])
        base_position.append([kp2[g.trainIdx].pt[0], kp2[g.trainIdx].pt[1]])
 
    apt1 = np.array(target_position)
    apt2 = np.array(base_position)
    return apt1, apt2
 
 
##
# @brief Affine行列から回転角、回転中心、拡大率を求める
# @param mat アフィン行列
# @return da 回転角度(度) center　回転中心　s 拡大率
def getRotateShift(mat):
    da = -math.atan2(mat[1, 0], mat[0, 0])  # ラジアン
    s = mat[0, 0] / math.cos(da)  # 拡大率
 
    m = np.zeros([2, 2])
    m[0, 0] = 1 - mat[0, 0]
    m[0, 1] = -mat[0, 1]
    m[1, 0] = mat[0, 1]
    m[1, 1] = m[0, 0]
 
    mm = np.zeros([2, 1])
    mm[0, 0] = mat[0, 2]
    mm[1, 0] = mat[1, 2]
 
    center = np.dot(np.linalg.inv(m), mm).reshape([2])
 
    return math.degrees(da), center, s
 
 
##
# @brief アフィン行列mtxfitの変化量を抑制する
# @param matfit アフィン行列 (3x3)
# @param cx  中心のx座標
# @param cy  中心のy座標
# @param feedback  フィードバック量（1.0で全フィードバック）
# @return mtx  mtxfitからfeedbackだけ変化が抑制された行列 (3x3)
def get_suppressed_mtx(mtxfit, cx, cy, feedback):
 
    angle, center, scale = getRotateShift(mtxfit)
 
    # 倍率、回転量の抑制 (逆回し)
    scale = 1/scale
    angle = (feedback-1) * angle
 
    # 中心座標の移動先を計算
    mx = mtxfit[0, 0] * cx + mtxfit[0, 1] * cy + mtxfit[0, 2]
    my = mtxfit[1, 0] * cx + mtxfit[1, 1] * cy + mtxfit[1, 2]
 
    # 抑制された行列 mtx の生成 (3x3)
    mtxback = cv2.getRotationMatrix2D((mx, my), angle, scale)
    mtxback = np.concatenate((mtxback, np.array([[0.0, 0.0, 1.0]])))
    mtx = np.dot(mtxback, mtxfit)
 
    # 中心座標の移動先を計算
    mx = mtx[0, 0] * cx + mtx[0, 1] * cy + mtx[0, 2]
    my = mtx[1, 0] * cx + mtx[1, 1] * cy + mtx[1, 2]
 
    # 移動量を抑制 (画像中心に寄せる)
    mtx[0, 2] = (mx * feedback + cx * (1-feedback)) - mtx[0, 0] * cx - mtx[0, 1] * cy
    mtx[1, 2] = (my * feedback + cy * (1-feedback)) - mtx[1, 0] * cx - mtx[1, 1] * cy
 
    return mtx
 
 
####################################
## main
if __name__ == '__main__':
    # 動画ファイルを開く
    cap = cv2.VideoCapture('C:/Users/k_tak/Downloads/boring_trimmed.mp4')

    # 出力用の動画ファイル設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # コーデック指定
    fps = cap.get(cv2.CAP_PROP_FPS) # 元動画のFPSを取得
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('C:/Users/k_tak/Downloads/output_boring_trimmed.mp4', fourcc, fps, (frame_width, frame_height))

    ret, img1 = cap.read()
    if not ret:
        print("Failed to read video")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        exit()

    h, w = img1.shape[:2]

    # 探索領域を絞る
    sx = w//4
    ex = w - sx
    sy = h//4
    ey = h - sy

    mtx1 = np.eye(3)
    kp1, des1 = get_keypoints(img1, (sx, sy), (ex, ey))

    frame_count = 0

    while(cap.isOpened()):
        ret, img2 = cap.read()
        if not ret:
            break

        kp2, des2 = get_keypoints(img2, (sx, sy), (ex, ey))
        match_result = get_matcher(kp2, des2, kp1, des1)
        
        # match_resultがNoneの場合の処理を追加
        if match_result is None:
            print(f"Frame {frame_count}: No matching keypoints found. Skipping frame.")
            out.write(img1)  # 前のフレームをそのまま使用
            frame_count += 1
            continue

        pt1, pt2 = match_result

        # アフィン行列の推定
        mtx, inliers = cv2.estimateAffinePartial2D(pt1, pt2, method=cv2.RANSAC)
        if mtx is not None:
            mtx = np.concatenate((mtx, np.array([[0.0, 0.0, 1.0]])))

            # 現在フレームへの行列
            mtx2 = np.dot(mtx1, mtx)
            mtx2 = get_suppressed_mtx(mtx2, w//2, h//2, 0.8)

            img1 = cv2.warpAffine(img2, mtx2[:2, :], (w, h))

            mtx1 = mtx2
            kp1 = kp2
            des1 = des2

            out.write(img1)  # 補正されたフレームを動画に書き出し

            frame_count += 1
        else:
            print(f"Frame {frame_count}: Unable to estimate affine transformation. Skipping frame.")
            out.write(img1)  # アフィン変換が推定できない場合、前のフレームをそのまま使用

    # リソース解放
    cap.release()
    out.release()
    cv2.destroyAllWindows()




    # flist = glob.glob("E:\\tmp\\blog\\20200729\\*.jpg")  # ソースの写真群がある場所
    # save = "E:\\tmp\\blog\\20200729\\dst\\"
    # img1 = cv2.imread(flist.pop(0))
    # cv2.imwrite(save + "{0:03d}".format(0) + ".jpg", img1[180:-180, 320:-320, :])
 
    # h, w = img1.shape[:2]
 
    # # 探索領域を絞る
    # sx = w//4
    # ex = w - sx
    # sy = h//4
    # ey = h - sy
 
    # mtx1 = np.eye(3)
    # kp1, des1 = get_keypoints(img1, (sx, sy), (ex, ey))
 
    # for index, fname in enumerate(flist):
    #     img2 = cv2.imread(fname)
 
    #     kp2, des2 = get_keypoints(img2, (sx, sy), (ex, ey))
    #     pt1, pt2 = get_matcher(kp2, des2, kp1, des1)
 
    #     # アフィン行列の推定
    #     mtx = cv2.estimateAffinePartial2D(pt1, pt2)[0]
    #     mtx = np.concatenate((mtx, np.array([[0.0, 0.0, 1.0]])))
 
    #     # 現在フレームへの行列
    #     mtx2 = np.dot(mtx1, mtx)
    #     mtx2 = get_suppressed_mtx(mtx2, w//2, h//2, 0.8)
 
    #     # 前フレームへの行列
    #     mtx4 = np.dot(mtx2, np.linalg.inv(mtx))
    #     mtx4 = np.dot(mtx4, np.linalg.inv(mtx1))
 
    #     img1 = cv2.warpAffine(img1, mtx4[:2, :], (w, h))
    #     img1 = cv2.warpAffine(img2, mtx2[:2, :], (w, h), borderMode=cv2.BORDER_TRANSPARENT, dst=img1)
 
    #     mtx1 = mtx2
    #     kp1 = kp2
    #     des1 = des2
 
    #     cv2.imwrite(save + "{0:03d}".format(index+1) + ".jpg", img1[180:-180, 320:-320, :])
