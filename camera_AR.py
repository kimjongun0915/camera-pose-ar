import numpy as np
import cv2 as cv

# === 비디오와 보정 데이터 ===
video_file = 'D:/python/calibrate_camera/chessboard.mp4'
K = np.array([[631.12372751, 0, 204.13870644],
              [0, 626.76168609, 363.15842532],
              [0, 0, 1]])
dist_coeff = np.array([0.03184266, -0.26914475, -0.00107398, 0.00244667, 0.73158658])

board_pattern = (6, 8)
board_cellsize = 0.025
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# === 비디오 열기 ===
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# === 체스보드의 3D 좌표(캘리브레이션용) ===
obj_points = board_cellsize * np.array([
    [c, r, 0] for r in range(board_pattern[1]) 
              for c in range(board_pattern[0])
])

# === 원기둥(Coke Can) 파라미터 ===
#   중심 (체스보드 좌표계), 반지름, 높이, 세그먼트 개수
center = np.array([3.0, 3.0])  # (col, row)
radius = 0.5                  # 체스보드 칸 단위
height = 1.0                  # 체스보드 칸 단위 (z=-height)
num_segments = 36             # 원 둘레를 근사할 다각형 분할 수

# === 색상 정의 (B, G, R) ===
COLOR_SILVER = (200, 200, 200)  # 은색
COLOR_RED = (0, 0, 255)         # 빨간색

while True:
    valid, img = video.read()
    if not valid:
        break

    # === 체스보드 코너 찾고, solvePnP로 rvec, tvec 추정 ===
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        _, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # 1) 원 둘레 점들(아래 원, 윗 원) 생성
        angles = np.linspace(0, 2*np.pi, num_segments, endpoint=False)
        bottom_circle = []
        top_circle = []
        for theta in angles:
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            bottom_circle.append([x, y, 0])       # z=0
            top_circle.append([x, y, -height])    # z=-height(위쪽)
        
        bottom_circle = np.array(bottom_circle, dtype=np.float32) * board_cellsize
        top_circle = np.array(top_circle, dtype=np.float32) * board_cellsize

        # 2) 투영 (3D→2D)
        pts_bottom, _ = cv.projectPoints(bottom_circle, rvec, tvec, K, dist_coeff)
        pts_top, _ = cv.projectPoints(top_circle, rvec, tvec, K, dist_coeff)
        pts_bottom = pts_bottom.reshape(-1, 2).astype(np.int32)
        pts_top = pts_top.reshape(-1, 2).astype(np.int32)

        # ============ 면 채우기 순서에 유의 ============

        # (a) "바닥"을 먼저 은색으로 채운다
        cv.fillPoly(img, [pts_bottom], COLOR_SILVER)

        # (b) "옆면"을 빨간색으로 여러 사각형(Trapezoid)으로 나누어 채운다
        for i in range(num_segments):
            j = (i + 1) % num_segments
            # 아래원 i→j, 윗원 i→j 순서로 사각형(또는 사다리꼴)
            side_polygon = np.array([
                pts_bottom[i],
                pts_bottom[j],
                pts_top[j],
                pts_top[i]
            ], dtype=np.int32)
            cv.fillPoly(img, [side_polygon], COLOR_RED)

        # (c) "윗면"을 은색으로 채운다
        cv.fillPoly(img, [pts_top], COLOR_SILVER)

        # 만약 윤곽선(테두리)도 그리고 싶다면 아래 코드 추가:
        # cv.polylines(img, [pts_bottom], True, (255,0,0), 2)
        # cv.polylines(img, [pts_top], True, (0,0,255), 2)
        # for b, t in zip(pts_bottom, pts_top):
        #     cv.line(img, tuple(b), tuple(t), (0,255,0), 1)

        # 3) 카메라 위치 출력
        R, _ = cv.Rodrigues(rvec)
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # === 결과 영상 표시 및 키 입력 ===
    cv.imshow('Pose Estimation (Colored Cylinder)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27:  # ESC
        break

video.release()
cv.destroyAllWindows()
