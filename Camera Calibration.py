import numpy as np
import cv2 as cv

def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10, wnd_name='Camera Calibration'):
    # Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened()

    # Select images
    img_select = []
    while True:
        # Grab an image from the video
        valid, img = video.read()
        if not valid:
            break

        if select_all:
            img_select.append(img)
        else:
            # Show the image
            display = img.copy()
            cv.putText(display, f'NSelect: {len(img_select)}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
            cv.imshow(wnd_name, display)

            # Process the key event
            key = cv.waitKey(wait_msec)
            if key == ord(' '):  # Space: Pause and show corners
                complete, pts = cv.findChessboardCorners(img, board_pattern)
                cv.drawChessboardCorners(display, board_pattern, pts, complete)
                cv.imshow(wnd_name, display)
                key = cv.waitKey()
                if key == ord('\r') or key == 13:  # Enter: Select the image
                    img_select.append(img)
            if key == 27:  # ESC: Exit (Complete image selection)
                break

    cv.destroyAllWindows()
    return img_select

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # Find 2D corner points from given images
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0

    # Prepare 3D points of the chess board
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points)  # Must be np.float32

    # Calibrate the camera
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)

if __name__ == '__main__':
    video_file = 'D:/python/calibrate_camera/chessboard.mp4'  # 본인 경로에 맞게 수정
    board_pattern = (6, 8)         # 체스보드 패턴 (내부 코너 개수)
    board_cellsize = 0.025         # 실제 정사각형 한 칸의 크기 (단위: m, cm 등)

    # 1. 이미지 선택
    img_select = select_img_from_video(video_file, board_pattern)
    assert len(img_select) > 0, 'There is no selected images!'

    # 2. 카메라 캘리브레이션 수행
    rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)

    # 3. 결과 출력
    print('## Camera Calibration Results')
    print(f'* The number of selected images = {len(img_select)}')
    print(f'* RMS error = {rms}')
    print(f'* Camera matrix (K) = \n{K}')
    print(f'* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {dist_coeff.flatten()}')

    # Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened(), 'Cannot read the given input, ' + video_file

    # Get video FPS and set appropriate wait time
    fps = video.get(cv.CAP_PROP_FPS)
    wait_time = int(1000 / fps) if fps > 0 else 33  # fallback: 30fps 기준

    # Run distortion correction
    show_rectify = True
    map1, map2 = None, None

    while True:
        valid, img = video.read()
        if not valid:
            break

        # Rectify geometric distortion (Alternative: cv.undistort())
        info = "Original"
        if show_rectify:
            if map1 is None or map2 is None:
                h, w = img.shape[:2]
                # 새로운 카메라 행렬로 왜곡 보정 정확도 향상 (선택)
                newK, _ = cv.getOptimalNewCameraMatrix(K, dist_coeff, (w, h), 1)
                map1, map2 = cv.initUndistortRectifyMap(K, dist_coeff, None, newK, (w, h), cv.CV_32FC1)
            img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
            info = "Rectified"

        # 화면에 상태 텍스트 표시
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
        cv.imshow("Geometric Distortion Correction", img)

        # Wait for next frame based on fps
        key = cv.waitKey(wait_time)
        if key == ord(' '):     # Space: 일시정지
            key = cv.waitKey()
        if key == 27:           # ESC: 종료
            break
        elif key == ord('\t'):  # Tab: 왜곡 보정 On/Off 
            show_rectify = not show_rectify

    video.release()
    cv.destroyAllWindows()
