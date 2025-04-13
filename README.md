# Camera Pose Estimation and AR (카메라 자세 추정 및 증강현실)

이 프로젝트는 OpenCV를 사용하여 **체스보드 패턴 기반의 카메라 자세 추정(Camera Pose Estimation)** 을 수행하고, 그 위에 **가상의 콜라캔(Coke Can)** 을 증강현실(AR)로 시각화하는 데모입니다.

## 🧠 주요 기능

- **6×8 체스보드 패턴**을 이용해 카메라의 위치 및 자세 추정
- 체스보드 위에 다음과 같은 **가상 원기둥(Coke Can)** 을 그려 AR 구현:
  - 빨간색 **옆면**
  - 은색 **윗면과 아랫면**
- 카메라의 **월드 좌표계 위치**를 실시간으로 출력

## 🖼️ 예시 프레임 (AR 렌더링)

> 이미지 캡처가 있다면 아래와 같이 삽입할 수 있습니다:
>
> `<img src="your-image-path.png" width="500">`

## 📹 시연 영상

### 🎯 AR 콜라캔 시연

[![AR Demo Video](https://img.youtube.com/vi/_mwHEp7jV58/0.jpg)](https://youtu.be/_mwHEp7jV58)

➡️ [**AR 콜라캔 데모 영상 보기**](https://youtu.be/_mwHEp7jV58)

> 실제 체스보드를 기반으로 실시간 포즈 추정과 가상 물체가 렌더링되는 모습을 확인할 수 있습니다.

### 🛠️ 카메라 보정(내부 파라미터 얻기) 과정

[![Calibration Video](https://img.youtube.com/vi/j1Sv2sFp-LA/0.jpg)](https://youtu.be/j1Sv2sFp-LA)

➡️ [**카메라 캘리브레이션 과정 보기**](https://youtu.be/j1Sv2sFp-LA)

> 아래의 `K`와 `dist_coeff`는 OpenCV의 `cv.calibrateCamera()`를 통해 얻은 결과입니다:

```python
K = np.array([[631.12372751, 0, 204.13870644],
              [0, 626.76168609, 363.15842532],
              [0, 0, 1]])
dist_coeff = np.array([0.03184266, -0.26914475, -0.00107398, 0.00244667, 0.73158658])
