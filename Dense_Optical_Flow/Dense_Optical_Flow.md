> # compute_optical_flow_with_mask
> -----------------------------------------------------
>
> ```python
> # Farneback 광류 → 크기(magnitude)의 ROI 평균값으로 움직임 정도 산출
> def compute_optical_flow_with_mask(prev_gray, curr_gray, roi_mask):
>   
>   flow = cv2.calcOpticalFlowFarneback(
>     prev_gray, curr_gray, None,
>     pyr_scale=0.5, levels=1, winsize=9,
>    iterations=1, poly_n=3, poly_sigma=0.9, flags=0
>  )
>  mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
>  # roi_mask는 0/1 바이너리여야 평균이 정확함
>  masked_mag = mag * roi_mask
>  mean_movement = np.sum(masked_mag) / (np.sum(roi_mask) + 1e-6)
>  return mean_movement
>```



> ### 1. 함수 설명
>
> 이 함수는 Farneback Optical Flow를 사용하여 연속된 두 프레임 간의 움직임 정도를 계산하고, 그중에서도 관심 영역(ROI, Region of Interest) 내부의 평균 움직임 크기만을 정량화하여 반환
> * Dense Optical Flow (Farneback) 기반
> * 방향이 아닌 움직임의 크기만 사용
> * 실시간 처리를 고려한 경량 파라미터 구성
> * 철도/도로 영상에서 특정 구간(ROI)의 진동·움직임 감지
> * 이상 움직임 탐지 (Anomaly Detection)
> * 객체가 ROI 내부에서 실제로 이동 중인지 여부 판단

> ### 2. 입력 파라미터
>
>| 파라미터 | 타입 | 설명 |
>|---------|------|------|
>| `prev_gray` | `np.ndarray` | 이전 프레임 (Grayscale 이미지) |
>| `curr_gray` | `np.ndarray` | 현재 프레임 (Grayscale 이미지) |
>| `roi_mask` | `np.ndarray` | 관심 영역 마스크 (0 또는 1로 구성된 바이너리 이미지) |
