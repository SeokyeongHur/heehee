import cv2
import mediapipe as mp
import math

# --- 1단계: 기본 설정 및 라이브러리 임포트 ---

# MediaPipe Hands 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # 한 손만 감지해도 줌 제어가 가능하지만, 두 손을 사용하고 싶다면 2로 설정
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 웹캠 설정
cap = cv2.VideoCapture(0) # 0은 기본 웹캠

# 현재 줌 레벨 (초기값은 1.0, 1배 줌)
zoom_level = 1.0
# 줌 조절 속도 및 범위
ZOOM_SPEED = 0.05
MAX_ZOOM = 4.0
MIN_ZOOM = 1.0

# 손가락 끝점 인덱스 (엄지 끝과 검지 끝)
THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP
INDEX_FINGER_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP

# 이전 프레임의 손가락 거리 저장 (움직임 감지용)
prev_distance = 0

# --- 2단계: 줌 제어 로직 구현 ---

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # 웹캠 이미지 좌우 반전 (거울 모드)
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # BGR을 RGB로 변환 (MediaPipe는 RGB를 선호)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 손 랜드마크 감지
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 랜드마크 그리기
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 엄지와 검지 끝점의 좌표 가져오기 (정규화된 좌표)
            thumb_tip_x = hand_landmarks.landmark[THUMB_TIP].x
            thumb_tip_y = hand_landmarks.landmark[THUMB_TIP].y
            index_tip_x = hand_landmarks.landmark[INDEX_FINGER_TIP].x
            index_tip_y = hand_landmarks.landmark[INDEX_FINGER_TIP].y

            # 픽셀 좌표로 변환
            thumb_tip_pixel_x = int(thumb_tip_x * width)
            thumb_tip_pixel_y = int(thumb_tip_y * height)
            index_tip_pixel_x = int(index_tip_x * width)
            index_tip_pixel_y = int(index_tip_y * height)

            # 엄지와 검지 끝점 사이의 유클리드 거리 계산
            current_distance = math.sqrt(
                (thumb_tip_pixel_x - index_tip_pixel_x)**2 +
                (thumb_tip_pixel_y - index_tip_pixel_y)**2
            )

            # 초기 prev_distance 설정 (첫 프레임 또는 손이 처음 감지될 때)
            if prev_distance == 0:
                prev_distance = current_distance
            
            # 거리 변화량에 따라 줌 레벨 조정
            distance_diff = current_distance - prev_distance

            # 특정 임계값 이상 변화가 있을 때만 줌 조절 (노이즈 방지)
            THRESHOLD_DISTANCE = 10 # 픽셀 단위, 조절 필요
            if abs(distance_diff) > THRESHOLD_DISTANCE:
                if distance_diff > 0: # 거리가 멀어지면 줌인
                    zoom_level += ZOOM_SPEED
                else: # 거리가 가까워지면 줌아웃
                    zoom_level -= ZOOM_SPEED
                
                # 줌 레벨 범위 제한
                zoom_level = max(MIN_ZOOM, min(MAX_ZOOM, zoom_level))
                
                # 이전 거리 업데이트
                prev_distance = current_distance
            
            # 현재 거리 표시
            cv2.putText(frame, f'Distance: {int(current_distance)}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # 손이 감지되지 않으면 이전 거리 초기화 (줌 제어 중지)
        prev_distance = 0


    # 줌 적용 (이미지 리사이즈)
    # 현재 프레임을 줌 레벨에 따라 확대/축소
    # 중앙을 기준으로 확대/축소하려면 크롭 후 리사이즈 필요
    if zoom_level > 1.0:
        # 확대 시 원본 프레임의 중앙 부분을 크롭
        new_width = int(width / zoom_level)
        new_height = int(height / zoom_level)

        start_x = (width - new_width) // 2
        start_y = (height - new_height) // 2

        # 크롭 영역이 유효한지 확인 (음수 값 방지)
        if new_width <= width and new_height <= height:
            cropped_frame = frame[start_y : start_y + new_height, start_x : start_x + new_width]
            display_frame = cv2.resize(cropped_frame, (width, height))
        else:
            # 유효하지 않은 크롭 영역이면 원본 프레임 사용 (오류 방지)
            display_frame = frame
    else: # 줌아웃 (1.0 미만은 현재는 MIN_ZOOM으로 제한)
        display_frame = frame

    # 현재 줌 레벨 표시
    cv2.putText(display_frame, f'Zoom: {zoom_level:.2f}x', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 화면에 표시
    cv2.imshow('Camera Zoom Control', display_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()

