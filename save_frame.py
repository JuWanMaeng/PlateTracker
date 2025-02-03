import cv2
import os

def save_specific_frames(video_path, frame_numbers, output_dir):
    """
    특정 프레임을 이미지로 저장하는 함수

    :param video_path: 비디오 파일 경로
    :param frame_numbers: 저장할 프레임 번호 리스트 (예: [10, 50, 100])
    :param output_dir: 저장할 폴더 경로
    """
    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 비디오 파일을 열 수 없습니다.")
        return

    # 출력 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎥 총 프레임 수: {total_frames}")

    # 프레임 번호 정렬 (필수는 아니지만 효율적)
    frame_numbers = sorted(set(frame_numbers))

    # 프레임 읽기
    current_frame = 0
    saved_count = 0

    while current_frame < total_frames:
        ret, frame = cap.read()
        if not ret:
            break  # 더 이상 프레임이 없으면 종료

        if current_frame in frame_numbers:
            output_path = os.path.join(output_dir, f"frame_{current_frame}.png")
            cv2.imwrite(output_path, frame)
            print(f"✅ 저장 완료: {output_path}")
            saved_count += 1

            # 모든 프레임 저장이 끝나면 루프 종료
            if saved_count == len(frame_numbers):
                break

        current_frame += 1

    cap.release()
    print("🚀 모든 작업 완료!")

# 사용 예제
video_path = '/workspace/ByteTrack/issue/issue1.mp4'               # 비디오 파일 경로
frame_numbers = [0,1]       # 저장할 프레임 번호 리스트
output_dir = 'output_frames'                  # 저장할 폴더 경로

save_specific_frames(video_path, frame_numbers, output_dir)
