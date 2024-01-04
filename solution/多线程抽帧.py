import cv2
import os
import threading

# 视频文件路径
video_path = r"D:\8022biaozhu_jiancha\142\vv\02号电梯\02号电梯_电梯轿厢内_左前_20210417.mp4"
# 保存帧的目录
output_dir = r"D:\8022biaozhu_jiancha\142\img\iii"
# 抽帧间隔（抽取每n帧）
frame_interval = 25

# 线程类定义
class FrameExtractionThread(threading.Thread):
    def __init__(self, video_path, output_dir, start_frame, end_frame):
        threading.Thread.__init__(self)
        self.video_path = video_path
        self.output_dir = output_dir
        self.start_frame = start_frame
        self.end_frame = end_frame

    def run(self):
        vidcap = cv2.VideoCapture(self.video_path)
        success, image = vidcap.read()
        frame_count = 0
        while success:
            if frame_count >= self.start_frame:
                # 保存帧图像
                frame_path = os.path.join(self.output_dir, "ebike_1_frame_{}.jpg".format(frame_count))
                cv2.imwrite(frame_path, image)

            frame_count += 1
            if frame_count > self.end_frame:
                break

            # 读取下一帧
            for _ in range(frame_interval):
                success, image = vidcap.read()

# 获取视频的总帧数
def get_total_frame_count(video_path):
    vidcap = cv2.VideoCapture(video_path)
    return int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

def main():
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取视频的总帧数
    total_frames = get_total_frame_count(video_path)
    print("Total frames: {}".format(total_frames))

    # 计算每个线程负责的帧范围
    num_threads = 4  # 线程数量
    frames_per_thread = total_frames // num_threads

    # 创建并启动线程
    threads = []
    for i in range(num_threads):
        start_frame = i * frames_per_thread
        end_frame = (i + 1) * frames_per_thread - 1 if i != num_threads - 1 else total_frames - 1
        thread = FrameExtractionThread(video_path, output_dir, start_frame, end_frame)
        threads.append(thread)
        thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    print("Frame extraction completed!")

if __name__ == "__main__":
    main()
