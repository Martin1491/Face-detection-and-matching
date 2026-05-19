# ==============================================================================
# CITS4402 Computer Vision Project -     
# ==============================================================================
# [Team Members Information ]
# Team Member 1 Name: [Zhi Wang] | Student ID: [24560057]
# Team Member 2 Name: [Simona Han] | Student ID: [25152074]
# Team Member 3 Name: [Bryan Zhang] | Student ID: [25384281]
# ==============================================================================

import tkinter as tk
from tkinter import filedialog
import cv2
import face_recognition
from PIL import Image, ImageTk
import time
import os
import numpy as np

"""
[Justification of Library Functions and Alternatives]

1. face_recognition (wrapping dlib):
   - Justification: We selected the face_recognition library as our primary engine 
   because it offers a highly robust, state-of-the-art HOG (Histogram of Oriented Gradients) 
   combined with a linear classifier for face detection, and a highly accurate 128-D ResNet (Residual Network) model 
   for face encoding. It strikes an optimal balance between accuracy 
   (achieving 99.38% on the Labeled Faces in the Wild benchmark) and CPU inference speed. 
   Furthermore, its native support for extracting 68-point facial landmarks perfectly 
   aligns with the project's specification for precise eye and nose localization.

   - Alternatives: 
     a) OpenCV's Haar Cascades (OpenCV's traditional Haar Cascade classifier is a faster alternative. 
     However, it is highly sensitive to illumination changes and non-frontal head poses (yaw/pitch angles), 
     leading to an unacceptable rate of false negatives (missed faces) in unconstrained environments. 
     Thus, it was rejected.).

     b) MTCNN (MTCNN is highly accurate and performs face detection and landmark alignment simultaneously. 
     However, it is computationally heavy and introduces significant external dependencies 
     (requiring TensorFlow or PyTorch), which makes the deployment architecture overly complex 
     for a standalone desktop application. dlib offers a much lighter and cleaner C++ backend.).
     
2. cv2 (OpenCV):
   - Justification: Used for core matrix transformations (e.g., `estimateAffinePartial2D`), 
   color space conversions, and drawing operations. OpenCV is the industry standard for low-level image processing.
   - Alternatives: Pillow (PIL) could be used for basic cropping and resizing, 
   but it lacks advanced mathematical matrix solvers required for rigid similarity transformations.
"""

class FaceDetectionApp:
    def __init__(self, root):
        """
        Initialize the Application GUI structure.
        """
        self.root = root
        self.root.title("CITS4402 Project - Face detection and matching")
        self.root.geometry("1000x600") 

        # Setup top buttons 
        button_frame = tk.Frame(root)
        button_frame.pack(side="top", fill="x", pady=10)

        self.btn_single = tk.Button(button_frame, text="Button A: Single Image", 
                                    command=self.process_single_image, bg="lightblue", font=("Arial", 10, "bold"))
        self.btn_single.pack(side="left", padx=20)

        self.btn_bulk = tk.Button(button_frame, text="Button B: Bulk Processing", 
                                  command=self.process_bulk_images, bg="lightgreen", font=("Arial", 10, "bold"))
        self.btn_bulk.pack(side="left", padx=20)

        # Setup middle image display panels 
        image_frame = tk.Frame(root)
        image_frame.pack(side="top", expand=True, fill="both", padx=10, pady=10)

        self.lbl_img_left = tk.Label(image_frame, text="Input Image", bg="gray")
        self.lbl_img_left.pack(side="left", expand=True, padx=10)

        self.lbl_img_right = tk.Label(image_frame, text="Output Result", bg="gray")
        self.lbl_img_right.pack(side="right", expand=True, padx=10)

        # Setup bottom status bar 
        status_frame = tk.Frame(root)
        status_frame.pack(side="bottom", fill="x", pady=10)

        self.lbl_status = tk.Label(status_frame, text="Status: Waiting for input...", font=("Arial", 12))
        self.lbl_status.pack()

    def display_image_on_label(self, cv_img, label_widget):
        """
        Convert OpenCV BGR image to Tkinter-compatible format and resize for UI display.

        """
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w = rgb_image.shape[:2]
        scaling_factor = min(400/w, 400/h)
        new_size = (int(w * scaling_factor), int(h * scaling_factor))
        resized_image = cv2.resize(rgb_image, new_size)

        pil_img = Image.fromarray(resized_image)
        img_tk = ImageTk.PhotoImage(image=pil_img)
        
        label_widget.config(image=img_tk, text="", width=new_size[0], height=new_size[1])
        label_widget.image = img_tk 

    # ==========================================================================
    # [Skin Colour Segmentation]
    # ==========================================================================
    def filter_false_positives_by_skin_colour(self, img_bgr, face_locations):
        """
        Uses traditional HSV skin color segmentation to verify detected bounding boxes.
        使用传统的 HSV 肤色分割来验证深度学习检测到的边界框是否真的包含人脸。
        """
        valid_face_locations = []
        
        # Convert BGR to HSV color space (highly robust against illumination changes)
        # 将色彩空间从 BGR 转换为 HSV（对光照变化具有很强的鲁棒性，最适合肤色检测）
        hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        # Define universal human skin colour thresholds in HSV space
        # 在 HSV 空间中定义涵盖大部分人类肤色的标准阈值范围
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create a binary mask where skin pixels = 255 (White), background = 0 (Black)
        # 生成二值化掩码图：肤色像素变为纯白，其他变为纯黑
        skin_mask = cv2.inRange(hsv_img, lower_skin, upper_skin)

        for (top, right, bottom, left) in face_locations:
            # Extract the proposed face bounding box region from the skin mask
            # 从掩码中切出深度学习模型提议的人脸边界框区域
            face_region_mask = skin_mask[top:bottom, left:right]
            
            skin_pixels = cv2.countNonZero(face_region_mask)
            total_pixels = face_region_mask.size
            if total_pixels == 0: continue
            
            # Verification: Only accept the box if > 15% of its pixels are skin-coloured
            # 验证逻辑：如果框内超过 15% 的像素是肤色，才承认这是一张人脸，否则作为误报剔除
            skin_ratio = skin_pixels / total_pixels
            if skin_ratio > 0.15:
                valid_face_locations.append((top, right, bottom, left))
                
        return valid_face_locations

    def process_single_image(self):
        """
        Button A Pipeline: Single Image Processing.
        按钮 A 流水线：单张图像处理（包含对齐、关键点绘制和四角覆盖显示）。
        """
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("JPEG Files", "*.jpg")])
        if not file_path: return
        
        self.lbl_status.config(text="Processing single image...")
        self.root.update() 
        
        img_bgr = cv2.imread(file_path)
        if img_bgr is None: return

        self.display_image_on_label(img_bgr, self.lbl_img_left)

        start_time = time.time()
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # ======================================================================
        # [Use of Face Detection Library]
        # 【使用人脸检测库】
        # We utilize face_recognition's HOG-based detector to find initial face bounding boxes.
        # 此处调用 face_recognition 库底层的 HOG 梯度直方图算法，定位初步的人脸边界框坐标。
        # ======================================================================
        raw_face_locations = face_recognition.face_locations(img_rgb)
        
        # Apply the custom Skin Colour filter to remove false positives
        # 应用自定义的肤色滤波器以移除误报
        face_locations = self.filter_false_positives_by_skin_colour(img_bgr, raw_face_locations)
        
        # Extract 68-point landmarks for VERIFIED faces
        # 仅为通过肤色验证的真实人脸提取 68 个面部特征点
        face_landmarks_list = face_recognition.face_landmarks(img_rgb, face_locations)
        
        img_result = img_bgr.copy()
        corners = [(0, 0), (0, 515), (355, 0), (355, 515)]

        for i, ((top, right, bottom, left), face_landmarks) in enumerate(zip(face_locations, face_landmarks_list)):
            
            # Compute arithmetic mean of the 6 points to find exact eye centers
            # 计算眼睛 6 个轮廓点的算术平均值，精确定位眼睛的几何中心
            screen_left_eye = face_landmarks['left_eye']
            pt_left = np.array([sum(p[0] for p in screen_left_eye) / 6, sum(p[1] for p in screen_left_eye) / 6], dtype=np.float32)
            
            screen_right_eye = face_landmarks['right_eye']
            pt_right = np.array([sum(p[0] for p in screen_right_eye) / 6, sum(p[1] for p in screen_right_eye) / 6], dtype=np.float32)
            
            nose_tip_real = face_landmarks['nose_tip'][2]

            # ==================================================================
            # [Similarity Transformation & Processing Sequence]
            # 【应用相似度变换与子步骤序列说明】
            # 
            # Sequence Justification (步骤序列论证):
            # A naive approach is to (1) Crop the box -> (2) Align/Rotate -> (3) Resize. 
            # However, rotating a pre-cropped box cuts off the corners of the face and 
            # leaves black "blank regions" near the borders.
            # 
            # OUR SOLUTION: We combine Align, Resize, and Crop into a SINGLE Matrix Operation.
            # We apply a Similarity Transformation matrix to the ENTIRE original image.
            # 
            # 错误的序列是：先裁剪 -> 再旋转对齐 -> 最后缩放。这会导致人脸的边角被切掉，并在边缘留下黑边。
            # 我们的解决方案：将对齐、缩放和裁剪合并为单一的矩阵运算，直接对“整张原图”应用相似变换。
            # ==================================================================
            
            # Sub-step 1: Define source points from the original image
            # 子步骤 1：定义原始图像上的实际关键点源坐标
            src_pts = np.array([pt_left, pt_right, nose_tip_real], dtype=np.float32)
            
            # Sub-step 2: Define strict target coordinates for the 125x125 viewport
            # 子步骤 2：定义项目要求的 125x125 画布内部的严格目标坐标
            dst_pts = np.array([[40, 40], [85, 40], [63, 70]], dtype=np.float32)

            # Sub-step 3: Compute the Similarity Transformation Matrix.
            # `estimateAffinePartial2D` limits the mapping to pure Rotation, Translation, and Uniform Scaling (4 Degrees of Freedom).
            # This STRICTLY prevents any shearing or unnatural distortion of the face.
            # 子步骤 3：计算相似变换矩阵。该函数将映射严格限制为平移、旋转和统一等比缩放，绝不会让人脸产生扭曲畸变。
            M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)

            # Sub-step 4: Apply the Warp transformation natively.
            # Using `BORDER_REPLICATE` perfectly solves the "blank regions near borders" problem 
            # by extending the edge pixels naturally if the face is too close to the image boundary.
            # 子步骤 4：执行原生矩阵映射。使用 BORDER_REPLICATE 模式智能延伸边缘像素，完美解决了“图像边缘留下黑色空白区域”的问题！
            face_chip = cv2.warpAffine(img_bgr, M, (125, 125), borderMode=cv2.BORDER_REPLICATE) 

            # Draw specification color-coded landmarks on the face chip (Red, Green, Blue)
            # 在提取出的头像切片上绘制严格规定颜色的标记点
            cv2.circle(face_chip, (40, 40), 3, (0, 0, 255), -1)  
            cv2.circle(face_chip, (85, 40), 3, (0, 255, 0), -1)  
            cv2.circle(face_chip, (63, 70), 3, (255, 0, 0), -1)  

            # Draw the same color-coded landmarks on the main visualization canvas
            # 在主展示画布的原图上同步绘制这些颜色标记点
            cv2.circle(img_result, (int(pt_left[0]), int(pt_left[1])), 4, (0, 0, 255), -1)
            cv2.circle(img_result, (int(pt_right[0]), int(pt_right[1])), 4, (0, 255, 0), -1)
            cv2.circle(img_result, nose_tip_real, 4, (255, 0, 0), -1)

            # Draw a green bounding box around the detected face
            cv2.rectangle(img_result, (left, top), (right, bottom), (0, 255, 0), 2)

            # Overlay the 125x125 aligned face onto the respective corner
            # 将对齐好的 125x125 头像覆盖到对应的图像角落
            if i < 4:
                y_start, x_start = corners[i]
                img_result[y_start:y_start+125, x_start:x_start+125] = face_chip

        processing_time = round(time.time() - start_time, 2)
        self.display_image_on_label(img_result, self.lbl_img_right)
        self.lbl_status.config(text=f"Single Image: Processed in {processing_time} seconds | Found {len(face_locations)} face(s).")

    def process_bulk_images(self):
        """
        Button B Pipeline: Batch processing, clustering, and exporting aligned chips.
        按钮 B 流水线：批量处理、身份聚类，并按规范导出对齐后的人脸切片图。
        """
        folder_path = filedialog.askdirectory(title="Select Sample Image Folder")
        if not folder_path: return

        self.lbl_status.config(text="Bulk processing sequence started...")
        self.root.update()

        # Generate "Processed_Images" directory at the SAME level as the target folder
        # 在目标文件夹的同级（平级）目录下生成 "Processed_Images" 文件夹，符合严格规范
        parent_dir = os.path.dirname(os.path.normpath(folder_path))
        results_dir = os.path.join(parent_dir, "Processed_Images")
        
        # Clear existing contents to prevent run contamination
        # 运行前自动清空该文件夹内以前留存的文件，防止数据污染
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                file_path = os.path.join(results_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            os.makedirs(results_dir)

        known_encodings = []  
        known_person_ids = []  
        
        next_person_id = 1
        identity_counts = {}  
        total_images_processed = 0
        total_faces_detected = 0

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
        if not image_files:
            self.lbl_status.config(text="Zero .jpg targets found in the directory.")
            return

        TOLERANCE = 0.6  # Euclidean distance threshold for clustering / 聚类的欧氏距离阈值
        start_time = time.time()

        for filename in image_files:
            img_bgr = cv2.imread(os.path.join(folder_path, filename))
            if img_bgr is None: continue

            # Instantly display original input image on the left / 在左侧瞬间更新显示正在处理的原图
            self.display_image_on_label(img_bgr, self.lbl_img_left)
            self.root.update() 

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # [Detection & Skin Verification Step]
            # 【检测与肤色验证步骤】
            raw_face_locations = face_recognition.face_locations(img_rgb)
            face_locations = self.filter_false_positives_by_skin_colour(img_bgr, raw_face_locations)
            
            # Extract 128D encodings and landmarks ONLY for verified faces
            # 仅为通过肤色验证的人脸提取 128 维特征编码和关键点
            face_encodings = face_recognition.face_encodings(img_rgb, face_locations)
            face_landmarks_list = face_recognition.face_landmarks(img_rgb, face_locations)
            
            img_result = img_bgr.copy() 
            total_images_processed += 1

            for (top, right, bottom, left), face_encoding, face_landmarks in zip(face_locations, face_encodings, face_landmarks_list):
                total_faces_detected += 1
                person_id = -1
                
                # --- Nearest-Neighbor Identity Clustering / 最近邻身份动态聚类 ---
                if len(known_encodings) > 0:
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if face_distances[best_match_index] < TOLERANCE:
                        person_id = known_person_ids[best_match_index]

                if person_id == -1:
                    person_id = next_person_id
                    known_encodings.append(face_encoding)
                    known_person_ids.append(person_id)
                    identity_counts[person_id] = 0 
                    next_person_id += 1

                # Update the face counter for this unique identity / 更新此特定身份的脸部计数器
                identity_counts[person_id] += 1
                face_num = identity_counts[person_id]

                # --- High-Fidelity Alignment / 高清相似变换对齐 (同单图逻辑) ---
                screen_left_eye = face_landmarks['left_eye']
                pt_left = np.array([sum(p[0] for p in screen_left_eye) / 6, sum(p[1] for p in screen_left_eye) / 6], dtype=np.float32)
                
                screen_right_eye = face_landmarks['right_eye']
                pt_right = np.array([sum(p[0] for p in screen_right_eye) / 6, sum(p[1] for p in screen_right_eye) / 6], dtype=np.float32)

                nose_tip_real = face_landmarks['nose_tip'][2]
                
                src_pts = np.array([pt_left, pt_right, nose_tip_real], dtype=np.float32)
                dst_pts = np.array([[40, 40], [85, 40], [63, 70]], dtype=np.float32)

                M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                face_chip = cv2.warpAffine(img_bgr, M, (125, 125), borderMode=cv2.BORDER_REPLICATE)
                
                # [Specification: Export without landmarks]
                # 【项目规范：导出纯净头像切片，不能带任何圆点标记，且命名格式必须严格遵循要求】
                save_filename = f"Identity_{person_id}_face_{face_num}.jpg"
                save_path = os.path.join(results_dir, save_filename)
                cv2.imwrite(save_path, face_chip)

                # Visual tracking aid for GUI only / 仅为提升界面反馈体验而绘制的绿框与标签
                cv2.rectangle(img_result, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(img_result, f"ID: {person_id}", (left, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)

            # Update visualization and pause slightly for viewing / 更新可视化并轻微暂停以形成幻灯片般的体验
            self.display_image_on_label(img_result, self.lbl_img_right)
            self.root.update()
            time.sleep(0.8)

        total_time = round(time.time() - start_time, 2)
        total_identities = next_person_id - 1

        # [Specification: Print Exact Metrics Format]
        # 【项目规范：底部状态栏必须严格打印规定的长字符串】
        final_status = f"Total {total_images_processed} images processed in {total_time} seconds. {total_faces_detected} faces detected corresponding to {total_identities} unique identities."
        self.lbl_status.config(text=final_status)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()