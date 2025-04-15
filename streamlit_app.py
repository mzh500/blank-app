import base64
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from database import *



try:
    from ultralytics import YOLO
except ImportError:
    st.error("请先安装 ultralytics 库: pip install ultralytics")
    raise

MODEL_PT_PATH = "model.pt"  # 改成你的 yolov8 权重路径，如 yolov8n.pt


@st.cache_resource
def load_yolo_model():
    if not os.path.exists(MODEL_PT_PATH):
        st.warning(f"未找到 yolov8 模型权重文件: {MODEL_PT_PATH}")
        return None
    model = YOLO(MODEL_PT_PATH)
    return model


def detect_objects_yolov8(model, img_pil, conf_thres=0.5, font=None):
    """对单张 PIL 图片进行推理，返回带框图和结果列表。"""
    if model is None:
        return {"annotated_image": img_pil, "results": []}

    res_list = model.predict(source=np.array(img_pil), conf=conf_thres)
    if not res_list:
        return {"annotated_image": img_pil, "results": []}

    res = res_list[0]
    annotated_image = img_pil.copy()
    draw = ImageDraw.Draw(annotated_image)
    names = model.names

    results_info = []
    for box in res.boxes:
        cls_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist()
        # 修改标签映射
        cls_name = names[cls_id]
        if cls_name == "face":
            cls_name = "没带口罩"
        elif cls_name == "face_mask":
            cls_name = "戴口罩"

        results_info.append({
            "cls_name": cls_name,
            "confidence": score,
            "xyxy": xyxy
        })

        x1, y1, x2, y2 = xyxy
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        text = f"{cls_name} {score:.2f}"
        if font:
            draw.text((x1, y1 - 15), text, fill=(255, 0, 0), font=font)
        else:
            draw.text((x1, y1 - 15), text, fill=(255, 0, 0))

    return {"annotated_image": annotated_image, "results": results_info}


def detect_video_yolov8_realtime(model, video_path, conf_thres=0.5):
    """
    对整段视频(文件)进行逐帧推理，并在 Streamlit 页面"实时"展示左右分栏对比。
    返回是否检测到未戴口罩 (布尔值)。
    """
    if not model or not os.path.exists(video_path):
        st.error("模型未加载或视频文件不存在。")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("无法打开视频文件。")
        return False

    # 加载中文字体（需确保字体文件存在）
    try:
        font = ImageFont.truetype("simhei.ttf", 20)  # Windows系统自带字体
    except:
        font = ImageFont.load_default()
        st.warning("未找到中文字体文件 simhei.ttf，使用默认字体可能显示异常")

    # 创建左右分栏占位符
    left_col, right_col = st.columns(2)
    left_placeholder = left_col.empty()
    right_placeholder = right_col.empty()

    no_mask_found = False
    names = model.names

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 原始视频帧（BGR转RGB用于显示）
        original_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 转换为PIL图像处理中文
        frame_pil = Image.fromarray(original_frame_rgb)
        draw = ImageDraw.Draw(frame_pil)

        # 检测处理帧
        res_list = model.predict(source=np.array(frame_pil), conf=conf_thres)

        if res_list:
            res = res_list[0]
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                score = float(box.conf[0].item())
                xyxy = box.xyxy[0].tolist()

                # 修改标签显示为中文
                cls_name = names[cls_id]
                if cls_name == "face":
                    cls_name = "没戴口罩"
                    no_mask_found = True
                elif cls_name == "face_mask":
                    cls_name = "戴口罩"

                # 使用PIL绘制中文标签
                x1, y1, x2, y2 = map(int, xyxy)
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                text = f"{cls_name} {score:.2f}"
                draw.text((x1, max(0, y1 - 25)), text, fill=(255, 0, 0), font=font)

        # 转换回OpenCV格式
        detected_frame_bgr = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # 显示左右分栏视频
        left_placeholder.image(original_frame_rgb, caption="原始视频", use_container_width=True)
        right_placeholder.image(detected_frame_bgr, caption="检测后视频", use_container_width=True)

    cap.release()
    left_placeholder.empty()
    right_placeholder.empty()

    # 视频处理完成后显示提示
    if no_mask_found:
        st.error("🚨 检测到未佩戴口罩人员！")
        st.markdown("""
        <script>
        alert("视频中发现未佩戴口罩人员！");
        </script>
        """, unsafe_allow_html=True)

    return no_mask_found



def get_user(username: str):
    conn = get_db_connection()
    user = conn.execute(
        'SELECT * FROM users WHERE username = ?',
        (username,)
    ).fetchone()
    conn.close()
    return dict(user) if user else None

def register_user(username: str, password: str) -> bool:
    try:
        conn = get_db_connection()
        conn.execute(
            'INSERT INTO users (username, password_hash) VALUES (?, ?)',
            (username, hash_password(password))
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:  # 用户名已存在
        return False
    finally:
        conn.close()

def login_user(username: str, password: str) -> bool:
    user = get_user(username)
    if not user:
        return False
    return user['password_hash'] == hash_password(password)


def update_user_password(username: str, new_password: str) -> bool:
    """更新用户密码"""
    if username == "admin" and st.session_state.username != "admin":
        return False  # 防止非管理员修改admin密码

    try:
        conn = get_db_connection()
        cur = conn.execute(
            '''
            UPDATE users 
            SET password_hash = ?
            WHERE username = ?
            ''',
            (hash_password(new_password), username)
        )
        conn.commit()
        return cur.rowcount > 0
    except sqlite3.Error as e:
        st.error(f"数据库错误: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def delete_user(username: str) -> bool:
    if username == "admin":
        return False
    conn = get_db_connection()
    cur = conn.execute('DELETE FROM users WHERE username = ?', (username,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted

def get_all_users():
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users').fetchall()
    conn.close()
    return [dict(user) for user in users]

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()




def save_detection_record(username, file_path, detect_label, confidence):
    """将检测记录保存到SQLite数据库"""
    try:
        conn = get_db_connection()
        conn.execute(
            '''
            INSERT INTO detection_records (username, time, file_path, detect_label, confidence)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (username,
             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
             file_path,
             detect_label,
             confidence)
        )
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"保存检测记录失败: {e}")
    finally:
        if conn:
            conn.close()


def load_detection_records(username):
    """从 SQLite 加载记录，返回字典列表"""
    conn = get_db_connection()
    cursor = conn.execute(
        '''
        SELECT time, file_path, detect_label, confidence 
        FROM detection_records 
        WHERE username = ?
        ORDER BY time DESC
        ''',
        (username,)
    )
    records = [dict(row) for row in cursor.fetchall()]  # 转换为字典列表
    conn.close()
    return records  # 返回示例: [{"time": "...", "file_path": "...", ...}, ...]


def export_detection_report(result):
    label_text = result['label']
    conf_val = result['confidence']
    report_content = f"""
检测报告 (yolov8)

用户：{st.session_state.username}
时间：{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
检测结果：{label_text}
最高置信度：{conf_val:.3f}
"""
    b64 = base64.b64encode(report_content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="detection_report.txt">下载检测报告 (txt)</a>'
    st.markdown(href, unsafe_allow_html=True)


# ==================== Streamlit 主程序 ====================
def main():
    st.set_page_config(
        page_title="yolov8口罩检测系统",
        page_icon="😷",
        layout="centered"

    )
    # 添加全局样式
    st.markdown("""
        <style>
        /* 隐藏默认的菜单和页脚 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* 主标题样式 */
        .main-title {
            color: #1f77b4;
            font-size: 2.5em;
            text-align: center;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }

        /* 容器样式 */
        .stContainer {
            background: #ffffff;
            border-radius: 15px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }

        /* 按钮样式 */
        .stButton>button {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.8rem 2rem;
            font-size: 1rem;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }

        /* 图片对比容器 */
        .image-compare {
            display: flex;
            justify-content: space-between;
            gap: 2rem;
            margin: 2rem 0;
        }

        .image-card {
            flex: 1;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        .image-card img {
            width: 100%;
            height: auto;
        }

        .image-caption {
            text-align: center;
            padding: 1rem;
            background: #f8f9fa;
            font-weight: 500;
            color: #333;
        }
        </style>
        """, unsafe_allow_html=True)
    init_database()

    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = ""

    if not st.session_state.is_authenticated:
        show_login_register_page()
    else:
        show_main_page()


def show_login_register_page():
    st.markdown(
        """
        <style>
        .login-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 2rem;
            border-radius: 10px;
            background-color: #f9f9f9;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .login-title {
            text-align: center;
        }
        .login-subtitle {
            color: #888;
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    st.markdown("<h2 class='login-title'>欢迎登录 yolov8 口罩检测系统</h2>", unsafe_allow_html=True)
    st.markdown("<p class='login-subtitle'>😷 为了您和他人的健康，请戴好口罩 😷</p>", unsafe_allow_html=True)

    choice = st.radio("请选择操作", ["登录", "注册"], index=0)

    if choice == "登录":
        username = st.text_input("用户名", key="login_username")
        password = st.text_input("密码", type='password', key="login_password")
        if st.button("登录", key="login_button"):
            if login_user(username, password):
                st.success(f"登录成功，欢迎 {username}！")
                st.session_state.is_authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("用户名或密码错误，请重试。")
    else:
        new_username = st.text_input("请输入新用户名", key="reg_username")
        new_password = st.text_input("请输入密码", type='password', key="reg_password")
        confirm = st.text_input("请再次输入密码", type='password', key="reg_confirm")
        if st.button("注册", key="reg_button"):
            if new_password != confirm:
                st.error("两次密码输入不一致")
            elif not new_username or not new_password:
                st.error("用户名或密码不能为空。")
            else:
                if register_user(new_username, new_password):
                    st.success("注册成功，请返回登录。")
                else:
                    st.error("用户名已存在，请更换用户名。")

    st.markdown("</div>", unsafe_allow_html=True)


def show_main_page():
    st.title(f"欢迎，{st.session_state.username}！")
    if st.button("退出登录"):
        st.session_state.is_authenticated = False
        st.session_state.username = ""
        st.rerun()

    if st.session_state.username == "admin":
        tabs = st.tabs(["口罩检测 🎯", "历史记录 📜", "管理员面板 🛠", "关于 ℹ️"])
        tab_detect = tabs[0]
        tab_history = tabs[1]
        tab_admin = tabs[2]
        tab_about = tabs[3]
    else:
        tabs = st.tabs(["口罩检测 🎯", "历史记录 📜", "关于 ℹ️"])
        tab_detect = tabs[0]
        tab_history = tabs[1]
        tab_about = tabs[2]
        tab_admin = None

    with tab_detect:
        detection_tab()

    with tab_history:
        show_history()

    if tab_admin is not None:
        with tab_admin:
            admin_dashboard()

    with tab_about:
        st.markdown("### 系统简介")
        st.write("此示例通过 **yolov8** (ultralytics) 加载 `model-bakup.pt`，对上传图片或视频进行目标检测（如识别是否戴口罩）。")
        st.write("- 支持逐帧实时展示视频检测过程。")
        st.write("- 若检测到 `NoMask` 则用中文提示。")
        st.write("- 管理员可增删改查用户。")


def detection_tab():
    st.subheader("口罩检测 🎯 (yolov8)")
    input_mode = st.radio("选择输入类型", ("照片", "视频", "摄像头"))
    threshold = st.slider("置信度阈值 (yolov8 conf)", 0.0, 1.0, 0.5, 0.01)

    yolo_model = load_yolo_model()
    if yolo_model is None:
        st.warning("未加载到有效的 yolov8 模型。")
        return

    if input_mode == "照片":
        uploaded_file = st.file_uploader("上传图片 (jpg/jpeg/png)", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            # 使用容器包装图片对比
            st.markdown('<div class="stContainer">', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 原始图片")
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, use_container_width=True)

            if st.button("开始检测", key="detect_img"):
                with st.spinner("检测中..."):
                    result_dict = detect_objects_yolov8(yolo_model, img, conf_thres=threshold)

                with col2:
                    st.markdown("### 检测结果")
                    st.image(result_dict["annotated_image"], use_container_width=True)

                    detect_str = ", ".join([f"{r['cls_name']}({r['confidence']:.2f})"
                                            for r in result_dict["results"]])
                    st.markdown(f"**检测结果**：{detect_str if detect_str else '无检测目标'}")

                    # 在图片检测结果显示位置添加弹窗提示
                    no_mask_found = any(r['cls_name'] == "没带口罩" for r in result_dict["results"])
                    if no_mask_found:
                        st.error("🚨 检测到未佩戴口罩！")
                        # JavaScript弹窗提示
                        st.markdown("""
                        <script>
                        alert("检测到未佩戴口罩！");
                        </script>
                        """, unsafe_allow_html=True)
                    else:
                        st.success("✅ 所有人员佩戴口罩符合规范")

                if result_dict["results"]:
                    first_cls = result_dict["results"][0]["cls_name"]
                    first_conf = result_dict["results"][0]["confidence"]
                else:
                    first_cls = "None"
                    first_conf = 0.0

                file_path = f"uploads/{st.session_state.username}/{uploaded_file.name}"
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                img.save(file_path)

                save_detection_record(st.session_state.username, file_path, first_cls, float(first_conf))
                export_detection_report({"label": detect_str, "confidence": float(first_conf)})

            st.markdown('</div>', unsafe_allow_html=True)

    elif input_mode == "视频":
        video_file = st.file_uploader("上传视频 (mp4/mov/avi)", type=['mp4', 'mov', 'avi'])
        if video_file is not None:
            file_path = f"uploads/{st.session_state.username}/{video_file.name}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(video_file.getvalue())

            st.video(file_path)
            # 增加一个“逐帧实时展示”按钮
            if st.button("开始实时展示检测"):
                with st.spinner("视频检测中，请稍候..."):
                    no_mask = detect_video_yolov8_realtime(yolo_model, file_path, conf_thres=threshold)

                # 检测结束后，给出提示并写历史记录
                if no_mask:
                    st.error("提醒：检测到有人没戴口罩！")
                    save_detection_record(st.session_state.username, file_path, "NoMask", 1.0)
                    export_detection_report({"label": "NoMask", "confidence": 1.0})
                else:
                    st.success("检测完成：视频中未检测到 `NoMask`")
                    save_detection_record(st.session_state.username, file_path, "Mask", 1.0)
                    export_detection_report({"label": "Mask", "confidence": 1.0})













    elif input_mode == "摄像头":

        st.markdown("### 浏览器摄像头实时检测")

        img_file_buffer = st.camera_input("点击按钮开始摄像头捕获")

        if img_file_buffer is not None:

            # 将上传的图片转换为PIL格式

            img_pil = Image.open(img_file_buffer)

            # 执行检测

            result_dict = detect_objects_yolov8(

                yolo_model,

                img_pil,

                conf_thres=threshold,

                font=load_chinese_font()  # 需要确保字体加载函数

            )

            # 显示结果

            col1, col2 = st.columns(2)

            with col1:

                st.image(img_pil, caption="原始画面", use_container_width=True)

            with col2:

                st.image(result_dict["annotated_image"], caption="检测结果", use_container_width=True)

            # 保存记录

            file_path = f"uploads/{st.session_state.username}/cam_{int(time.time())}.jpg"

            img_pil.save(file_path)

            # 检查是否检测到未戴口罩

            no_mask_found = any(r['cls_name'] == "没带口罩" for r in result_dict["results"])

            label = "没带口罩" if no_mask_found else "戴口罩"

            confidence = max([r['confidence'] for r in result_dict["results"]], default=0.0)

            save_detection_record(

                st.session_state.username,

                file_path,

                label,

                confidence

            )

            # 警报提示

            if no_mask_found:
                st.error("🚨 检测到未佩戴口罩！")

                st.markdown("""

                <script>

                alert("检测到未佩戴口罩！");

                </script>

                """, unsafe_allow_html=True)


def load_chinese_font():
    try:
        return ImageFont.truetype("simhei.ttf", 20)
    except:
        return ImageFont.load_default()

def _start_camera_detection(model, threshold):
    """初始化摄像头和视频保存"""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("无法打开摄像头")

        # 创建视频保存路径
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"uploads/{st.session_state.username}/cam_{timestamp}.mp4"  # 改为 .mp4
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 初始化视频写入器，使用 H.264 编码
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # 或 'mp4v'
        frame_size = (int(cap.get(3)), int(cap.get(4)))
        writer = cv2.VideoWriter(output_path, fourcc, 20.0, frame_size)

        # 检查写入器是否成功打开
        if not writer.isOpened():
            st.error("无法初始化视频写入器，请检查编码器或路径。")
            cap.release()
            return

        st.session_state.cam.update({
            'active': True,
            'cap': cap,
            'writer': writer,
            'output_path': output_path,
            'no_mask': False
        })

    except Exception as e:
        st.error(f"摄像头初始化失败: {str(e)}")


def _stop_camera_detection():
    """停止检测并保存记录"""
    if st.session_state.cam['active']:
        # 释放资源
        st.session_state.cam['cap'].release()
        st.session_state.cam['writer'].release()

        # 保存检测记录
        if os.path.exists(st.session_state.cam['output_path']):
            label = "没带口罩" if st.session_state.cam['no_mask'] else "戴口罩"
            confidence = 1.0 if st.session_state.cam['no_mask'] else 0.0

            save_detection_record(
                st.session_state.username,
                st.session_state.cam['output_path'],
                label,
                confidence
            )

            # 导出报告
            export_detection_report({
                'label': label,
                'confidence': confidence
            })

        # 重置状态
        st.session_state.cam['active'] = False


def _show_camera_feed(model, threshold):
    """实时显示和处理摄像头画面"""
    frame_placeholder = st.empty()
    cap = st.session_state.cam['cap']
    writer = st.session_state.cam['writer']

    while st.session_state.cam['active']:
        ret, frame = cap.read()
        if not ret:
            break

        # 执行目标检测
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detect_objects_yolov8(
            model,
            img_pil,
            conf_thres=threshold,
            font=st.session_state.cam['font']  # 传入预加载字体
        )

        # 更新未戴口罩状态
        st.session_state.cam['no_mask'] = any(
            r['cls_name'] == "没带口罩" for r in result['results']
        )

        # 显示处理后的画面
        frame_placeholder.image(result['annotated_image'], use_container_width=True)

        # 将处理后的帧写入视频文件（注意颜色转换）
        annotated_frame_bgr = cv2.cvtColor(np.array(result['annotated_image']), cv2.COLOR_RGB2BGR)
        writer.write(annotated_frame_bgr)

        time.sleep(0.01)  # 控制帧率

    frame_placeholder.empty()





def show_history():
    st.subheader("历史记录 📜")
    records = load_detection_records(st.session_state.username)

    if not records:
        st.info("暂无历史记录。")
    else:
        # 使用卡片式布局展示历史记录
        for record in records:
            st.markdown("""
            <div style="
                border: 1px solid #e6e6e6;
                border-radius: 10px;
                padding: 1.5rem;
                margin: 1rem 0;
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            ">
            """, unsafe_allow_html=True)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**检测时间**: {record['time']}")
                status_style = "color: #d32f2f;" if record['detect_label'] == "NoMask" else "color: #2e7d32;"
                st.markdown(f"**检测结果**: <span style='{status_style}'>{record['detect_label']}</span>",
                            unsafe_allow_html=True)
                st.markdown(f"**置信度**: {record['confidence']:.3f}")

            with col2:
                file_path = record['file_path']
                if os.path.exists(file_path):
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in [".jpg", ".jpeg", ".png"]:
                        st.image(file_path, width=150)
                    elif ext in [".mp4", ".mov", ".avi"]:
                        # 显式指定 MIME 类型
                        st.video(file_path, format="video/mp4")  # 强制指定为 mp4

            st.markdown("</div>", unsafe_allow_html=True)


def admin_dashboard():
    st.subheader("管理员面板 🛠")
    tab_view, tab_add, tab_update, tab_delete = st.tabs(["查看用户", "添加用户", "修改密码", "删除用户"])

    with tab_view:
        st.markdown("### 查看用户")
        users = get_all_users()
        if not users:
            st.info("当前还没有用户数据。")
        else:
            df_users = pd.DataFrame(users)
            st.dataframe(df_users[['username', 'password_hash']])

    with tab_add:
        st.markdown("### 添加用户")
        new_username = st.text_input("新用户名", key="add_username")
        new_user_pwd = st.text_input("新用户密码", type='password', key="add_password")
        if st.button("添加", key="add_btn"):
            if not new_username or not new_user_pwd:
                st.error("用户名或密码不能为空。")
            else:
                if register_user(new_username, new_user_pwd):
                    st.success(f"用户 {new_username} 已添加。")
                else:
                    st.error("该用户已存在，添加失败。")

    with tab_update:
        st.markdown("### 修改用户密码")
        users = get_all_users()
        if not users:
            st.info("尚无用户可供操作。")
        else:
            # 修复这里：从用户字典列表中提取用户名
            user_list = [user['username'] for user in users]
            target_username = st.selectbox("选择要修改的用户", user_list, key="update_select")
            new_pwd = st.text_input("新密码", type='password', key="update_pwd")
            if st.button("重置密码", key="update_btn"):
                if not new_pwd:
                    st.error("新密码不能为空！")
                else:
                    if update_user_password(target_username, new_pwd):
                        st.success(f"用户 {target_username} 密码已重置。")
                    else:
                        st.error("重置失败，用户不存在？")

    with tab_delete:
        st.markdown("### 删除用户")
        users = get_all_users()
        if not users:
            st.info("尚无用户可供删除。")
        else:
            # 修复这里：从用户字典列表中提取用户名
            user_list = [user['username'] for user in users]
            del_username = st.selectbox("选择要删除的用户", user_list, key="delete_select")
            if st.button("删除", key="delete_btn"):
                if del_username == "admin":
                    st.error("禁止删除管理员账号！")
                else:
                    if delete_user(del_username):
                        st.success(f"用户 {del_username} 已删除。")
                        st.rerun()  # 刷新页面更新列表
                    else:
                        st.error("删除失败或用户不存在。")


if __name__ == '__main__':
    main()