# utils/voice_interaction.py

import os
import tempfile
import requests
from typing import Optional, Tuple, Union, Dict, Any
import streamlit as st
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VoiceInteraction:
    """语音交互类，支持语音转文本和文本转语音"""
    
    def __init__(self, openai_api_key: Optional[str] = None, use_local_recorder: bool = False):
        """
        初始化语音交互
        
        Args:
            openai_api_key: OpenAI API密钥
            use_local_recorder: 是否使用本地录音器组件而非Streamlit组件
        """
        self.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("需要提供OpenAI API密钥")
        
        self.use_local_recorder = use_local_recorder
        self._setup_audio_recorder()
    
    def _setup_audio_recorder(self):
        """设置音频录制器"""
        try:
            if self.use_local_recorder:
                # 使用HTML和JavaScript实现的录音功能
                self._setup_html_recorder()
            else:
                # 尝试导入streamlit-audio-recorder组件
                try:
                    import streamlit_audio_recorder as st_audiorec
                    self.audio_recorder_available = True
                    self.audio_recorder = st_audiorec
                except ImportError:
                    logger.warning("未找到streamlit-audio-recorder组件，尝试使用st_audiorec")
                    try:
                        from st_audiorec import st_audiorec
                        self.audio_recorder_available = True
                        self.audio_recorder = st_audiorec
                    except ImportError:
                        logger.warning("未找到st_audiorec，尝试使用streamlit_webrtc")
                        try:
                            from streamlit_webrtc import webrtc_streamer
                            import av
                            self.audio_recorder_available = True
                            self.use_webrtc = True
                        except ImportError:
                            logger.warning("无可用的音频录制组件，将退回到文件上传方式")
                            self.audio_recorder_available = False
        except Exception as e:
            logger.error(f"设置音频录制器时出错: {str(e)}")
            self.audio_recorder_available = False
    
    def _setup_html_recorder(self):
        """设置基于HTML的录音机"""
        html_code = """
        <script>
        let audioBlob = null;
        let mediaRecorder = null;
        let audioChunks = [];
        
        function startRecording() {
            document.getElementById("start-btn").disabled = true;
            document.getElementById("stop-btn").disabled = false;
            document.getElementById("status").innerText = "录音中...";
            
            audioChunks = [];
            
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    mediaRecorder = new MediaRecorder(stream);
                    mediaRecorder.start();
                    
                    mediaRecorder.addEventListener("dataavailable", event => {
                        audioChunks.push(event.data);
                    });
                    
                    mediaRecorder.addEventListener("stop", () => {
                        audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        const audioUrl = URL.createObjectURL(audioBlob);
                        document.getElementById("audio-player").src = audioUrl;
                        document.getElementById("audio-player").style.display = "block";
                        
                        // 将录音数据转换为base64
                        const reader = new FileReader();
                        reader.readAsDataURL(audioBlob);
                        reader.onloadend = () => {
                            const base64data = reader.result;
                            // 发送到Streamlit
                            window.parent.postMessage({
                                type: "streamlit:setComponentValue",
                                value: base64data
                            }, "*");
                        }
                    });
                });
        }
        
        function stopRecording() {
            document.getElementById("start-btn").disabled = false;
            document.getElementById("stop-btn").disabled = true;
            document.getElementById("status").innerText = "录音已完成";
            
            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
            }
        }
        </script>
        
        <div style="border: 1px solid #ddd; padding: 15px; border-radius: 5px;">
            <h3>语音录制</h3>
            <p id="status">准备就绪</p>
            <button id="start-btn" onclick="startRecording()">开始录音</button>
            <button id="stop-btn" onclick="stopRecording()" disabled>停止录音</button>
            <audio id="audio-player" controls style="display: none; margin-top: 10px; width: 100%;"></audio>
        </div>
        """
        
        return html_code
    
    def render_recorder(self) -> Union[bytes, None]:
        """
        渲染音频录制器UI并返回录制的音频数据
        
        Returns:
            录制的音频数据或None
        """
        st.subheader("语音输入")
        
        audio_data = None
        
        if self.use_local_recorder:
            # 使用HTML/JS录音机
            component_value = st.components.v1.html(
                self._setup_html_recorder(),
                height=200
            )
            
            if component_value:
                # 处理base64编码的音频数据
                import base64
                try:
                    # 提取base64部分
                    audio_base64 = component_value.split(',')[1]
                    audio_data = base64.b64decode(audio_base64)
                except Exception as e:
                    st.error(f"处理录音数据时出错: {str(e)}")
        
        elif hasattr(self, 'use_webrtc') and self.use_webrtc:
            # 使用streamlit_webrtc
            from streamlit_webrtc import webrtc_streamer
            import av
            
            class AudioProcessor:
                def __init__(self):
                    self.audio_chunks = []
                
                def recv(self, frame):
                    self.audio_chunks.append(frame.to_ndarray())
                    return frame
            
            audio_processor = AudioProcessor()
            
            ctx = webrtc_streamer(
                key="speech-to-text",
                audio_processor_factory=lambda: audio_processor,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
            
            if st.button("转换为文本") and len(audio_processor.audio_chunks) > 0:
                # 将音频转换为WAV格式
                import numpy as np
                import soundfile as sf
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    audio_data = np.concatenate(audio_processor.audio_chunks, axis=0)
                    sf.write(temp_file.name, audio_data, 16000)
                    
                    with open(temp_file.name, "rb") as f:
                        audio_data = f.read()
                    
                    os.unlink(temp_file.name)
        
        elif self.audio_recorder_available:
            # 使用streamlit-audio-recorder
            try:
                if hasattr(self.audio_recorder, 'st_audiorec'):
                    # st_audiorec用法
                    audio_data = self.audio_recorder.st_audiorec()
                else:
                    # streamlit_audio_recorder用法
                    audio_data = self.audio_recorder()
            except Exception as e:
                st.error(f"录音时出错: {str(e)}")
                st.info("请尝试使用文件上传方式")
                audio_data = None
        
        # 后备方案：文件上传
        if audio_data is None:
            uploaded_file = st.file_uploader("或上传音频文件", type=["wav", "mp3", "ogg"])
            if uploaded_file is not None:
                audio_data = uploaded_file.getvalue()
        
        return audio_data
    
    def speech_to_text(self, audio_data) -> str:
        """
        将语音转换为文本
        
        Args:
            audio_data: 音频数据（字节或文件对象）
            
        Returns:
            转录的文本
        """
        if audio_data is None or len(audio_data) == 0:
            st.warning("未获取到有效的音频数据")
            return ""
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            if hasattr(audio_data, 'getbuffer'):
                temp_file.write(audio_data.getbuffer())
            else:
                temp_file.write(audio_data)
            file_path = temp_file.name
        
        # 使用OpenAI的Whisper API转录
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }
            
            with open(file_path, "rb") as f:
                response = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers=headers,
                    files={"file": f},
                    data={"model": "whisper-1"}
                )
            
            # 删除临时文件
            os.unlink(file_path)
            
            # 检查响应
            if response.status_code == 200:
                transcription = response.json().get("text", "")
                if transcription:
                    st.success("语音转文本成功!")
                return transcription
            else:
                error_msg = f"转录失败: {response.text}"
                logger.error(error_msg)
                st.error(error_msg)
                return ""
                
        except Exception as e:
            # 确保临时文件被删除
            if os.path.exists(file_path):
                os.unlink(file_path)
            error_msg = f"处理音频时出错: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            return ""
    
    def text_to_speech(self, text: str, voice: str = "alloy") -> Tuple[bytes, str]:
        """
        将文本转换为语音
        
        Args:
            text: 要转换为语音的文本
            voice: 语音类型，可选: "alloy", "echo", "fable", "onyx", "nova", "shimmer"
            
        Returns:
            音频数据和内容类型
        """
        if not text:
            return b"", ""
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # 准备请求数据
            data = {
                "model": "tts-1-hd",  # 或 "tts-1" 获取标准质量
                "input": text[:4000],  # 限制字符数
                "voice": voice
            }
            
            # 发送请求
            response = requests.post(
                "https://api.openai.com/v1/audio/speech",
                headers=headers,
                json=data
            )
            
            # 检查响应
            if response.status_code == 200:
                st.success("文本转语音成功!")
                return response.content, "audio/mp3"
            else:
                error_msg = f"语音生成失败: {response.text}"
                logger.error(error_msg)
                st.error(error_msg)
                return b"", ""
                
        except Exception as e:
            error_msg = f"生成语音时出错: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            return b"", ""
    
    def play_audio(self, audio_data: bytes, content_type: str):
        """
        播放音频
        
        Args:
            audio_data: 音频数据
            content_type: 内容类型
        """
        if not audio_data:
            return
            
        try:
            st.audio(audio_data, format=content_type)
        except Exception as e:
            st.error(f"播放音频时出错: {str(e)}")
            
    def available_voices(self) -> Dict[str, Dict[str, Any]]:
        """
        返回可用的TTS语音选项
        
        Returns:
            语音ID到语音信息的字典
        """
        return {
            "alloy": {
                "name": "Alloy",
                "description": "中性平衡的声音",
                "gender": "中性"
            },
            "echo": {
                "name": "Echo",
                "description": "深沉、平静的声音",
                "gender": "男声"
            },
            "fable": {
                "name": "Fable",
                "description": "英国口音、优雅的声音",
                "gender": "女声"
            },
            "onyx": {
                "name": "Onyx",
                "description": "深沉、权威的声音",
                "gender": "男声"
            },
            "nova": {
                "name": "Nova",
                "description": "温暖、友好的声音",
                "gender": "女声"
            },
            "shimmer": {
                "name": "Shimmer",
                "description": "活泼、年轻的声音",
                "gender": "女声"
            }
        }
        
    def render_voice_selector(self, default: str = "alloy") -> str:
        """
        渲染语音选择器
        
        Args:
            default: 默认语音
            
        Returns:
            选择的语音ID
        """
        voices = self.available_voices()
        voice_options = [f"{info['name']} ({info['gender']})" for _, info in voices.items()]
        voice_ids = list(voices.keys())
        
        selected_index = voice_ids.index(default) if default in voice_ids else 0
        selected_voice_name = st.selectbox(
            "选择语音",
            options=voice_options,
            index=selected_index
        )
        
        # 从名称映射回ID
        selected_index = voice_options.index(selected_voice_name)
        return voice_ids[selected_index]

# 使用示例
def voice_interaction_demo():
    st.title("语音交互演示")
    
    # 获取API密钥
    api_key = os.environ.get("OPENAI_API_KEY") or st.text_input(
        "OpenAI API密钥",
        type="password",
        placeholder="输入你的OpenAI API密钥",
        help="请输入你的OpenAI API密钥来启用语音功能"
    )
    
    if not api_key:
        st.warning("请提供OpenAI API密钥以使用语音功能")
        return
    
    try:
        voice = VoiceInteraction(api_key, use_local_recorder=True)
        
        # 语音转文本
        st.header("语音转文本")
        audio_data = voice.render_recorder()
        
        if audio_data and st.button("转换为文本"):
            with st.spinner("转换中..."):
                text = voice.speech_to_text(audio_data)
                if text:
                    st.text_area("转录结果", text, height=150)
        
        # 文本转语音
        st.header("文本转语音")
        input_text = st.text_area("输入文本", "欢迎使用SmartNote AI的语音功能！", height=150)
        selected_voice = voice.render_voice_selector()
        
        if st.button("生成语音"):
            with st.spinner("生成中..."):
                audio_data, content_type = voice.text_to_speech(input_text, voice=selected_voice)
                if audio_data:
                    st.subheader("生成的语音:")
                    voice.play_audio(audio_data, content_type)
    
    except Exception as e:
        st.error(f"初始化语音交互时出错: {str(e)}")

if __name__ == "__main__":
    voice_interaction_demo()
