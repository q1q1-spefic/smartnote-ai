# app/app.py

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# 加载本地环境变量
load_dotenv()

# 导入自定义模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
from utils.query_engine import QueryEngine

# 尝试导入语音交互模块
voice_interaction_available = False
try:
    from utils.voice_interaction import VoiceInteraction
    voice_interaction_available = True
except ImportError:
    st.sidebar.warning("语音交互模块导入失败，请确保已安装相关依赖")

# 尝试导入知识图谱模块
knowledge_graph_available = False
try:
    from utils.knowledge_graph import KnowledgeGraph
    knowledge_graph_available = True
except ImportError as e:
    missing_module = str(e).split("'")[1] if "'" in str(e) else str(e)
    st.sidebar.warning(f"知识图谱模块导入失败: 缺少 {missing_module} 模块。请运行 'pip install {missing_module}'")

# 设置页面标题和图标
st.set_page_config(
    page_title="SmartNote AI - 私人知识问答库",
    page_icon="🧠",
    layout="wide"
)

# 应用标题
st.title("🧠 SmartNote AI - 私人知识问答库")
st.markdown("上传你的文档，创建你的私人知识库，让AI帮你记住并回答问题")

# 侧边栏 - 配置
with st.sidebar:
    st.header("⚙️ 配置")
    
    # API密钥设置
    api_key = st.text_input("OpenAI API Key", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # 嵌入模型选择
    embedding_type = st.selectbox(
        "选择嵌入模型",
        options=["openai", "huggingface"],
        index=0,
        help="OpenAI嵌入质量更高，需要API密钥；HuggingFace模型免费但质量略低"
    )
    
    # 语言模型选择
    model_name = st.selectbox(
        "选择语言模型",
        options=["gpt-3.5-turbo", "gpt-4"],
        index=0
    )
    
    # 分块设置
    chunk_size = st.slider("文档分块大小", min_value=100, max_value=2000, value=500, step=100)
    chunk_overlap = st.slider("分块重叠大小", min_value=0, max_value=500, value=50, step=10)
    
    # 数据库设置
    vector_db_type = st.selectbox(
        "向量数据库类型",
        options=["faiss", "chroma"],
        index=0
    )
    
    # 数据库路径
    db_path = st.text_input("数据库存储路径", value="./data/vector_db")

    # 语音设置 - 仅当模块可用时显示
    if voice_interaction_available:
        st.header("🎙️ 语音设置")
        enable_voice = st.checkbox("启用语音功能", value=False)
        if enable_voice:
            voice_type = st.selectbox(
                "选择语音类型",
                options=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                index=0
            )
    else:
        enable_voice = False
    
    # 保存按钮
    if st.button("保存配置"):
        st.success("配置已保存！")

# 初始化会话状态
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'query_engine' not in st.session_state:
    st.session_state.query_engine = None
if 'documents_added' not in st.session_state:
    st.session_state.documents_added = False
if 'voice_interaction' not in st.session_state:
    st.session_state.voice_interaction = None
if 'knowledge_graph' not in st.session_state:
    st.session_state.knowledge_graph = None

# 创建标签页 - 根据可用模块调整
tabs = ["📚 上传文档", "❓ 问答", "📊 知识分析"]
if knowledge_graph_available:
    tabs.append("🔍 知识图谱")

all_tabs = st.tabs(tabs)
tab1 = all_tabs[0]  # 上传文档
tab2 = all_tabs[1]  # 问答
tab3 = all_tabs[2]  # 知识分析
tab4 = all_tabs[3] if knowledge_graph_available else None  # 知识图谱（如果可用）

# 标签页1：上传文档
with tab1:
    st.header("📚 上传文档")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 文件上传
        uploaded_files = st.file_uploader("上传PDF、TXT或Markdown文件",
                                         type=["pdf", "txt", "md"],
                                         accept_multiple_files=True)
        
        # 文档元数据
        doc_category = st.text_input("文档分类", value="", help="例如：物理课程、编程笔记等")
        doc_source = st.text_input("文档来源", value="", help="例如：课程名称、书名等")
    
    with col2:
        # 网页URL输入
        web_url = st.text_input("或输入网页URL", value="", help="例如：https://example.com/article")
        
        # 处理按钮
        process_button = st.button("处理文档")
    
    # 处理文档
    if process_button and (uploaded_files or web_url):
        # 验证API密钥
        if not api_key and embedding_type == "openai":
            st.error("请提供OpenAI API密钥，或选择HuggingFace嵌入模型")
        else:
            with st.spinner("正在处理文档..."):
                try:
                    # 初始化文档处理器
                    doc_processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    
                    # 初始化向量存储
                    vector_store = VectorStore(embedding_type=embedding_type, api_key=api_key)
                    
                    # 确保目录存在
                    os.makedirs(os.path.dirname(db_path), exist_ok=True)
                    
                    # 处理上传的文件
                    all_documents = []
                    
                    # 处理文件
                    if uploaded_files:
                        for uploaded_file in uploaded_files:
                            # 创建临时文件
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
                                temp_file.write(uploaded_file.getbuffer())
                                file_path = temp_file.name
                            
                            # 加载文档
                            docs = doc_processor.load_document(file_path)
                            
                            # 添加元数据
                            metadata = {
                                "source": uploaded_file.name,
                                "category": doc_category if doc_category else "未分类",
                                "origin": doc_source if doc_source else "用户上传"
                            }
                            
                            # 处理文档
                            processed_docs = doc_processor.process_documents(docs, metadata)
                            all_documents.extend(processed_docs)
                            
                            # 删除临时文件
                            os.unlink(file_path)
                    
                    # 处理网页
                    if web_url:
                        try:
                            docs = doc_processor.load_web_page(web_url)
                            
                            # 添加元数据
                            metadata = {
                                "source": web_url,
                                "category": doc_category if doc_category else "网页内容",
                                "origin": doc_source if doc_source else "网页"
                            }
                            
                            # 处理文档
                            processed_docs = doc_processor.process_documents(docs, metadata)
                            all_documents.extend(processed_docs)
                        except Exception as e:
                            st.error(f"处理网页时出错: {str(e)}")
                    
                    # 如果有处理后的文档，创建向量存储
                    if all_documents:
                        # 创建向量存储
                        vector_db = vector_store.create_vector_store(
                            all_documents,
                            store_type=vector_db_type,
                            persist_directory=db_path
                        )
                        
                        # 保存到会话状态
                        st.session_state.vector_store = vector_store
                        
                        # 初始化查询引擎
                        retriever = vector_store.get_retriever()
                        query_engine = QueryEngine(
                            retriever=retriever,
                            model_name=model_name,
                            api_key=api_key
                        )
                        st.session_state.query_engine = query_engine
                        st.session_state.documents_added = True
                        
                        # 初始化知识图谱 - 如果模块可用
                        if knowledge_graph_available and api_key:
                            try:
                                from langchain.chat_models import ChatOpenAI
                                llm = ChatOpenAI(model_name=model_name, openai_api_key=api_key)
                                st.session_state.knowledge_graph = KnowledgeGraph(llm=llm)
                            except Exception as e:
                                st.warning(f"初始化知识图谱时出错: {str(e)}")
                        
                        # 初始化语音交互 - 如果模块可用
                        if voice_interaction_available and enable_voice and api_key:
                            try:
                                st.session_state.voice_interaction = VoiceInteraction(openai_api_key=api_key)
                            except Exception as e:
                                st.warning(f"初始化语音交互时出错: {str(e)}")
                        
                        st.success(f"成功处理 {len(all_documents)} 个文档块！")
                    else:
                        st.warning("没有处理到任何文档内容，请检查文件或URL是否有效。")
                        
                except Exception as e:
                    st.error(f"处理文档时出错: {str(e)}")

    # 加载现有数据库
    st.markdown("---")
    if st.button("加载现有知识库"):
        try:
            # 验证API密钥
            if not api_key and embedding_type == "openai":
                st.error("请提供OpenAI API密钥，或选择HuggingFace嵌入模型")
            else:
                with st.spinner("正在加载知识库..."):
                    # 检查数据库目录是否存在
                    if not os.path.exists(db_path):
                        st.error(f"数据库路径不存在: {db_path}")
                    else:
                        # 初始化向量存储
                        vector_store = VectorStore(embedding_type=embedding_type, api_key=api_key)
                        
                        # 加载向量存储
                        vector_db = vector_store.load_vector_store(
                            store_type=vector_db_type,
                            persist_directory=db_path
                        )
                        
                        # 保存到会话状态
                        st.session_state.vector_store = vector_store
                        
                        # 初始化查询引擎
                        retriever = vector_store.get_retriever()
                        query_engine = QueryEngine(
                            retriever=retriever,
                            model_name=model_name,
                            api_key=api_key
                        )
                        st.session_state.query_engine = query_engine
                        st.session_state.documents_added = True
                        
                        # 初始化知识图谱 - 如果模块可用
                        if knowledge_graph_available and api_key:
                            try:
                                from langchain.chat_models import ChatOpenAI
                                llm = ChatOpenAI(model_name=model_name, openai_api_key=api_key)
                                st.session_state.knowledge_graph = KnowledgeGraph(llm=llm)
                            except Exception as e:
                                st.warning(f"初始化知识图谱时出错: {str(e)}")
                        
                        # 初始化语音交互 - 如果模块可用
                        if voice_interaction_available and enable_voice and api_key:
                            try:
                                st.session_state.voice_interaction = VoiceInteraction(openai_api_key=api_key)
                            except Exception as e:
                                st.warning(f"初始化语音交互时出错: {str(e)}")
                        
                        st.success("成功加载知识库！")
        except Exception as e:
            st.error(f"加载知识库时出错: {str(e)}")

# 标签页2：问答
with tab2:
    st.header("❓ 问答")
    
    # 检查是否有向量数据库
    if not st.session_state.documents_added:
        st.info("请先上传或加载文档，然后再进行问答。")
    else:
        # 问题输入方式选择 - 仅当语音模块可用时显示选择
        if voice_interaction_available and st.session_state.voice_interaction:
            input_method = st.radio("选择输入方式", ["文本输入", "语音输入"], horizontal=True)
        else:
            input_method = "文本输入"
        
        if input_method == "文本输入":
            # 问题输入
            question = st.text_input("输入你的问题", help="例如：什么是热力学第二定律？")
        else:
            # 语音输入
            st.info("点击下方按钮，然后说出你的问题")
            audio_bytes = st.audio_recorder(
                sample_rate=16000,
                key="voice_input"
            )
            
            if audio_bytes:
                try:
                    from io import BytesIO
                    audio_file = BytesIO(audio_bytes)
                    with st.spinner("正在转录语音..."):
                        question = st.session_state.voice_interaction.speech_to_text(audio_file)
                        st.success(f"已识别问题: {question}")
                except Exception as e:
                    st.error(f"语音转录失败: {str(e)}")
                    question = ""
            else:
                question = ""
        
        # 语音输出选项 - 仅当语音模块可用时显示
        if voice_interaction_available and st.session_state.voice_interaction:
            enable_voice_output = st.checkbox("启用语音回答", value=False)
        else:
            enable_voice_output = False
            
        # 查询按钮
        if st.button("提问") and question:
            with st.spinner("思考中..."):
                try:
                    # 查询
                    result = st.session_state.query_engine.query(question)
                    
                    # 显示回答
                    st.markdown("### 回答")
                    answer_text = result["answer"]
                    st.write(answer_text)
                    
                    # 语音回答 - 仅当启用语音输出且语音模块可用时
                    if enable_voice_output and voice_interaction_available and st.session_state.voice_interaction:
                        try:
                            with st.spinner("生成语音回答..."):
                                # 将文本分成小段以适应API限制
                                max_length = 4000
                                answer_chunks = [answer_text[i:i+max_length] for i in range(0, len(answer_text), max_length)]
                                
                                for i, chunk in enumerate(answer_chunks):
                                    audio_content, content_type = st.session_state.voice_interaction.text_to_speech(chunk)
                                    if audio_content and content_type:
                                        st.audio(audio_content, format=content_type)
                                    else:
                                        st.warning("无法生成语音回答")
                                        break
                        except Exception as e:
                            st.error(f"生成语音失败: {str(e)}")
                    
                    # 显示来源
                    st.markdown("### 参考来源")
                    for i, doc in enumerate(result["source_documents"]):
                        with st.expander(f"来源 {i+1} ({doc.metadata.get('source', '未知来源')})"):
                            st.markdown(doc.page_content)
                            st.markdown(f"**分类**: {doc.metadata.get('category', '未分类')}")
                            st.markdown(f"**来源**: {doc.metadata.get('origin', '未知')}")
                            
                except Exception as e:
                    st.error(f"查询过程中出错: {str(e)}")

# 标签页3：知识分析
with tab3:
    st.header("📊 知识分析")
    
    # 检查是否有向量数据库
    if not st.session_state.documents_added:
        st.info("请先上传或加载文档，然后再进行知识分析。")
    else:
        # 选择分析类型
        analysis_type = st.radio(
            "选择分析类型",
            ["生成摘要", "提取关键点"]
        )
        
        # 主题或关键词
        topic = st.text_input("输入主题或关键词", help="例如：机器学习、量子物理")
        
        # 分析按钮
        if st.button("分析") and topic:
            with st.spinner("分析中..."):
                try:
                    # 获取相关文档
                    retriever = st.session_state.vector_store.get_retriever({"k": 8})
                    docs = retriever.get_relevant_documents(topic)
                    
                    if not docs:
                        st.warning(f"未找到与'{topic}'相关的内容。")
                    else:
                        if analysis_type == "生成摘要":
                            # 生成摘要
                            summary = st.session_state.query_engine.generate_summary(docs)
                            
                            # 显示摘要
                            st.markdown("### 关于 " + topic + " 的摘要")
                            st.write(summary)
                            
                        elif analysis_type == "提取关键点":
                            # 提取关键点
                            key_points = st.session_state.query_engine.extract_key_points(docs)
                            
                            # 显示关键点
                            st.markdown("### 关于 " + topic + " 的关键点")
                            for i, point in enumerate(key_points):
                                st.markdown(f"**{i+1}.** {point}")
                        
                        # 显示来源文档数量
                        st.info(f"分析基于 {len(docs)} 个相关文档块")
                        
                except Exception as e:
                    st.error(f"分析过程中出错: {str(e)}")

# 标签页4：知识图谱 - 仅当模块可用时显示
if knowledge_graph_available and tab4:
    with tab4:
        st.header("🔍 知识图谱")
        
        # 检查是否有向量数据库和API密钥
        if not st.session_state.documents_added:
            st.info("请先上传或加载文档，然后再生成知识图谱。")
        elif not api_key:
            st.warning("生成知识图谱需要OpenAI API密钥，请在配置中提供。")
        elif not st.session_state.knowledge_graph:
            st.warning("知识图谱模块未初始化，请确保API密钥正确，然后重新加载知识库。")
        else:
            # 主题或关键词
            graph_topic = st.text_input("输入要生成知识图谱的主题或关键词", help="例如：人工智能、气候变化")
            
            # 设置
            col1, col2 = st.columns(2)
            with col1:
                doc_count = st.slider("检索文档数量", min_value=3, max_value=15, value=5, step=1)
            with col2:
                entity_limit = st.slider("实体数量限制", min_value=5, max_value=20, value=10, step=1)
            
            # 生成按钮
            if st.button("生成知识图谱") and graph_topic:
                with st.spinner("正在生成知识图谱..."):
                    try:
                        # 获取相关文档
                        retriever = st.session_state.vector_store.get_retriever({"k": doc_count})
                        docs = retriever.get_relevant_documents(graph_topic)
                        
                        if not docs:
                            st.warning(f"未找到与'{graph_topic}'相关的内容。")
                        else:
                            # 提取实体和关系
                            relations = st.session_state.knowledge_graph.extract_entities_and_relations(docs)
                            
                            if not relations:
                                st.warning("未能提取出实体和关系，请尝试其他主题或关键词。")
                            else:
                                # 限制实体数量以避免图过于复杂
                                if len(relations) > entity_limit:
                                    relations = relations[:entity_limit]
                                    
                                # 构建图
                                graph = st.session_state.knowledge_graph.build_graph(relations)
                                
                                # 显示图
                                try:
                                    fig = st.session_state.knowledge_graph.visualize_plotly(graph)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # 显示中心概念
                                    central_concepts = st.session_state.knowledge_graph.get_central_concepts()
                                    if central_concepts:
                                        st.markdown("### 中心概念")
                                        for concept, score in central_concepts:
                                            st.markdown(f"- **{concept}** (重要性得分: {score:.2f})")
                                except Exception as e:
                                    st.error(f"可视化图谱时出错: {str(e)}")
                                    
                                # 显示处理的文档数
                                st.info(f"知识图谱基于 {len(docs)} 个相关文档块，包含 {len(graph.nodes())} 个实体和 {len(graph.edges())} 个关系。")
                    
                    except Exception as e:
                        st.error(f"生成知识图谱时出错: {str(e)}")
                        
            # 知识图谱的说明
            with st.expander("什么是知识图谱？"):
                st.markdown("""
                **知识图谱**是以图的形式呈现知识的结构化表示，由实体（节点）和关系（边）组成。
                
                在SmartNote AI中，知识图谱帮助你:
                - 直观地了解文档中的关键概念及其关系
                - 发现潜在的知识联系
                - 识别重要的中心概念
                
                使用方法：输入一个主题或关键词，系统会在你的知识库中检索相关内容，然后提取出实体和关系，以交互式图表的形式展示。
                """)

# 页脚
st.markdown("---")
st.markdown("*SmartNote AI - 让知识永远触手可及* 🚀")
