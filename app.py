# app/app.py

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Try to load API key from Streamlit secrets first
if 'OPENAI_API_KEY' in st.secrets:
    os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
# Then try to load from local .env file
else:
    load_dotenv()

# 导入自定义模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
from utils.query_engine import QueryEngine

# 导入新的模块 - 测验生成器和闪卡生成器
from utils.quiz_generator import QuizGenerator
from utils.flashcard_generator import FlashcardGenerator

# 尝试导入语音交互模块
voice_interaction_available = False
try:
    from utils.voice_interaction import VoiceInteraction
    voice_interaction_available = True
except ImportError:
    st.sidebar.warning("Voice interaction module failed to import. Please ensure dependencies are installed.")

# 尝试导入知识图谱模块
knowledge_graph_available = False
try:
    from utils.knowledge_graph import KnowledgeGraph
    knowledge_graph_available = True
except ImportError as e:
    missing_module = str(e).split("'")[1] if "'" in str(e) else str(e)
    st.sidebar.warning(f"Knowledge graph module failed to import: Missing {missing_module} module. Please run 'pip install {missing_module}'")

# 设置页面标题和图标
st.set_page_config(
    page_title="SmartNote AI - Personal Knowledge Base",
    page_icon="🧠",
    layout="wide"
)

# Apply English font to entire app
st.markdown("""
<style>
    * {
        font-family: 'Arial', 'Helvetica', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Arial', 'Helvetica', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# 应用标题
st.title("🧠 SmartNote AI - Personal Knowledge Base")
st.markdown("Upload your documents, create your personal knowledge base, and let AI remember and answer questions for you")

# 侧边栏 - 配置
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API密钥设置 - 修改此部分

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # 嵌入模型选择
    embedding_type = st.selectbox(
        "Select Embedding Model",
        options=["openai", "huggingface"],
        index=0,
        help="OpenAI embeddings are higher quality but require API key; HuggingFace models are free but lower quality"
    )
    
    # 语言模型选择
    model_name = st.selectbox(
        "Select Language Model",
        options=["gpt-3.5-turbo", "gpt-4"],
        index=0
    )
    
    # 分块设置
    chunk_size = st.slider("Document Chunk Size", min_value=100, max_value=2000, value=500, step=100)
    chunk_overlap = st.slider("Chunk Overlap Size", min_value=0, max_value=500, value=50, step=10)
    
    # 数据库设置
    vector_db_type = st.selectbox(
        "Vector Database Type",
        options=["faiss", "chroma"],
        index=0
    )
    
    # 数据库路径
    db_path = st.text_input("Database Storage Path", value="./data/vector_db")

    # 语音设置 - 仅当模块可用时显示
    if voice_interaction_available:
        st.header("🎙️ Voice Settings")
        enable_voice = st.checkbox("Enable Voice Features", value=False)
        if enable_voice:
            voice_type = st.selectbox(
                "Select Voice Type",
                options=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                index=0
            )
    else:
        enable_voice = False
    
    # 保存按钮
    if st.button("Save Configuration"):
        st.success("Configuration saved!")

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
if 'quiz_generator' not in st.session_state:
    st.session_state.quiz_generator = None
if 'flashcard_generator' not in st.session_state:
    st.session_state.flashcard_generator = None
if 'quiz_results' not in st.session_state:
    st.session_state.quiz_results = None
if 'current_quiz' not in st.session_state:
    st.session_state.current_quiz = None
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = {}
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False

# 创建标签页 - 根据可用模块调整
tabs = ["📚 Upload Documents", "❓ Q&A", "📊 Knowledge Analysis", "📝 Quiz", "🔄 Learning Reinforcement"]
if knowledge_graph_available:
    tabs.append("🔍 Knowledge Graph")

all_tabs = st.tabs(tabs)
tab1 = all_tabs[0]  # 上传文档
tab2 = all_tabs[1]  # 问答
tab3 = all_tabs[2]  # 知识分析
tab4 = all_tabs[3]  # 测验
tab5 = all_tabs[4]  # 学习强化
tab6 = all_tabs[5] if knowledge_graph_available else None  # 知识图谱（如果可用）


with tab1:
    st.header("📚 Upload Documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 文件上传
        uploaded_files = st.file_uploader("Upload PDF, TXT or Markdown files",
                                         type=["pdf", "txt", "md"],
                                         accept_multiple_files=True)
        
        # 文档元数据
        doc_category = st.text_input("Document Category", value="", help="E.g.: Physics course, Programming notes, etc.")
        doc_source = st.text_input("Document Source", value="", help="E.g.: Course name, Book title, etc.")
    
    with col2:
        # 网页URL输入
        web_url = st.text_input("Or enter a web page URL", value="", help="E.g.: https://example.com/article")
        
        # 处理按钮
        process_button = st.button("Process Documents")
    
    # 处理文档
    if process_button and (uploaded_files or web_url):
        # 验证API密钥
        if not os.environ.get("OPENAI_API_KEY") and embedding_type == "openai":
            st.error("Please provide an OpenAI API key, or select HuggingFace embedding model")
        else:
            with st.spinner("Processing documents..."):
                try:
                    # 初始化文档处理器
                    doc_processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    
                    # 初始化向量存储
                    vector_store = VectorStore(embedding_type=embedding_type, api_key=os.environ.get("OPENAI_API_KEY", ""))
                    
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
                                "category": doc_category if doc_category else "Uncategorized",
                                "origin": doc_source if doc_source else "User Upload"
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
                                "category": doc_category if doc_category else "Web Content",
                                "origin": doc_source if doc_source else "Web Page"
                            }
                            
                            # 处理文档
                            processed_docs = doc_processor.process_documents(docs, metadata)
                            all_documents.extend(processed_docs)
                        except Exception as e:
                            st.error(f"Error processing web page: {str(e)}")
                    
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
                            api_key=os.environ.get("OPENAI_API_KEY", "")
                        )
                        st.session_state.query_engine = query_engine
                        st.session_state.documents_added = True
                        
                        # 初始化知识图谱 - 如果模块可用
                        if knowledge_graph_available and os.environ.get("OPENAI_API_KEY"):
                            try:
                                from langchain.chat_models import ChatOpenAI
                                llm = ChatOpenAI(model_name=model_name, openai_api_key=os.environ.get("OPENAI_API_KEY"))
                                st.session_state.knowledge_graph = KnowledgeGraph(llm=llm)
                            except Exception as e:
                                st.warning(f"Error initializing knowledge graph: {str(e)}")
                        
                        # 初始化语音交互 - 如果模块可用
                        if voice_interaction_available and enable_voice and os.environ.get("OPENAI_API_KEY"):
                            try:
                                st.session_state.voice_interaction = VoiceInteraction(openai_api_key=os.environ.get("OPENAI_API_KEY"))
                            except Exception as e:
                                st.warning(f"Error initializing voice interaction: {str(e)}")
                        
                        # 初始化测验生成器
                        if os.environ.get("OPENAI_API_KEY"):
                            try:
                                from langchain.chat_models import ChatOpenAI
                                llm = ChatOpenAI(model_name=model_name, openai_api_key=os.environ.get("OPENAI_API_KEY"))
                                st.session_state.quiz_generator = QuizGenerator(llm=llm)
                                st.session_state.flashcard_generator = FlashcardGenerator(llm=llm)
                            except Exception as e:
                                st.warning(f"Error initializing quiz and flashcard generators: {str(e)}")
                        
                        st.success(f"Successfully processed {len(all_documents)} document chunks!")
                    else:
                        st.warning("No document content was processed. Please check if the file or URL is valid.")
                        
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")

    # 加载现有数据库
    st.markdown("---")
    if st.button("Load Existing Knowledge Base"):
        try:
            # 验证API密钥
            if not os.environ.get("OPENAI_API_KEY") and embedding_type == "openai":
                st.error("Please provide an OpenAI API key, or select HuggingFace embedding model")
            else:
                with st.spinner("Loading knowledge base..."):
                    # 检查数据库目录是否存在
                    if not os.path.exists(db_path):
                        st.error(f"Database path doesn't exist: {db_path}")
                    else:
                        # 初始化向量存储
                        vector_store = VectorStore(embedding_type=embedding_type, api_key=os.environ.get("OPENAI_API_KEY", ""))
                        
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
                            api_key=os.environ.get("OPENAI_API_KEY", "")
                        )
                        st.session_state.query_engine = query_engine
                        st.session_state.documents_added = True
                        
                        # 初始化知识图谱 - 如果模块可用
                        if knowledge_graph_available and os.environ.get("OPENAI_API_KEY"):
                            try:
                                from langchain.chat_models import ChatOpenAI
                                llm = ChatOpenAI(model_name=model_name, openai_api_key=os.environ.get("OPENAI_API_KEY"))
                                st.session_state.knowledge_graph = KnowledgeGraph(llm=llm)
                            except Exception as e:
                                st.warning(f"Error initializing knowledge graph: {str(e)}")
                        
                        # 初始化语音交互 - 如果模块可用
                        if voice_interaction_available and enable_voice and os.environ.get("OPENAI_API_KEY"):
                            try:
                                st.session_state.voice_interaction = VoiceInteraction(openai_api_key=os.environ.get("OPENAI_API_KEY"))
                            except Exception as e:
                                st.warning(f"Error initializing voice interaction: {str(e)}")
                        
                        # 初始化测验生成器和闪卡生成器
                        if os.environ.get("OPENAI_API_KEY"):
                            try:
                                from langchain.chat_models import ChatOpenAI
                                llm = ChatOpenAI(model_name=model_name, openai_api_key=os.environ.get("OPENAI_API_KEY"))
                                st.session_state.quiz_generator = QuizGenerator(llm=llm)
                                st.session_state.flashcard_generator = FlashcardGenerator(llm=llm)
                            except Exception as e:
                                st.warning(f"Error initializing quiz and flashcard generators: {str(e)}")
                        
                        st.success("Successfully loaded knowledge base!")
        except Exception as e:
            st.error(f"Error loading knowledge base: {str(e)}")

# 标签页2：问答
with tab2:
    st.header("❓ Q&A")
    
    # 检查是否有向量数据库
    if not st.session_state.documents_added:
        st.info("Please upload or load documents first before asking questions.")
    else:
        # 问题输入方式选择 - 仅当语音模块可用时显示选择
        if voice_interaction_available and st.session_state.voice_interaction:
            input_method = st.radio("Select input method", ["Text Input", "Voice Input"], horizontal=True)
        else:
            input_method = "Text Input"
        
        if input_method == "Text Input":
            # 问题输入
            question = st.text_input("Enter your question", help="E.g.: What is the second law of thermodynamics?")
        else:
            # 语音输入
            st.info("Click the button below and speak your question")
            audio_bytes = st.audio_recorder(
                sample_rate=16000,
                key="voice_input"
            )
            
            if audio_bytes:
                try:
                    from io import BytesIO
                    audio_file = BytesIO(audio_bytes)
                    with st.spinner("Transcribing speech..."):
                        question = st.session_state.voice_interaction.speech_to_text(audio_file)
                        st.success(f"Recognized question: {question}")
                except Exception as e:
                    st.error(f"Speech transcription failed: {str(e)}")
                    question = ""
            else:
                question = ""
        
        # 语音输出选项 - 仅当语音模块可用时显示
        if voice_interaction_available and st.session_state.voice_interaction:
            enable_voice_output = st.checkbox("Enable voice response", value=False)
        else:
            enable_voice_output = False
            
        # 查询按钮
        if st.button("Ask") and question:
            with st.spinner("Thinking..."):
                try:
                    # 查询
                    result = st.session_state.query_engine.query(question)
                    
                    # 显示回答
                    st.markdown("### Answer")
                    answer_text = result["answer"]
                    st.write(answer_text)
                    
                    # 语音回答 - 仅当启用语音输出且语音模块可用时
                    if enable_voice_output and voice_interaction_available and st.session_state.voice_interaction:
                        try:
                            with st.spinner("Generating voice response..."):
                                # 将文本分成小段以适应API限制
                                max_length = 4000
                                answer_chunks = [answer_text[i:i+max_length] for i in range(0, len(answer_text), max_length)]
                                
                                for i, chunk in enumerate(answer_chunks):
                                    audio_content, content_type = st.session_state.voice_interaction.text_to_speech(chunk)
                                    if audio_content and content_type:
                                        st.audio(audio_content, format=content_type)
                                    else:
                                        st.warning("Could not generate voice response")
                                        break
                        except Exception as e:
                            st.error(f"Failed to generate speech: {str(e)}")
                    
                    # 显示来源
                    st.markdown("### References")
                    for i, doc in enumerate(result["source_documents"]):
                        with st.expander(f"Source {i+1} ({doc.metadata.get('source', 'Unknown source')})"):
                            st.markdown(doc.page_content)
                            st.markdown(f"**Category**: {doc.metadata.get('category', 'Uncategorized')}")
                            st.markdown(f"**Origin**: {doc.metadata.get('origin', 'Unknown')}")
                            
                except Exception as e:
                    st.error(f"Error during query: {str(e)}")

# 标签页3：知识分析
with tab3:
    st.header("📊 Knowledge Analysis")
    
    # 检查是否有向量数据库
    if not st.session_state.documents_added:
        st.info("Please upload or load documents first before performing knowledge analysis.")
    else:
        # 选择分析类型
        analysis_type = st.radio(
            "Select analysis type",
            ["Generate Summary", "Extract Key Points"]
        )
        
        # 主题或关键词
        topic = st.text_input("Enter topic or keyword", help="E.g.: Machine Learning, Quantum Physics")
        
        # 分析按钮
        if st.button("Analyze") and topic:
            with st.spinner("Analyzing..."):
                try:
                    # 获取相关文档
                    retriever = st.session_state.vector_store.get_retriever({"k": 8})
                    docs = retriever.get_relevant_documents(topic)
                    
                    if not docs:
                        st.warning(f"No content related to '{topic}' was found.")
                    else:
                        if analysis_type == "Generate Summary":
                            # 生成摘要
                            summary = st.session_state.query_engine.generate_summary(docs)
                            
                            # 显示摘要
                            st.markdown("### Summary about " + topic)
                            st.write(summary)
                            
                        elif analysis_type == "Extract Key Points":
                            # 提取关键点
                            key_points = st.session_state.query_engine.extract_key_points(docs)
                            
                            # 显示关键点
                            st.markdown("### Key points about " + topic)
                            for i, point in enumerate(key_points):
                                st.markdown(f"**{i+1}.** {point}")
                        
                        # 显示来源文档数量
                        st.info(f"Analysis based on {len(docs)} relevant document chunks")
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    
                    
# 标签页4：知识图谱 - 仅当模块可用时显示
if knowledge_graph_available and tab6:
    with tab6:
        st.header("🔍 Knowledge Graph")
        
        # 检查是否有向量数据库和API密钥
        if not st.session_state.documents_added:
            st.info("Please upload or load documents first before generating a knowledge graph.")
        elif not os.environ.get("OPENAI_API_KEY"):
            st.warning("Generating a knowledge graph requires an OpenAI API key. Please provide one in the configuration.")
        elif not st.session_state.knowledge_graph:
            st.warning("Knowledge graph module not initialized. Please ensure API key is correct and reload the knowledge base.")
        else:
            # 主题或关键词
            graph_topic = st.text_input("Enter topic or keyword for knowledge graph", help="E.g.: Artificial Intelligence, Climate Change")
            
            # 设置
            col1, col2 = st.columns(2)
            with col1:
                doc_count = st.slider("Number of documents to retrieve", min_value=3, max_value=15, value=5, step=1)
            with col2:
                entity_limit = st.slider("Entity count limit", min_value=5, max_value=20, value=10, step=1)
            
            # 生成按钮
            if st.button("Generate Knowledge Graph") and graph_topic:
                with st.spinner("Generating knowledge graph..."):
                    try:
                        # 获取相关文档
                        retriever = st.session_state.vector_store.get_retriever({"k": doc_count})
                        docs = retriever.get_relevant_documents(graph_topic)
                        
                        if not docs:
                            st.warning(f"No content related to '{graph_topic}' was found.")
                        else:
                            # 提取实体和关系
                            relations = st.session_state.knowledge_graph.extract_entities_and_relations(docs)
                            
                            if not relations:
                                st.warning("Could not extract entities and relations. Please try another topic or keyword.")
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
                                        st.markdown("### Central Concepts")
                                        for concept, score in central_concepts:
                                            st.markdown(f"- **{concept}** (Importance score: {score:.2f})")
                                except Exception as e:
                                    st.error(f"Error visualizing graph: {str(e)}")
                                    
                                # 显示处理的文档数
                                st.info(f"Knowledge graph based on {len(docs)} relevant document chunks, containing {len(graph.nodes())} entities and {len(graph.edges())} relationships.")
                    
                    except Exception as e:
                        st.error(f"Error generating knowledge graph: {str(e)}")
                        
            # 知识图谱的说明
            with st.expander("What is a Knowledge Graph?"):
                st.markdown("""
                A **Knowledge Graph** is a structured representation of knowledge in the form of a graph, consisting of entities (nodes) and relationships (edges).
                
                In SmartNote AI, knowledge graphs help you:
                - Visually understand key concepts in your documents and their relationships
                - Discover potential knowledge connections
                - Identify important central concepts
                
                How to use: Enter a topic or keyword, and the system will retrieve relevant content from your knowledge base, extract entities and relationships, and display them as an interactive graph.
                """)

# 标签页4：测验
with tab4:
    st.header("📝 Quiz")
    
    # 检查是否有向量数据库和API密钥
    if not st.session_state.documents_added:
        st.info("Please upload or load documents first before generating quizzes.")
    elif not os.environ.get("OPENAI_API_KEY"):
        st.warning("Generating quizzes requires an OpenAI API key. Please provide one in the configuration.")
    elif not st.session_state.quiz_generator:
        st.warning("Quiz module not initialized. Please ensure API key is correct and reload the knowledge base.")
    else:
        # 定义创建新测验和显示当前测验的列
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Generate New Quiz")
            
            # 测验主题
            quiz_topic = st.text_input("Enter quiz topic", key="quiz_topic",
                                       help="E.g.: Newton's Laws of Motion, Python Programming Basics")
            
            # 测验配置
            quiz_difficulty = st.select_slider(
                "Quiz Difficulty",
                options=["easy", "medium", "hard"],
                value="medium"
            )
            
            num_questions = st.slider("Number of Questions", min_value=3, max_value=15, value=5)
            
            question_types = st.multiselect(
                "Question Types",
                options=["multiple_choice", "true_false", "open_ended"],
                default=["multiple_choice"],
                help="Multiple choice: Select one correct answer; True/False: Evaluate statement; Open-ended: Write your own answer"
            )
            
            # 生成测验按钮
            generate_button = st.button("Generate Quiz")
            
            if generate_button and quiz_topic:
                with st.spinner("Generating quiz..."):
                    try:
                        # 获取相关文档
                        retriever = st.session_state.vector_store.get_retriever({"k": min(10, num_questions * 2)})
                        docs = retriever.get_relevant_documents(quiz_topic)
                        
                        if not docs:
                            st.warning(f"No content related to '{quiz_topic}' was found.")
                        else:
                            # 配置测验
                            quiz_config = {
                                "topic": quiz_topic,
                                "difficulty": quiz_difficulty,
                                "num_questions": num_questions,
                                "question_types": question_types
                            }
                            
                            # 生成测验
                            quiz = st.session_state.quiz_generator.generate_quiz(docs, quiz_config)
                            
                            # 保存到会话状态
                            st.session_state.current_quiz = quiz
                            st.session_state.quiz_answers = {}
                            st.session_state.quiz_submitted = False
                            st.session_state.quiz_results = None
                            
                            # 重新加载页面以显示新测验
                            st.rerun()
                    
                    except Exception as e:
                        st.error(f"Error generating quiz: {str(e)}")
        
        # 显示当前测验
        with col2:
            if st.session_state.current_quiz:
                st.subheader(f"Quiz: {st.session_state.current_quiz['topic']}")
                st.markdown(f"**Difficulty**: {st.session_state.current_quiz['difficulty'].title()}")
                st.markdown(f"**Total Questions**: {st.session_state.current_quiz['total_questions']}")
                
                # 如果测验已提交，显示重置按钮
                if st.session_state.quiz_submitted:
                    if st.button("Start New Attempt"):
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_results = None
                        st.rerun()
                
                # 如果测验未提交，显示问题
                if not st.session_state.quiz_submitted:
                    with st.form("quiz_form"):
                        for i, question in enumerate(st.session_state.current_quiz["questions"]):
                            st.markdown(f"### Question {i+1}")
                            
                            if question["type"] == "multiple_choice":
                                st.markdown(f"**{question['question']}**")
                                
                                # 使用radio按钮显示选项
                                st.session_state.quiz_answers[i] = st.radio(
                                    f"Select answer for question {i+1}:",
                                    options=question["options"],
                                    key=f"q{i}_mc"
                                )
                                
                            elif question["type"] == "true_false":
                                st.markdown(f"**{question['statement']}**")
                                
                                # 使用radio按钮显示True/False选项
                                st.session_state.quiz_answers[i] = st.radio(
                                    f"Select answer for question {i+1}:",
                                    options=["True", "False"],
                                    key=f"q{i}_tf"
                                )
                                
                            elif question["type"] == "open_ended":
                                st.markdown(f"**{question['question']}**")
                                
                                # 使用文本区域输入回答
                                st.session_state.quiz_answers[i] = st.text_area(
                                    f"Your answer for question {i+1}:",
                                    key=f"q{i}_oe",
                                    height=100
                                )
                                
                            st.markdown("---")
                        
                        # 提交按钮
                        submit_quiz = st.form_submit_button("Submit Quiz")
                        
                        if submit_quiz:
                            # 评分
                            with st.spinner("Grading quiz..."):
                                results = []
                                total_score = 0
                                max_score = len(st.session_state.current_quiz["questions"]) * 100
                                
                                for i, question in enumerate(st.session_state.current_quiz["questions"]):
                                    user_answer = st.session_state.quiz_answers.get(i, "")
                                    
                                    if question["type"] == "multiple_choice":
                                        correct = question["options"][question["correct_answer"]] == user_answer
                                        score = 100 if correct else 0
                                        feedback = question["explanation"]
                                        result = {
                                            "question_num": i + 1,
                                            "correct": correct,
                                            "score": score,
                                            "feedback": feedback,
                                            "user_answer": user_answer,
                                            "correct_answer": question["options"][question["correct_answer"]]
                                        }
                                        
                                    elif question["type"] == "true_false":
                                        # 将字符串"True"/"False"转换为布尔值
                                        user_bool = user_answer == "True"
                                        correct = question["correct_answer"] == user_bool
                                        score = 100 if correct else 0
                                        feedback = question["explanation"]
                                        result = {
                                            "question_num": i + 1,
                                            "correct": correct,
                                            "score": score,
                                            "feedback": feedback,
                                            "user_answer": user_answer,
                                            "correct_answer": "True" if question["correct_answer"] else "False"
                                        }
                                        
                                    elif question["type"] == "open_ended":
                                        # 使用LLM评估开放性问题
                                        evaluation = st.session_state.quiz_generator.grade_open_answer(question, user_answer)
                                        score = evaluation["score"]
                                        feedback = evaluation["feedback"]
                                        missing = ", ".join(evaluation["missing_keywords"]) if evaluation["missing_keywords"] else "None"
                                        result = {
                                            "question_num": i + 1,
                                            "score": score,
                                            "feedback": feedback,
                                            "user_answer": user_answer,
                                            "missing_keywords": missing,
                                            "reference_answer": question["answer"]
                                        }
                                    
                                    total_score += score
                                    results.append(result)
                                
                                # 计算总分
                                average_score = total_score / len(results)
                                
                                # 保存结果
                                st.session_state.quiz_results = {
                                    "results": results,
                                    "total_score": total_score,
                                    "average_score": average_score,
                                    "max_score": max_score
                                }
                                
                                st.session_state.quiz_submitted = True
                                st.rerun()
                
                # 如果测验已提交，显示结果
                else:
                    if st.session_state.quiz_results:
                        results = st.session_state.quiz_results
                        
                        # 显示总分
                        st.markdown(f"### Your Score: {results['average_score']:.1f}%")
                        st.progress(results['average_score'] / 100)
                        
                        # 根据分数给予评价
                        if results['average_score'] >= 90:
                            st.success("Excellent work! You've mastered this topic.")
                        elif results['average_score'] >= 70:
                            st.success("Good job! You have a solid understanding of this topic.")
                        elif results['average_score'] >= 50:
                            st.warning("You're on the right track, but might need more review.")
                        else:
                            st.error("You should spend more time studying this topic.")
                        
                        # 显示每个问题的详细结果
                        st.markdown("### Detailed Results")
                        for result in results["results"]:
                            with st.expander(f"Question {result['question_num']}"):
                                question = st.session_state.current_quiz["questions"][result['question_num']-1]
                                
                                if question["type"] == "multiple_choice":
                                    st.markdown(f"**Question**: {question['question']}")
                                    st.markdown(f"**Your answer**: {result['user_answer']}")
                                    st.markdown(f"**Correct answer**: {result['correct_answer']}")
                                    if result['correct']:
                                        st.success(f"✓ Correct! Score: {result['score']}")
                                    else:
                                        st.error(f"✗ Incorrect. Score: {result['score']}")
                                    st.markdown(f"**Explanation**: {result['feedback']}")
                                    
                                elif question["type"] == "true_false":
                                    st.markdown(f"**Statement**: {question['statement']}")
                                    st.markdown(f"**Your answer**: {result['user_answer']}")
                                    st.markdown(f"**Correct answer**: {result['correct_answer']}")
                                    if result['correct']:
                                        st.success(f"✓ Correct! Score: {result['score']}")
                                    else:
                                        st.error(f"✗ Incorrect. Score: {result['score']}")
                                    st.markdown(f"**Explanation**: {result['feedback']}")
                                    
                                elif question["type"] == "open_ended":
                                    st.markdown(f"**Question**: {question['question']}")
                                    st.markdown(f"**Your answer**: {result['user_answer']}")
                                    st.markdown(f"**Score**: {result['score']}/100")
                                    st.markdown(f"**Feedback**: {result['feedback']}")
                                    st.markdown(f"**Missing keywords**: {result['missing_keywords']}")
                                    with st.expander("Reference Answer"):
                                        st.markdown(result["reference_answer"])
        
        # 显示测验指南和说明
        with st.expander("Quiz Guidelines"):
            st.markdown("""
            ## How to Use the Quiz Feature
            
            1. **Generate a quiz** by entering a topic and configuring options
            2. **Take the quiz** by answering all questions
            3. **Submit your answers** to receive immediate feedback
            4. **Review your results** with detailed explanations
            
            ### Question Types:
            - **Multiple Choice**: Select one correct answer from four options
            - **True/False**: Determine if a statement is true or false
            - **Open-Ended**: Write your own answer, which will be evaluated based on relevance and completeness
            
            ### Difficulty Levels:
            - **Easy**: Basic recall and simpler concepts
            - **Medium**: Application of concepts and deeper understanding
            - **Hard**: Analysis, synthesis, and complex problem-solving
            """)

# 标签5：学习强化（闪卡）
# --- Tab 5 : Learning Reinforcement (Flashcards) ---------------------------
with tab5:
    st.header("🔄 Learning Reinforcement")

    # ╭────────────────────────  前置检查  ───────────────────────╮
    if not st.session_state.documents_added:
        st.info("Please upload or load documents first before creating flashcards.")
        st.stop()
    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("Creating flashcards requires an OpenAI API key. Please provide one in the configuration.")
        st.stop()
    if not st.session_state.flashcard_generator:
        st.warning("Flashcard module not initialized. Please ensure API key is correct and reload the knowledge base.")
        st.stop()
    # ╰──────────────────────────────────────────────────────────╯

    # 创建 3 个子标签页
    fc_tabs = st.tabs(["Create Flashcards", "Today's Review", "View All Decks"])

    # ───────────────────────── Tab 1 · Create Flashcards ──────────────────────────
    with fc_tabs[0]:
        st.subheader("Create Flashcards")

        flashcard_topic = st.text_input(
            "Enter topic for flashcards",
            key="flashcard_topic",
            help="E.g.: Chemical Elements, American Presidents",
        )

        deck_name = st.text_input("Deck Name", value="My Knowledge Cards")

        col1, col2 = st.columns(2)
        with col1:
            complexity = st.select_slider(
                "Complexity Level",
                options=["basic", "medium", "advanced"],
                value="medium",
            )
        with col2:
            num_cards = st.slider("Number of Cards", min_value=5, max_value=30, value=10)

        card_types = st.multiselect(
            "Card Types",
            options=["basic", "cloze"],
            default=["basic"],
            help="Basic: Question-answer format; Cloze: Fill-in-the-blank format",
        )

        if st.button("Create Flashcards") and flashcard_topic:
            with st.spinner("Generating flashcards..."):
                try:
                    retriever = st.session_state.vector_store.get_retriever({"k": min(10, num_cards)})
                    docs = retriever.get_relevant_documents(flashcard_topic)

                    if not docs:
                        st.warning(f"No content related to '{flashcard_topic}' was found.")
                    else:
                        config = {
                            "card_types": card_types,
                            "num_cards": num_cards,
                            "complexity": complexity,
                            "deck_name": deck_name,
                        }
                        flashcards = st.session_state.flashcard_generator.generate_flashcards(docs, config)

                        st.success(
                            f"Successfully created {flashcards['total_cards']} flashcards in deck '{flashcards['deck_name']}'!"
                        )

                        with st.expander("Preview Flashcards"):
                            for i, card in enumerate(flashcards["cards"][:5]):
                                st.markdown(f"**Card {i+1}:**")
                                if card["type"] == "basic":
                                    st.markdown(f"**Front**: {card['front']}")
                                    st.markdown(f"**Back**: {card['back']}")
                                else:  # cloze
                                    st.markdown(f"**Text**: {card['front']}")
                                    st.markdown(f"**Answer**: {card['back']}")
                                st.markdown(f"**Tags**: {', '.join(card['tags'])}")
                                st.markdown("---")
                            if len(flashcards["cards"]) > 5:
                                st.info(f"{len(flashcards['cards']) - 5} more cards created but not shown here.")
                except Exception as e:
                    st.error(f"Error generating flashcards: {e}")

    # ──────────────────────── Tab 2 · Today's Review ─────────────────────────────
    with fc_tabs[1]:
        st.subheader("Today's Review")

        if st.button("Load Today's Cards"):
            with st.spinner("Loading cards due for review..."):
                try:
                    due_cards = st.session_state.flashcard_generator.get_cards_due_today()
                except Exception as e:
                    st.error(f"Error loading review cards: {e}")
                    due_cards = []

                if due_cards:
                    st.success(f"Loaded {len(due_cards)} cards for review")
                    st.session_state.review_cards = due_cards
                    st.session_state.review_index = 0
                    st.session_state.show_answer = False
                    st.rerun()
                else:
                    st.info("No cards due for review today. Great job staying on top of your studies!")

        # 显示复习界面
        if "review_cards" in st.session_state and st.session_state.review_cards:
            progress = (st.session_state.review_index + 1) / len(st.session_state.review_cards)
            st.progress(progress)
            st.markdown(
                f"Card **{st.session_state.review_index + 1} / {len(st.session_state.review_cards)}**"
            )

            current = st.session_state.review_cards[st.session_state.review_index]
            st.markdown("### Question")
            st.markdown(f"**{current['front']}**")

            if st.button("Show Answer"):
                st.session_state.show_answer = True

            if st.session_state.show_answer:
                st.markdown("### Answer")
                st.markdown(f"**{current['back']}**")

                st.markdown("### How well did you know this?")
                cols = st.columns(5)
                ratings = [1, 2, 3, 4, 5]
                labels = ["Forgot", "Hard", "Good", "Easy", "Perfect"]
                for col, rating, label in zip(cols, ratings, labels):
                    if col.button(f"{rating} - {label}"):
                        st.session_state.flashcard_generator.update_card_review(current["id"], rating)
                        st.session_state.review_index = (st.session_state.review_index + 1) % len(
                            st.session_state.review_cards
                        )
                        st.session_state.show_answer = False
                        st.rerun()

            if st.button("Reset Review Session"):
                for k in ["review_index", "review_cards", "show_answer"]:
                    st.session_state.pop(k, None)
                st.rerun()

    # ──────────────────────── Tab 3 · View All Decks ────────────────────────────
    with fc_tabs[2]:
        st.subheader("Manage Flashcard Decks")
        if st.button("Load All Decks"):
            with st.spinner("Loading flashcard decks..."):
                try:
                    db_path = os.path.join(Path(__file__).parent.parent, "data", "flashcards.sqlite")
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT deck_name, COUNT(*)
                        FROM flashcards
                        GROUP BY deck_name
                        ORDER BY deck_name
                        """
                    )
                    decks = cursor.fetchall()
                    conn.close()

                    if not decks:
                        st.info("No flashcard decks found. Create some flashcards first!")
                    else:
                        for deck_name, card_count in decks:
                            with st.expander(f"{deck_name} ({card_count} cards)"):
                                col1, col2 = st.columns(2)
                                if col1.button("Export CSV", key=f"csv_{deck_name}"):
                                    csv_data = st.session_state.flashcard_generator.export_to_csv(deck_name)
                                    st.download_button(
                                        "Download CSV",
                                        data=csv_data,
                                        file_name=f"{deck_name.replace(' ', '_')}.csv",
                                        mime="text/csv",
                                    )
                                if col2.button("Export Anki", key=f"anki_{deck_name}"):
                                    apkg_b64 = st.session_state.flashcard_generator.export_to_anki(deck_name)
                                    binary = base64.b64decode(apkg_b64)
                                    st.download_button(
                                        "Download Anki Deck",
                                        data=binary,
                                        file_name=f"{deck_name.replace(' ', '_')}.apkg",
                                        mime="application/octet-stream",
                                    )
                except Exception as e:
                    st.error(f"Error loading flashcard decks: {e}")

    
    # Display usage guide
    with st.expander("Using Flashcards for Effective Learning"):
        st.markdown("""
        ## Spaced Repetition Learning
        
        Spaced repetition is a learning technique that incorporates increasing intervals of time between subsequent review of previously learned material to exploit the psychological spacing effect.
        
        ### How it works:
        
        1. When you review a card, you rate how well you knew the answer.
        2. Cards you find difficult will appear more frequently.
        3. Cards you know well will appear less frequently, with increasing intervals.
        
        ### Rating Scale:
        
        - **1 (Forgot)**: Complete blackout - you couldn't recall the information at all.
        - **2 (Hard)**: You recalled with significant difficulty or incorrectly.
        - **3 (Good)**: You recalled correctly but with some effort.
        - **4 (Easy)**: You recalled easily with minimal hesitation.
        - **5 (Perfect)**: Perfect recall with no effort.
        
        ### Tips for effective flashcard learning:
        
        - Review cards daily rather than cramming
        - Create clear, concise cards with one concept per card
        - Use both basic and cloze deletion cards for variety
        - Add tags to organize related concepts
        - Export to apps like Anki for mobile learning
        """)


st.markdown("---")
st.markdown("*SmartNote AI - Knowledge at your fingertips* 🚀")



