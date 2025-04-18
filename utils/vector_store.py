import os
from typing import List, Optional, Dict, Any
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS, Chroma

class VectorStore:
    """向量存储类，支持FAISS和Chroma"""
    
    def __init__(self, embedding_type: str = "openai", api_key: Optional[str] = None):
        """
        初始化向量存储
        
        Args:
            embedding_type: 使用的嵌入模型类型，"openai"或"huggingface"
            api_key: OpenAI API密钥（如果使用OpenAI嵌入）
        """
        self.embedding_type = embedding_type
        
        if embedding_type == "openai":
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            self.embedding_model = OpenAIEmbeddings()
        elif embedding_type == "huggingface":
            # 使用sentence-transformers中的多语言模型
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"不支持的嵌入类型: {embedding_type}")
            
        self.vector_db = None
        
    def create_vector_store(self, documents: List[Document],
                          store_type: str = "faiss",
                          persist_directory: Optional[str] = None) -> Any:
        """
        从文档创建向量存储
        
        Args:
            documents: 文档列表
            store_type: 向量存储类型，"faiss"或"chroma"
            persist_directory: 持久化目录路径
            
        Returns:
            向量存储实例
        """
        if store_type == "faiss":
            self.vector_db = FAISS.from_documents(documents, self.embedding_model)
            if persist_directory:
                self.vector_db.save_local(persist_directory)
        elif store_type == "chroma":
            self.vector_db = Chroma.from_documents(
                documents,
                self.embedding_model,
                persist_directory=persist_directory
            )
            if persist_directory:
                self.vector_db.persist()
        else:
            raise ValueError(f"不支持的向量存储类型: {store_type}")
            
        return self.vector_db
    
    def load_vector_store(self, store_type: str = "faiss",
                        persist_directory: str = "./vector_db") -> Any:
        """
        加载现有的向量存储
        
        Args:
            store_type: 向量存储类型，"faiss"或"chroma"
            persist_directory: 持久化目录路径
            
        Returns:
            向量存储实例
        """
        if store_type == "faiss":
            self.vector_db = FAISS.load_local(persist_directory, self.embedding_model,
             allow_dangerous_deserialization=True)
        elif store_type == "chroma":
            self.vector_db = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embedding_model
            )
        else:
            raise ValueError(f"不支持的向量存储类型: {store_type}")
            
        return self.vector_db
    
    def get_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """获取检索器"""
        if self.vector_db is None:
            raise ValueError("向量数据库未初始化")
            
        if search_kwargs is None:
            search_kwargs = {"k": 4}  # 默认检索4个最相关的文档块
            
        return self.vector_db.as_retriever(search_kwargs=search_kwargs)
