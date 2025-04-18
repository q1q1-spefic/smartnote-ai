import os
from typing import List, Optional

# Use a single try/except block to handle all imports
try:
    # Try newer LangChain structure first
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        UnstructuredMarkdownLoader,
        WebBaseLoader,
    )
    from langchain_core.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
except ModuleNotFoundError:
    # Fall back to older LangChain structure
    from langchain.document_loaders import (
        PyPDFLoader,
        TextLoader,
        UnstructuredMarkdownLoader,
        WebBaseLoader,
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document

class DocumentProcessor:
    """处理不同类型文档的类"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """根据文件类型加载文档"""
        _, file_extension = os.path.splitext(file_path)
        
        if file_extension.lower() == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_extension.lower() == '.txt':
            loader = TextLoader(file_path)
        elif file_extension.lower() in ['.md', '.markdown']:
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"不支持的文件类型: {file_extension}")
        
        return loader.load()
    
    def load_web_page(self, url: str) -> List[Document]:
        """加载网页内容"""
        loader = WebBaseLoader(url)
        return loader.load()
    
    def process_documents(self, documents: List[Document], metadata: Optional[dict] = None) -> List[Document]:
        """处理文档：分割文本并添加元数据"""
        chunks = self.text_splitter.split_documents(documents)
        
        # 如果提供了元数据，添加到每个文档块
        if metadata:
            for chunk in chunks:
                chunk.metadata.update(metadata)
                
        return chunks
