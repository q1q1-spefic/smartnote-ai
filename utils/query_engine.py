# utils/query_engine.py

from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

class QueryEngine:
    """查询引擎类，将用户问题与检索到的相关内容结合生成回答"""
    
    def __init__(self, 
               retriever, 
               model_name: str = "gpt-3.5-turbo", 
               api_key: Optional[str] = None,
               temperature: float = 0.3):
        """
        初始化查询引擎
        
        Args:
            retriever: 向量数据库检索器
            model_name: 使用的模型名称
            api_key: OpenAI API密钥
            temperature: 模型温度参数
        """
        import os
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model_name=model_name, 
            temperature=temperature
        )
        
        # 定义更好的提示模板
        template = """你是一个智能学习助手，可以根据用户的笔记和资料回答问题。
        
        请根据以下检索到的相关内容，回答用户的问题：
        
        ### 相关内容:
        {context}
        
        ### 用户问题:
        {question}
        
        回答要求：
        1. 请直接回答问题，不要重复问题
        2. 回答应该准确、清晰且与检索到的内容相关
        3. 如果检索到的内容不足以回答问题，请坦率地承认不知道
        4. 引用具体来源时，请标注出处
        5. 如果可以，回答中应包含相关的例子或图解
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # 创建检索QA链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # stuff方法将所有文档内容合并
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        处理用户问题并返回回答
        
        Args:
            question: 用户问题
            
        Returns:
            包含回答和来源文档的字典
        """
        result = self.qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
        
    def generate_summary(self, documents: List[Document]) -> str:
        """
        为一组文档生成摘要
        
        Args:
            documents: 文档列表
            
        Returns:
            摘要文本
        """
        text_content = "\n\n".join([doc.page_content for doc in documents])
        
        summary_prompt = f"""请为以下内容生成一个简洁但全面的摘要：
        
        {text_content}
        
        摘要应该：
        1. 包含主要概念和关键点
        2. 结构清晰，易于理解
        3. 不超过300字
        """
        
        response = self.llm.predict(summary_prompt)
        return response
        
    def extract_key_points(self, documents: List[Document]) -> List[str]:
        """
        从文档中提取关键点
        
        Args:
            documents: 文档列表
            
        Returns:
            关键点列表
        """
        text_content = "\n\n".join([doc.page_content for doc in documents])
        
        extraction_prompt = f"""请从以下内容中提取5-7个关键点：
        
        {text_content}
        
        每个关键点应该：
        1. 简洁明了
        2. 代表一个重要概念或事实
        3. 独立成行
        """
        
        response = self.llm.predict(extraction_prompt)
        
        # 处理响应，按行分割并移除可能的数字前缀
        key_points = []
        for line in response.strip().split('\n'):
            # 移除可能的数字前缀和特殊字符
            clean_line = line.strip()
            if clean_line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '•', '-', '*')):
                clean_line = clean_line[2:].strip()
            if clean_line:
                key_points.append(clean_line)
                
        return key_points