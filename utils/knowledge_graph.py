# utils/knowledge_graph.py

import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import List, Dict, Any, Tuple, Optional, Union
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
import json
import re
import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """知识图谱生成和可视化类"""
    
    def __init__(self, llm=None, temperature: float = 0):
        """
        初始化知识图谱生成器
        
        Args:
            llm: 语言模型实例，如果为None则创建默认模型
            temperature: LLM温度参数，用于控制输出的创造性
        """
        if llm is None:
            self.llm = ChatOpenAI(temperature=temperature)
        else:
            self.llm = llm
        
        self.graph = nx.DiGraph()
        # 保存关系数据以便后期分析
        self.relations_data = []
    
    def extract_entities_and_relations(self, documents: List[Document],
                                      entity_limit: int = 7,
                                      text_chunk_size: int = 2000) -> List[Dict[str, Any]]:
        """
        从文档中提取实体和关系
        
        Args:
            documents: 文档列表
            entity_limit: 每个文档最多提取的实体关系数量
            text_chunk_size: 处理的文本块大小
            
        Returns:
            包含实体和关系的字典列表
        """
        results = []
        
        for i, doc in enumerate(documents):
            logger.info(f"处理文档 {i+1}/{len(documents)}")
            
            # 分块处理长文本
            text = doc.page_content
            chunks = [text[i:i+text_chunk_size] for i in range(0, len(text), text_chunk_size)]
            
            for j, chunk in enumerate(chunks):
                logger.info(f"处理文档块 {j+1}/{len(chunks)}")
                
                # 提示模板
                prompt = f"""
                请从以下文本中提取重要概念、实体及其关系，输出为JSON格式：
                
                {chunk}
                
                请以以下格式输出（仅包含最重要的{entity_limit}个实体和关系）：
                [
                    {{
                        "source": "实体1名称",
                        "relation": "与实体2的关系",
                        "target": "实体2名称"
                    }},
                    ...
                ]
                
                仅返回JSON数组，不要有任何其他文本。
                """
                
                try:
                    response = self.llm.predict(prompt)
                    chunk_relations = self._parse_json_response(response)
                    if chunk_relations:
                        results.extend(chunk_relations)
                
                except Exception as e:
                    logger.error(f"处理文档块时出错: {str(e)}")
                    continue
        
        # 删除重复的关系
        unique_relations = []
        seen = set()
        for rel in results:
            rel_key = f"{rel['source']}|{rel['relation']}|{rel['target']}"
            if rel_key not in seen:
                unique_relations.append(rel)
                seen.add(rel_key)
        
        self.relations_data = unique_relations
        return unique_relations
    
    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """改进的JSON响应解析方法"""
        try:
            # 首先尝试直接解析整个响应
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试提取JSON部分
            json_match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if json_match:
                try:
                    json_str = f"[{json_match.group(1)}]"
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
                
            # 更宽松的模式：寻找可能包含整个JSON数组的部分
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
        
        logger.warning("无法解析JSON响应")
        return []
    
    def build_graph(self, relations: Optional[List[Dict[str, Any]]] = None) -> nx.DiGraph:
        """
        从关系列表构建图
        
        Args:
            relations: 关系字典列表，每个字典包含source, relation, target
                       如果为None，则使用之前提取的关系
            
        Returns:
            构建的有向图
        """
        if relations is None:
            relations = self.relations_data
            
        if not relations:
            logger.warning("没有关系数据可用于构建图谱")
            return self.graph
            
        G = nx.DiGraph()
        
        for rel in relations:
            source = rel.get("source", "").strip()
            target = rel.get("target", "").strip()
            relation = rel.get("relation", "").strip()
            
            if source and target:
                # 添加节点属性
                if not G.has_node(source):
                    G.add_node(source, type="entity")
                if not G.has_node(target):
                    G.add_node(target, type="entity")
                    
                # 添加边和关系属性
                G.add_edge(source, target, label=relation, weight=1.0)
                
                # 如果相同的边已存在，增加权重
                if G.has_edge(source, target):
                    current_weight = G[source][target].get('weight', 1.0)
                    G[source][target]['weight'] = current_weight + 0.5
                    
        self.graph = G
        return G
    
    def visualize_plotly(self, graph: Optional[nx.DiGraph] = None,
                         show_labels: bool = True,
                         color_theme: str = 'YlGnBu',
                         title: str = 'SmartNote AI 知识图谱') -> go.Figure:
        """
        使用Plotly生成交互式图形可视化
        
        Args:
            graph: 要可视化的图，如果为None则使用当前图
            show_labels: 是否显示关系标签
            color_theme: 颜色主题
            title: 图表标题
            
        Returns:
            Plotly图形对象
        """
        if graph is None:
            graph = self.graph
        
        if not graph.nodes():
            raise ValueError("图为空，无法可视化")
        
        # 使用spring布局计算节点位置
        pos = nx.spring_layout(graph, seed=42, k=0.5)
        
        # 边的追踪
        edge_x = []
        edge_y = []
        edge_text = []
        
        # 边标签位置
        middle_x = []
        middle_y = []
        
        for edge in graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # 计算边的中点用于标签
            middle_x.append((x0 + x1) / 2)
            middle_y.append((y0 + y1) / 2)
            
            # 关系文本
            relation = edge[2].get('label', "")
            edge_text.append(relation)
        
        # 创建边的轨迹
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='text',
            mode='lines',
            text=edge_text
        )
        
        # 创建边标签轨迹
        if show_labels and edge_text:
            edge_label_trace = go.Scatter(
                x=middle_x,
                y=middle_y,
                mode='text',
                text=edge_text,
                textposition="middle center",
                textfont=dict(
                    size=10,
                    color='#555'
                ),
                hoverinfo='none'
            )
        
        # 节点的追踪
        node_x = []
        node_y = []
        node_text = []
        
        for node in graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        
        # 根据连接数着色
        node_adjacencies = []
        for node in graph.nodes():
            node_adjacencies.append(len(list(graph.neighbors(node))))
        
        # 节点大小基于度中心性
        degree_centrality = nx.degree_centrality(graph)
        node_sizes = [20 + 30 * degree_centrality[node] for node in graph.nodes()]
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=True,
                colorscale=color_theme,
                size=node_sizes,
                color=node_adjacencies,
                colorbar=dict(
                    thickness=15,
                    title='节点连接数',
                    xanchor='left'
                ),
                line_width=2
            )
        )
        
        # 创建网络图
        fig_data = [edge_trace, node_trace]
        if show_labels and edge_text:
            fig_data.append(edge_label_trace)
            
        fig = go.Figure(
            data=fig_data,
            layout=go.Layout(
                title=dict(
                    text=title,
                    font=dict(size=16)
                ),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                template='plotly_white'
            )
        )
        
        return fig
    
    def visualize_matplotlib(self, graph: Optional[nx.DiGraph] = None,
                           figsize: Tuple[int, int] = (12, 10),
                           show_labels: bool = True) -> plt.Figure:
        """
        使用Matplotlib生成静态图形可视化
        
        Args:
            graph: 要可视化的图，如果为None则使用当前图
            figsize: 图形尺寸
            show_labels: 是否显示关系标签
            
        Returns:
            Matplotlib图形对象
        """
        if graph is None:
            graph = self.graph
            
        if not graph.nodes():
            raise ValueError("图为空，无法可视化")
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 计算节点位置
        pos = nx.spring_layout(graph, seed=42)
        
        # 计算节点大小和颜色
        degree_centrality = nx.degree_centrality(graph)
        node_sizes = [300 + 1000 * degree_centrality[node] for node in graph.nodes()]
        node_colors = list(degree_centrality.values())
        
        # 绘制节点
        nx.draw_networkx_nodes(
            graph, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.YlGnBu,
            alpha=0.8,
            ax=ax
        )
        
        # 绘制边
        nx.draw_networkx_edges(
            graph, pos,
            width=1.0,
            alpha=0.5,
            edge_color='gray',
            ax=ax
        )
        
        # 节点标签
        nx.draw_networkx_labels(
            graph, pos,
            font_size=10,
            font_weight='bold',
            ax=ax
        )
        
        # 边标签
        if show_labels:
            edge_labels = nx.get_edge_attributes(graph, 'label')
            nx.draw_networkx_edge_labels(
                graph, pos,
                edge_labels=edge_labels,
                font_size=8,
                ax=ax
            )
        
        plt.axis('off')
        plt.tight_layout()
        
        return fig
    
    def get_central_concepts(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        获取图中最中心的概念
        
        Args:
            top_n: 返回的概念数量
            
        Returns:
            按重要性排序的概念和分数元组列表
        """
        if not self.graph.nodes():
            return []
        
        # 计算多种节点中心性指标
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        degree_centrality = nx.degree_centrality(self.graph)
        
        try:
            eigenvector_centrality = nx.eigenvector_centrality(self.graph)
        except:
            eigenvector_centrality = {}
        
        # 结合不同的中心性指标
        combined_scores = {}
        for node in self.graph.nodes():
            combined_scores[node] = (
                betweenness_centrality.get(node, 0) * 0.4 +
                degree_centrality.get(node, 0) * 0.4 +
                eigenvector_centrality.get(node, 0) * 0.2
            )
        
        # 返回得分最高的节点
        return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def save_graph(self, filename: str, format: str = 'graphml'):
        """
        保存图谱到文件
        
        Args:
            filename: 文件名
            format: 文件格式，可选 'graphml', 'gexf', 'adjlist', 'edgelist'
        """
        if not self.graph.nodes():
            raise ValueError("图为空，无法保存")
            
        # 确保扩展名与格式匹配
        base, ext = os.path.splitext(filename)
        if not ext:
            filename = f"{filename}.{format}"
            
        try:
            if format == 'graphml':
                nx.write_graphml(self.graph, filename)
            elif format == 'gexf':
                nx.write_gexf(self.graph, filename)
            elif format == 'adjlist':
                nx.write_adjlist(self.graph, filename)
            elif format == 'edgelist':
                nx.write_edgelist(self.graph, filename)
            else:
                raise ValueError(f"不支持的文件格式: {format}")
                
            logger.info(f"图谱已保存至: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"保存图谱时出错: {str(e)}")
            return False
    
    def load_graph(self, filename: str) -> bool:
        """
        从文件加载图谱
        
        Args:
            filename: 文件名
            
        Returns:
            成功加载返回True，否则返回False
        """
        if not os.path.exists(filename):
            logger.error(f"文件不存在: {filename}")
            return False
            
        try:
            _, ext = os.path.splitext(filename)
            ext = ext.lower()[1:]  # 移除点并转为小写
            
            if ext == 'graphml':
                self.graph = nx.read_graphml(filename)
            elif ext == 'gexf':
                self.graph = nx.read_gexf(filename)
            elif ext == 'adjlist':
                self.graph = nx.read_adjlist(filename)
            elif ext == 'edgelist':
                self.graph = nx.read_edgelist(filename)
            else:
                raise ValueError(f"不支持的文件格式: {ext}")
                
            logger.info(f"已从 {filename} 加载图谱")
            return True
            
        except Exception as e:
            logger.error(f"加载图谱时出错: {str(e)}")
            return False
    
    def query_relations(self, entity: str, max_depth: int = 2) -> Dict[str, Any]:
        """
        查询与指定实体相关的关系
        
        Args:
            entity: 要查询的实体名称
            max_depth: 搜索的最大深度
            
        Returns:
            包含相关实体和关系的字典
        """
        if not self.graph.nodes():
            return {"entity": entity, "exists": False, "relations": []}
            
        if entity not in self.graph.nodes():
            return {"entity": entity, "exists": False, "relations": []}
            
        # BFS搜索相关实体
        relations = []
        visited = set([entity])
        queue = [(entity, 0)]  # (节点, 深度)
        
        while queue:
            current, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
                
            # 获取出边关系
            for successor in self.graph.successors(current):
                if successor not in visited:
                    rel = {
                        "source": current,
                        "relation": self.graph[current][successor].get("label", ""),
                        "target": successor,
                        "direction": "outgoing"
                    }
                    relations.append(rel)
                    visited.add(successor)
                    queue.append((successor, depth + 1))
            
            # 获取入边关系
            for predecessor in self.graph.predecessors(current):
                if predecessor not in visited:
                    rel = {
                        "source": predecessor,
                        "relation": self.graph[predecessor][current].get("label", ""),
                        "target": current,
                        "direction": "incoming"
                    }
                    relations.append(rel)
                    visited.add(predecessor)
                    queue.append((predecessor, depth + 1))
        
        return {
            "entity": entity,
            "exists": True,
            "relations": relations
        }
    
    def find_paths(self, source: str, target: str, cutoff: int = 3) -> List[List[str]]:
        """
        查找从源实体到目标实体的所有路径
        
        Args:
            source: 源实体
            target: 目标实体
            cutoff: 最大路径长度
            
        Returns:
            路径列表，每个路径是节点列表
        """
        if not self.graph.nodes():
            return []
            
        if source not in self.graph.nodes() or target not in self.graph.nodes():
            return []
            
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=cutoff))
            return paths
        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            logger.error(f"查找路径时出错: {str(e)}")
            return []
    
    def summarize_graph(self) -> Dict[str, Any]:
        """
        生成图谱摘要统计信息
        
        Returns:
            包含图谱统计信息的字典
        """
        if not self.graph.nodes():
            return {
                "nodes": 0,
                "edges": 0,
                "density": 0,
                "avg_clustering": 0,
                "top_entities": []
            }
            
        # 基本统计
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()
        density = nx.density(self.graph)
        
        # 计算平均聚类系数
        try:
            avg_clustering = nx.average_clustering(self.graph)
        except:
            avg_clustering = 0
            
        # 获取最中心的实体
        top_entities = self.get_central_concepts(5)
        
        return {
            "nodes": num_nodes,
            "edges": num_edges,
            "density": density,
            "avg_clustering": avg_clustering,
            "top_entities": [{"entity": e[0], "centrality": e[1]} for e in top_entities]
        }
    
    def extract_communities(self) -> Dict[str, List[str]]:
        """
        从图中提取社区/聚类
        
        Returns:
            社区ID到节点列表的映射
        """
        if not self.graph.nodes() or self.graph.number_of_nodes() < 3:
            return {}
            
        try:
            # 转换为无向图以用于社区检测
            undirected_graph = self.graph.to_undirected()
            
            # 使用Louvain算法检测社区
            from community import best_partition
            partition = best_partition(undirected_graph)
            
            # 组织结果
            communities = {}
            for node, community_id in partition.items():
                community_id = str(community_id)
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node)
                
            return communities
            
        except ImportError:
            # 如果没有安装python-louvain，则使用networkx的算法
            try:
                # 使用Girvan-Newman算法
                from networkx.algorithms import community
                comp = community.girvan_newman(undirected_graph)
                
                # 获取第一个划分结果
                communities_iter = next(comp)
                communities = {}
                
                for i, comm in enumerate(communities_iter):
                    communities[str(i)] = list(comm)
                    
                return communities
                
            except Exception as e:
                logger.error(f"提取社区时出错: {str(e)}")
                return {}
                
        except Exception as e:
            logger.error(f"提取社区时出错: {str(e)}")
            return {}
