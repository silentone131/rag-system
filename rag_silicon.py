import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # 硅基流动兼容 OpenAI 格式
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==================== 硅基流动配置 ====================
API_KEY = "sk-huxttxoesskrggnsejhmmzgsvyiuaskelaxztirwjxailyfv"  # 换成你的 API Key
BASE_URL = "https://api.siliconflow.cn/v1"

# ==================== 1. 准备文档 ====================
file_path = "C:/Users/admin/Desktop/knowledge.txt"
if not os.path.exists(file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("""
传送带故障排查指南：
1. 如果传送带突然停机，先检查急停按钮是否被按下。
2. 如果急停按钮正常，检查电源指示灯是否亮起。
3. 如果电源正常，听电机是否有嗡嗡声。有声音但不动，可能是电机卡死；没声音可能是电机烧坏。
4. 如果传送带跑偏，检查滚筒两端是否水平，调整张紧装置。
5. 每季度需要给轴承加润滑油，防止过度磨损。
""")
    print("已创建示例文档 knowledge.txt")

loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()
print(f"已加载 {len(documents)} 个文档")

# ==================== 2. 文档分块 ====================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"已切分成 {len(chunks)} 个文本块")

# ==================== 3. Embedding：用硅基流动的 API ====================
# 不需要本地模型，直接调用云端 Embedding 服务
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",                    # 硅基流动支持的 Embedding 模型[citation:10]
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL
)
vectorstore = Chroma.from_documents(chunks, embeddings)
print("向量数据库创建完成")

# ==================== 4. 检索器 ====================
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ==================== 5. LLM：用硅基流动的 DeepSeek-V3 ====================
llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",        # 硅基流动支持的模型[citation:1][citation:8]
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    temperature=0.3
)

# ==================== 6. 提示模板 ====================
prompt = ChatPromptTemplate.from_template("""
根据以下资料回答问题。如果资料里没有相关信息，就说不知道。

资料：{context}

问题：{question}

请在回答中用[1]、[2]标注信息来自哪段资料。

答案：""")

# ==================== 7. 组装 RAG 链 ====================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ==================== 8. 测试 ====================
question = "传送带该如何选择品质？"
print("\n问题:", question)
answer = rag_chain.invoke(question)
print("答案:", answer)