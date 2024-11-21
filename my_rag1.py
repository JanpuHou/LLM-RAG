from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


from langchain.text_splitter import CharacterTextSplitter

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

import faiss

# 初始化Ollama模型
llm = Ollama(model='llama3.2')

# 建立文件列表，每個文件包含一段文字內容
docs = [
    Document(page_content='下列各款不予發明專利：一、飲食品及嗜好品。但其製造方法不在此限。二、動、植物及微生物新品種。但植物新品種及微生物新菌種育成方法不在此限。三、人體或動物疾病之診斷、治療或手術方法。四、科學原理或數學方法。五、遊戲及運動之規則或方法。六、其他必須藉助於人類推理力、記憶力始能實施之方法或計畫。七、物品新用途之發現。但化學品及醫藥品不在此限。發明妨害公共秩序、善良風俗或衛生者，或發明品之使用違反法律者，不予專利。'),
    Document(page_content='外國人所屬之國家與中華民國如未共同參加保護專利之國際條約或無相互保護專利之條約、協定或由團體、機構互訂經主管機關核准保護專利之協議，或對中華民國國民申請專利，不予受理者，其專利申請，得不予受理。'),
    Document(page_content='申請專利之發明，經審查確定後，給予專利權，並發證書。專利權期間為十五年，自公告之日起算。但自申請之日起不得逾十八年。'),
]

# 設定文本分割器，chunk_size是分割的大小，chunk_overlap是重疊的部分
text_splitter = CharacterTextSplitter(chunk_size=20, chunk_overlap=5)
documents = text_splitter.split_documents(docs)  # 將文件分割成更小的部分

# 初始化嵌入模型
embeddings = OllamaEmbeddings()

# 使用FAISS建立向量資料庫
vectordb = FAISS.from_documents(docs, embeddings)
# 將向量資料庫設為檢索器
retriever = vectordb.as_retriever()

# 設定提示模板，將系統和使用者的提示組合
prompt = ChatPromptTemplate.from_messages([
    ('system', 'Answer the user\'s questions in Chinese, based on the context provided below:\n\n{context}'),
    ('user', 'Question: {input}'),
])

# 創建文件鏈，將llm和提示模板結合
document_chain = create_stuff_documents_chain(llm, prompt)

# 創建檢索鏈，將檢索器和文件鏈結合
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# context = []
input_text = input('>>> ')
while input_text.lower() != 'bye':
    response = retrieval_chain.invoke({
        'input': input_text,
        # 'context': context
    })
    print(response['answer'])
    # context = response['context']
    input_text = input('>>> ')