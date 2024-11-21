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

# 初始化Ollama模型
llm = Ollama(model='llama3.2')

# 建立文件列表，每個文件包含一段文字內容
docs = [
    Document(page_content='曼德珍珠奶茶草：這種植物具有強大的魔法屬性，常用於恢復被石化的受害者。'),
    Document(page_content='山羊可愛蓮花石 ：是一種從山羊胃中取出的石頭，可以解百毒。在緊急情況下，它被認為是最有效的解毒劑。'),
    Document(page_content='日本小可愛佐籐鱗片：這些鱗片具有強大的治愈能力，常用於製作治療藥水，特別是用於治療深層傷口。'),
]