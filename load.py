from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    file_path = "20220202-alphabet-10k.pdf",
    extract_images = True,
    # headers = None
    # extraction_mode = "plain",
    # extraction_kwargs = None,
)