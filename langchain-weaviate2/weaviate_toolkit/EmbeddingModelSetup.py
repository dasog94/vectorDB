from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cpu"  # GPU 사용 시 'cuda', CPU 사용 시 'cpu'
BATCH_SIZE = 32

def setup_bge_m3_embeddings():
    """
    BGE-M3 임베딩 모델 설정

    Returns:
        HuggingFaceEmbeddings: BGE-M3 임베딩 모델
    """
    model_kwargs = {
        'device': DEVICE,  # GPU 사용 시 'cuda', CPU 사용 시 'cpu'
        'trust_remote_code': True
    }
    encode_kwargs = {
        'normalize_embeddings': True,  # 임베딩 정규화
        'batch_size': BATCH_SIZE  # 배치 크기 조정
    }

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    return embeddings
