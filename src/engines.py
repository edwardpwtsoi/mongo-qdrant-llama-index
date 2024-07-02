import os

from dotenv import load_dotenv
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.postprocessor import EmbeddingRecencyPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.bedrock import Bedrock
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import qdrant_client


load_dotenv()


class BedRockHaiKuMixin:
    llm = Bedrock(
       model=os.environ['BEDROCK_MODEL_NAME'],
       region_name=os.environ['BEDROCK_REGION_NAME'],
       context_size=int(os.environ['BEDROCK_CONTEXT_SIZE'])
    )


class AzureOpenAIHuggingFaceQueryEngine:
    llm = AzureOpenAI(
        model=os.environ['AZURE_OPENAI_MODEL_NAME'],
        deployment_name=os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'],
        api_key=os.environ['AZURE_OPENAI_API_KEY'],
        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_version=os.environ['AZURE_OPENAI_API_VERSION'],
    )
    embed_model = HuggingFaceEmbedding(model_name=os.environ['HUGGINGFACE_EMBEDDING_MODEL'])

    def __init__(self, persist_dir="storage/default", *, storage_context=None):
        self.persist_dir = persist_dir
        # create the pipeline with transformations
        self.pipeline = self.build_ingestion_pipeline()
        # storage context & index
        self.storage_context = self.load_storage_context(persist_dir) if storage_context is None else storage_context
        self.index = self.build_index()
        self.retriever = self.build_retriever()
        self.query_engine = self.build_query_engine()

    @staticmethod
    def load_storage_context(persist_dir):
        if os.path.exists(persist_dir):  # when there is storage
            storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore.from_persist_dir(persist_dir=persist_dir),
                vector_store=SimpleVectorStore.from_persist_dir(persist_dir=persist_dir, namespace="default"),
                index_store=SimpleIndexStore.from_persist_dir(persist_dir=persist_dir),
            )
            return storage_context
        else:
            return StorageContext.from_defaults()

    @classmethod
    def build_ingestion_pipeline(cls):
        return IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=64),
                # TitleExtractor(llm=llm),
                cls.embed_model,
            ]
        )

    @classmethod
    def from_documents(cls, documents, persist_dir):
        storage_context = cls.load_storage_context(persist_dir)
        ingestion_pipeline = cls.build_ingestion_pipeline()
        nodes = ingestion_pipeline.run(documents=documents, show_progress=True)
        VectorStoreIndex(nodes, embed_model=cls.embed_model, storage_context=storage_context, show_progress=True)
        return cls(persist_dir=persist_dir, storage_context=storage_context)

    def build_index(self):
        index = load_index_from_storage(self.storage_context, embed_model=self.embed_model)
        return index

    def build_query_engine(self):
        return RetrieverQueryEngine.from_args(self.retriever, llm=self.llm)

    def build_retriever(self):
        return self.index.as_retriever(similarity_top_k=6)

    def write_to_disk(self):
        self.storage_context.persist(persist_dir=f"{self.persist_dir}")

    def add_documents(self, documents):
        nodes = self.pipeline.run(documents=documents, show_progress=True)
        self.index.insert_nodes(nodes)

    def __call__(self, query, use_async=False):
        if use_async:
            return self.query_engine.aquery(query)
        return self.query_engine.query(query)


class AzureOpenAIHuggingFaceAutoMergingQueryEngine(AzureOpenAIHuggingFaceQueryEngine):
    def build_retriever(self):
        _base_retriever = self.index.as_retriever(similarity_top_k=6)
        return AutoMergingRetriever(_base_retriever, self.storage_context, verbose=True)

    @classmethod
    def build_ingestion_pipeline(cls):
        return IngestionPipeline(
            transformations=[
                HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 1024, 512], chunk_overlap=32),
                cls.embed_model,
            ]
        )

    @classmethod
    def from_documents(cls, documents, persist_dir):
        storage_context = cls.load_storage_context(persist_dir)
        ingestion_pipeline = cls.build_ingestion_pipeline()
        nodes = ingestion_pipeline.run(documents=documents, show_progress=True)
        storage_context.docstore.add_documents(nodes)  # manually adding since index will only get the leaf nodes
        leaf_nodes = get_leaf_nodes(nodes)
        VectorStoreIndex(leaf_nodes, embed_model=cls.embed_model,
                         storage_context=storage_context, show_progress=True)
        return cls(persist_dir=persist_dir, storage_context=storage_context)

    def add_documents(self, documents):
        nodes = self.pipeline.run(documents=documents, show_progress=True)
        self.storage_context.docstore.add_documents(nodes)
        leaf_nodes = get_leaf_nodes(nodes)
        self.index.insert_nodes(leaf_nodes)


class AzureOpenAIHuggingFaceAutoMergingRecencyQueryEngine(AzureOpenAIHuggingFaceAutoMergingQueryEngine):
    def build_query_engine(self):
        return RetrieverQueryEngine.from_args(
            self.retriever, llm=self.llm, node_postprocessors=[
                EmbeddingRecencyPostprocessor(
                    embed_model=self.embed_model, date_key="date", similarity_cutoff=0.85
                )
            ]
        )


class MongoQdrantStorageContextMixin:
    @staticmethod
    def load_storage_context(persist_dir, host="localhost"):
        mongo_uri = f"mongodb://{os.environ['MONGODB_USERNAME']}:{os.environ['MONGODB_PASSWORD']}@{host}:27017/"
        index_store = MongoIndexStore.from_uri(mongo_uri, db_name=f"{persist_dir}-index_store")
        doc_store = MongoDocumentStore.from_uri(mongo_uri, db_name=f"{persist_dir}-docstore")
        client = qdrant_client.QdrantClient(
            # you can use :memory: mode for fast and light-weight experiments,
            # it does not require to have Qdrant deployed anywhere
            # but requires qdrant-client >= 1.1.1
            # location=":memory:"
            # otherwise set Qdrant instance address with:
            # url="http://<host>:<port>"
            # otherwise set Qdrant instance with host and port:
            host=host,
            port=6333
            # set API KEY for Qdrant Cloud
            # api_key="<qdrant-api-key>",
        )
        vector_store = QdrantVectorStore(client=client, collection_name=persist_dir)
        storage_context = StorageContext.from_defaults(
            index_store=index_store, vector_store=vector_store, docstore=doc_store
        )
        return storage_context


class AzureOpenAIHuggingFaceQueryEngineMongoQdrant(MongoQdrantStorageContextMixin, AzureOpenAIHuggingFaceQueryEngine):
    pass


class AzureOpenAIHuggingFaceAutoMergingRecencyQueryEngineMongoQdrant(
    MongoQdrantStorageContextMixin, AzureOpenAIHuggingFaceAutoMergingRecencyQueryEngine
):
    pass
