import chromadb
from jinja2.lexer import newline_re
from sympy import false


class ChromaDatabaseRepository:

    def __init__(self):
        self.chroma_client = chromadb.Client()

    """ 創建collection資訊 """
    def create_collection(self , collection_name):
        new_collect = self.chroma_client.get_or_create_collection(
            name=f"{collection_name}",                          # 新增該collection
        )
        return new_collect

    """ 新增collection資訊 """
    def add_collection_data(self , collection_name , documents , metadatas , ids):
        try:
            the_collection = self.chroma_client.get_collection( # 找到該collection
                name=f"{collection_name}",
            )
            the_collection.add(                                 # 插入資料[單筆,陣列]
                documents = documents ,
                metadatas = metadatas ,
                ids       = ids ,
            )
            return True
        except Exception as e:
            print(f"新增collection錯誤：str({e})")
            return False

    """ 查詢collection結果 """
    def query_collection(self , collection_name , question):
        the_collection = self.chroma_client.get_collection(     # 找到該collection
            name=f"{collection_name}",
        )
        results = the_collection.query(                         # 下query 向量給結果
            query_texts = [f"{question}"] ,
            n_results=1
        )
        return results
        # print("IDs:", results["ids"])                         ＃ result 資料結構
        # print("Documents:", results["documents"])
        # print("Metadatas:", results["metadatas"])
        # print("Distances:", results["distances"])
        # print("Embeddings:", results["embeddings"])