import json

from llama_index.core.schema import Document


class GoogleNewsSearchResult:
    @staticmethod
    def from_query_result_json(json_file):
        data = json.load(open(json_file, "r"))
        documents = [Document(
            text=new["content"],
            metadata={
                "title": new["title"],
                "date": new["date"],
                "source": new["source"],
            },
            excluded_embed_metadata_keys=["date"]
        ) for new in data["news"] if new["content"] is not None]
        return documents
