from langchain_community.vectorstores import faiss
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_extraction_chain
import pprint

from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

schema = {
    "properties": {
        "recipe ingredients": {"type": "string"},
        "steps": {"type": "string"},
    },
    "required": ["recipe ingredients", "steps"],
}


def extract(content: str, schema: dict):
    return create_extraction_chain(schema=schema, llm=llm).run(content)


def get_recipe():
    """returns a recipe from a site"""
    loader = AsyncChromiumLoader(
        ["https://www.allrecipes.com/crusted-chicken-romano-recipe-8414173"]
    )
    recipe = loader.load()

    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        recipe, tags_to_extract=["body"]
    )

    print("extracting content...")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=4000, chunk_overlap=0
    )
    splits = splitter.split_documents(docs_transformed)

    vector = faiss.FAISS.from_documents(splits, OpenAIEmbeddings())
    docs = vector.similarity_search("Get the recipe", k=2)
    res = llm(docs)
    pprint.pprint(res)
    # extracted_content = extract(schema=schema, content=splits[0].page_content)
    # pprint.pprint(extracted_content)
