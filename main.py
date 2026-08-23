import os
from git import  Repo
from rich.console import Console
from langchain_community.document_loaders.generic import  GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_text_splitters import  Language
from langchain_community.vectorstores import  Chroma
from transformers import  pipeline
from langchain_huggingface import  HuggingFaceEmbeddings
from langchain_community.chains import  PebbloRetrievalQA
from langchain_classic.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from transformers.testing_utils import Expectations
from dotenv import  load_dotenv

load_dotenv()

repo_url = "https://github.com/scikit-learn/scikit-learn/"
console = Console()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#console.print(f"openai api key:-{OPENAI_API_KEY}")
def get_repo_name(url:str)->str:
    if len(url)==0:
        console.print("[yellow] URL is empty.Please enter correct url [/yellow]")
    url = url.rstrip("/")
    url= url.split("/")
    repo_name = url[-1]
    return repo_name if len(repo_name) !=0 else "Unknow_repo"

file_path = get_repo_name(repo_url)

def make_folder(name:str)->None:

    if os.path.isdir(name):
        os.removedirs(name)

    os.mkdir(name)

#make_folder(file_path)

def clone_repo(url:str,name:str)->None:
    try:
        console.print(f"[yellow] Cloning the repo:-{name}.[/yellow]")
        Repo.clone_from(url=url,to_path=name)
        console.print(f"[green] Successfully Cloned Repository.[/green]")

    except Expectations as e:
        console.print(f"[red] Error while cloning repo:{e}[/red]")

#clone_repo(repo_url,file_path)

def load_documents(path:str):
    try:
        console.print(f"[yellow] Loading And Parsing Files from:-{path},and total file in this path:-{len(os.listdir(path))} [/yellow]")
        loader = GenericLoader.from_filesystem(
            path=path,
            suffixes=['.py'],
            parser=LanguageParser(language="python"),
            glob="**/*")
        documents = loader.load()
        console.print(f"[yellow] Files Loaded:-{len(documents)}.[/yellow]")
        return documents
    except Expectations as e:
        console.print(f"[red] Error while loading files:-{e}[/red]")

file_documents = load_documents(path= file_path + "/examples")
#console.print(f"[yellow] First document:-{file_documents[0]} [/yellow]")

def convert_in_chunks(docs):
    try:
        console.print(f"converting the docs in chunks.")
        code_text_splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language.PYTHON,
                chunk_size = 3000,
                chunk_overlap= 300
        )
        code_chunks = code_text_splitter.split_documents(docs)
        console.print(f"[yellow] Total Code chunks:-{len(code_chunks)}[/yellow]")
        return  code_chunks
    except Expectations as e:
        console.print(f"Error while converting in chunks:-{e}")

total_code_chunks = convert_in_chunks(file_documents)
#console.print(f"First code chunk:-{total_code_chunks[0]}")


def convert_in_embedding(chunks,db_path):
    try:
        console.print(f"Loading Embedding Model.")
        embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        console.print(f"Embedding model loaded successfully.")
    except Exception as e:
        console.print(f"Error while loading embedding model:-{e}")

    chroma_db = Chroma.from_documents(
        documents= chunks,
        embedding=embedding_model,
        persist_directory=db_path
    )
    chroma_db.persist()

    return chroma_db

chroma = convert_in_embedding(total_code_chunks,"./chroma_db")
#console.print(f"Chroma_db created.{chroma}")

llm = pipeline("text-generation")

def ask(chroma_db,model):
    try:
        try:
            retriever = chroma.as_retriever()
            agent = PebbloRetrievalQA.from_chain_type(
                llm=model,
            chain_type="stuff",
            retriever = retriever)
            return  agent

        except Exception as e:
            console.print(f"Error while retrieving :-{e}")


    except Exception as e:
        console.print(f"Error :-{e}")

agent_a = ask(chroma,llm)
console.print(f"Agent Created Successfully.")

response = agent_a.run({"query":"code of decision tree"})
console.print(f"[yellow] Agent Response... [/yellow]")
console.print(f"{response}")









