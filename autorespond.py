import os
import time
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
load_dotenv()

# === CONFIG ===
PDF_PATH = "autopdf.pdf"
VIDEO_ID = "D-IxSUxvoJQ" #https://youtu.be/D-IxSUxvoJQ
CLIENT_SECRET_FILE = "client_secret.json"  # Download from Google Cloud Console
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

# === STEP 1: AUTHENTICATE YOUTUBE ===
def get_authenticated_youtube_service():
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
    creds = flow.run_local_server(port=0)
    return build("youtube", "v3", credentials=creds)

# === STEP 2: LOAD PDF & BUILD VECTORSTORE ===
def build_vectorstore_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    return vectorstore

# === STEP 3: BUILD RAG AGENT WITH DEEPSEEK ===
def build_rag_agent(vectorstore):
    llm = ChatDeepSeek(
        model="deepseek-chat",  # or another DeepSeek model if desired
        temperature=0.8,
        max_tokens=50,
        # api_key can be set via env var DEEPSEEK_API_KEY or passed here
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )
    return qa_chain

# === STEP 4: FETCH YOUTUBE COMMENTS ===
def fetch_comments(youtube, video_id):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=20
    )
    response = request.execute()

    for item in response["items"]:
        # Only include comments that have never been replied to
        if item["snippet"].get("totalReplyCount", 0) == 0:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comment_id = item["snippet"]["topLevelComment"]["id"]
            comments.append((comment_id, comment))
    return comments

# === STEP 5: POST REPLY TO COMMENT ===
def post_reply(youtube, comment_id, reply_text):
    youtube.comments().insert(
        part="snippet",
        body={
            "snippet": {
                "parentId": comment_id,
                "textOriginal": reply_text
            }
        }
    ).execute()

# === STEP 6: MAIN AUTO-REPLY FUNCTION ===
def auto_reply_to_comments():
    print("Authenticating with YouTube...")
    youtube = get_authenticated_youtube_service()

    print("Loading knowledge from PDF...")
    vectorstore = build_vectorstore_from_pdf(PDF_PATH)

    print("Initializing DeepSeek-powered RAG agent...")
    agent = build_rag_agent(vectorstore)

    print("Fetching YouTube comments...")
    comments = fetch_comments(youtube, VIDEO_ID)

    for comment_id, comment_text in comments:
        try:
            print(f"\n🗣 New comment: {comment_text}")
            reply = agent.run(comment_text)
            print(f"🤖 Generated reply: {reply}")

            post_reply(youtube, comment_id, reply)
            print("✅ Reply posted.")
            time.sleep(2)  # Avoid spam detection
        except Exception as e:
            print(f"❌ Error processing comment: {e}")

# === ENTRY POINT ===u
if __name__ == "__main__":
    auto_reply_to_comments()
