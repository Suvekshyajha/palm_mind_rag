"""Terminal tester for the Palm Mind RAG API. A testing helper, not part of the backend."""
from pathlib import Path

import httpx

BASE = "http://localhost:8000"
MENU = """
==============================
  What do you want to do?
  1  Upload a document
  2  Chat
  3  Quit
=============================="""


def show_error(response: httpx.Response) -> None:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    print(f"\nSomething went wrong ({response.status_code}): {detail}\n")


def upload_flow(client: httpx.Client) -> None:
    print("\nDrag your file into this window (or paste its path) and press Enter.")
    path = Path(input("File: ").strip().strip('"').strip("'"))
    if not path.is_file():
        print(f"\nI can't find that file: {path}\n")
        return

    print("\nHow should the document be split into chunks?")
    print("  1  Recursive (default, splits at paragraphs and sentences)")
    print("  2  Semantic (groups sentences that are about the same topic)")
    strategy = "semantic" if input("Choose 1 or 2 (Enter = 1): ").strip() == "2" else "recursive"

    print("\nUploading... the first upload can take a little while.")
    with path.open("rb") as file:
        response = client.post(
            f"{BASE}/documents/upload",
            files={"file": (path.name, file)},
            data={"chunking_strategy": strategy},
            timeout=300,
        )
    if response.status_code == 201:
        info = response.json()
        print(f"\nDone! {info['filename']} was split into {info['chunk_count']} chunks "
              f"({strategy}) in {info['processing_time_seconds']} seconds.\n")
    else:
        show_error(response)


def looks_like_upload_request(text: str) -> bool:
    words = text.lower()
    return "upload" in words and any(w in words for w in ("doc", "file", "pdf", "txt"))


def chat_flow(client: httpx.Client) -> None:
    print("\nChat started. Ask about your documents, say hi, or book an interview.")
    print("Type /menu to go back, /new for a fresh conversation.\n")
    session_id: str | None = None

    while True:
        text = input("You: ").strip()
        if not text:
            continue
        if text == "/menu":
            return
        if text == "/new":
            session_id = None
            print("Started a new conversation.\n")
            continue
        if looks_like_upload_request(text):  # handled here, because uploading is a separate API
            upload_flow(client)
            continue

        body: dict[str, str] = {"message": text}
        if session_id:
            body["session_id"] = session_id
        response = client.post(f"{BASE}/chat", json=body, timeout=120)
        if response.status_code != 200:
            show_error(response)
            continue

        data = response.json()
        session_id = data["session_id"]
        print(f"\nBot: {data['reply']}")
        for source in data["sources"]:
            page = f", page {source['page']}" if source["page"] else ""
            print(f"     (source: {source['source']}{page})")
        if data["booking_status"] == "completed":
            print("     (booking saved)")
        print(f"     [intent: {data['intent']}]\n")


def main() -> None:
    print("Palm Mind RAG - terminal tester")
    with httpx.Client() as client:
        try:
            client.get(f"{BASE}/docs", timeout=5)
        except httpx.HTTPError:
            print("\nI can't reach the server. Start it first in another terminal with:")
            print("    uvicorn app.main:app\n")
            return

        while True:
            print(MENU)
            choice = input("Your choice: ").strip().lower()
            try:
                if choice in ("1", "upload"):
                    upload_flow(client)
                elif choice in ("2", "chat"):
                    chat_flow(client)
                elif choice in ("3", "quit", "q", "exit"):
                    print("Bye!")
                    return
                else:
                    print("Please type 1, 2 or 3.")
            except httpx.HTTPError as error:
                print(f"\nLost connection to the server ({error}).\n")
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                return


if __name__ == "__main__":
    main()