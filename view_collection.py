import chromadb
import os
import sys
from config import DB_PATH

def view_collection():
    # Ensure DB path exists
    if not os.path.exists(DB_PATH):
        print(f"Error: Database path '{DB_PATH}' not found.")
        return

    print(f"Connecting to ChromaDB at: {DB_PATH}")
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
    except Exception as e:
        print(f"Failed to initialize ChromaDB client: {e}")
        return

    # List collections
    try:
        collections = client.list_collections()
    except Exception as e:
        print(f"Error listing collections: {e}")
        return

    if not collections:
        print("No collections found in the database.")
        return

    print("\nAvailable Collections:")
    for idx, col in enumerate(collections):
        print(f"{idx + 1}. {col.name}")

    # Optional: Delete a collection
    delete_choice = input("\nDo you want to delete a collection? (y/N): ").strip().lower()
    if delete_choice in ("y", "yes"):
        del_name = input("Enter the exact collection name to delete: ").strip()
        if not del_name:
            print("No collection name provided. Skipping deletion.")
        else:
            confirm = input(f"Type DELETE to permanently remove '{del_name}': ").strip()
            if confirm == "DELETE":
                try:
                    client.delete_collection(name=del_name)
                    print(f"Collection '{del_name}' deleted.")
                except Exception as e:
                    print(f"Error deleting collection '{del_name}': {e}")
            else:
                print("Deletion cancelled.")

    # Select collection
    col_name = input("\nEnter the name of the collection to inspect (default: video_frames_v1): ").strip()
    if not col_name:
        col_name = "video_frames_v1"

    try:
        collection = client.get_collection(name=col_name)
    except Exception as e:
        print(f"Error getting collection '{col_name}': {e}")
        return

    count = collection.count()
    print(f"\nCollection '{col_name}' contains {count} items.")

    if count > 0:
        limit_input = input(f"How many items to view? (default 5): ").strip()
        limit = int(limit_input) if limit_input.isdigit() else 5

        show_embeddings = input("Show embeddings? (y/N): ").strip().lower() in ("y", "yes")
        include = ['metadatas', 'documents']
        if show_embeddings:
            include.append('embeddings')

        results = collection.get(limit=limit, include=include)

        print(f"\n--- First {len(results['ids'])} Items ---")
        for i, item_id in enumerate(results['ids']):
            print(f"\n[ID]: {item_id}")
            print(f"[Metadata]: {results['metadatas'][i]}")
            print(f"[Document]: {results['documents'][i]}")
            if show_embeddings:
                emb = results['embeddings'][i]
                emb_len = len(emb) if emb is not None else 0
                preview = emb[:8] if emb_len else []
                print(f"[Embedding]: len={emb_len} preview={preview}")

if __name__ == "__main__":
    view_collection()
