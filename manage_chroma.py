import chromadb
import sys
from config import DB_PATH

def main():
    # Initialize Chroma Client
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        print(f"Connected to ChromaDB at: {DB_PATH}")
    except Exception as e:
        print(f"Error initializing client: {e}")
        return

    print("Fetching collections from ChromaDB...")
    
    try:
        collections = client.list_collections()
    except Exception as e:
        print("Error: Failed to connect to ChromaDB or list collections.")
        print(f"Details: {e}")
        sys.exit(1)

    if not collections:
        print("No collections found.")
        return

    print("\nAvailable Collections:")
    for i, collection in enumerate(collections):
        # collection is an object with a .name attribute
        print(f"{i + 1}. {collection.name}")

    try:
        name_to_delete = input('\nEnter the EXACT NAME of the collection to delete (or press Enter to exit): ').strip()
    except KeyboardInterrupt:
        print("\nExiting.")
        return

    if not name_to_delete:
        print("Exiting without changes.")
        return

    confirm = input(f"Are you sure you want to PERMANENTLY DELETE '{name_to_delete}'? (yes/no): ").strip().lower()

    if confirm in ['yes', 'y']:
        try:
            client.delete_collection(name=name_to_delete)
            print(f"\n[SUCCESS] Collection '{name_to_delete}' has been deleted.")
        except Exception as e:
            print(f"\n[ERROR] Failed to delete collection '{name_to_delete}'. {e}")
    else:
        print("Deletion cancelled.")

if __name__ == "__main__":
    main()