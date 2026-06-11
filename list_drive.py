import os
import sys
import subprocess

# Ensure google-api-python-client and google-auth are installed
try:
    from googleapiclient.discovery import build
    import google.auth
except ImportError:
    print("Installing required Google API libraries...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "google-api-python-client", "google-auth", "google-auth-httplib2", "google-auth-oauthlib"])
    from googleapiclient.discovery import build
    import google.auth

def list_folder_contents(folder_id):
    print(f"Authenticating and accessing folder ID: {folder_id}...")
    try:
        # Load credentials from environment (Application Default Credentials)
        credentials, project = google.auth.default(
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        
        service = build('drive', 'v3', credentials=credentials)
        
        # Query files in the specific folder
        query = f"'{folder_id}' in parents and trashed = false"
        results = service.files().list(
            q=query,
            pageSize=100,
            fields="nextPageToken, files(id, name, mimeType, size)"
        ).execute()
        
        files = results.get('files', [])
        
        if not files:
            print("No files found in this folder or you might not have access to it.")
            return
        
        print("\nFiles in folder:")
        print("-" * 60)
        for file in files:
            size_kb = f"{int(file.get('size', 0)) / 1024:.1f} KB" if 'size' in file else "N/A"
            print(f"Name: {file['name']}")
            print(f"  ID: {file['id']}")
            print(f"  Type: {file['mimeType']}")
            print(f"  Size: {size_kb}")
            print("-" * 60)
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("\nIf you get a permission error, please make sure:")
        print("1. You ran: gcloud auth application-default login")
        print("2. The account you logged into has access to the Drive folder link you provided.")

if __name__ == "__main__":
    folder_id = "1mjKuJMGamKrUrQhgbdaYxOmu3-O-YnyP"
    list_folder_contents(folder_id)
